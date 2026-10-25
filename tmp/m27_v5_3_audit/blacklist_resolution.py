# =====================================================================
# LISTA NEGRA DE LA GUIA DE ARQUETIPOS -> LIGAS CONCRETAS
# ---------------------------------------------------------------------
# FUENTE DE LA PREMISA:
#   docs/ligas/COMPORTAMIENTO_LIGAS_Y_GUIA_ENTRENAMIENTO.md (semáforo de ligas):
#     🔴 R1  Juveniles / Canteras U16-U23      -> "BLACKLIST OBLIGATORIA"
#     🔴 R2  College NCAA                      -> "EXCLUIR DEL DATASET GENERAL"
#     🔴 R3  Regionales / Amistosos / Pretemporada
#     🟡 A1  NBA y ligas de 12 min             -> sub-modelo exclusivo
#     🟡 A2  Femenino Pro                      -> sub-modelo o feature is_women
#     🟡 A3  Playoffs / Finales                -> flags is_playoffs / is_final
#   y el SQL de filtrado recomendado:
#     quarter_duration_minutes = 10 AND is_youth = 0 AND is_college = 0
#     AND competition_type != 'friendly' AND match_count >= 50
#
# QUE HACE:
#   Resuelve cada regla del semáforo a las ligas REALES de la base usando los
#   flags de `leagues_classification` (3.643 filas) y mide el volumen de
#   partidos que cada una representa, para decidir que se descarta del
#   entrenamiento de m27_v5_3 y que queda para un modelo propio.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/blacklist_resolution.py
# =====================================================================

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"
OUT_JSON = ROOT / "tmp" / "m27_v5_3_audit" / "blacklist_resolution.json"
OUT_TXT = ROOT / "tmp" / "m27_v5_3_audit" / "blacklist_resolution.txt"

# Reglas del semáforo -> predicado sobre leagues_classification
REGLAS = [
    ("R1 JUVENILES / CANTERAS U16-U23", "🔴 EXCLUIR",
     "is_youth = 1",
     "BLACKLIST OBLIGATORIA (PROHIBIDO ENTRENAR). Produce puro ruido y penaliza los gradientes de momentum."),
    ("R2 COLLEGE NCAA", "🔴 EXCLUIR",
     "is_college = 1",
     "EXCLUIR DEL DATASET GENERAL. Reloj de 30s y mitades de 20 min: requiere modelo propio."),
    ("R3 AMISTOSOS / TORNEO / PRESEASON", "🔴 EXCLUIR",
     "competition_type IN ('friendly','tournament') OR is_tournament = 1",
     "Regionales, amistosos y pretemporadas: no son espejo de la competencia regular."),
    ("A3 PLAYOFFS / FINALES / ELIMINATORIAS", "🟡 MODELO PROPIO",
     "is_playoffs = 1 OR is_final = 1 OR is_semifinal = 1 OR is_quarterfinal = 1 OR is_relegation = 1",
     "Efecto playoff: -14.6 pts de media, localia 62.3 %. Requiere flag is_playoffs/is_final."),
    ("A1 LIGAS DE 12 MINUTOS", "🟡 MODELO PROPIO",
     "quarter_duration_minutes = 12",
     "Sub-modelo exclusivo de 48 min. YA IMPLEMENTADO en m27_v5_3 (se rechazan en inferencia y no entran al train)."),
    ("A2 FEMENINO ADULTO", "🟡 MODELO PROPIO",
     "is_women = 1 AND is_youth = 0 AND is_college = 0",
     "Sub-modelo femenino o feature is_women. YA IMPLEMENTADO: m27_v5_3 tiene la feature is_women."),
]


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    lc = {r["league"]: dict(r) for r in conn.execute(
        "SELECT * FROM leagues_classification").fetchall()}

    # volumen real de partidos terminados por liga
    vol = {}
    for r in conn.execute(
        "SELECT league, COUNT(*) n FROM matches WHERE status_type='finished' "
        "AND league IS NOT NULL AND league!='' GROUP BY league"
    ):
        vol[r["league"]] = r["n"]
    total = sum(vol.values())
    print(f"Base: {len(vol):,} ligas con partidos terminados | {total:,} partidos")
    print(f"leagues_classification: {len(lc):,} filas "
          f"({len(set(lc) & set(vol)):,} de ellas con partidos)")

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    informe = {}
    for nombre, nivel, pred, nota in REGLAS:
        try:
            rows = conn.execute(
                f"SELECT league, clean_name, gender, age_category, competition_type, "
                f"is_youth, is_college, is_playoffs, is_final, is_tournament, is_relegation, "
                f"quarter_duration_minutes, match_count, finished_match_count, "
                f"avg_total_points, avg_q4_total_points, home_win_pct, blowout_rate, clutch_rate "
                f"FROM leagues_classification WHERE {pred}").fetchall()
        except sqlite3.Error as exc:
            emit(f"ERROR en la regla {nombre}: {exc}")
            continue
        items = []
        for r in rows:
            d = dict(r)
            d["partidos_reales"] = vol.get(r["league"], 0)
            items.append(d)
        items.sort(key=lambda d: -d["partidos_reales"])
        n_part = sum(d["partidos_reales"] for d in items)
        n_lig = sum(1 for d in items if d["partidos_reales"] > 0)

        emit("")
        emit("=" * 104)
        emit(f"{nombre}   [{nivel}]")
        emit(f"  regla: {pred}")
        emit(f"  {nivel}: {nota}")
        emit("=" * 104)
        emit(f"  {n_lig:,} liga(s) con partidos | {n_part:,} partidos "
             f"({n_part * 100 / total:.1f}% del total)")
        emit(f"  {'liga':<62}{'part.':>8}{'m_cnt':>7}{'pts':>7}{'Q4':>7}{'local%':>8}")
        for d in items:
            if d["partidos_reales"] == 0:
                continue
            emit(f"  {d['league'][:60]:<62}{d['partidos_reales']:>8,}{d['match_count'] or 0:>7,}"
                 f"{(d['avg_total_points'] or 0):>7.1f}{(d['avg_q4_total_points'] or 0):>7.1f}"
                 f"{(d['home_win_pct'] or 0) * 100:>7.1f}%")
        informe[nombre] = {"nivel": nivel, "regla": pred, "nota": nota,
                            "ligas": n_lig, "partidos": n_part,
                            "detalle": items}

    # ¿Cuantas ve el modelo hoy? Cruce con el dataset construido
    import pickle
    ds = ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl"
    if ds.exists():
        D = pickle.load(open(ds, "rb"))
        lg_dataset = {m["lg"] for m in D["meta"]}
        emit("")
        emit("=" * 104)
        emit("ESTADO ACTUAL: ¿que de esto esta dentro del dataset de entrenamiento de m27_v5_3?")
        emit("=" * 104)
        for nombre, nivel, pred, nota in REGLAS:
            items = informe[nombre]["detalle"]
            dentro = [d for d in items if d["league"] in lg_dataset]
            n_d = sum(d["partidos_reales"] for d in dentro)
            emit(f"  {nombre:<42} {nivel:<16} dentro del train: {len(dentro):>4} liga(s), "
                 f"{n_d:>8,} partidos")
        emit("")
        emit("  -> Lo que sigue en el train pese a estar en la lista negra es RUIDO NO DESCONTADO.")
    conn.close()

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(informe, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nInforme: {OUT_TXT}\n         {OUT_JSON}")


if __name__ == "__main__":
    main()

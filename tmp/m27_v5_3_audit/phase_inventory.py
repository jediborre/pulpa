# =====================================================================
# INVENTARIO DE FASES DE TORNEO (para revision y decision del usuario)
# ---------------------------------------------------------------------
# PROPOSITO:
#   La premisa de m27_v5_3 era que los partidos de formatos especiales
#   (eliminatorias, copas, championship round, promotion, formativas) se
#   DESCARTARIAN del entrenamiento para quitar ruido, porque para ellos se
#   planeaba un modelo independiente. En la implementacion solo se aplico
#   `not is_toxic_league(liga)` con 16 patrones escritos a mano (5 muertos).
#   Este inventario lista TODAS las variantes de fase que existen de verdad
#   en la base, agrupadas por tipo, con su volumen, para que se decida
#   cuales se descartan.
#
# QUE HACE:
#   1. Lee todas las ligas de `matches` con nº de partidos y rango de fechas.
#   2. Separa (liga base, fase) y clasifica la fase por tipo.
#   3. Lista, por tipo, las ligas base afectadas con sus variantes y volumen.
#   4. Señala aparte los casos atipicos (amistosos,的事实, mislabeled).
#   5. Exporta el inventario completo a JSON y a texto.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/phase_inventory.py
# =====================================================================

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"
OUT_JSON = ROOT / "tmp" / "m27_v5_3_audit" / "phase_inventory.json"
OUT_TXT = ROOT / "tmp" / "m27_v5_3_audit" / "phase_inventory.txt"

# --- Clasificacion de la fase (por prioridad) ---------------------------------
# "marcadores regulares": no son fases especiales, son etiquetas de grupo,
# region, genero o temporada dentro de la competition regular.
REGULAR_MARKERS = [
    "group", "groups", "south", "east", "west", "north", "norte", "sur", "sul",
    "nord", "sud", "oeste", "este", "seccion", "sección", "women", "women's",
    "men", "fase i", "fase ii", "fase 1", "fase 2", "1ª fase", "2ª fase",
    "primeira fase", "faza i", "faza ii", "runda", "grundspil", "reguliarusis",
    "temporada", "split", "apertura", "clausura", "zk", "conference", "division",
    "k.k.m", "all", "aller", "retour", "round robin", "relegation group",
    "promotion group", "poule", "gironе", "girone", "klasse", "classifica",
]

PHASE_RULES = [
    ("AMISTOSOS", ["friendly", "friendlies", "amistoso", "exhibition", "exhibition games"]),
    ("ELIMINATORIA", ["playoff", "play off", "play-off", "playoffs", "knockout",
                      "elimination", "eliminatoria", "fase final", "final phase",
                      "superfinal", "octavos", "cuartos de final", "semifinal",
                      "quarter-final", "semi-final", "1/4 final", "1/2 final"]),
    ("COPA", ["cup", "copa", "poklop", "pokal", "coupe", "trofeo", "trophy"]),
    ("PRELIMINAR", ["qualifying round", "qualifier", "qualifiers", "preliminar",
                    "classification round", "pre-season", "presason"]),
    ("SEGUNDA FASE", ["championship", "ronda de campeonato", "intergroup",
                      "inter group", "inter-group", "second round", "second phase",
                      "second round of", "fase 2", "2ª fase de campeonato"]),
    ("PROMOCION", ["promotion", "relegation", "playout", "play-out", "descenso",
                   "ascenso", "placement", "placement round", "manutenção",
                   "intermediate phase", "intermediate", "runda de promocion",
                   "classificação", "classificacao"]),
]


def split_league(name: str) -> tuple[str, str]:
    s = str(name or "").strip()
    if "," in s:
        base, _, rest = s.partition(",")
        return base.strip(), rest.strip()
    return s, ""


def classify(name: str) -> tuple[str, str, str]:
    """(liga base, fase, tipo de fase)."""
    base, phase = split_league(name)
    ph = phase.lower()
    bl = base.lower()
    for cat, marks in PHASE_RULES:
        for mk in marks:
            if mk in ph:
                return base, phase, cat
    # Some leagues carry the format in the base name itself.
    for cat, marks in PHASE_RULES:
        for mk in marks:
            if mk in bl:
                return base, phase, cat
    if not phase:
        return base, phase, "REGULAR"
    for mk in REGULAR_MARKERS:
        if mk in ph:
            return base, phase, "REGULAR"
    return base, phase, "OTRA"


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT m.league AS league, COUNT(*) n, MIN(m.date) d0, MAX(m.date) d1
        FROM matches m
        WHERE m.league IS NOT NULL AND m.league != ''
          AND m.status_type = 'finished' AND m.date IS NOT NULL
        GROUP BY m.league
        """
    ).fetchall()
    conn.close()

    info = {}
    for r in rows:
        base, phase, cat = classify(r["league"])
        d = info.setdefault(r["league"], {"base": base, "phase": phase, "cat": cat,
                                           "n": 0, "d0": r["d0"], "d1": r["d1"], "variants": []})
        d["n"] += r["n"]
        d["variants"].append({"league": r["league"], "n": r["n"]})

    por_cat = defaultdict(list)
    for lg, d in info.items():
        por_cat[d["cat"]].append((d["base"], d, lg))

    out_lines = []
    def emit(s=""):
        print(s)
        out_lines.append(s)

    emit("=" * 108)
    emit("INVENTARIO DE FASES DE TORNEO EN LA BASE (partidos terminados)")
    emit("=" * 108)
    total_matches = sum(d["n"] for d in info.values())
    bases = {d["base"] for d in info.values()}
    emit(f"  variantes de liga: {len(info):,} | ligas base: {len(bases):,} | partidos: {total_matches:,}")
    emit("")
    emit(f"  {'tipo de fase':<16}{'variantes':>10}{'ligas base':>11}{'partidos':>12}{'%':>7}")
    for cat in sorted(por_cat, key=lambda c: -sum(x[1]["n"] for x in por_cat[c])):
        items = por_cat[cat]
        n = sum(x[1]["n"] for x in items)
        nb = len({x[0] for x in items})
        emit(f"  {cat:<16}{len(items):>10,}{nb:>11,}{n:>12,}{n * 100 / total_matches:>6.1f}%")

    for cat in ("AMISTOSOS", "ELIMINATORIA", "COPA", "PRELIMINAR", "SEGUNDA FASE", "PROMOCION", "OTRA"):
        items = por_cat.get(cat, [])
        if not items:
            continue
        # Agrupar por liga base
        by_base = defaultdict(list)
        for base, d, lg in items:
            by_base[base].append(d)
        emit("")
        emit("=" * 108)
        emit(f"{cat}  —  {len(by_base)} liga(s) base, {len(items)} variantes, "
             f"{sum(x[1]['n'] for x in items):,} partidos")
        emit("=" * 108)
        ordered = sorted(by_base.items(), key=lambda kv: -sum(d["n"] for d in kv[1]))
        for base, ds in ordered:
            n = sum(d["n"] for d in ds)
            variants = sorted({d["phase"] for d in ds if d["phase"]})
            d0 = min(d["d0"] for d in ds)
            d1 = max(d["d1"] for d in ds)
            emit(f"  {base}  ({n:,} part., {d0} → {d1})")
            for ph in sorted(variants)[:8]:
                sub = [d for d in ds if d["phase"] == ph]
                emit(f"      - {ph}  ({sum(d['n'] for d in sub):,} part.)")
            if len(variants) > 8:
                emit(f"      ... y {len(variants) - 8} variante(s) mas")

    emit("")
    emit("=" * 108)
    emit("CASOS ATIPICOS (revisar a mano: no encajan en ninguna categoria)")
    emit("=" * 108)
    atip = [d for d in info.values() if d["cat"] in ("OTRA", "REGULAR")
            and any(k in (d["base"] + " " + d["phase"]).lower()
                    for k in ("friendly", "final", "superfinal", "playoff", "cup", "tour", "exhibition"))]
    for d in sorted(atip, key=lambda x: -x["n"])[:40]:
        emit(f"  {d['base'] + (', ' + d['phase'] if d['phase'] else ''):<70} {d['n']:>7,} part.")

    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(
        {lg: d for lg, d in info.items()}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nInventario completo: {OUT_TXT}  /  {OUT_JSON}")


if __name__ == "__main__":
    main()

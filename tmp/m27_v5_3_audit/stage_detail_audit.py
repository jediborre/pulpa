# =====================================================================
# FASE NORMALIZADA (stage_detail) + AUDITORIA DE LIGAS DE 12 MINUTOS
# ---------------------------------------------------------------------
# HALLAZGO:
#   `leagues_classification.stage_detail` ya normaliza la fase de torneo en
#   los 10 buckets, MULTILINGUE (el campo `stage` guarda el nombre original:
#   'Playoffs', 'Półfinały', 'Četrtfinale', 'Плэйофф', 'Izlučne utakmice',
#   'FİNAL', 'Klasman Maçları'... -> todos al mismo bucket). Es la fuente
#   correcta, ya que la.detecta el pipeline de clasificacion.
#
# QUE HACE:
#   1. Volcumen de partidos y de ligas por cada bucket de stage_detail.
#   2. Cuanto de cada bucket esta DENTRO del dataset de entrenamiento de
#      m27_v5_3 (lo que la premisa de "descartar" deberia haber quitado).
#   3. AUDITORIA DE 12 MINUTOS: lista las ligas marcadas q_dur=12 y busca las
#      que NO lo estan pero tienen la firma de 12 min (Q4 de 50+ puntos).
#   4. Las ligas "championship" que el usuario vio (bucket Final/Championship).
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/stage_detail_audit.py
# =====================================================================

import pickle
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"


def block(t: str) -> None:
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    vol = {}
    for r in conn.execute(
        "SELECT league, COUNT(*) n FROM matches WHERE status_type='finished' "
        "AND league IS NOT NULL AND league!='' GROUP BY league"
    ):
        vol[r["league"]] = r["n"]
    total = sum(vol.values())

    ds = ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl"
    en_train = set()
    if ds.exists():
        D = pickle.load(open(ds, "rb"))
        en_train = {m["lg"] for m in D["meta"]}
    print(f"Base: {len(vol):,} ligas | {total:,} partidos terminados")
    print(f"Dataset de entrenamiento m27_v5_3: {len(en_train):,} ligas\n")

    # ------------------------------------------------------------------
    block("1. FASE NORMALIZADA (stage_detail): volumen y presencia en el train")
    print(f"  {'bucket (stage_detail)':<28}{'ligas':>7}{'part.':>10}{'% base':>8}"
          f"{'en train':>10}{'part. train':>13}")
    agg = defaultdict(lambda: {"lig": 0, "n": 0, "lig_tr": 0, "n_tr": 0, "ejemplos": []})
    for r in conn.execute(
        "SELECT league, stage, stage_detail, quarter_duration_minutes, "
        "avg_total_points, avg_q4_total_points, match_count, is_youth, is_college, "
        "competition_type, is_tournament, is_playoffs, is_final "
        "FROM leagues_classification"
    ):
        d = r["stage_detail"] or "(sin clasificar)"
        a = agg[d]
        n = vol.get(r["league"], 0)
        a["lig"] += 1
        a["n"] += n
        if n:
            a["ejemplos"].append((r["league"], n, r["stage"]))
        if r["league"] in en_train:
            a["lig_tr"] += 1
            a["n_tr"] += n
    for d, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {d:<28}{a['lig']:>7,}{a['n']:>10,}{a['n'] * 100 / total:>7.1f}%"
              f"{a['lig_tr']:>10,}{a['n_tr']:>13,}")

    # ------------------------------------------------------------------
    block("2. LAS 'CHAMPIONSHIP' (bucket Final / Championship) - las que viste")
    ej = sorted(agg["Final / Championship"]["ejemplos"], key=lambda x: -x[1])
    for lg, n, stage in ej[:30]:
        print(f"  {lg[:58]:<60}{n:>8,} part.   stage='{stage}'")
    print(f"  ... total bucket: {agg['Final / Championship']['lig']:,} ligas, "
          f"{agg['Final / Championship']['n']:,} partidos")

    # ------------------------------------------------------------------
    block("3. AUDITORIA DE LIGAS DE 12 MINUTOS")
    doce = [r for r in conn.execute(
        "SELECT league, stage_detail, avg_total_points, avg_q4_total_points, "
        "match_count FROM leagues_classification WHERE quarter_duration_minutes = 12")]
    print(f"  marcadas q_dur=12: {len(doce)} ligas")
    for r in sorted(doce, key=lambda x: -vol.get(x["league"], 0)):
        print(f"    {r['league'][:58]:<60}{vol.get(r['league'], 0):>8,} part.  "
              f"pts={(r['avg_total_points'] or 0):>6.1f} Q4={(r['avg_q4_total_points'] or 0):>5.1f}")

    print("\n  Buscando 12 min NO marcados (firma: Q4 >= 46 pts y total >= 185):")
    candidatas = []
    for r in conn.execute(
        "SELECT league, stage_detail, avg_total_points, avg_q4_total_points, match_count, "
        "quarter_duration_minutes FROM leagues_classification "
        "WHERE (quarter_duration_minutes IS NULL OR quarter_duration_minutes != 12) "
        "AND avg_q4_total_points >= 46 AND avg_total_points >= 185 AND match_count >= 20"
    ):
        n = vol.get(r["league"], 0)
        if n >= 20:
            candidatas.append((r["league"], n, r["avg_total_points"], r["avg_q4_total_points"],
                               r["quarter_duration_minutes"]))
    for lg, n, pts, q4, qd in sorted(candidatas, key=lambda x: -x[1]):
        marca = "  <-- 12 min NO MARCADA" if qd != 12 else ""
        print(f"    {lg[:58]:<60}{n:>8,} part.  pts={pts:>6.1f} Q4={q4:>5.1f} q_dur={qd}{marca}")
    if not candidatas:
        print("    (ninguna)")

    # ------------------------------------------------------------------
    block("4. FASE vs TIPO DE COMPETICION (competition_type)")
    print(f"  {'competition_type':<22}{'ligas':>7}{'part.':>10}")
    for r in conn.execute(
        "SELECT COALESCE(competition_type,'(null)') ct, COUNT(*) lg FROM leagues_classification "
        "GROUP BY ct ORDER BY lg DESC"
    ):
        print(f"  {r['ct']:<22}{r['lg']:>7,}")

    # ------------------------------------------------------------------
    block("5. QUE HAY QUE DESCARTAR SEGUN LA PREMISA, RESUELTO CON stage_detail")
    premise = {
        "🔴 R1 Juveniles": "is_youth = 1",
        "🔴 R2 College NCAA": "is_college = 1",
        "🔴 R3 Amistosos/Torneos": "competition_type IN ('friendly','tournament') OR is_tournament=1",
        "🟡 A3 Playoffs/Finales": "stage_detail IN ('Playoffs','Final / Championship',"
                                  "'Semifinal / Final Four','Quarterfinal','Play-in')",
        "🟡 A3b Elimination->clasif": "stage_detail IN ('Relegation / Playout','Bronze / 3rd Place')",
        "🟡 A3c Copas": "stage_detail = 'Cup Stage'",
        "🟡 A3d Segunda fase/grupos": "stage_detail = 'Group Stage'",
        "🟡 A1 12 minutos": "quarter_duration_minutes = 12",
    }
    print(f"  {'regla':<30}{'ligas':>7}{'part.':>10}{'en train (lig)':>18}{'part. train':>14}")
    for nombre, pred in premise.items():
        try:
            rows = conn.execute(
                f"SELECT league FROM leagues_classification WHERE {pred}").fetchall()
        except sqlite3.Error as exc:
            print(f"  {nombre:<30} ERROR {exc}")
            continue
        ligs = [r["league"] for r in rows]
        n = sum(vol.get(l, 0) for l in ligs)
        n_tr = sum(vol.get(l, 0) for l in ligs if l in en_train)
        l_tr = sum(1 for l in ligs if l in en_train)
        print(f"  {nombre:<30}{len(ligs):>7,}{n:>10,}{l_tr:>18,}{n_tr:>14,}")
    conn.close()


if __name__ == "__main__":
    main()

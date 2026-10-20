# =====================================================================
# SEGURIDAD DE LA MIGRACION legacy -> v2
#   1. ¿Discrepan los valores en los partidos comunes? (si sí, no se puede
#      sobrescribir a ciegas)
#   2. ¿Cuándo son los partidos con cuartos incompletos? (¿vale la pena
#      descargarlos?)
#   3. Coste estimado de la migración y del backfill opcional.
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row


def one(sql):
    return conn.execute(sql).fetchone()[0]


print("1. DISCREPANCIA entre legacy y v2 en los partidos comunes")
PIVOT = """
SELECT q.match_id,
  MAX(CASE WHEN q.quarter='Q1' THEN q.home END) q1h, MAX(CASE WHEN q.quarter='Q1' THEN q.away END) q1a,
  MAX(CASE WHEN q.quarter='Q2' THEN q.home END) q2h, MAX(CASE WHEN q.quarter='Q2' THEN q.away END) q2a,
  MAX(CASE WHEN q.quarter='Q3' THEN q.home END) q3h, MAX(CASE WHEN q.quarter='Q3' THEN q.away END) q3a,
  MAX(CASE WHEN q.quarter='Q4' THEN q.home END) q4h, MAX(CASE WHEN q.quarter='Q4' THEN q.away END) q4a
FROM quarter_scores q GROUP BY q.match_id
"""
conn.execute("DROP TABLE IF EXISTS temp._pivot")
conn.execute(f"CREATE TEMP TABLE _pivot AS {PIVOT}")
common = one("SELECT COUNT(*) FROM _pivot p JOIN quarter_scores_v2 v ON v.match_id=p.match_id")
print(f"  partidos comunes: {common:,}")
diffs = 0
ejemplos = []
for r in conn.execute(
    "SELECT p.match_id, p.q1h,p.q1a,p.q2h,p.q2a,p.q3h,p.q3a,p.q4h,p.q4a, "
    "v.q1_home,v.q1_away,v.q2_home,v.q2_away,v.q3_home,v.q3_away,v.q4_home,v.q4_away "
    "FROM _pivot p JOIN quarter_scores_v2 v ON v.match_id=p.match_id"
):
    a = (r["q1h"], r["q1a"], r["q2h"], r["q2a"], r["q3h"], r["q3a"], r["q4h"], r["q4a"])
    b = (r["q1_home"], r["q1_away"], r["q2_home"], r["q2_away"],
         r["q3_home"], r["q3_away"], r["q4_home"], r["q4_away"])
    if any(x is not None and y is not None and x != y for x, y in zip(a, b)):
        diffs += 1
        if len(ejemplos) < 5:
            ejemplos.append((r["match_id"], a, b))
print(f"  partidos con ALGUN valor distinto: {diffs:,} ({diffs * 100 / max(common, 1):.2f} %)")
for mid, a, b in ejemplos:
    print(f"     {mid}: legacy={a} v2={b}")

print()
print("2. ¿CUÁNDO SON LOS PARTIDOS CON CUARTOS INCOMPLETOS? (¿vale la pena bajarlos?)")
for r in conn.execute(
    "SELECT substr(m.date,1,4) yr, COUNT(*) n FROM (SELECT match_id FROM quarter_scores "
    "GROUP BY match_id HAVING COUNT(*)<4) g JOIN matches m ON m.match_id=g.match_id "
    "GROUP BY yr ORDER BY yr"
):
    print(f"   {r['yr']}: {r['n']:,} partidos")
print(f"   total a recuperar: {one('SELECT COUNT(*) FROM (SELECT match_id FROM quarter_scores GROUP BY match_id HAVING COUNT(*)<4)'):,}"
      f"  -> a 1 req/s son ~{one('SELECT COUNT(*) FROM (SELECT match_id FROM quarter_scores GROUP BY match_id HAVING COUNT(*)<4)') // 60} min")

print()
print("3. COSTE DE LA MIGRACION (pivot SQL, sin API)")
print(f"   legacy: {one('SELECT COUNT(*) FROM quarter_scores'):,} filas -> "
      f"{one('SELECT COUNT(DISTINCT match_id) FROM quarter_scores'):,} partidos")
print("   INSERT INTO quarter_scores_v2 SELECT ... FROM (pivot) -> una pasada, segundos")
print(f"   v2 pasaria de {one('SELECT COUNT(*) FROM quarter_scores_v2'):,} a "
      f"~{one('SELECT COUNT(DISTINCT match_id) FROM quarter_scores'):,} filas")
conn.close()

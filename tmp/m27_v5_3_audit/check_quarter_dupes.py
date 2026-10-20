# =====================================================================
# DUPLICADOS en la tabla legacy (afectan al MAX(CASE...) del pivot)
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


print("duplicados (match_id, quarter) en quarter_scores:",
      one("SELECT COUNT(*) FROM (SELECT match_id, quarter, COUNT(*) c FROM quarter_scores "
          "GROUP BY match_id, quarter HAVING c>1)"))
print("  partidos afectados:",
      one("SELECT COUNT(DISTINCT match_id) FROM (SELECT match_id FROM quarter_scores "
          "GROUP BY match_id, quarter HAVING COUNT(*)>1)"))
print()
print("valores distintos dentro de un duplicado (¿correcciones o basura?):")
for r in conn.execute(
    "SELECT q.match_id, q.quarter, COUNT(DISTINCT q.home||'-'||q.away) v "
    "FROM quarter_scores q GROUP BY q.match_id, q.quarter HAVING COUNT(*)>1 LIMIT 5"
):
    print("   ", dict(r))
print()
print("distribucion de filas por partido en la legacy:")
for r in conn.execute(
    "SELECT n_filas, COUNT(*) partidos FROM (SELECT match_id, COUNT(*) n_filas FROM quarter_scores "
    "GROUP BY match_id) GROUP BY n_filas ORDER BY n_filas"
):
    print(f"   {r['n_filas']} filas -> {r['partidos']:,} partidos")
print()
print("valores de la columna quarter:")
for r in conn.execute("SELECT quarter, COUNT(*) n FROM quarter_scores GROUP BY quarter ORDER BY n DESC"):
    print(f"   {r['quarter']:<8}{r['n']:>10,}")
print()
print("partidos con Q4 ausente (3 filas) que se podrian recuperar de la API:", one(
    "SELECT COUNT(*) FROM (SELECT match_id FROM quarter_scores GROUP BY match_id HAVING COUNT(*)=3)"))
print("de esos, con Q3 completo:", one(
    "SELECT COUNT(*) FROM (SELECT match_id, SUM(CASE WHEN quarter='Q3' AND home IS NOT NULL "
    "AND away IS NOT NULL THEN 1 ELSE 0 END) c FROM quarter_scores GROUP BY match_id "
    "HAVING COUNT(*)=3 AND c=1)"))
conn.close()

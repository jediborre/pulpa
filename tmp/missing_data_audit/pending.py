"""
tmp/missing_data_audit/pending.py
Proposito: Calcular, por tabla, cuantos partidos estan pendientes (falta la tabla y NO
esta en detail_404) y la union total, para verificar el numero del backfill.
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "matches.db")
TABLES = ["play_by_play", "graph_points", "match_events", "match_h2h",
          "team_statistics", "lineups", "player_stats", "match_odds", "team_strength"]
print(f"{'tabla':<18} {'missing':>10} {'en404':>8} {'pendiente':>10}")
for t in TABLES:
    miss = con.execute(f"SELECT COUNT(*) FROM matches m WHERE NOT EXISTS (SELECT 1 FROM {t} x WHERE x.match_id=m.match_id)").fetchone()[0]
    in404 = con.execute(f"SELECT COUNT(*) FROM matches m WHERE NOT EXISTS (SELECT 1 FROM {t} x WHERE x.match_id=m.match_id) AND EXISTS (SELECT 1 FROM detail_404 d WHERE d.match_id=m.match_id AND d.table_name='{t}')").fetchone()[0]
    print(f"{t:<18} {miss:>10} {in404:>8} {miss-in404:>10}")
cond = " OR ".join(
    f"(NOT EXISTS (SELECT 1 FROM {t} x WHERE x.match_id=m.match_id) "
    f"AND NOT EXISTS (SELECT 1 FROM detail_404 d WHERE d.match_id=m.match_id AND d.table_name='{t}'))"
    for t in TABLES
)
union = con.execute(f"SELECT COUNT(*) FROM matches m WHERE ({cond})").fetchone()[0]
print(f"\nUNION pendiente: {union}")
con.close()

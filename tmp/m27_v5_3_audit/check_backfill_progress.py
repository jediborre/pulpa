# =====================================================================
# ESTADO DEL BACKFILL DE team_stats_v3
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("1) ESTADO:")
for r in conn.execute("""SELECT status, COUNT(*) n, SUM(n_rows) filas FROM team_stats_backfill_state
                         GROUP BY status ORDER BY n DESC"""):
    print(f"   {r['status']:<16} {r['n']:>7,} partidos | {r['filas'] or 0:>9,} filas de stats")
tot = conn.execute("SELECT COUNT(*) n FROM team_stats_v3").fetchone()["n"]
print(f"   team_stats_v3: {tot:,} filas")
print()
print("2) PARTIDOS CON DESGLOSE POR CUARTO (3Q):")
n3 = conn.execute("""SELECT COUNT(DISTINCT match_id) n FROM team_stats_v3
                     WHERE period='3Q'""").fetchone()["n"]
nall = conn.execute("""SELECT COUNT(DISTINCT match_id) n FROM team_stats_v3
                       WHERE period='ALL'""").fetchone()["n"]
print(f"   con 3Q: {n3:,} | con ALL: {nall:,}")
print()
print("3) LIGAS YA DESCARGADAS (con su cobertura):")
for r in conn.execute("""
    SELECT m.league, COUNT(DISTINCT t.match_id) partidos,
           SUM(CASE WHEN t.period='3Q' THEN 1 ELSE 0 END) filas_3q
    FROM team_stats_v3 t JOIN matches m ON CAST(m.match_id AS INTEGER)=t.match_id
    GROUP BY m.league ORDER BY partidos DESC LIMIT 20"""):
    print(f"   {r['league'][:44]:<46} {r['partidos']:>5} partidos | filas 3Q: {r['filas_3q']:>6}")
conn.close()

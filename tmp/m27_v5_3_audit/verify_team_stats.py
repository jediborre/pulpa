# =====================================================================
# VERIFICACION DE LOS DATOS DESCARGADOS EN team_stats_v3
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("1) FILAS POR PERIODO Y STAT:")
for r in conn.execute("""SELECT period, stat_name, COUNT(*) n FROM team_stats_v3
                         GROUP BY period, stat_name ORDER BY period, n DESC"""):
    print(f"   {r['period']:<5} {r['stat_name']:<24} {r['n']:>6,}")

print()
print("2) EJEMPLO DE UN PARTIDO:")
mid = conn.execute("SELECT match_id FROM team_stats_v3 LIMIT 1").fetchone()["match_id"]
for r in conn.execute("""SELECT period, stat_name, home_value, away_value FROM team_stats_v3
                         WHERE match_id=? ORDER BY period, stat_name""", (mid,)):
    print(f"   {r['period']:<5} {r['stat_name']:<24} {r['home_value']} - {r['away_value']}")

print()
print("3) ¿PARTIDOS CON DESGLOSE POR CUARTO (1Q-3Q)?")
n_total = conn.execute("SELECT COUNT(DISTINCT match_id) n FROM team_stats_v3").fetchone()["n"]
n_cuarto = conn.execute("""SELECT COUNT(DISTINCT match_id) n FROM team_stats_v3
                            WHERE period='3Q' AND stat_name='Fouls'""").fetchone()["n"]
n_all = conn.execute("""SELECT COUNT(DISTINCT match_id) n FROM team_stats_v3
                         WHERE period='ALL' AND stat_name='Fouls'""").fetchone()["n"]
print(f"   partidos con stats: {n_total:,} | con 3Q (faltas): {n_cuarto:,} "
      f"({n_cuarto * 100 / max(n_total, 1):.0f} %) | con ALL: {n_all:,}")

print()
print("4) TAMAÑO DE LA TABLA:")
pg = conn.execute("PRAGMA page_count").fetchone()[0]
ps = conn.execute("PRAGMA page_size").fetchone()[0]
print(f"   base de datos: {pg * ps / 1024 / 1024:.0f} MB")
conn.close()

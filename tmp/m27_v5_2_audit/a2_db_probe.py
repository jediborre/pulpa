"""
tmp/m27_v5_2_audit/a2_db_probe.py
Sondeo de matches.db: tablas _v3 de monitoreo (rendimiento live real), esquema de match_odds/team_strength,
y semantica de match_events.time_seconds/points. Solo lectura.
"""
import sqlite3, sys
sys.stdout.reconfigure(encoding="utf-8")
c = sqlite3.connect("file:matches.db?mode=ro", uri=True, timeout=30)
cur = c.cursor()
tabs = [r[0] for r in cur.execute("select name from sqlite_master where type='table' order by 1")]
print(tabs)
for t in tabs:
    if "v3" in t or "monitor" in t or "eval" in t:
        n = cur.execute(f"select count(*) from {t}").fetchone()[0]
        cols = [r[1] for r in cur.execute(f"pragma table_info({t})")]
        print(t, n, cols)

"""
Ubicación original: scratch/check_ft_failed.py
Propósito / Qué hacía:
Diagnóstico de partidos marcados como FT donde falló la captura final de marcadores.
"""

import sqlite3
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

conn = sqlite3.connect('matches.db')
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT match_id, home_team, away_team, league, status, skip_reason, event_date "
    "FROM bet_monitor_schedule_v2 "
    "WHERE skip_reason = 'final_fetch_failed' "
    "ORDER BY event_date DESC LIMIT 100"
).fetchall()

print(f"Partidos con final_fetch_failed: {len(rows)}")

# Group by date
by_date = {}
for r in rows:
    d = r['event_date']
    by_date.setdefault(d, []).append(r)

for d in sorted(by_date.keys(), reverse=True):
    print(f"\n  === {d} ({len(by_date[d])} partidos) ===")
    for r in by_date[d]:
        print(f"    {r['home_team']} vs {r['away_team']} | {r['league']}")

"""
Ubicación original: scratch/get_ft_ids.py
Propósito / Qué hacía:
Obtiene la lista de IDs de partidos finalizados FT que requieren liquidación.
"""

import sqlite3
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

conn = sqlite3.connect('matches.db')
conn.row_factory = sqlite3.Row

# Tomar los primeros 5 partidos fallidos del 27 con su match_id
rows = conn.execute(
    "SELECT match_id, home_team, away_team, league FROM bet_monitor_schedule_v2 "
    "WHERE skip_reason = 'final_fetch_failed' AND event_date = '2026-05-27' "
    "LIMIT 5"
).fetchall()

for r in rows:
    print(f"{r['match_id']} | {r['home_team']} vs {r['away_team']}")

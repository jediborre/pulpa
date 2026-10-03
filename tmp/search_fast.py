"""
Ubicación original: scratch/search_fast.py
Propósito / Qué hacía:
Búsqueda rápida de equipos y partidos por prefijo de texto en SQLite.
"""

import sqlite3

db_path = r"c:\Users\App\Desktop\pulpa\matches.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

teams = ["Amambay", "Félix", "Cavaliers", "Knicks", "Manatí", "Santurce"]

print("=== SEARCHING IN 'matches' ===")
for t in teams:
    cursor.execute("""
        SELECT match_id, date, home_team, away_team, league 
        FROM matches 
        WHERE home_team LIKE ? OR away_team LIKE ?
    """, (f"%{t}%", f"%{t}%"))
    rows = cursor.fetchall()
    if rows:
        print(f"\nMatches matching '{t}':")
        for r in rows:
            print("  ", dict(r))

print("\n=== SEARCHING IN 'bet_monitor_schedule_v2' ===")
for t in teams:
    cursor.execute("""
        SELECT match_id, event_date, home_team, away_team, league, status, skip_reason 
        FROM bet_monitor_schedule_v2 
        WHERE home_team LIKE ? OR away_team LIKE ?
    """, (f"%{t}%", f"%{t}%"))
    rows = cursor.fetchall()
    if rows:
        print(f"\nSchedule V2 matching '{t}':")
        for r in rows:
            print("  ", dict(r))

conn.close()

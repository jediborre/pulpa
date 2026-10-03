"""
Ubicación original: scratch/search_matches.py
Propósito / Qué hacía:
Búsqueda parametrizada de partidos por fecha y nombres de equipos.
"""

import sqlite3

db_path = r"c:\Users\App\Desktop\pulpa\match\matches.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# List all tables to make sure we don't miss anything
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row["name"] for row in cursor.fetchall()]

teams = ["Amambay", "Félix", "Cavaliers", "Knicks", "Manatí", "Santurce"]

print("Searching matches table:")
if "matches" in tables:
    for t in teams:
        cursor.execute("SELECT match_id, date, home_team, away_team, league FROM matches WHERE home_team LIKE ? OR away_team LIKE ?", (f"%{t}%", f"%{t}%"))
        rows = cursor.fetchall()
        for r in rows:
            print("  [matches]", dict(r))

print("\nSearching bet_monitor_schedule_v2 table:")
if "bet_monitor_schedule_v2" in tables:
    for t in teams:
        cursor.execute("SELECT match_id, event_date, home_team, away_team, league, status, skip_reason FROM bet_monitor_schedule_v2 WHERE home_team LIKE ? OR away_team LIKE ?", (f"%{t}%", f"%{t}%"))
        rows = cursor.fetchall()
        for r in rows:
            print("  [schedule_v2]", dict(r))

print("\nSearching other tables for Félix Pérez or Amambay:")
for table in tables:
    if table in ["matches", "bet_monitor_schedule_v2"]:
        continue
    try:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [c[1] for c in cursor.fetchall()]
        search_cols = [col for col in cols if any(n in col.lower() for n in ["team", "home", "away", "match"])]
        if search_cols:
            for t in ["Amambay", "Félix", "Manatí", "Santurce"]:
                clauses = [f"{col} LIKE '%{t}%'" for col in search_cols]
                query = f"SELECT * FROM {table} WHERE " + " OR ".join(clauses)
                cursor.execute(query)
                rows = cursor.fetchall()
                if rows:
                    print(f"  [{table}] found {len(rows)} matching rows for '{t}'")
                    for row in rows[:3]:
                        print("    ", dict(row))
    except Exception as e:
        pass

conn.close()

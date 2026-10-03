"""
Ubicación original: scratch/check_match_status.py
Propósito / Qué hacía:
Comprobación del estado en vivo de partidos monitoreados.
"""

import sqlite3

db_path = r"c:\Users\App\Desktop\pulpa\match\matches.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

match_id = "16208417"

print(f"=== CHECKING DATABASE FOR MATCH {match_id} ===")

cursor.execute("SELECT * FROM bet_monitor_schedule_v2 WHERE match_id = ?", (match_id,))
row = cursor.fetchone()
if row:
    print("\nSchedule row:")
    print("  ", dict(row))
else:
    print("\nNo schedule row found in bet_monitor_schedule_v2!")

cursor.execute("SELECT * FROM bet_monitor_log_v2 WHERE match_id = ?", (match_id,))
rows = cursor.fetchall()
if rows:
    print(f"\nLog rows ({len(rows)} found):")
    for r in rows:
        r_dict = dict(r)
        r_dict.pop("raw_json", None)
        r_dict.pop("inference_json", None)
        print("  ", r_dict)
else:
    print("\nNo log rows found in bet_monitor_log_v2!")

conn.close()

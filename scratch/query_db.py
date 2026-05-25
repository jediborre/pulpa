# -*- coding: utf-8 -*-
import sqlite3
import sys
from pathlib import Path

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.database.connection import get_real_db_path

def query_database():
    db_path = get_real_db_path()
    print(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # 1. Search in schedules
    print("\n--- SCHEDULE MATCHES ---")
    cursor = conn.execute("SELECT * FROM bet_monitor_schedule_v2 WHERE home_team LIKE '%Toros%' OR away_team LIKE '%Toros%' OR home_team LIKE '%Piratas%' OR away_team LIKE '%Piratas%'")
    rows = cursor.fetchall()
    if not rows:
        print("No schedule found for 'Toros' or 'Piratas'.")
    for r in rows:
        print(dict(r))
        match_id = r["match_id"]
        
        # 2. Search in logs
        print(f"\n--- LOGS FOR MATCH {match_id} ---")
        log_cursor = conn.execute("SELECT * FROM bet_monitor_log_v2 WHERE match_id = ?", (match_id,))
        log_rows = log_cursor.fetchall()
        for lr in log_rows:
            d = dict(lr)
            d.pop("raw_json", None)
            d.pop("inference_json", None)
            print(d)

    print("\n--- ALL RECENT LOGS ---")
    cursor = conn.execute("SELECT id, match_id, model_version, picked_side, signal_type, result, confidence FROM bet_monitor_log_v2 ORDER BY id DESC LIMIT 20")
    for r in cursor.fetchall():
        print(dict(r))

    conn.close()

if __name__ == "__main__":
    query_database()

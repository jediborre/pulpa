# -*- coding: utf-8 -*-
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.database.connection import get_real_db_path

def main():
    db_path = get_real_db_path()
    print(f"Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    match_id = "16208417"
    
    # Check current state
    print("\n--- BEFORE RESET ---")
    row = conn.execute("SELECT status, final_fetched, final_fetch_at FROM bet_monitor_schedule_v2 WHERE match_id = ?", (match_id,)).fetchone()
    if row:
        print("Schedule:", dict(row))
    else:
        print("Schedule row not found!")
        
    logs = conn.execute("SELECT id, model_version, result FROM bet_monitor_log_v2 WHERE match_id = ?", (match_id,)).fetchall()
    for l in logs:
        print("Log:", dict(l))
        
    # Execute Reset
    print("\nResetting match in database...")
    with conn:
        conn.execute("""
            UPDATE bet_monitor_schedule_v2
            SET status = 'pending',
                skip_reason = NULL,
                final_fetched = 0,
                final_fetch_at = NULL,
                updated_at = datetime('now')
            WHERE match_id = ?
        """, (match_id,))
        
        conn.execute("""
            UPDATE bet_monitor_log_v2
            SET result = 'pending'
            WHERE match_id = ?
        """, (match_id,))
        
        # Also clean up the evaluation results table if it was populated prematurely
        conn.execute("""
            DELETE FROM eval_match_results_v2
            WHERE match_id = ?
        """, (match_id,))
        
        # Also clean up quarter scores if they were saved prematurely
        conn.execute("""
            DELETE FROM quarter_scores_v2
            WHERE match_id = ?
        """, (match_id,))

    # Check new state
    print("\n--- AFTER RESET ---")
    row = conn.execute("SELECT status, final_fetched, final_fetch_at FROM bet_monitor_schedule_v2 WHERE match_id = ?", (match_id,)).fetchone()
    if row:
        print("Schedule:", dict(row))
    logs = conn.execute("SELECT id, model_version, result FROM bet_monitor_log_v2 WHERE match_id = ?", (match_id,)).fetchall()
    for l in logs:
        print("Log:", dict(l))
        
    conn.close()
    print("\nReset successful!")

if __name__ == "__main__":
    main()

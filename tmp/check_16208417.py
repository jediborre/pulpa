"""
Ubicación original: scratch/check_16208417.py
Propósito / Qué hacía:
Diagnóstico forense y verificación de datos del partido específico ID 16208417.
"""

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

    # Query schedule
    print("\n=== SCHEDULE ROW ===")
    cursor = conn.execute("SELECT * FROM bet_monitor_schedule_v2 WHERE match_id = ?", ("16208417",))
    row = cursor.fetchone()
    if row:
        print(dict(row))
    else:
        print("No schedule row found.")

    # Query logs
    print("\n=== LOGS ===")
    cursor = conn.execute("SELECT * FROM bet_monitor_log_v2 WHERE match_id = ?", ("16208417",))
    rows = cursor.fetchall()
    if rows:
        for r in rows:
            d = dict(r)
            d.pop("raw_json", None)
            d.pop("inference_json", None)
            print(d)
    else:
        print("No log rows found.")

    # Query quarter scores
    print("\n=== QUARTER SCORES ===")
    cursor = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id = ?", ("16208417",))
    row_qs = cursor.fetchone()
    if row_qs:
        print(dict(row_qs))
    else:
        print("No quarter scores found.")


    conn.close()

if __name__ == "__main__":
    main()

"""
Ubicación original: scratch/redownload_requested.py
Propósito / Qué hacía:
Re-descarga controlada de partidos específicos solicitados por el usuario.
"""

import sys
import asyncio
import time
from pathlib import Path

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.database.connection import get_db_connection
from bet_monitor_v2.database.repository import update_schedule_status
from bet_monitor_v2.main import _final_fetch_and_save
from bet_monitor_v2.utils.helpers import ensure_obscura_running

async def main():
    # Make sure Obscura is running
    print("Checking Obscura (CDP) status...")
    if ensure_obscura_running():
        print("Obscura is running.")
    else:
        print("Obscura is not running. Scraper might use fallback or fail if Obscura is required.")
        
    matches_to_download = [
        {"mid": "15939376", "desc": "Deportivo Amambay vs Félix Pérez Cardozo"},
        {"mid": "15935080", "desc": "Cleveland Cavaliers vs New York Knicks"},
        {"mid": "16077982", "desc": "Osos de Manatí vs Cangrejeros De Santurce"}
    ]
    
    for item in matches_to_download:
        mid = item["mid"]
        desc = item["desc"]
        print(f"\n=========================================")
        print(f"Processing: {desc} (ID: {mid})")
        print(f"=========================================")
        
        # 1. Reset state so it's clean
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE bet_monitor_schedule_v2
                SET status = 'pending', skip_reason = NULL, final_fetched = 0, final_fetch_at = NULL
                WHERE match_id = ?
            """, (mid,))
            conn.execute("""
                UPDATE bet_monitor_log_v2
                SET result = 'pending'
                WHERE match_id = ?
            """, (mid,))
            conn.commit()
        print(f"Database fields reset for {mid}.")
        
        # 2. Get teams
        with get_db_connection() as conn:
            r = conn.execute("SELECT home_team, away_team FROM bet_monitor_schedule_v2 WHERE match_id = ?", (mid,)).fetchone()
            if not r:
                print(f"Match {mid} not found in bet_monitor_schedule_v2! Skipping.")
                continue
            home = r["home_team"]
            away = r["away_team"]
            
        print(f"Triggering final fetch and save for: {home} vs {away} (ID: {mid})...")
        try:
            await _final_fetch_and_save(mid, home, away)
            print(f"[SUCCESS] Processed match {mid} successfully!")
        except Exception as e:
            print(f"[ERROR] Failed processing match {mid}: {e}")

if __name__ == '__main__':
    # Support emoji output on Windows
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    asyncio.run(main())

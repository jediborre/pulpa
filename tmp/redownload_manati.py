"""
Ubicación original: scratch/redownload_manati.py
Propósito / Qué hacía:
Re-descarga de datos del partido Atenienses de Manatí para corregir registros faltantes.
"""

import sys
import asyncio
from pathlib import Path

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.database.connection import get_db_connection
from bet_monitor_v2.main import _final_fetch_and_save

async def main():
    mid = '16077982'
    
    # 1. Reset database fields
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE bet_monitor_schedule_v2
            SET status = 'pending', skip_reason = NULL, final_fetched = 0
            WHERE match_id = ?
        """, (mid,))
        conn.execute("""
            UPDATE bet_monitor_log_v2
            SET result = 'pending'
            WHERE match_id = ?
        """, (mid,))
        conn.commit()
    print(f"Database state reset successfully for match: {mid}")
            
    # 2. Trigger redownload
    with get_db_connection() as conn:
        r = conn.execute("SELECT home_team, away_team FROM bet_monitor_schedule_v2 WHERE match_id = ?", (mid,)).fetchone()
        home = r["home_team"]
        away = r["away_team"]
        
    print(f"\nTriggering final fetch and save for: {home} vs {away} (ID: {mid})...")
    try:
        await _final_fetch_and_save(mid, home, away)
        print(f"Successfully processed match: {mid}")
    except Exception as e:
        print(f"Error processing match {mid}: {e}")

if __name__ == '__main__':
    asyncio.run(main())

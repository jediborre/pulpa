"""
Ubicación original: scratch/fetch_match_live_status.py
Propósito / Qué hacía:
Consulta directa de estado en vivo de un partido vía API.
"""

import sys
import asyncio
from pathlib import Path

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.scrapers.live_scraper import fetch_event_snapshot
from bet_monitor_v2.scrapers.browser_client import fetch_match_by_id

async def main():
    match_id = "16208417"
    print(f"=== CHECKING LIVE STATUS FOR {match_id} ===")
    
    try:
        snapshot = await fetch_event_snapshot(match_id)
        print("Snapshot from live_scraper:")
        import json
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error fetching snapshot: {e}")
        
    try:
        print("\nFetching full match data from browser_client...")
        data = await fetch_match_by_id(match_id, is_ft=False)
        print("Match keys:", list(data.keys()))
        print("Score:", data.get("score"))
        match_meta = data.get("match", {})
        print("Match Meta (status, description):")
        print(f"  status_type: {match_meta.get('status_type')}")
        print(f"  status_description: {match_meta.get('status_description')}")
    except Exception as e:
        print(f"Error fetching full match: {e}")

if __name__ == '__main__':
    asyncio.run(main())

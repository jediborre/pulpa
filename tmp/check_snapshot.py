"""
Ubicación original: scratch/check_snapshot.py
Propósito / Qué hacía:
Comprobación de snapshot de datos a minutos específicos de juego.
"""

import sys
import asyncio
from pathlib import Path

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.scrapers.live_scraper import fetch_event_snapshot

async def main():
    match_id = "16220335"
    print(f"Fetching event snapshot for ID: {match_id}...")
    try:
        snapshot = await fetch_event_snapshot(match_id)
        import json
        print("\nSnapshot keys:", list(snapshot.keys()))
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    asyncio.run(main())

"""
Ubicación original: scratch/infer_live_match.py
Propósito / Qué hacía:
Prueba de inferencia en vivo sobre un partido activo usando los modelos en caché.
"""

import sys
import asyncio
from pathlib import Path

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.scrapers.browser_client import fetch_match_by_id
from match.training.infer_match import _infer_minute_from_pbp

async def main():
    # Accept match ID as command line parameter, or default to 16220335
    match_id = "16220335"
    if len(sys.argv) > 1:
        match_id = sys.argv[1].strip()
        
    print(f"Fetching live match data for ID: {match_id}...")
    try:
        # Fetch live data (same as the bonds does)
        data = await fetch_match_by_id(match_id, is_ft=False)
        
        # Calculate inferred minute
        minute = _infer_minute_from_pbp(data)
        
        # Check graph points
        gp = data.get("graph_points", [])
        gp_count = len(gp)
        last_gp = gp[-1] if gp else None
        
        # Calculate adjusted minute using the new fallback
        adjusted_minute = minute
        if gp_count > 0 and (adjusted_minute is None or gp_count > adjusted_minute):
            adjusted_minute = gp_count
        
        print("\n=========================================")
        print(f"RESULT FOR LIVE MATCH {match_id}:")
        print(f"Original minute from PBP: {minute}")
        print(f"Adjusted minute using new fallback: {adjusted_minute}")
        print(f"Number of Graph Points (pressure graph): {gp_count}")
        if last_gp:
            print(f"Last Graph Point: {last_gp}")
        print("=========================================")
        
        # Print play-by-play to verify
        pbp = data.get("play_by_play", {})
        print("\nPlay-by-Play quarters available in payload:", list(pbp.keys()))
        for q, plays in pbp.items():
            if plays:
                # Print the newest play (plays[0]) instead of oldest (plays[-1])
                print(f"  {q}: {len(plays)} plays. Newest play: {plays[0]}")
                
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Support emoji/UTF-8 on Windows CMD
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    asyncio.run(main())

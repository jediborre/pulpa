# -*- coding: utf-8 -*-
import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.scrapers.browser_client import fetch_match_by_id

async def main():
    match_id = "16208417"
    print(f"Fetching match raw PBP data for {match_id}...")
    try:
        data = await fetch_match_by_id(match_id, is_ft=False)
        print("\nMatch metadata:")
        print("Home Team:", data.get("match", {}).get("home_team"))
        print("Away Team:", data.get("match", {}).get("away_team"))
        print("Status Type:", data.get("match", {}).get("status_type"))
        print("Status Description:", data.get("match", {}).get("status_description"))
        print("Quarters:", data.get("score", {}).get("quarters"))
        
        pbp = data.get("play_by_play", {})
        print("\nPlay-by-play keys:", list(pbp.keys()))
        for q, plays in pbp.items():
            print(f"\n--- {q} plays (total {len(plays)}) ---")
            # print up to 10 plays
            for p in plays[:10]:
                print(p)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())

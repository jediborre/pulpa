"""
Ubicación original: scratch/test_chrome_backend.py
Propósito / Qué hacía:
Prueba de conexión y extracción de datos vía Google Chrome headless CDP.
"""

# -*- coding: utf-8 -*-
import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match.scraper import fetch_match_by_id as ss_fetch_match
from match.training.infer_match import _infer_minute_from_pbp

async def main():
    match_id = "16244040"
    print(f"Fetching match {match_id} using real system-installed Google Chrome...")
    
    try:
        # Fetch using real system Chrome
        data = await asyncio.to_thread(ss_fetch_match, match_id, backend="chrome")
        
        print("\nSUCCESS!")
        print("Home Team:", data.get("match", {}).get("home_team"))
        print("Away Team:", data.get("match", {}).get("away_team"))
        print("Status Type:", data.get("match", {}).get("status_type"))
        print("Status Description:", data.get("match", {}).get("status_description"))
        print("Quarters:", data.get("score", {}).get("quarters"))
        
        minute = _infer_minute_from_pbp(data)
        print("Inferred minute:", minute)
        
    except Exception as e:
        print("\nFAILED!")
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())

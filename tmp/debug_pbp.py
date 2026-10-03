"""
Ubicación original: scratch/debug_pbp.py
Propósito / Qué hacía:
Depuración de extracción y guardado de jugadas PBP.
"""

import sys
import asyncio
from pathlib import Path

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.scrapers.browser_client import fetch_match_by_id
from match.training.infer_match import _infer_minute_from_pbp, _quarter_index, _clock_to_seconds

async def main():
    match_id = "16220335"
    print(f"Fetching data for ID: {match_id}...")
    data = await fetch_match_by_id(match_id, is_ft=False)
    
    # Check league
    match_meta = data.get("match", {}) or {}
    league = str(match_meta.get("league", "") or "")
    print(f"League name from API: '{league}'")
    
    # Replicate logic to see what q_len is
    q_len = 10.0
    league_lower = league.lower()
    if any(n in league_lower for n in ["nba", "nbl", "pba", "cba"]):
        q_len = 12.0
        print("League matched 12-min quarter keywords!")
        
    pbp = data.get("play_by_play", {}) or {}
    
    # Reloj >= 10 check
    found_10_plus = False
    for quarter_label, plays in pbp.items():
        for play in plays or []:
            time_str = str(play.get("time", "") or "")
            if ":" in time_str:
                parts = time_str.split(":")
                if len(parts) == 2 and int(parts[0]) >= 10:
                    q_len = 12.0
                    found_10_plus = True
                    print(f"Found play with time >= 10: {quarter_label} - {time_str}")
                    break
        if found_10_plus:
            break
            
    print(f"Determined q_len: {q_len} minutes")
    
    # Calculate global min for each play
    for quarter_label, plays in pbp.items():
        q_idx = _quarter_index(str(quarter_label))
        if q_idx is None:
            continue
        q_start = (q_idx - 1) * q_len
        print(f"\nQuarter {quarter_label} (q_idx={q_idx}, q_start={q_start}):")
        
        play_mins = []
        for play in plays or []:
            rem_sec = _clock_to_seconds(str(play.get("time", "") or ""))
            if rem_sec is None:
                continue
            global_min = q_start + (q_len - rem_sec / 60.0)
            play_mins.append((play.get("time"), global_min))
            
        if play_mins:
            print("  First 3 plays:")
            for p_time, g_min in play_mins[:3]:
                print(f"    Time: {p_time} -> global_min: {g_min:.2f} (int={int(g_min)})")
            print("  Last 3 plays:")
            for p_time, g_min in play_mins[-3:]:
                print(f"    Time: {p_time} -> global_min: {g_min:.2f} (int={int(g_min)})")
                
    # Run original and adjusted
    minute = _infer_minute_from_pbp(data)
    print("\nResult from _infer_minute_from_pbp:", minute)
    
if __name__ == '__main__':
    asyncio.run(main())

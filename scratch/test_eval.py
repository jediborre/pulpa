import os
import sys
import time
import logging
from pathlib import Path

# Set up logging to console to see the monitor messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match.bet_monitor import _fetch_all_events_for_date_sync, _monitor_local_today_str

def main():
    today = _monitor_local_today_str()
    print(f"Testing schedule cache for local date: {today}")
    
    # 1. Clean existing cache if present to ensure we test download first
    cache_dir = ROOT / "api_cache"
    print("Clearing any existing cache files for today's dates in api_cache...")
    for f in cache_dir.glob("schedule_*.json"):
        try:
            f.unlink()
            print(f"  Removed old cache: {f.name}")
        except Exception:
            pass
            
    # 2. First call: Should perform network fetch and save to cache
    print("\n--- First Call: Downloading from Network ---")
    start_time = time.time()
    events_1 = _fetch_all_events_for_date_sync(today)
    elapsed_1 = time.time() - start_time
    print(f"First call returned {len(events_1)} events in {elapsed_1:.2f} seconds.")
    
    # 3. Second call: Should read from local cache instantly (0 network requests)
    print("\n--- Second Call: Reading from Local Cache ---")
    start_time = time.time()
    events_2 = _fetch_all_events_for_date_sync(today)
    elapsed_2 = time.time() - start_time
    print(f"Second call returned {len(events_2)} events in {elapsed_2:.2f} seconds.")
    
    # Verification
    print("\n--- Verification ---")
    if len(events_1) == len(events_2):
        print(f"SUCCESS: Both calls returned identical counts ({len(events_1)} events).")
    else:
        print("WARNING: Count mismatch between calls.")
        
    if elapsed_2 < 0.2:
        print(f"SUCCESS: Cache read was instant ({elapsed_2*1000:.1f} ms) vs network read ({elapsed_1:.2f} s)!")
    else:
        print(f"WARNING: Cache read took longer than expected ({elapsed_2:.2f} s).")

if __name__ == '__main__':
    main()

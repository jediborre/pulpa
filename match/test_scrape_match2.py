#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import scraper

with open('test_scrape_output.log', 'w', buffering=1) as f:
    print(f"Testing scrape for match 14428863...", flush=True)
    f.write(f"Testing scrape for match 14428863...\n")
    f.flush()
    
    try:
        data = scraper.fetch_match_by_id('14428863')
        if data:
            match_info = data.get('match', {})
            msg = f"✓ Success!\nHome: {match_info.get('home_team')}\nAway: {match_info.get('away_team')}\nLeague: {match_info.get('league')}\nDate: {match_info.get('date')} {match_info.get('time')}\nStatus: {match_info.get('status_type')}"
            print(msg, flush=True)
            f.write(msg + "\n")
        else:
            msg = "✗ Scraper returned None"
            print(msg, flush=True)
            f.write(msg + "\n")
    except Exception as e:
        msg = f"✗ Error: {e}"
        print(msg, flush=True)
        f.write(msg + "\n")
        import traceback
        traceback.print_exc(file=f)
    
    f.flush()
    print("Done", flush=True)

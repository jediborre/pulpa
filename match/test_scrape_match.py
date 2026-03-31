#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import scraper

print(f"Testing scrape for match 14428863...")
try:
    data = scraper.fetch_match_by_id('14428863')
    if data:
        match_info = data.get('match', {})
        print(f"✓ Success!")
        print(f"  Home: {match_info.get('home_team')}")
        print(f"  Away: {match_info.get('away_team')}")
        print(f"  League: {match_info.get('league')}")
        print(f"  Date: {match_info.get('date')} {match_info.get('time')}")
        print(f"  Status: {match_info.get('status_type')}")
    else:
        print("✗ Scraper returned None")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

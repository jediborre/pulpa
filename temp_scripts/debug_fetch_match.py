"""Debug: test fetch_match with team_strength."""
import sys, json, traceback
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from scraper import fetch_match

match_id = "15935010"
url = f"https://www.sofascore.com/event/{match_id}"

try:
    result = fetch_match(url=url, match_id=match_id, fetch_team_data=True)
    ts = result.get("team_strength", [])
    print(f"Got {len(ts)} team_strength rows")
    for row in ts:
        print(json.dumps(row, indent=2))
except Exception as e:
    traceback.print_exc()

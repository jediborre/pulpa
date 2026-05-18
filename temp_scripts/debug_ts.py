"""Debug: test team_strength fetch standalone."""
import sys, json, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from scraper import fetch_team_strength

ts = fetch_team_strength(3424, 3432, "Detroit Pistons", "Cleveland Cavaliers")
print(f"Got {len(ts)} rows")
for row in ts:
    print(json.dumps(row, indent=2))

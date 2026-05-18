"""Show H2H split by home/away side."""
import sys, json
from pathlib import Path
ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))
from scraper import fetch_h2h_via_teams

h2h = fetch_h2h_via_teams(3424, 3432)

current_home = "Detroit Pistons"
current_away = "Cleveland Cavaliers"

for label, team, is_home_team in [
    (current_home, current_home, True),
    (current_away, current_away, False),
]:
    as_home = [h for h in h2h if h["home_team"] == team]
    as_away = [h for h in h2h if h["away_team"] == team]
    print(f"=== {label} ===")
    print(f"  As HOME ({len(as_home)}):")
    for h in as_home:
        w = "W" if h["home_score"] > h["away_score"] else "L"
        print(f"    {w} {h['home_score']}-{h['away_score']}  Q:{h['q1_home']}-{h['q1_away']} {h['q2_home']}-{h['q2_away']} {h['q3_home']}-{h['q3_away']} {h['q4_home']}-{h['q4_away']}")
    print(f"  As AWAY ({len(as_away)}):")
    for h in as_away:
        w = "W" if h["away_score"] > h["home_score"] else "L"
        print(f"    {w} {h['home_score']}-{h['away_score']}  Q:{h['q1_home']}-{h['q1_away']} {h['q2_home']}-{h['q2_away']} {h['q3_home']}-{h['q3_away']} {h['q4_home']}-{h['q4_away']}")

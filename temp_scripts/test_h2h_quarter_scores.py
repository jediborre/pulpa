"""End-to-end test: discover H2H matches with quarter scores via team events."""
import json, sys
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from scraper import fetch_h2h_via_teams

# Try various rival pairings
pairs = [
    (25134, 25212, "Real Madrid vs Barcelona"),
    (25236, 25211, "Fenerbahçe vs Anadolu Efes"),
    (25209, 25210, "Olympiacos vs Panathinaikos"),
    (3560, 18961, "Maccabi TA vs Hapoel Jerusalem"),
    (25210, 25209, "Panathinaikos vs Olympiacos"),
]
for ht, at, label in pairs:
    result = fetch_h2h_via_teams(ht, at, limit=20)
    print(f"\n{label}: {len(result)} common H2H matches")
    for r in result:
        print(f"  ID={r['h2h_match_id']}  {r['date']}  "
              f"Q: {r['q1_home']}-{r['q1_away']}, {r['q2_home']}-{r['q2_away']}, "
              f"{r['q3_home']}-{r['q3_away']}, {r['q4_home']}-{r['q4_away']}")

print(f"Found {len(result)} common H2H matches")
for r in result:
    print(f"  ID={r['h2h_match_id']}  {r['date']}  "
          f"{r['home_team']} vs {r['away_team']}  "
          f"Score: {r['home_score']}-{r['away_score']}  "
          f"Q: {r['q1_home']}-{r['q1_away']}, {r['q2_home']}-{r['q2_away']}, "
          f"{r['q3_home']}-{r['q3_away']}, {r['q4_home']}-{r['q4_away']}")

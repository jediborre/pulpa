"""Test fetch_h2h_via_teams standalone - how many H2H matches for NBA?"""
import sys, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright
from scraper import fetch_h2h_via_teams

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
BASE = "https://api.sofascore.com/api/v1"

match_id = "15935010"

# Use a Playwright session JUST to get team IDs
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(1)

    resp = ctx.request.get(
        f"{BASE}/event/{match_id}",
        headers={"Referer": "https://www.sofascore.com/", "Accept": "application/json"},
        timeout=15_000
    )
    ev = resp.json().get("event", {})
    home_id = ev.get("homeTeam", {}).get("id")
    away_id = ev.get("awayTeam", {}).get("id")
    home_name = ev.get("homeTeam", {}).get("name")
    away_name = ev.get("awayTeam", {}).get("name")
    print(f"Home: {home_name} (ID={home_id}), Away: {away_name} (ID={away_id})")
    browser.close()

# Now call fetch_h2h_via_teams standalone (it opens its own browser)
h2h_rows = fetch_h2h_via_teams(home_id, away_id, limit=50)
print(f"\nH2H matches via teams: {len(h2h_rows)} (limit=50 per team)")
for h in h2h_rows[:15]:
    print(f"  {h['h2h_match_id']}: {h['home_team']:>25s} {h['home_score']:>3s}-{h['away_score']:<3s} {h['away_team']:<25s} | "
          f"Q1:{h['q1_home']}-{h['q1_away']} Q2:{h['q2_home']}-{h['q2_away']} Q3:{h['q3_home']}-{h['q3_away']} Q4:{h['q4_home']}-{h['q4_away']} | {h['tournament']}")
if len(h2h_rows) > 15:
    print(f"  ... and {len(h2h_rows) - 15} more")

# Count how many have quarter scores
with_q = sum(1 for h in h2h_rows if h['q1_home'] is not None)
print(f"\nWith quarter scores: {with_q}/{len(h2h_rows)}")

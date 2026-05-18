"""Show full H2H row as saved in DB."""
import sys, time, json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright
from scraper import fetch_h2h_via_teams

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE = "https://api.sofascore.com/api/v1"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json"}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(1)

    # Get team IDs
    resp = ctx.request.get(f"{BASE}/event/15935010", headers=extra_headers, timeout=15_000)
    ev = resp.json().get("event", {})
    home_id = ev.get("homeTeam", {}).get("id")
    away_id = ev.get("awayTeam", {}).get("id")
    browser.close()

h2h = fetch_h2h_via_teams(home_id, away_id)
print(f"Total H2H rows: {len(h2h)}\n")

for row in h2h:
    print(json.dumps(row, indent=2))
    print()

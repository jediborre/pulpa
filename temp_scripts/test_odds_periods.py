"""Test odds endpoint and period stats computation with a real match."""
import json, sys, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

# Test 1: Period stats computation from existing data
from scraper import _parse_period_stats, _parse_odds

# Load the saved initialProps to test period computation
props = json.load(open(ROOT / "temp_scripts" / "full_initial_props.json", "r", encoding="utf-8"))
incidents = props.get("incidents", [])
print(f"Test 1: Period stats from {len(incidents)} incidents")
period_stats = _parse_period_stats("test_14491671", incidents)
print(f"  Generated {len(period_stats)} period stat rows")
for ps in period_stats:
    print(f"  {ps['period']} | {ps['stat_name']}: home={ps['home_value']} away={ps['away_value']}")

# Test 2: Odds parsing with mock data
print(f"\nTest 2: Odds parsing")
mock_odds = {
    "markets": [
        {
            "marketType": "1x2",
            "name": "Match Winner",
            "choices": [
                {"name": "1", "decimalValue": 1.85},
                {"name": "2", "decimalValue": 3.10},
                {"name": "X", "decimalValue": 3.50},
            ]
        },
        {
            "marketType": "spread",
            "name": "Handicap",
            "choices": [
                {"name": "1", "decimalValue": 1.90},
                {"name": "2", "decimalValue": 1.90},
            ]
        }
    ],
    "timestamp": "2026-05-18T10:00:00Z"
}
odds_rows = _parse_odds("test_id", mock_odds)
for r in odds_rows:
    print(f"  {r['odds_type']}: home={r['home_value']} away={r['away_value']} draw={r['draw_value']}")

# Test 3: Real odds endpoint via Playwright
print(f"\nTest 3: Real odds endpoint")
from playwright.sync_api import sync_playwright
STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(2)

    # Test with NBA match (more likely to have odds)
    resp = ctx.request.get(
        "https://api.sofascore.com/api/v1/event/15935010/odds/1/all",
        headers=extra_headers, timeout=15_000
    )
    print(f"  NBA odds endpoint: HTTP {resp.status}")
    if resp.ok:
        body = resp.json()
        markets = body.get("markets", [])
        print(f"  Markets: {len(markets)}")
        for m in markets[:5]:
            choices = m.get("choices", [])
            print(f"    {m.get('marketType')}: {[(c.get('name'), c.get('decimalValue')) for c in choices]}")

    # Also test period stats computation with live scraped data
    inc_resp = ctx.request.get(
        "https://api.sofascore.com/api/v1/event/15935010/incidents",
        headers=extra_headers, timeout=15_000
    )
    if inc_resp.ok:
        incs = inc_resp.json().get("incidents", [])
        print(f"\n  NBA incidents: {len(incs)}")
        ps = _parse_period_stats("test_nba", incs)
        print(f"  Period stat rows: {len(ps)}")
        for p in ps[:20]:
            print(f"    {p['period']} | {p['stat_name']}: {p['home_value']} vs {p['away_value']}")

    browser.close()

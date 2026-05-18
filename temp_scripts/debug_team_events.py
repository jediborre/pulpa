"""Debug: inspect team events response structure."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")

# Check what the team event looks like
from playwright.sync_api import sync_playwright

STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    try:
        page.goto(
            "https://www.sofascore.com/basketball",
            wait_until="networkidle",
            timeout=45_000,
        )
    except Exception:
        pass

    # Fetch Real Madrid events
    resp = ctx.request.get(
        "https://api.sofascore.com/api/v1/team/25134/events/last/20",
        headers=extra_headers,
        timeout=20_000,
    )
    print(f"Status: {resp.status}")
    if resp.ok:
        body = resp.json()
        events = body.get("events", [])
        print(f"Events count: {len(events)}")
        for ev in events[:3]:
            print(f"\nEvent ID: {ev.get('id')}")
            print(f"  Home: {(ev.get('homeTeam') or {}).get('name')} (ID: {(ev.get('homeTeam') or {}).get('id')})")
            print(f"  Away: {(ev.get('awayTeam') or {}).get('name')} (ID: {(ev.get('awayTeam') or {}).get('id')})")
            print(f"  Score keys: {list((ev.get('homeScore') or {}).keys())}")
            print(f"  HomeScore: {json.dumps(ev.get('homeScore'))}")
            print(f"  AwayScore: {json.dumps(ev.get('awayScore'))}")

    browser.close()

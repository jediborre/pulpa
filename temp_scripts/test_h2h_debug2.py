"""Check dates of returned team events and try different limit values."""
import sys, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
BASE = "https://api.sofascore.com/api/v1"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json"}

match_id = "15935010"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA, viewport={"width": 1920, "height": 1080})
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(1)

    # Get team IDs
    resp = ctx.request.get(f"{BASE}/event/{match_id}", headers=extra_headers, timeout=15_000)
    ev = resp.json().get("event", {})
    home_id = ev.get("homeTeam", {}).get("id")
    away_id = ev.get("awayTeam", {}).get("id")
    home_name = ev.get("homeTeam", {}).get("name")
    away_name = ev.get("awayTeam", {}).get("name")
    print(f"Match: {home_name} (ID={home_id}) vs {away_name} (ID={away_id})")

    # Try different limits
    for limit in [10, 30, 50, 100]:
        resp_h = ctx.request.get(f"{BASE}/team/{home_id}/events/last/{limit}", headers=extra_headers, timeout=15_000)
        events = (resp_h.json() or {}).get("events", [])
        dates = []
        ids = []
        for ev in events:
            ts = ev.get("startTimestamp", 0)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
            dates.append(str(dt)[:10] if dt else "?")
            ids.append(ev.get("id"))
        print(f"\nlimit={limit}: returned {len(events)} events")
        print(f"  Dates: {dates[:5]}... (last: {dates[-1]})")
        print(f"  IDs: {ids[:5]}... (last: {ids[-1]})")

    # Also check /team/{id}/events/next/30 (future events)
    resp_h = ctx.request.get(f"{BASE}/team/{home_id}/events/next/30", headers=extra_headers, timeout=15_000)
    events = (resp_h.json() or {}).get("events", [])
    print(f"\n/team/{home_id}/events/next/30: returned {len(events)} events")
    for ev in events[:3]:
        ts = ev.get("startTimestamp", 0)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        print(f"  ID={ev.get('id')} date={str(dt)[:10] if dt else '?'} vs {ev.get('awayTeam',{}).get('name')}")

    # Also check /team/{id}/events/last/0 - maybe 0 means "unlimited"?
    # Actually let's just check the match page for how it handles H2H
    page.goto(f"https://www.sofascore.com/team/basketball/{home_id}", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(3)
    
    # Check what API calls the team page makes
    print("\nChecking team page for H2H data sources...")
    
    browser.close()

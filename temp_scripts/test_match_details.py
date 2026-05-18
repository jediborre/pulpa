"""Check what tournament/season this match belongs to."""
import sys, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE = "https://api.sofascore.com/api/v1"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json"}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(1)

    match_id = "15935010"
    resp = ctx.request.get(f"{BASE}/event/{match_id}", headers=extra_headers, timeout=15_000)
    body = resp.json()
    ev = body.get("event", {})
    from datetime import datetime, timezone
    ts = ev.get("startTimestamp", 0)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
    print(f"Tournament: {ev.get('tournament', {}).get('name')}")
    print(f"UniqueTournament: {ev.get('uniqueTournament', {}).get('name')}")
    print(f"Season: {ev.get('season', {}).get('name')}")
    print(f"Start time: {dt}")
    print(f"Status: {ev.get('status', {}).get('description')}")
    print(f"Round: {ev.get('roundInfo', {}).get('round')}")
    
    # Check what the team events endpoint actually returns
    # Try limit=0 (maybe means "most recent")
    for limit in [0, 1, 5]:
        resp_h = ctx.request.get(f"{BASE}/team/3424/events/last/{limit}", headers=extra_headers, timeout=15_000)
        events = (resp_h.json() or {}).get("events", [])
        print(f"\nlimit={limit}: {len(events)} events")
        for ev in events[:3]:
            ts2 = ev.get("startTimestamp", 0)
            dt2 = datetime.fromtimestamp(ts2, tz=timezone.utc) if ts2 else None
            print(f"  ID={ev.get('id')} date={str(dt2)[:10]} vs {ev.get('awayTeam',{}).get('name')}")
    
    browser.close()

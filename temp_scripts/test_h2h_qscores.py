"""Check quarter scores for those 7 H2H matches."""
import sys, time
from pathlib import Path
from datetime import datetime, timezone

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

    # Fetch page 0 events for team 3424 (Pistons)
    resp = ctx.request.get(f"{BASE}/team/3424/events/last/0", headers=extra_headers, timeout=15_000)
    h_events = (resp.json() or {}).get("events", [])

    resp = ctx.request.get(f"{BASE}/team/3432/events/last/0", headers=extra_headers, timeout=15_000)
    a_events = (resp.json() or {}).get("events", [])

    h_ids = {str(e.get("id")) for e in h_events}
    a_ids = {str(e.get("id")) for e in a_events}
    common = h_ids & a_ids
    print(f"Found {len(common)} common matches on page 0\n")

    all_events = h_events + a_events
    seen = set()
    for ev in all_events:
        mid = str(ev.get("id"))
        if mid not in common or mid in seen:
            continue
        seen.add(mid)
        hs = ev.get("homeScore") or {}
        as_ = ev.get("awayScore") or {}
        ts2 = ev.get("startTimestamp", 0)
        dt2 = datetime.fromtimestamp(ts2, tz=timezone.utc) if ts2 else None
        ht = ev.get("homeTeam", {}).get("name", "?")
        at = ev.get("awayTeam", {}).get("name", "?")
        print(f"Match {mid} | {dt2.strftime('%Y-%m-%d')}")
        print(f"  {ht} {hs.get('current')}-{as_.get('current')} {at}")
        print(f"  Q1: {hs.get('period1')}-{as_.get('period1')}")
        print(f"  Q2: {hs.get('period2')}-{as_.get('period2')}")
        print(f"  Q3: {hs.get('period3')}-{as_.get('period3')}")
        print(f"  Q4: {hs.get('period4')}-{as_.get('period4')}")

    browser.close()

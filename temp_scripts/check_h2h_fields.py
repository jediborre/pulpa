"""Check what data is available in team events for H2H (timestamp, venue)."""
import sys, json
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

    # 1. Check a team event for venue + timestamp
    resp = ctx.request.get(f"{BASE}/team/3424/events/last/0", headers=extra_headers, timeout=15_000)
    events = (resp.json() or {}).get("events", [])
    if events:
        ev = events[0]
        print("=== Team event keys ===")
        for k in sorted(ev.keys()):
            v = ev[k]
            if isinstance(v, (dict, list)):
                print(f"  {k}: {type(v).__name__} {json.dumps(list(v.keys()) if isinstance(v, dict) else v[:2], default=str)[:100]}")
            else:
                print(f"  {k}: {v}")

        # Check for venue
        print("\n=== Venue check ===")
        print(f"  'venue' in event: {'venue' in ev}")
        print(f"  'time' in event: {'time' in ev}")
        t = ev.get("time", {})
        if t:
            print(f"  time keys: {list(t.keys())}")
        
        # Check startTimestamp
        from datetime import datetime, timezone
        ts = ev.get("startTimestamp", 0)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        print(f"\n  startTimestamp: {ts} -> {dt}")

    # 2. Also check the aggregate H2H response for venue
    print("\n=== Aggregate H2H ===")
    resp = ctx.request.get(f"{BASE}/event/15935010/h2h", headers=extra_headers, timeout=15_000)
    if resp.ok:
        h2h = resp.json().get("h2h", [])
        if h2h:
            h = h2h[0]
            print(f"  H2H entry keys: {sorted(h.keys())}")
            for k in sorted(h.keys()):
                print(f"    {k}: {h[k]}")
        else:
            print("  No h2h entries in aggregate response")

    browser.close()

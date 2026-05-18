"""Find upcoming basketball matches to check for odds data."""
import json, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
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

BASE = "https://api.sofascore.com/api/v1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()

    page.goto("https://www.sofascore.com/basketball",
               wait_until="domcontentloaded", timeout=30_000)
    time.sleep(3)

    # Get today's scheduled basketball matches
    from datetime import date
    today = date.today().isoformat()
    resp = ctx.request.get(
        f"{BASE}/sport/basketball/scheduled-events/{today}",
        headers=extra_headers, timeout=15_000
    )
    if resp.ok:
        body = resp.json()
        events = body.get("events", [])
        print(f"Today's scheduled basketball events: {len(events)}")
        for ev in events[:5]:
            mid = ev.get("id")
            ht = ((ev.get("homeTeam") or {})).get("name", "?")
            at = ((ev.get("awayTeam") or {})).get("name", "?")
            print(f"  {mid}: {ht} vs {at}")

            # Check if odds endpoint works for upcoming matches
            for ep in [f"/event/{mid}/odds", f"/event/{mid}/betting", f"/event/{mid}/prematch"]:
                try:
                    # Navigate to match page and check HTML for odds
                    page.goto(f"https://www.sofascore.com/event/{mid}",
                               wait_until="domcontentloaded", timeout=30_000)
                    time.sleep(2)
                    html = page.content()

                    # Check HTML for embedded odds data
                    import re
                    odds_in_html = re.findall(r'"odds"[^}]{0,500}', html)
                    for o in odds_in_html[:3]:
                        print(f"    odds fragment: {o[:200]}")

                    # Try direct API
                    resp2 = ctx.request.get(f"{BASE}{ep}",
                                            headers=extra_headers, timeout=10_000)
                    print(f"    {ep} -> HTTP {resp2.status}")
                    if resp2.ok:
                        print(f"      Body: {resp2.text()[:300]}")
                except Exception as e:
                    print(f"    Error: {e}")
            break  # Just one match for now
    else:
        print(f"No events for {today}: HTTP {resp.status}")

    browser.close()

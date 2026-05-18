"""Probe a Euroleague match for detailed incidents with fouls/timeouts."""
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

# Find finished Euroleague matches
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball",
               wait_until="domcontentloaded", timeout=30_000)
    time.sleep(2)

    # Fetch scheduled events to find Euroleague matches
    resp = ctx.request.get(
        f"{BASE}/sport/basketball/scheduled-events/2026-05-17",
        headers=extra_headers, timeout=15_000
    )
    euro_matches = []
    if resp.ok:
        events = resp.json().get("events", [])
        for ev in events:
            tourney = ev.get("tournament", {}).get("name", "")
            if "Euroleague" in tourney or "Euroleague" in tourney:
                euro_matches.append(ev.get("id"))
    
    print(f"Found Euroleague matches: {euro_matches[:5]}")
    
    # If no euro matches found for that date, try getting recent finished matches
    if not euro_matches:
        # Try fetching completed events for a tournament
        resp2 = ctx.request.get(
            f"{BASE}/unique-tournament/132/season/65375/events/last/5",
            headers=extra_headers, timeout=15_000
        )
        if resp2.ok:
            events = resp2.json().get("events", [])
            for ev in events[:5]:
                euro_matches.append(ev.get("id"))
                print(f"Euro match: {ev.get('id')} - {ev.get('homeTeam',{}).get('name')} vs {ev.get('awayTeam',{}).get('name')}")

    # Probe incidents for a euro match
    for mid in euro_matches[:3]:
        print(f"\n\n=== Euro Match {mid} ===")
        resp_inc = ctx.request.get(
            f"{BASE}/event/{mid}/incidents",
            headers=extra_headers, timeout=30_000
        )
        if resp_inc.ok:
            incs = resp_inc.json().get("incidents", [])
            print(f"Total incidents: {len(incs)}")
            from collections import Counter
            types = Counter(i.get("incidentType","?") for i in incs)
            print(f"Incident types:")
            for t, c in types.most_common():
                print(f"  {t}: {c}")
            # Check for one of each type
            seen = set()
            for i in incs:
                typ = i.get("incidentType")
                if typ not in seen:
                    seen.add(typ)
                    print(f"\n  Sample {typ}:")
                    print(f"    class={i.get('incidentClass')}, from={i.get('from')}")
                    print(f"    team={i.get('team')}, isHome={i.get('isHome')}")
                    print(f"    isScored={i.get('isScored')}, pointValue={i.get('pointValue')}")
                    print(f"    period={i.get('period')}, reversedPeriodTime={i.get('reversedPeriodTime')}")
                    print(f"    keys={sorted(i.keys())}")

    browser.close()

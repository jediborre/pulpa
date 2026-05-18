"""Check NBA match for richer incident data with attempts."""
import json, time
from pathlib import Path
from datetime import date, timedelta

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
    time.sleep(2)

    # Try to find finished NBA games from yesterday
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    print(f"Checking NBA games from {yesterday}...")

    resp = ctx.request.get(
        f"{BASE}/sport/basketball/scheduled-events/{yesterday}",
        headers=extra_headers, timeout=15_000
    )
    nba_matches = []
    if resp.ok:
        events = resp.json().get("events", [])
        print(f"Total events: {len(events)}")
        for ev in events:
            tourney = ev.get("tournament", {}).get("name", "")
            if "NBA" in tourney:
                mid = ev.get("id")
                nba_matches.append(mid)
                print(f"  NBA: {mid} - {ev.get('homeTeam',{}).get('name')} vs {ev.get('awayTeam',{}).get('name')}")

    if not nba_matches:
        # Try unique-tournament/132 (Euroleague) or another approach
        resp2 = ctx.request.get(
            f"{BASE}/unique-tournament/132/season/65375/events/last/5",
            headers=extra_headers, timeout=15_000
        )
        if resp2.ok:
            events = resp2.json().get("events", [])
            print(f"Euroleague last events: {len(events)}")
            for ev in events:
                nba_matches.append(ev.get("id"))
        else:
            print(f"Euroleague endpoint: HTTP {resp2.status}")

    print(f"\nBest match candidates: {nba_matches[:5]}")

    # Check incidents for each
    for mid in nba_matches[:5]:
        print(f"\n=== Match {mid} ===")
        resp_inc = ctx.request.get(
            f"{BASE}/event/{mid}/incidents",
            headers=extra_headers, timeout=30_000
        )
        if resp_inc.ok:
            incs = resp_inc.json().get("incidents", [])
            print(f"Total incidents: {len(incs)}")

            from collections import Counter
            types = Counter(i.get("incidentType","?") for i in incs)
            print(f"Types: {dict(types)}")

            # Check for isScored
            has_scored = any(i.get("isScored") is not None for i in incs)
            has_shot = any(i.get("isShot") is not None for i in incs)
            has_team = any(i.get("team") is not None for i in incs)
            print(f"has_isScored={has_scored}, has_isShot={has_shot}, has_team={has_team}")

            # Sample each type
            seen = set()
            for i in incs:
                typ = i.get("incidentType")
                if typ not in seen:
                    seen.add(typ)
                    print(f"  {typ}: class={i.get('incidentClass')}, from={i.get('from')}, "
                          f"isScored={i.get('isScored')}, isShot={i.get('isShot')}, "
                          f"team={i.get('team')}, isHome={i.get('isHome')}")
                    extra = [k for k in i.keys() if k not in ('incidentType','incidentClass','from','team','teamId','homeScore','awayScore','isHome','isScored','isShot','pointValue','period','time','periodTimeSeconds','reversedPeriodTime','reversedPeriodTimeSeconds','player','text','id','addedTime','isLive')]
                    if extra:
                        print(f"    extra keys: {extra}")
            break  # Just first match

    browser.close()

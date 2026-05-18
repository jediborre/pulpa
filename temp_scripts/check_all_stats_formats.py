"""Check statistics response for basketball matches from DIFFERENT leagues.
Maybe some leagues have per-period data in the statistics endpoint."""
import json, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}
BASE = "https://api.sofascore.com/api/v1"

# Different basketball matches from different leagues
matches_to_check = [
    # (match_id, league name)
    ("14491671", "ACB (Spain)"),
    ("15935010", "NBA"),
    # Try to find a finished Euroleague match with ID
]

# First get some Euroleague match IDs
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(2)

    # Get recent Euroleague matches
    resp = ctx.request.get(
        f"{BASE}/unique-tournament/132/season/65375/events/last/10",
        headers=extra_headers, timeout=15_000
    )
    if resp.ok:
        events = resp.json().get("events", [])
        for ev in events[:5]:
            mid = str(ev.get("id"))
            ht = ev.get("homeTeam", {}).get("name", "?")
            at = ev.get("awayTeam", {}).get("name", "?")
            matches_to_check.append((mid, f"Euroleague: {ht} vs {at}"))

    # Also try ACB
    resp2 = ctx.request.get(
        f"{BASE}/unique-tournament/157/season/65382/events/last/10",
        headers=extra_headers, timeout=15_000
    )
    if resp2.ok:
        events = resp2.json().get("events", [])
        for ev in events[:3]:
            mid = str(ev.get("id"))
            matches_to_check.append((mid, "ACB (Spain)"))

    print(f"Checking {len(matches_to_check)} matches for per-period stats\n")

    for mid, label in matches_to_check:
        print(f"=== {label} (ID: {mid}) ===")

        # Statistics endpoint
        resp = ctx.request.get(
            f"{BASE}/event/{mid}/statistics",
            headers=extra_headers, timeout=15_000
        )
        if resp.ok:
            body = resp.json()
            stats = body.get("statistics", [])
            print(f"  Statistics periods: {[s.get('period') for s in stats]}")
            for s in stats[:5]:
                period = s.get("period")
                groups = s.get("groups", [])
                print(f"  Period '{period}': {len(groups)} groups")
                for g in groups:
                    items = g.get("statisticsItems", [])
                    print(f"    Group '{g.get('groupName')}': {len(items)} items")
                    # Show first 2 items
                    for item in items[:2]:
                        print(f"      {item.get('name')}: {item.get('homeValue')}/{item.get('homeTotal')} vs {item.get('awayValue')}/{item.get('awayTotal')}")

        # Incidents endpoint - check types
        resp_inc = ctx.request.get(
            f"{BASE}/event/{mid}/incidents",
            headers=extra_headers, timeout=15_000
        )
        if resp_inc.ok:
            incs = resp_inc.json().get("incidents", [])
            from collections import Counter
            types = Counter(i.get("incidentType","?") for i in incs)
            print(f"  Incidents: {len(incs)}, types: {dict(types)}")

        print()
        time.sleep(1)

    browser.close()

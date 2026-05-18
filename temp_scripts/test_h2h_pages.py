"""Test: iterate through multiple pages of team events to find actual H2H matches."""
import sys, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE = "https://api.sofascore.com/api/v1"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json"}

match_id = "15935010"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(1)

    resp = ctx.request.get(f"{BASE}/event/{match_id}", headers=extra_headers, timeout=15_000)
    ev = resp.json().get("event", {})
    home_id = ev.get("homeTeam", {}).get("id")
    away_id = ev.get("awayTeam", {}).get("id")
    home_name = ev.get("homeTeam", {}).get("name")
    away_name = ev.get("awayTeam", {}).get("name")
    print(f"Match: {home_name} (ID={home_id}) vs {away_name} (ID={away_id})")

    # Iterate pages to find common matches
    all_home_ids = {}
    all_away_ids = {}
    home_name_cache = {}
    away_name_cache = {}
    found_any = False

    for page_num in range(20):  # check up to 20 pages (600 events total per team)
        resp_h = ctx.request.get(f"{BASE}/team/{home_id}/events/last/{page_num}", headers=extra_headers, timeout=15_000)
        h_events = (resp_h.json() or {}).get("events", [])
        if not h_events:
            break
        for ev in h_events:
            eid = str(ev.get("id"))
            all_home_ids.setdefault(eid, 0)
            all_home_ids[eid] += 1
            if eid not in home_name_cache:
                ht = ev.get("homeTeam", {}).get("name", "?")
                at = ev.get("awayTeam", {}).get("name", "?")
                home_name_cache[eid] = f"{ht} vs {at}"

        resp_a = ctx.request.get(f"{BASE}/team/{away_id}/events/last/{page_num}", headers=extra_headers, timeout=15_000)
        a_events = (resp_a.json() or {}).get("events", [])
        if not a_events:
            break
        for ev in a_events:
            eid = str(ev.get("id"))
            all_away_ids.setdefault(eid, 0)
            all_away_ids[eid] += 1
            if eid not in away_name_cache:
                ht = ev.get("homeTeam", {}).get("name", "?")
                at = ev.get("awayTeam", {}).get("name", "?")
                away_name_cache[eid] = f"{ht} vs {at}"

        common = set(all_home_ids.keys()) & set(all_away_ids.keys())
        if common:
            print(f"\nPage {page_num}: Found {len(common)} common matches!")
            for mid in sorted(common)[:10]:
                print(f"  Match {mid}: {home_name_cache.get(mid, '?')} | {away_name_cache.get(mid, '?')}")
            found_any = True
            break

        time.sleep(0.5)

    if not found_any:
        print("\nNo common matches found across 20 pages (600 events per team)")
        # Show what dates each covers
        print(f"\nHome IDs ({len(all_home_ids)}): {sorted(all_home_ids.keys())[:3]}...{sorted(all_home_ids.keys())[-3:]}")
        print(f"Away IDs ({len(all_away_ids)}): {sorted(all_away_ids.keys())[:3]}...{sorted(all_away_ids.keys())[-3:]}")

    browser.close()

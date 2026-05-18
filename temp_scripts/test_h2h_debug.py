"""Debug: check what team events endpoints return and why intersection is empty."""
import sys, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
BASE = "https://api.sofascore.com/api/v1"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9"}

match_id = "15935010"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(1)

    # Get team IDs and names
    resp = ctx.request.get(f"{BASE}/event/{match_id}", headers=extra_headers, timeout=15_000)
    ev = resp.json().get("event", {})
    home_id = ev.get("homeTeam", {}).get("id")
    away_id = ev.get("awayTeam", {}).get("id")
    home_name = ev.get("homeTeam", {}).get("name")
    away_name = ev.get("awayTeam", {}).get("name")
    print(f"Match: {home_name} (ID={home_id}) vs {away_name} (ID={away_id})")

    # Fetch home events (limit 50)
    resp_h = ctx.request.get(f"{BASE}/team/{home_id}/events/last/50", headers=extra_headers, timeout=15_000)
    home_events = (resp_h.json() or {}).get("events", [])
    
    # Fetch away events (limit 50)
    resp_a = ctx.request.get(f"{BASE}/team/{away_id}/events/last/50", headers=extra_headers, timeout=15_000)
    away_events = (resp_a.json() or {}).get("events", [])

    print(f"\nHome events ({home_name}): {len(home_events)}")
    print(f"Away events ({away_name}): {len(away_events)}")

    home_ids = {str(e.get("id")) for e in home_events if e.get("id")}
    away_ids = {str(e.get("id")) for e in away_events if e.get("id")}
    common = home_ids & away_ids
    print(f"Common match IDs: {len(common)}")

    if common:
        for mid in sorted(common)[:10]:
            print(f"  Match ID: {mid}")
    else:
        print("\nNo common IDs found. Checking if home_ids contains the match_id...")
        print(f"  Current match_id {match_id} in home_ids? {match_id in home_ids}")
        print(f"  Current match_id {match_id} in away_ids? {match_id in away_ids}")
        
        # Show some home event IDs
        print(f"\nSample Home IDs (first 5): {sorted(list(home_ids))[:5]}")
        print(f"Sample Away IDs (first 5): {sorted(list(away_ids))[:5]}")
        
        # Check if the events are from different sports
        print("\nChecking sport for home events...")
        for ev in home_events[:3]:
            tournament = ev.get("tournament", {})
            sport = tournament.get("sport", {}).get("name", "?")
            print(f"  Home event {ev.get('id')}: sport={sport}, tournament={tournament.get('name')}")
        print("\nChecking sport for away events...")
        for ev in away_events[:3]:
            tournament = ev.get("tournament", {})
            sport = tournament.get("sport", {}).get("name", "?")
            print(f"  Away event {ev.get('id')}: sport={sport}, tournament={tournament.get('name')}")

    browser.close()

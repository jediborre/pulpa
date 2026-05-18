"""Check if substitution info exists anywhere (lineups, incidents, etc)."""
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

    # 1. Check lineups response for substitutions
    mid = "15935010"
    resp = ctx.request.get(f"{BASE}/event/{mid}/lineups", headers=extra_headers, timeout=15_000)
    if resp.ok:
        lups = resp.json()
        print("=== Lineups keys ===")
        print(f"  Top keys: {list(lups.keys())}")
        # Check for substitutions
        for k, v in lups.items():
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], dict):
                    print(f"  {k}[0] keys: {list(v[0].keys())}")
        # Check for substitutions in home/away
        for side in ("home", "away"):
            key = f"{side}Team" if f"{side}Team" in lups else side
            team_data = lups.get(key, {})
            if isinstance(team_data, dict):
                print(f"\n  {key} keys: {list(team_data.keys())[:20]}")
                # Check for substitutes or bench
                for sk in ("substitutes", "bench", "players", "starting"):
                    if sk in team_data:
                        items = team_data[sk]
                        print(f"    {sk}: {len(items)} items")
                        if items and isinstance(items[0], dict):
                            print(f"    sample keys: {list(items[0].keys())[:10]}")

    # 2. Also check if there's any substitution-like field in goal incidents
    resp = ctx.request.get(f"{BASE}/event/{mid}/incidents", headers=extra_headers, timeout=15_000)
    incs = resp.json().get("incidents", [])
    for inc in incs:
        if inc.get("incidentType") not in ("goal", "period"):
            print(f"\nNon-standard incident: {json.dumps(inc, default=str)[:300]}")
            break
    else:
        print("\nNo non-standard incidents found")

    # 3. Check the event page for roster changes
    for side in ("homeTeam", "awayTeam"):
        resp = ctx.request.get(f"{BASE}/event/{mid}", headers=extra_headers, timeout=15_000)
        ev = resp.json().get("event", {})
        print(f"\n=== {side} from event ===")
        team = ev.get(side, {})
        print(f"  keys: {list(team.keys())}")
    
    browser.close()

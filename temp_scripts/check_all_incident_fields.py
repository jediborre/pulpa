"""Check ALL fields in ALL incidents to find any substitution data."""
import sys, json
from pathlib import Path
from collections import Counter
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

    # Try multiple matches from different leagues
    mids = ["15935010", "14491671"]
    for mid in mids:
        resp = ctx.request.get(f"{BASE}/event/{mid}/incidents", headers=extra_headers, timeout=15_000)
        incs = resp.json().get("incidents", [])

        # Collect ALL unique keys across ALL incidents
        all_keys = Counter()
        for inc in incs:
            for k in inc.keys():
                all_keys[k] += 1

        print(f"\n=== Match {mid} ===")
        print(f"Incidents: {len(incs)}")
        print(f"All keys: {dict(all_keys)}")

        # Check goal incidents for extra fields
        goal_inc = [i for i in incs if i.get("incidentType") == "goal"]
        if goal_inc:
            g = goal_inc[0]
            # Show everything
            print(f"\nGoal incident sample (full):")
            print(json.dumps(g, indent=2, default=str)[:1000])

    browser.close()

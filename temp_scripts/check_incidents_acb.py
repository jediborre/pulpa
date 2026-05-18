"""Check incident types across different leagues."""
import sys, json
from pathlib import Path
ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))
from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE = "https://api.sofascore.com/api/v1"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json"}

matches = [
    ("15935010", "NBA Playoffs"),
    ("14491671", "ACB Spain"),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)

    for mid, label in matches:
        resp = ctx.request.get(f"{BASE}/event/{mid}/incidents", headers=extra_headers, timeout=15_000)
        incs = resp.json().get("incidents", [])
        types = {}
        for inc in incs:
            it = inc.get("incidentType", "?")
            types[it] = types.get(it, 0) + 1

        print(f"\n=== {label} (ID={mid}) ===")
        print(f"  Types: {json.dumps(types)}")
        for inc in incs:
            it = inc.get("incidentType")
            if it not in ("goal", "period"):
                print(f"  sample {it}: {json.dumps({k: v for k, v in inc.items() if k != 'player'}, default=str)[:400]}")
                if "player" in inc:
                    print(f"    player: {inc['player'].get('name')} (id={inc['player'].get('id')})")
                break

    browser.close()

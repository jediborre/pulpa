"""Check if incidents include player info (who scored, who fouled, etc)."""
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

    resp = ctx.request.get(f"{BASE}/event/15935010/incidents", headers=extra_headers, timeout=15_000)
    incs = resp.json().get("incidents", [])

    # Show unique incident types and their keys
    types = {}
    for inc in incs:
        itype = inc.get("incidentType", "?")
        if itype not in types:
            types[itype] = {"count": 0, "keys": list(inc.keys())}
        types[itype]["count"] += 1

    print("=== Incident types ===")
    for itype, info in sorted(types.items()):
        print(f"\n{itype} ({info['count']}):")
        print(f"  keys: {info['keys']}")

        # Show first example of each type
        for inc in incs:
            if inc.get("incidentType") == itype:
                print(f"  sample: {json.dumps(inc, indent=4)[:500]}")
                break

    browser.close()

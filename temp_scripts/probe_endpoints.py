"""
Probe SofaScore API to discover available endpoints for
statistics, lineups, odds, and player data.
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))
sys.path.insert(0, str(ROOT / "match" / "training"))

from playwright.sync_api import sync_playwright

MATCH_ID = "14491671"  # BAXI Manresa vs Basket Zaragoza
BASE = "https://api.sofascore.com/api/v1"

ENDPOINTS_TO_PROBE = [
    # Known working
    f"/event/{MATCH_ID}",
    f"/event/{MATCH_ID}/incidents",
    f"/event/{MATCH_ID}/graph",
    f"/event/{MATCH_ID}/h2h",
    # Statistics — common patterns
    f"/event/{MATCH_ID}/statistics",
    f"/event/{MATCH_ID}/statistics/team",
    f"/event/{MATCH_ID}/statistics/top",
    f"/event/{MATCH_ID}/statistics/full",
    # Player statistics
    f"/event/{MATCH_ID}/players/statistics",
    f"/event/{MATCH_ID}/player/statistics",
    # Lineups
    f"/event/{MATCH_ID}/lineups",
    f"/event/{MATCH_ID}/lineup",
    # Odds / betting
    f"/event/{MATCH_ID}/odds",
    f"/event/{MATCH_ID}/odds/latest",
    f"/event/{MATCH_ID}/odds/all",
    f"/event/{MATCH_ID}/odds/1",
    # Team info (uses team IDs from event response)
    # Player info
    f"/event/{MATCH_ID}/players",
    # Also try with /v1 removed
]

extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

results = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ))
    page = ctx.new_page()

    # Warm up
    try:
        page.goto("https://www.sofascore.com/basketball",
                   wait_until="networkidle", timeout=45_000)
    except Exception:
        pass

    for ep in ENDPOINTS_TO_PROBE:
        url = f"{BASE}{ep}"
        try:
            resp = ctx.request.get(url, headers=extra_headers, timeout=15_000)
            status = resp.status
            body = {}
            if resp.ok:
                body = resp.json()
            results.append({
                "endpoint": ep,
                "status": status,
                "ok": resp.ok,
                "keys_preview": list(body.keys())[:10] if isinstance(body, dict) else "not_a_dict",
                "sample": json.dumps(body, indent=2, ensure_ascii=False)[:2000] if resp.ok else "",
            })
        except Exception as e:
            results.append({
                "endpoint": ep,
                "status": "ERROR",
                "ok": False,
                "keys_preview": str(e),
                "sample": "",
            })

    browser.close()

# Print results
print(f"\n{'='*80}")
print(f"  Endpoint Probe Results for match {MATCH_ID}")
print(f"{'='*80}")
for r in results:
    status_str = f"HTTP {r['status']}" if isinstance(r['status'], int) else f"ERR {r['status']}"
    ok_str = "[OK]" if r['ok'] else "[FAIL]"
    print(f"\n{ok_str} {r['endpoint']}  ({status_str})")
    if r['ok']:
        print(f"  Keys: {r['keys_preview']}")
        # Show first 500 chars of the actual data
        lines = r['sample'].split('\n')
        print(f"  Preview:")
        for line in lines[:25]:
            print(f"    {line}")
        if len(lines) > 25:
            print(f"    ... ({len(lines) - 25} more lines)")

"""
Deep probe: show full structure of statistics, lineups, event JSON,
and try additional patterns for player stats + H2H match history.
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))
sys.path.insert(0, str(ROOT / "match" / "training"))

from playwright.sync_api import sync_playwright

MATCH_ID = "14491671"
BASE = "https://api.sofascore.com/api/v1"

MORE_ENDPOINTS = [
    # Already known working - inspect more fully
    f"/event/{MATCH_ID}/statistics",
    f"/event/{MATCH_ID}/lineups",
    # Player statistics — alternative patterns
    f"/event/{MATCH_ID}/playerStats",
    f"/event/{MATCH_ID}/playerstats",
    f"/event/{MATCH_ID}/player-stats",
    f"/event/{MATCH_ID}/team/players",
    # H2H match history (different patterns)
    f"/event/{MATCH_ID}/h2h/matches",
    f"/event/{MATCH_ID}/h2h/events",
    f"/event/{MATCH_ID}/h2h-history",
    # Team endpoints (using team IDs from event response)
    # Odds
    f"/event/{MATCH_ID}/betting",
    f"/event/{MATCH_ID}/prediction",
    # Incidents with more params
    f"/event/{MATCH_ID}/incidents?withStats=true",
    # SofaScore rating per player — check main event for inline stats
]

extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

results = {}
all_responses = {}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ))
    page = ctx.new_page()

    try:
        page.goto("https://www.sofascore.com/basketball",
                   wait_until="networkidle", timeout=45_000)
    except Exception:
        pass

    for ep in MORE_ENDPOINTS:
        url = f"{BASE}{ep}"
        try:
            resp = ctx.request.get(url, headers=extra_headers, timeout=20_000)
            body = resp.json() if resp.ok else {}
            results[ep] = {"status": resp.status, "ok": resp.ok}
            if resp.ok:
                all_responses[ep] = body
        except Exception as e:
            results[ep] = {"status": "ERROR", "ok": False, "error": str(e)}

    browser.close()

# --- Print results for new endpoints ---
print(f"\n{'='*80}")
print(f"  Deep Probe — New Endpoints")
print(f"{'='*80}")
for ep, info in results.items():
    s = f"HTTP {info['status']}" if isinstance(info['status'], int) else f"ERR"
    ok = "[OK]" if info['ok'] else "[FAIL]"
    print(f"\n{ok} {ep}  ({s})")

# --- Dump the FULL statistics structure ---
if "/event/14491671/statistics" in all_responses:
    print(f"\n{'='*80}")
    print(f"  FULL STATISTICS RESPONSE")
    print(f"{'='*80}")
    stats = all_responses["/event/14491671/statistics"]
    print(json.dumps(stats, indent=2, ensure_ascii=False)[:5000])

# --- Dump the FULL lineups structure ---
if "/event/14491671/lineups" in all_responses:
    print(f"\n{'='*80}")
    print(f"  FULL LINEUPS RESPONSE")
    print(f"{'='*80}")
    lineups = all_responses["/event/14491671/lineups"]
    print(json.dumps(lineups, indent=2, ensure_ascii=False)[:5000])

# --- Also check the main event for inline player stats/ratings ---
print(f"\n{'='*80}")
print(f"  MAIN EVENT — checking for inline statistics/ratings")
print(f"{'='*80}")
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ))
    page = ctx.new_page()
    try:
        page.goto("https://www.sofascore.com/basketball",
                   wait_until="networkidle", timeout=45_000)
    except Exception:
        pass
    resp = ctx.request.get(
        f"{BASE}/event/{MATCH_ID}",
        headers=extra_headers, timeout=15_000
    )
    if resp.ok:
        ev = resp.json().get("event", {})
        top_keys = list(ev.keys())
        print(f"\nTop-level event keys: {top_keys}")
        # Check for any statistics/rating fields
        for key in top_keys:
            val = ev[key]
            if isinstance(val, (dict, list)):
                print(f"\n  {key}: type={type(val).__name__}, "
                      f"keys/len={list(val.keys())[:15] if isinstance(val, dict) else len(val)}")
    browser.close()

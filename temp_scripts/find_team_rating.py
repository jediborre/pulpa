"""Find team SofaScore rating endpoint."""
import json, sys
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

MATCH_ID = "14491671"
BASE = "https://api.sofascore.com/api/v1"
HOME_TEAM_ID = 3560
AWAY_TEAM_ID = 25134

extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Team rating endpoint patterns
patterns = [
    # Team info
    f"/team/{HOME_TEAM_ID}",
    f"/team/{HOME_TEAM_ID}/statistics",
    f"/team/{HOME_TEAM_ID}/statistics/season",
    # Event-based
    f"/event/{MATCH_ID}/team-statistics",
    f"/event/{MATCH_ID}/teamStats",
    f"/event/{MATCH_ID}/team/stats",
    # Rating-specific
    f"/event/{MATCH_ID}/ratings/team",
    f"/event/{MATCH_ID}/team/rating",
    # Maybe it's in the statistics with a different period
    f"/event/{MATCH_ID}/statistics?period=Q4",
    f"/event/{MATCH_ID}/statistics/4",
    # Season stats
    f"/team/{HOME_TEAM_ID}/seasons",
    f"/team/{HOME_TEAM_ID}/performance",
]

out = {}

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

    for ep in patterns:
        url = f"{BASE}{ep}"
        try:
            resp = ctx.request.get(url, headers=extra_headers, timeout=15_000)
            ok = resp.ok
            body = resp.json() if ok else None
            status = resp.status
            out[ep] = {"status": status, "ok": ok}
            if ok and isinstance(body, dict):
                out[ep]["keys"] = list(body.keys())[:15]
                # Look for rating in the response
                if "team" in body:
                    team = body.get("team", body)
                    if isinstance(team, dict):
                        for rk in ["rating", "averageRating", "sofascoreRating", "score"]:
                            if rk in team:
                                out[ep][f"found_{rk}"] = team[rk]
                # Direct check
                for rk in ["rating", "averageRating", "sofascoreRating", "score"]:
                    if rk in body:
                        out[ep][f"found_{rk}"] = body[rk]
                # Check if statistic groups have rating
                if "statistics" in body:
                    out[ep]["statistics_keys"] = list(body["statistics"].keys())[:10]
                    if isinstance(body["statistics"], list):
                        for item in body["statistics"][:3]:
                            if isinstance(item, dict):
                                out[ep]["stat_item_keys"] = list(item.keys())[:10]
        except Exception as e:
            out[ep] = {"status": "ERROR", "ok": False}

    browser.close()

# Print results
for ep, info in out.items():
    ok_str = "[OK]" if info.get("ok") else "[FAIL]"
    print(f"{ok_str} {ep}  (HTTP {info.get('status')})")
    if info.get("keys"):
        print(f"       Keys: {info['keys']}")
    for k, v in info.items():
        if k.startswith("found_"):
            print(f"       {k}: {v}")
    if "statistics_keys" in info:
        print(f"       stats keys: {info['statistics_keys']}")
    if "stat_item_keys" in info:
        print(f"       stat item keys: {info['stat_item_keys']}")

# Also check the main statistics response for a rating stat
print(f"\n{'='*60}")
print("  Also checking statistics for composite 'rating' stat...")
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
        f"{BASE}/event/{MATCH_ID}/statistics",
        headers=extra_headers, timeout=15_000
    )
    if resp.ok:
        stats = resp.json()
        all_keys = set()
        for period_group in stats.get("statistics", []):
            for g in period_group.get("groups", []):
                for item in g.get("statisticsItems", []):
                    all_keys.add(item.get("key", ""))
        print(f"All stat keys: {sorted(all_keys)}")
    browser.close()

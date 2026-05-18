"""
Deep probe v2 — save outputs to file, explore event JSON for player stats,
try more endpoint patterns.
"""
import json, sys
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

MATCH_ID = "14491671"
BASE = "https://api.sofascore.com/api/v1"

# More endpoint patterns to try
EXTRA = [
    f"/event/{MATCH_ID}/statistics/players",
    f"/event/{MATCH_ID}/statistics/player",
    f"/event/{MATCH_ID}/statistics/1",
    f"/event/{MATCH_ID}/statistics/2",
    # H2H real match list
    f"/team/0/events/h2h/{MATCH_ID}",    # maybe h2h is under team
    f"/event/{MATCH_ID}/duels",
    f"/event/{MATCH_ID}/history",
    # Odds patterns
    f"/odds/event/{MATCH_ID}",
    f"/betting/event/{MATCH_ID}",
    # Player ratings
    f"/event/{MATCH_ID}/ratings",
    f"/event/{MATCH_ID}/playerRatings",
    # Team seasons / roster
    f"/event/{MATCH_ID}/statistics/roster",
    # Try without v1
]

extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

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

    # 1. Try extra endpoints
    for ep in EXTRA:
        try:
            resp = ctx.request.get(f"{BASE}{ep}", headers=extra_headers, timeout=15_000)
            if resp.ok:
                out[ep] = {"status": resp.status, "data": resp.json()}
            else:
                out[ep] = {"status": resp.status, "data": None}
        except Exception as e:
            out[ep] = {"status": "ERROR", "data": str(e)}

    # 2. Get main event and inspect for statistics/ratings
    resp = ctx.request.get(f"{BASE}/event/{MATCH_ID}", headers=extra_headers, timeout=15_000)
    if resp.ok:
        ev = resp.json().get("event", {})
        out["_event_keys"] = list(ev.keys())
        # Check for statistics or rating fields
        for suspicious_key in ["statistics", "ratings", "playerStats", "playerRatings",
                                "homeRating", "awayRating", "homeStatistics", "awayStatistics",
                                "teams", "managers", "highlights"]:
            if suspicious_key in ev:
                out[f"_event_{suspicious_key}"] = ev[suspicious_key]
        # Check homeTeam and awayTeam for embedded rating
        for side in ["homeTeam", "awayTeam"]:
            team = ev.get(side, {})
            team_keys = list(team.keys())
            out[f"_event_{side}_keys"] = team_keys
            for sus in ["rating", "score", "statistics", "players", "id"]:
                if sus in team:
                    out[f"_event_{side}_{sus}"] = team[sus]

    # 3. Get lineups and save full response
    resp_lu = ctx.request.get(f"{BASE}/event/{MATCH_ID}/lineups", headers=extra_headers, timeout=15_000)
    if resp_lu.ok:
        out["_lineups"] = resp_lu.json()

    # 4. Get statistics and save full response
    resp_st = ctx.request.get(f"{BASE}/event/{MATCH_ID}/statistics", headers=extra_headers, timeout=15_000)
    if resp_st.ok:
        out["_statistics"] = resp_st.json()

    # 5. Get h2h and save
    resp_h2h = ctx.request.get(f"{BASE}/event/{MATCH_ID}/h2h", headers=extra_headers, timeout=15_000)
    if resp_h2h.ok:
        out["_h2h"] = resp_h2h.json()

    browser.close()

# Save to file to avoid encoding issues
out_path = ROOT / "temp_scripts" / "probe_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
print(f"Saved to {out_path}")

# Print a summary
for ep, info in out.items():
    if ep.startswith("_"):
        continue
    status = info.get("status", "?")
    ok = "[OK]" if info.get("data") else "[FAIL]"
    print(f"{ok} {ep}  (HTTP {status})")

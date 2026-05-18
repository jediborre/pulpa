"""Check if /team/{id}/events/last/{n} includes period/quarter scores."""
import json, sys
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

BASE = "https://api.sofascore.com/api/v1"
HOME_TEAM_ID = 3560
AWAY_TEAM_ID = 25134

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

    # Fetch recent events for BOTH teams
    for tid, label in [(HOME_TEAM_ID, "home_team"), (AWAY_TEAM_ID, "away_team")]:
        resp = ctx.request.get(
            f"{BASE}/team/{tid}/events/last/10",
            headers=extra_headers, timeout=15_000
        )
        if resp.ok:
            body = resp.json()
            events = body.get("events", [])
            out[label] = {"n_events": len(events)}
            if events:
                ev = events[0]
                hs = ev.get("homeScore") or {}
                as_ = ev.get("awayScore") or {}
                out[label]["homeScore_keys"] = list(hs.keys())
                out[label]["awayScore_keys"] = list(as_.keys())
                out[label]["event_keys"] = list(ev.keys())[:15]
                out[label]["has_period_scores"] = any(k.startswith("period") for k in hs.keys())

    # Also fetch the full event for a specific match to see period keys
    # Find common matches
    home_resp = ctx.request.get(
        f"{BASE}/team/{HOME_TEAM_ID}/events/last/20",
        headers=extra_headers, timeout=15_000
    )
    away_resp = ctx.request.get(
        f"{BASE}/team/{AWAY_TEAM_ID}/events/last/20",
        headers=extra_headers, timeout=15_000
    )

    if home_resp.ok and away_resp.ok:
        home_ids = {str(e.get("id")) for e in home_resp.json().get("events", []) if e.get("id")}
        away_ids = {str(e.get("id")) for e in away_resp.json().get("events", []) if e.get("id")}
        common = home_ids & away_ids
        out["common_matches"] = list(common)

        # For a common match, fetch the full event to see period scores
        if common:
            mid = list(common)[0]
            resp = ctx.request.get(
                f"{BASE}/event/{mid}",
                headers=extra_headers, timeout=15_000
            )
            if resp.ok:
                ev = resp.json().get("event", {})
                hs = ev.get("homeScore") or {}
                out["full_event_homeScore_keys"] = list(hs.keys())
                # Check for period scores
                periods = {k: hs[k] for k in hs.keys() if k.startswith("period")}
                out["full_event_period_scores"] = periods

    browser.close()

# Save
with open(ROOT / "temp_scripts" / "team_events_check.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

print(json.dumps(out, indent=2, ensure_ascii=False))

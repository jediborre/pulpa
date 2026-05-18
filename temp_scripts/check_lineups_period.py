"""Check lineups response for per-period player statistics."""
import json, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
from playwright.sync_api import sync_playwright

STANDARD_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
extra_headers = {
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

match_id = "14491671"
BASE = "https://api.sofascore.com/api/v1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball",
               wait_until="domcontentloaded", timeout=30_000)
    time.sleep(2)

    # Get lineups
    resp = ctx.request.get(
        f"{BASE}/event/{match_id}/lineups",
        headers=extra_headers, timeout=20_000
    )
    if resp.ok:
        data = resp.json()
        home = data.get("home", {}).get("players", [])
        away = data.get("away", {}).get("players", [])

        print(f"Home players: {len(home)}, Away players: {len(away)}")

        # Check all keys in a player statistics dict
        for p in (home + away)[:1]:
            stats = p.get("statistics", {})
            print(f"\nPlayer: {p.get('player',{}).get('name')}")
            print(f"Stat keys: {sorted(stats.keys())}")

            # Check for per-period keys
            period_keys = [k for k in stats.keys() if any(x in k.lower() for x in ["period", "quarter", "q1", "q2", "q3", "q4"])]
            print(f"Period-related keys: {period_keys}")

            # Show full stat dict
            print(f"Full stats:")
            print(json.dumps(stats, indent=2, ensure_ascii=False)[:2000])

        # Search for ANY per-period data
        all_keys = set()
        for p in home + away:
            stats = p.get("statistics", {})
            all_keys.update(stats.keys())
        print(f"\nAll stat keys across all players: {sorted(all_keys)}")

    browser.close()

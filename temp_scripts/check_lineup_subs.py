"""Check lineup player details - substitute flag + timing."""
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

    mid = "15935010"
    resp = ctx.request.get(f"{BASE}/event/{mid}/lineups", headers=extra_headers, timeout=15_000)
    lups = resp.json()

    for side in ("home", "away"):
        team = lups.get(side, {})
        players = team.get("players", [])
        print(f"\n=== {side} ({len(players)} players) ===")
        for p in players:
            pobj = p.get("player", {})
            substitute = p.get("substitute")
            stats = p.get("statistics", {})
            played = stats.get("minutesPlayed", "?")
            print(f"  #{p.get('jerseyNumber','?')} {pobj.get('shortName','?'):20s} sub={substitute} min={played}")
            # If substitute=True, show all stats keys
            if substitute:
                print(f"    stats keys: {list(stats.keys())[:15]}")
                # Check for substitution time
                for time_key in ("substitutionIn", "substitutionOut", "timePlayed", "mins"):
                    if time_key in stats:
                        print(f"    {time_key}: {stats[time_key]}")

    browser.close()

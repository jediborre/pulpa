"""Deep dive into team data - pregameForm, rankings, ratings."""
import sys, time, json
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE = "https://api.sofascore.com/api/v1"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json"}

team_ids = {"Pistons": 3424, "Cavaliers": 3432}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(1)

    for name, tid in team_ids.items():
        print(f"\n{'='*60}")
        print(f"=== {name} (ID={tid}) ===")

        # Full team base
        resp = ctx.request.get(f"{BASE}/team/{tid}", headers=extra_headers, timeout=15_000)
        body = resp.json()
        team = body.get("team", {})
        pf = body.get("pregameForm", {})

        print(f"\n--- Team Info ---")
        for k in ["name", "shortName", "slug", "gender", "national", "country"]:
            print(f"  {k}: {team.get(k)}")
        # Ranking data
        print(f"\n--- Ranking ---")
        rankings = team.get("ranking", {})
        if rankings:
            print(f"  {json.dumps(rankings, indent=4)}")
        else:
            print(f"  (none)")

        # Any team statistics/field
        print(f"\n--- Other team keys ---")
        other_keys = [k for k in team.keys() if k not in ("name", "shortName", "slug", "gender", "national", "country", "ranking", "sport", "category", "tournament", "primaryUniqueTournament")]
        for k in other_keys:
            print(f"  {k}: {json.dumps(team[k], default=str)[:200]}")

        print(f"\n--- PregameForm ---")
        print(f"  {json.dumps(pf, indent=4)}")

        # Performance data
        resp = ctx.request.get(f"{BASE}/team/{tid}/performance", headers=extra_headers, timeout=15_000)
        perf = resp.json()
        perf_events = perf.get("events", [])
        perf_points = perf.get("points", {})
        print(f"\n--- Performance ---")
        print(f"  Events: {len(perf_events)}")
        if perf_points:
            print(f"  Points keys: {list(perf_points.keys())[:5]}")
            for k, v in list(perf_points.items())[:3]:
                print(f"    {k}: {v}")
        # What's in each event?
        if perf_events:
            ev = perf_events[0]
            print(f"  Event[0] keys: {list(ev.keys())}")
            print(f"    homeTeam: {ev.get('homeTeam',{}).get('name')} (seed={ev.get('homeTeamSeed')})")
            print(f"    awayTeam: {ev.get('awayTeam',{}).get('name')} (seed={ev.get('awayTeamSeed')})")
            print(f"    homeScore: {ev.get('homeScore')}")
            print(f"    awayScore: {ev.get('awayScore')}")
            print(f"    tournament: {ev.get('tournament',{}).get('name')}")
            print(f"    status: {ev.get('status',{}).get('description') if ev.get('status') else '?'}")

    browser.close()

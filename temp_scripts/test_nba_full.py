"""Test what data the scraper extracts for an NBA match (periods + team stats)."""
import sys, json, time
from pathlib import Path

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match"))

from playwright.sync_api import sync_playwright

STANDARD_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
extra_headers = {"Referer": "https://www.sofascore.com/", "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}
BASE = "https://api.sofascore.com/api/v1"

# NBA match
match_id = "15935010"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=STANDARD_UA)
    page = ctx.new_page()
    page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30_000)
    time.sleep(2)

    # 1. Fetch event
    resp = ctx.request.get(f"{BASE}/event/{match_id}", headers=extra_headers, timeout=15_000)
    event_json = resp.json() if resp.ok else {}

    # 2. Fetch statistics (this has per-period data)
    resp = ctx.request.get(f"{BASE}/event/{match_id}/statistics", headers=extra_headers, timeout=15_000)
    stats_json = resp.json() if resp.ok else {}

    # 3. Fetch incidents
    resp = ctx.request.get(f"{BASE}/event/{match_id}/incidents", headers=extra_headers, timeout=15_000)
    incidents_json = resp.json() if resp.ok else {}

    # 4. Fetch lineups
    resp = ctx.request.get(f"{BASE}/event/{match_id}/lineups", headers=extra_headers, timeout=15_000)
    lineups_json = resp.json() if resp.ok else {}

    # 5. Fetch odds
    resp = ctx.request.get(f"{BASE}/event/{match_id}/odds/1/all", headers=extra_headers, timeout=15_000)
    odds_json = resp.json() if resp.ok else {}

    # 6. Fetch h2h
    resp = ctx.request.get(f"{BASE}/event/{match_id}/h2h", headers=extra_headers, timeout=15_000)
    h2h_json = resp.json() if resp.ok else {}

    # Now parse using our functions
    from scraper import (
        _parse, _parse_team_statistics, _parse_period_stats,
        _parse_lineups, _parse_odds, _parse_h2h
    )

    # Event parse
    parsed = _parse(event_json, incidents_json.get("incidents", []), [])

    # Team statistics (GAME totals + PERIOD data)
    team_stats = _parse_team_statistics(match_id, stats_json)
    
    # Period stats fallback (from incidents)
    period_stats = _parse_period_stats(match_id, incidents_json.get("incidents", []))

    # Lineups + player stats
    lineup_rows, player_stat_rows = _parse_lineups(match_id, lineups_json)

    # Odds
    odds = _parse_odds(match_id, odds_json)

    # H2H
    h2h = _parse_h2h(match_id, h2h_json)

    print(f"=== NBA Match {match_id} ===")
    print(f"Score: {parsed['score']}")
    print(f"Incidents: {len(parsed['events'])}")
    print(f"\n=== Team Statistics per Period ===")
    for ts in team_stats:
        if ts["stat_name"] in ("Free throws", "2 pointers", "3 pointers", 
                                "Field goals", "Rebounds", "Assists",
                                "Turnovers", "Steals", "Blocks", "Fouls",
                                "Timeouts", "Defensive rebounds", "Offensive rebounds",
                                "Max points in a row", "Time spent in lead", "Biggest lead"):
            print(f"  {ts['period']:4s} | {ts['stat_name']:24s} | {ts['home_display']:>20s} vs {ts['away_display']:>20s}")

    print(f"\n=== Period Stats Fallback (from incidents) ===")
    for ps in period_stats:
        print(f"  {ps['period']} | {ps['stat_name']:20s} | home={ps['home_value']} away={ps['away_value']}")

    print(f"\n=== Player Stats (sample) ===")
    for ps in player_stat_rows[:3]:
        print(f"  {ps['team']:4s} | {ps['player_name']:25s} | rating={ps['sofascore_rating']} pts={ps['points']} fouls={ps['fouls']} +/-={ps['plus_minus']} min={ps['minutes_played']}")

    print(f"\n=== Odds ===")
    for o in odds:
        print(f"  {o['market_name']:25s} | {o['market_period']:15s} | home={o['home_value']} away={o['away_value']}")

    print(f"\n=== H2H ===")
    for h in h2h[:3]:
        print(f"  {h}")

    print(f"\n=== Team Stats Periods Available ===")
    from collections import Counter
    periods = Counter(ts["period"] for ts in team_stats)
    print(f"  {dict(periods)}")

    browser.close()

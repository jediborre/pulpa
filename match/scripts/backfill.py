"""
Backfill: re-fetch existing matches reusing ONE Playwright session.

Usage:
    python -m match.scripts.backfill path/to/db.sqlite --all
    python -m match.scripts.backfill path/to/db.sqlite --ids 15935010 --limit 5
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "match"))

from db import get_conn, save_match
from scraper import (
    _browser_context,
    _parse, _parse_h2h, _parse_team_statistics, _parse_period_stats,
    _parse_lineups, _parse_odds,
    STANDARD_UA,
)

BASE = "https://api.sofascore.com/api/v1"
_HEADERS = {"Referer": "https://www.sofascore.com/", "Accept": "application/json, text/plain, */*"}


def _fetch_json(ctx, url: str) -> dict | None:
    try:
        resp = ctx.request.get(url, headers=_HEADERS, timeout=20_000)
        return resp.json() if resp.ok else None
    except Exception:
        return None


def _team_strength(ctx, home_tid, away_tid, home_name, away_name):
    """Fetch pregameForm + performance points for both teams."""
    rows = []
    for tid, tname in ((home_tid, home_name), (away_tid, away_name)):
        row = {"team_id": int(tid), "team_name": tname}
        # pregameForm
        body = _fetch_json(ctx, f"{BASE}/team/{tid}")
        if body:
            pf = body.get("pregameForm") or {}
            row["position"] = pf.get("position")
            val = (pf.get("value") or "").split("-")
            row["wins"] = int(val[0]) if len(val) > 0 and val[0].isdigit() else None
            row["losses"] = int(val[1]) if len(val) > 1 and val[1].isdigit() else None
            row["form"] = json.dumps(pf.get("form", []))
        # performance points
        body = _fetch_json(ctx, f"{BASE}/team/{tid}/performance")
        if body:
            row["perf_points"] = json.dumps(body.get("points") or {})
        rows.append(row)
    return rows


def _h2h_via_teams(ctx, home_tid, away_tid, limit=0):
    """Discover H2H matches with quarter scores via shared session."""
    from datetime import datetime, timezone
    h_events = (_fetch_json(ctx, f"{BASE}/team/{home_tid}/events/last/{limit}") or {}).get("events", [])
    a_events = (_fetch_json(ctx, f"{BASE}/team/{away_tid}/events/last/{limit}") or {}).get("events", [])

    h_ids = {str(e.get("id")) for e in h_events}
    a_ids = {str(e.get("id")) for e in a_events}
    common = h_ids & a_ids

    seen = set()
    results = []
    for ev in h_events + a_events:
        mid = str(ev.get("id"))
        if mid not in common or mid in seen:
            continue
        seen.add(mid)
        hs, as_ = ev.get("homeScore") or {}, ev.get("awayScore") or {}
        ts = ev.get("startTimestamp", 0)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        results.append({
            "h2h_match_id": mid,
            "date": dt.strftime("%Y-%m-%d") if dt else "",
            "timestamp": ts or None,
            "home_team": (ev.get("homeTeam") or {}).get("name", ""),
            "away_team": (ev.get("awayTeam") or {}).get("name", ""),
            "home_score": hs.get("current", hs.get("normaltime")),
            "away_score": as_.get("current", as_.get("normaltime")),
            "q1_home": hs.get("period1"), "q1_away": as_.get("period1"),
            "q2_home": hs.get("period2"), "q2_away": as_.get("period2"),
            "q3_home": hs.get("period3"), "q3_away": as_.get("period3"),
            "q4_home": hs.get("period4"), "q4_away": as_.get("period4"),
            "tournament": (ev.get("tournament") or {}).get("name", ""),
        })
    return results


def _all_payloads(ctx, match_id: str):
    """Fetch ALL API payloads for a match."""
    return (
        _fetch_json(ctx, f"{BASE}/event/{match_id}") or {},
        _fetch_json(ctx, f"{BASE}/event/{match_id}/incidents") or {"incidents": []},
        _fetch_json(ctx, f"{BASE}/event/{match_id}/graph") or {"graphPoints": []},
        _fetch_json(ctx, f"{BASE}/event/{match_id}/h2h"),
        _fetch_json(ctx, f"{BASE}/event/{match_id}/statistics"),
        _fetch_json(ctx, f"{BASE}/event/{match_id}/lineups"),
        _fetch_json(ctx, f"{BASE}/event/{match_id}/odds/1/all"),
    )


def backfill(
    db_path: str,
    match_ids: list[str],
    delay: float = 0.5,
    backend: str | None = None,
) -> None:
    conn = get_conn(db_path)
    total = len(match_ids)
    ok = err = 0

    with _browser_context("https://www.sofascore.com/basketball", backend=backend) as (_, ctx, _page):
        for idx, mid in enumerate(match_ids):
            try:
                print(f"[{idx + 1}/{total}] {mid}...", end=" ", flush=True)

                event_json, incidents_json, graph_json, h2h_json, statistics_json, lineups_json, odds_json = \
                    _all_payloads(ctx, mid)

                if not event_json or "event" not in event_json:
                    print("SKIP (no event)")
                    err += 1
                    continue

                ev = event_json.get("event") or {}
                home_tid = (ev.get("homeTeam") or {}).get("id")
                away_tid = (ev.get("awayTeam") or {}).get("id")
                home_name = (ev.get("homeTeam") or {}).get("name", "")
                away_name = (ev.get("awayTeam") or {}).get("name", "")

                # H2H via teams (preferred)
                h2h_rows = _parse_h2h(mid, h2h_json)
                if home_tid and away_tid and home_tid != away_tid:
                    team_h2h = _h2h_via_teams(ctx, home_tid, away_tid)
                    if team_h2h:
                        for r in team_h2h:
                            r["match_id"] = mid
                        h2h_rows = team_h2h

                # Team strength
                team_strength_rows = _team_strength(ctx, home_tid, away_tid, home_name, away_name) \
                    if home_tid and away_tid else []

                # Parse all
                parsed = _parse(event_json, incidents_json.get("incidents", []), graph_json.get("graphPoints", []))
                parsed["h2h"] = h2h_rows
                parsed["team_statistics"] = _parse_team_statistics(mid, statistics_json) if statistics_json else []
                parsed["period_stats"] = _parse_period_stats(
                    mid, incidents_json.get("incidents", []),
                    period_seconds=(ev.get("time") or {}).get("periodLength", 600),
                )
                if lineups_json:
                    lr, pr = _parse_lineups(mid, lineups_json)
                    parsed["lineups"], parsed["player_stats"] = lr, pr
                else:
                    parsed["lineups"] = parsed["player_stats"] = []
                parsed["odds"] = _parse_odds(mid, odds_json) if odds_json else []
                parsed["team_strength"] = team_strength_rows

                save_match(conn, mid, parsed)
                try:
                    from db import mark_discovered_processed
                    mark_discovered_processed(conn, mid)
                except Exception:
                    pass
                ok += 1
                print("OK")
                time.sleep(delay)

            except Exception as e:
                try:
                    from db import mark_discovered_error
                    mark_discovered_error(conn, mid, str(e))
                except Exception:
                    pass
                err += 1
                print(f"ERROR: {e}")

    print(f"\nDone. OK={ok}, Errors={err}, Total={total}")
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("db")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--backend", choices=["traditional", "obscura"], default="traditional")
    args = parser.parse_args()

    conn = get_conn(args.db)
    if args.all:
        rows = conn.execute("SELECT match_id FROM matches ORDER BY date DESC").fetchall()
        match_ids = [r["match_id"] for r in rows]
    elif args.ids:
        match_ids = args.ids
    else:
        print("Specify --ids or --all"); sys.exit(1)
    conn.close()

    if args.limit > 0:
        match_ids = match_ids[:args.limit]

    print(f"Backfilling {len(match_ids)} matches in {args.db}")
    backfill(args.db, match_ids, delay=args.delay, backend=args.backend)

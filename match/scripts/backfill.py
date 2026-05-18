"""
Backfill: re-fetch existing matches reusing ONE browser session.

Usage:
    python -m match.scripts.backfill path/to/db.sqlite --all
    python -m match.scripts.backfill path/to/db.sqlite --ids 15935010 --limit 5
"""
import sys, json, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "match"))

from db import get_conn, init_db, save_match
from scraper import (
    _browser_context,
    _parse, _parse_h2h, _parse_team_statistics, _parse_period_stats,
    _parse_lineups, _parse_odds,
)
from tqdm import tqdm

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


def _group_rows_by_date(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("date") or "unknown")].append(row)
    return dict(grouped)


def _match_refresh_state(conn, match_id: str) -> dict[str, int | bool]:
    match_row = conn.execute(
        "SELECT status_type FROM matches WHERE match_id = ?",
        (match_id,),
    ).fetchone()
    status_type = str((match_row["status_type"] if match_row else "") or "").lower()
    core = conn.execute(
        "SELECT COUNT(*) AS n FROM quarter_scores WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]
    pbp = conn.execute(
        "SELECT COUNT(*) AS n FROM play_by_play WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]
    events = conn.execute(
        "SELECT COUNT(*) AS n FROM match_events WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]
    graph = conn.execute(
        "SELECT COUNT(*) AS n FROM graph_points WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]
    h2h = conn.execute(
        "SELECT COUNT(*) AS n FROM match_h2h WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]
    team_stats = conn.execute(
        "SELECT COUNT(*) AS n FROM team_statistics WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]
    lineups = conn.execute(
        "SELECT COUNT(*) AS n FROM lineups WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]
    player_stats = conn.execute(
        "SELECT COUNT(*) AS n FROM player_stats WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]
    odds = conn.execute(
        "SELECT COUNT(*) AS n FROM match_odds WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]
    strength = conn.execute(
        "SELECT COUNT(*) AS n FROM team_strength WHERE match_id = ?",
        (match_id,),
    ).fetchone()["n"]

    complete = bool(
        status_type == "finished"
        and
        core >= 4
        and pbp > 0
        and events > 0
        and graph > 0
        and h2h > 0
        and team_stats > 0
        and lineups > 0
        and player_stats > 0
        and odds > 0
        and strength > 0
    )
    return {
        "complete": complete,
        "status_type": status_type,
        "quarters": int(core),
        "pbp": int(pbp),
        "events": int(events),
        "graph": int(graph),
        "h2h": int(h2h),
        "team_stats": int(team_stats),
        "lineups": int(lineups),
        "player_stats": int(player_stats),
        "odds": int(odds),
        "strength": int(strength),
    }


def _skip_reason(state: dict[str, int | bool]) -> str:
    if state.get("status_type") != "finished":
        return f"not_ft({state.get('status_type') or 'unknown'})"
    if state.get("complete"):
        return "already_complete"
    missing = [
        name for name in (
            ("quarters", state.get("quarters", 0) >= 4),
            ("pbp", state.get("pbp", 0) > 0),
            ("events", state.get("events", 0) > 0),
            ("graph", state.get("graph", 0) > 0),
            ("h2h", state.get("h2h", 0) > 0),
            ("team_stats", state.get("team_stats", 0) > 0),
            ("lineups", state.get("lineups", 0) > 0),
            ("player_stats", state.get("player_stats", 0) > 0),
            ("odds", state.get("odds", 0) > 0),
            ("strength", state.get("strength", 0) > 0),
        ) if not name[1]
    ]
    return "missing:" + ",".join(name for name, _ok in missing) if missing else "unknown"


def backfill(
    db_path: str,
    match_rows: list[dict],
    delay: float = 0.5,
    backend: str | None = None,
) -> None:
    conn = get_conn(db_path)
    init_db(conn)
    grouped = _group_rows_by_date(match_rows)
    day_totals = {day: len(rows) for day, rows in grouped.items()}
    total = sum(day_totals.values())
    ok = err = skipped = 0

    with _browser_context("https://www.sofascore.com/basketball", backend=backend) as (_, ctx, _page):
        for day, rows in grouped.items():
            day_ok = day_err = day_skip = 0
            print(f"\n[backfill] day={day} matches={len(rows)}")
            with tqdm(total=len(rows), desc=day, unit="match", ncols=140, leave=True) as bar:
                for row in rows:
                    mid = str(row.get("match_id", ""))
                    reason = ""
                    if not mid:
                        day_skip += 1
                        skipped += 1
                        bar.update(1)
                        bar.set_postfix_str(f"ok={day_ok} skip={day_skip} err={day_err} r=no_id")
                        continue

                    state = _match_refresh_state(conn, mid)
                    if state["complete"]:
                        day_skip += 1
                        skipped += 1
                        reason = _skip_reason(state)
                        bar.update(1)
                        bar.set_postfix_str(f"ok={day_ok} skip={day_skip} err={day_err} r={reason}")
                        continue

                    try:
                        event_json, incidents_json, graph_json, h2h_json, statistics_json, lineups_json, odds_json = \
                            _all_payloads(ctx, mid)

                        if not event_json or "event" not in event_json:
                            day_skip += 1
                            skipped += 1
                            bar.update(1)
                            bar.set_postfix_str(f"ok={day_ok} skip={day_skip} err={day_err} r=no_event")
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
                        day_ok += 1
                        ok += 1
                        reason = "saved"
                        time.sleep(delay)

                    except Exception as e:
                        try:
                            from db import mark_discovered_error
                            mark_discovered_error(conn, mid, str(e))
                        except Exception:
                            pass
                        day_err += 1
                        err += 1
                        reason = "error"
                    finally:
                        bar.update(1)
                        if reason == "":
                            reason = "saved" if day_ok + day_err + day_skip else ""
                        bar.set_postfix_str(f"ok={day_ok} skip={day_skip} err={day_err} r={reason}")

    print(f"\nDone. OK={ok}, Errors={err}, Total={total}")
    print(f"Skipped already-complete: {skipped}")
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
        rows = conn.execute("SELECT match_id, date FROM matches ORDER BY date DESC, time DESC").fetchall()
        match_rows = [dict(r) for r in rows]
    elif args.ids:
        placeholders = ",".join("?" for _ in args.ids)
        rows = conn.execute(
            f"SELECT match_id, date FROM matches WHERE match_id IN ({placeholders}) ORDER BY date DESC, time DESC",
            tuple(args.ids),
        ).fetchall()
        found = {r["match_id"] for r in rows}
        match_rows = [dict(r) for r in rows]
        for mid in args.ids:
            if mid not in found:
                match_rows.append({"match_id": mid, "date": "unknown"})
    else:
        print("Specify --ids or --all"); sys.exit(1)
    conn.close()

    if args.limit > 0:
        match_rows = match_rows[:args.limit]

    print(f"Backfilling {len(match_rows)} matches in {args.db}")
    backfill(args.db, match_rows, delay=args.delay, backend=args.backend)

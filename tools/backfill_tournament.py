#!/usr/bin/env python3
"""
tools/backfill_tournament.py (fast version)
Descarga historico de un torneo de SofaScore usando solo el endpoint de eventos
(la lista ya incluye cuartos Q1-Q4), sin petición por partido.

Uso:
  .venv\\Scripts\\python.exe tools\\backfill_tournament.py --tid 33655 --league "NCAA Women's Division I, Regular season" --seasons all
"""
import argparse, asyncio, json, sqlite3, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monitor_v3.core.mobile_client import get_mobile_client

DB = ROOT / "matches.db"
DEFAULT_RATE = 0.5

def load_state(tid):
    f = ROOT / "tools" / ("backfill_tournament_state_%d.json" % tid)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8")), f
    return {"done_seasons": [], "done_matches": []}, f

def save_state(st, f):
    f.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

def upsert_match(conn, league_name, ev):
    home = ev.get("homeTeam", {})
    away = ev.get("awayTeam", {})
    hs = ev.get("homeScore", {})
    as_ = ev.get("awayScore", {})
    status = ev.get("status", {})
    ts = ev.get("startTimestamp")
    if not ts:
        return False
    if status.get("type") != "finished":
        return False
    q1h, q1a = hs.get("period1"), as_.get("period1")
    q2h, q2a = hs.get("period2"), as_.get("period2")
    q3h, q3a = hs.get("period3"), as_.get("period3")
    q4h, q4a = hs.get("period4"), as_.get("period4")
    if not all(v is not None for v in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a)):
        return False
    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    time_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:00")
    match_id = str(ev.get("id"))
    conn.execute(
        "INSERT OR REPLACE INTO matches (match_id, home_team, away_team, home_slug, away_slug, event_slug, "
        "custom_id, status_type, status_description, date, time, venue, league, home_score, away_score, "
        "home_team_id, away_team_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (match_id, home.get("name"), away.get("name"), home.get("slug"), away.get("slug"),
         ev.get("slug"), str(ev.get("customId")), status.get("type"), status.get("description"),
         date_str, time_str, None, league_name, hs.get("current"), as_.get("current"),
         home.get("id"), away.get("id")))
    conn.execute(
        "INSERT OR REPLACE INTO quarter_scores_v2 (match_id, q1_home, q1_away, q2_home, q2_away, "
        "q3_home, q3_away, q4_home, q4_away) VALUES (?,?,?,?,?,?,?,?,?)",
        (match_id, q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a))
    conn.commit()
    return True

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tid", type=int, required=True)
    ap.add_argument("--league", required=True)
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE)
    args = ap.parse_args()

    st, state_file = load_state(args.tid)
    done_seasons = set(st["done_seasons"])
    done_matches = set(st["done_matches"])

    async with get_mobile_client() as c:
        r = await c.request("GET", "/unique-tournament/%d/seasons" % args.tid)
        seasons = r.json().get("seasons", [])
        if args.seasons == "all" or args.seasons is None:
            target = seasons
        else:
            wanted = set(x.strip() for x in args.seasons.split(","))
            target = [s for s in seasons if s["year"] in wanted]
        print("target seasons:", [(s["id"], s["year"]) for s in target])

        total_new = 0
        for s in target:
            sid = s["id"]
            if sid in done_seasons:
                print("season", sid, s["year"], "already done, skip")
                continue
            print("\n=== Season", sid, s["year"], "===")
            page = 0
            events = []
            while True:
                await asyncio.sleep(args.rate)
                r = await c.request("GET", "/unique-tournament/%d/season/%d/events/last/%d" % (args.tid, sid, page))
                data = r.json()
                evs = data.get("events", [])
                events.extend(evs)
                print("  page", page, "->", len(evs), "events (total", len(events), ")")
                if not data.get("hasNextPage"):
                    break
                page += 1
            print("  season total events:", len(events))
            new = 0
            for ev in events:
                mid = str(ev.get("id"))
                if mid in done_matches:
                    continue
                conn = sqlite3.connect(str(DB))
                try:
                    ok = upsert_match(conn, args.league, ev)
                    if ok:
                        new += 1
                except Exception as ex:
                    print("  DB ERR", mid, ex)
                finally:
                    conn.close()
                done_matches.add(mid)
                if len(done_matches) % 100 == 0:
                    st["done_matches"] = list(done_matches)
                    save_state(st, state_file)
            print("  new matches inserted:", new)
            total_new += new
            done_seasons.add(sid)
            st["done_seasons"] = list(done_seasons)
            st["done_matches"] = list(done_matches)
            save_state(st, state_file)
            print("  checkpoint saved")
    print("\nDONE. Total matches downloaded:", len(done_matches), "| new this run:", total_new)

if __name__ == "__main__":
    asyncio.run(main())

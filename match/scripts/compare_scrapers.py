"""
Compare the traditional Playwright backend with Obscura-backed CDP sessions.

Usage:
    python -m match.scripts.compare_scrapers --ids 14428863 14428864
    python -m match.scripts.compare_scrapers --all --limit 5
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "match"))

from db import get_conn
import scraper


RELEVANT_LIST_KEYS = ("h2h", "team_statistics", "period_stats", "lineups", "player_stats", "odds", "team_strength")


def _load_ids(args) -> list[str]:
    if args.ids:
        return args.ids

    if not args.all:
        raise SystemExit("Specify --ids or --all")

    conn = get_conn(args.db)
    try:
        rows = conn.execute("SELECT match_id FROM matches ORDER BY date DESC").fetchall()
        ids = [r["match_id"] for r in rows]
    finally:
        conn.close()

    if args.limit > 0:
        ids = ids[: args.limit]
    return ids


def _signature(data: dict | None) -> dict:
    if not data:
        return {"ok": False}

    out = {"ok": True, "keys": sorted(data.keys())}
    for key in RELEVANT_LIST_KEYS:
        value = data.get(key)
        out[f"{key}_len"] = len(value) if isinstance(value, list) else 0
    return out


def _run_backend(backend: str, ids: list[str], args) -> list[dict]:
    rows: list[dict] = []
    for idx, match_id in enumerate(ids, start=1):
        started = time.perf_counter()
        try:
            data = scraper.fetch_match_by_id(
                match_id,
                backend=backend,
                fetch_h2h=not args.no_h2h,
                fetch_statistics=not args.no_statistics,
                fetch_lineups=not args.no_lineups,
                fetch_team_data=not args.no_team_data,
            )
            elapsed = time.perf_counter() - started
            rows.append({
                "backend": backend,
                "match_id": match_id,
                "ok": True,
                "seconds": elapsed,
                "signature": _signature(data),
            })
            print(f"[{backend}] {idx}/{len(ids)} {match_id} OK {elapsed:.2f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - started
            rows.append({
                "backend": backend,
                "match_id": match_id,
                "ok": False,
                "seconds": elapsed,
                "error": str(exc),
            })
            print(f"[{backend}] {idx}/{len(ids)} {match_id} ERROR {elapsed:.2f}s: {exc}")
    return rows


def _summarize(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("ok")]
    times = [r["seconds"] for r in ok_rows]
    return {
        "total": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "avg": statistics.mean(times) if times else None,
        "median": statistics.median(times) if times else None,
        "min": min(times) if times else None,
        "max": max(times) if times else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--db", default=str(ROOT / "match" / "matches.db"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--no-h2h", action="store_true")
    parser.add_argument("--no-statistics", action="store_true")
    parser.add_argument("--no-lineups", action="store_true")
    parser.add_argument("--no-team-data", action="store_true")
    parser.add_argument("--obscura-cdp-url", default=os.getenv("OBSCURA_CDP_URL", "http://127.0.0.1:9222"))
    args = parser.parse_args()

    os.environ.setdefault("OBSCURA_CDP_URL", args.obscura_cdp_url)

    ids = _load_ids(args)
    if not ids:
        print("No match IDs found")
        return 1

    print(f"Comparing {len(ids)} matches")
    trad = _run_backend("traditional", ids, args)
    obsc = _run_backend("obscura", ids, args)

    trad_sum = _summarize(trad)
    obsc_sum = _summarize(obsc)

    print("\nSummary")
    print(f"traditional: {trad_sum}")
    print(f"obscura:     {obsc_sum}")

    paired = []
    by_match = {r["match_id"]: r for r in obsc}
    for row in trad:
        other = by_match.get(row["match_id"])
        if not other or not row.get("ok") or not other.get("ok"):
            continue
        paired.append(row["seconds"] - other["seconds"])

    if paired:
        print(f"delta(traditional-obscura) avg={statistics.mean(paired):.2f}s median={statistics.median(paired):.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
SofaScore basketball CLI.

Commands
--------
  scrape <URL> [--db PATH]
    Scrape a match and save to SQLite.
    Saves: metadata, quarter scores, play-by-play, and graph points.

    export-features [--db PATH] [--out PATH] [--format csv|jsonl]
        Build a ML-ready tabular dataset from all stored matches.

    export-features-quarters [--db PATH] [--out PATH] [--format csv|jsonl]
        Build one ML-ready row per quarter (Q1-Q4) per stored match.

    backfill-ft [--db PATH] [--days N] [--start-date YYYY-MM-DD]
        Discover FT IDs day-by-day and ingest each match (resumable).

    backfill-status [--db PATH]
        Show resume cursor and pending/processed counters.

    process-pending [--db PATH] [--limit N]
        Ingest only already-discovered pending FT matches.

    plot-graph <match_id> [--db PATH] [--out PATH]
        Reconstruct the SofaScore pressure graph using seaborn.

  show <match_id> [--db PATH]
      Print the stored match as pretty JSON.

  list [--db PATH]
      Print all stored matches as a JSON array.

Examples
--------
  python cli.py scrape "https://www.sofascore.com/basketball/match/brooklyn-nets-new-york-knicks/wtbsLtb#id:14442355"
  python cli.py show 14442355
  python cli.py list
    python cli.py export-features --out features.csv
    python cli.py export-features-quarters --out features_quarters.csv
        python cli.py backfill-ft --days 7 --ingest-limit 200
        python cli.py backfill-status
    python cli.py process-pending --limit 500
    python cli.py plot-graph 14442355 --out graph_14442355.png
"""

import argparse
import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import db as db_mod
import ml_tools as ml_mod
import scraper as scraper_mod

DEFAULT_DB = str(Path(__file__).parent / "matches.db")
DEFAULT_RESUME_KEY = "basketball_ft"


# ── helpers ───────────────────────────────────────────────────────────────────

def _open_db(db_path: str) -> db_mod.sqlite3.Connection:
    conn = db_mod.get_conn(db_path)
    db_mod.init_db(conn)
    return conn


def _print_summary(match_id: str, data: dict) -> None:
    """Pretty-print a match summary table to stdout."""
    m = data["match"]
    s = data["score"]

    rec_h = f" ({m['home_record']})" if m.get("home_record") else ""
    rec_a = f" ({m['away_record']})" if m.get("away_record") else ""
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  {m['home_team']}{rec_h}  vs  {m['away_team']}{rec_a}")
    print(f"  {m.get('league', '')}  |  {m.get('venue', '')}")
    print(f"  {m['date']} {m['time']} UTC  |  match_id: {match_id}")
    quarters = s.get("quarters", {})
    q_order = ["Q1", "Q2", "Q3", "Q4"]
    q_parts = []
    for q in q_order:
        if q in quarters:
            q_parts.append(f"{q} {quarters[q]['home']}-{quarters[q]['away']}")

    q_text = f"  |  {' | '.join(q_parts)}" if q_parts else ""
    print(
        f"  FT:  {m['home_team']} {s['home']}  -  {s['away']} {m['away_team']}"
        f"{q_text}"
    )
    print(sep)

    if s.get("quarters"):
        qs = sorted(s["quarters"].keys())
        col = 8
        print(f"\n  {'TEAM':<24}" + "".join(f"{q:<{col}}" for q in qs) + "TOTAL")
        print(
            f"  {m['home_team'][:24]:<24}"
            + "".join(f"{s['quarters'][q]['home']:<{col}}" for q in qs)
            + str(s["home"])
        )
        print(
            f"  {m['away_team'][:24]:<24}"
            + "".join(f"{s['quarters'][q]['away']:<{col}}" for q in qs)
            + str(s["away"])
        )

    for q in sorted(data.get("play_by_play", {}).keys()):
        plays = data["play_by_play"][q]
        print(f"\n  -- {q} --  ({len(plays)} scoring plays)")
        for p in plays:
            team_name = m["home_team"] if p["team"] == "home" else m["away_team"]
            hs = p.get("home_score", "?")
            as_ = p.get("away_score", "?")
            print(
                f"    {p['time']:>6}  {p['points']}pts"
                f"  {p['player']:<22}  {team_name[:18]:<18}  {hs}-{as_}"
            )

    gp = data.get("graph_points", [])
    if gp:
        values = [int(p["value"]) for p in gp]
        print("\n  -- GRAPH --")
        print(
            "  points: "
            f"{len(gp)}  range: [{min(values)}, {max(values)}]"
        )
        print(
            "  first: "
            f"minute={gp[0]['minute']} value={gp[0]['value']}"
            " | "
            f"last: minute={gp[-1]['minute']} value={gp[-1]['value']}"
        )

    print()


def _parse_date(date_text: str) -> datetime.date:
    return datetime.strptime(date_text, "%Y-%m-%d").date()


def _utc_yesterday() -> datetime.date:
    return datetime.now(timezone.utc).date() - timedelta(days=1)


def _ingest_pending_matches(
    conn: db_mod.sqlite3.Connection,
    date_from: str,
    date_to: str,
    limit: int,
) -> tuple[int, int, int, int]:
    pending = db_mod.list_pending_discovered_ft(
        conn,
        date_from,
        date_to,
        limit,
    )
    print(f"[ingest] pending={len(pending)}")

    ing_ok = 0
    ing_fail = 0
    skipped_ft = 0
    total = len(pending)
    if total:
        _print_progress("ingest", 0, total)

    for idx, row in enumerate(pending, start=1):
        match_id = row["match_id"]
        existing = db_mod.get_match(conn, match_id)
        if _is_ft_complete(existing):
            db_mod.mark_discovered_processed(conn, match_id)
            skipped_ft += 1
            _print_progress("ingest", idx, total)
            continue

        try:
            data = scraper_mod.fetch_match_by_id(match_id)
            db_mod.save_match(conn, match_id, data)
            db_mod.mark_discovered_processed(conn, match_id)
            ing_ok += 1
        except KeyboardInterrupt:
            print()
            print("[ingest] interrupted by user")
            _print_progress("ingest", idx - 1, total)
            break
        except Exception as e:
            db_mod.mark_discovered_error(conn, match_id, str(e))
            ing_fail += 1
            print()
            print(f"[ingest][error] {match_id}: {e}")
        finally:
            _print_progress("ingest", idx, total)

    if total:
        print()

    return len(pending), ing_ok, ing_fail, skipped_ft


# ── command handlers ──────────────────────────────────────────────────────────

def cmd_scrape(args: argparse.Namespace) -> None:
    match_id = scraper_mod.parse_match_id(args.url)
    print(f"[scrape] match_id = {match_id}")
    print("[scrape] Launching headless browser (Playwright) …")
    data = scraper_mod.fetch_match(args.url, match_id)
    print("[scrape] Playwright successful.")

    conn = _open_db(args.db)
    db_mod.save_match(conn, match_id, data)
    conn.close()

    print(f"[scrape] Saved to {args.db}\n")
    _print_summary(match_id, data)


def cmd_show(args: argparse.Namespace) -> None:
    conn = _open_db(args.db)
    data = db_mod.get_match(conn, args.match_id)
    conn.close()

    if data is None:
        print(f"Match '{args.match_id}' not found in {args.db}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_list(args: argparse.Namespace) -> None:
    conn = _open_db(args.db)
    matches = db_mod.list_matches(conn)
    conn.close()

    print(json.dumps(matches, indent=2, ensure_ascii=False))


def cmd_export_features(args: argparse.Namespace) -> None:
    conn = _open_db(args.db)
    matches = db_mod.list_matches(conn)

    rows = []
    for meta in matches:
        data = db_mod.get_match(conn, meta["match_id"])
        if data is None:
            continue
        rows.append(ml_mod.build_feature_row(data))

    conn.close()

    out_path = ml_mod.export_feature_rows(rows, args.out, args.format)
    print(
        f"[export-features] rows={len(rows)} format={args.format} "
        f"saved={out_path}"
    )


def cmd_export_features_quarters(args: argparse.Namespace) -> None:
    conn = _open_db(args.db)
    matches = db_mod.list_matches(conn)

    rows = []
    for meta in matches:
        data = db_mod.get_match(conn, meta["match_id"])
        if data is None:
            continue
        rows.extend(ml_mod.build_feature_rows_by_quarter(data))

    conn.close()

    out_path = ml_mod.export_feature_rows(rows, args.out, args.format)
    print(
        f"[export-features-quarters] rows={len(rows)} format={args.format} "
        f"saved={out_path}"
    )


def cmd_plot_graph(args: argparse.Namespace) -> None:
    conn = _open_db(args.db)
    data = db_mod.get_match(conn, args.match_id)
    conn.close()

    if data is None:
        print(f"Match '{args.match_id}' not found in {args.db}", file=sys.stderr)
        sys.exit(1)

    out = args.out or str(
        Path(__file__).parent / f"graph_{args.match_id}.png"
    )
    saved = ml_mod.plot_graph(data, out)
    print(f"[plot-graph] saved={saved}")


def cmd_backfill_ft(args: argparse.Namespace) -> None:
    conn = _open_db(args.db)
    cursor_key = f"{args.resume_key}.cursor_date"

    if args.reset_cursor:
        start = _parse_date(args.start_date) if args.start_date else _utc_yesterday()
        db_mod.set_state(conn, cursor_key, start.isoformat())
        print(f"[backfill-ft] cursor reset: {start.isoformat()}")

    cursor_txt = db_mod.get_state(conn, cursor_key)
    if cursor_txt:
        cursor = _parse_date(cursor_txt)
    else:
        cursor = _parse_date(args.start_date) if args.start_date else _utc_yesterday()

    stop = _parse_date(args.stop_date) if args.stop_date else datetime(2000, 1, 1).date()
    discovered_total = 0
    days_processed = 0

    while days_processed < args.days and cursor >= stop:
        day = cursor.isoformat()
        print(f"[discover] date={day}")
        try:
            rows = scraper_mod.fetch_finished_match_ids_for_date(day)
            inserted = db_mod.save_discovered_ft_matches(conn, rows)
            discovered_total += inserted
            print(f"[discover] finished_ids={inserted}")
        except Exception as e:
            print(f"[discover][error] {day}: {e}")

        cursor = cursor - timedelta(days=1)
        db_mod.set_state(conn, cursor_key, cursor.isoformat())
        days_processed += 1

    pending_count = 0
    ing_ok = 0
    ing_fail = 0
    skipped_ft = 0
    if not args.no_ingest:
        pending_count, ing_ok, ing_fail, skipped_ft = _ingest_pending_matches(
            conn,
            stop.isoformat(),
            _utc_yesterday().isoformat(),
            args.ingest_limit,
        )

    stats = db_mod.get_discovered_stats(conn)
    next_cursor = db_mod.get_state(conn, cursor_key)
    conn.close()

    print("[backfill-ft] done")
    print(
        f"[backfill-ft] discovered_rows={discovered_total} "
        f"days_processed={days_processed}"
    )
    print(f"[backfill-ft] pending_seen={pending_count}")
    print(
        f"[backfill-ft] ingested_ok={ing_ok} "
        f"ingested_fail={ing_fail} skipped_ft={skipped_ft}"
    )
    print(
        f"[backfill-ft] total={stats['total']} processed={stats['processed']} "
        f"pending={stats['pending']} pending_with_error={stats['pending_with_error']}"
    )
    print(f"[backfill-ft] resume_cursor={next_cursor}")


def cmd_backfill_status(args: argparse.Namespace) -> None:
    conn = _open_db(args.db)
    cursor_key = f"{args.resume_key}.cursor_date"
    cursor = db_mod.get_state(conn, cursor_key)
    stats = db_mod.get_discovered_stats(conn)
    conn.close()

    print("[backfill-status]")
    print(f"resume_key={args.resume_key}")
    print(f"resume_cursor={cursor}")
    print(
        f"total={stats['total']} processed={stats['processed']} "
        f"pending={stats['pending']} pending_with_error={stats['pending_with_error']}"
    )
    print(f"date_min={stats['min_date']} date_max={stats['max_date']}")


def cmd_process_pending(args: argparse.Namespace) -> None:
    conn = _open_db(args.db)
    stats_before = db_mod.get_discovered_stats(conn)
    min_date = stats_before["min_date"] or "2000-01-01"
    max_date = stats_before["max_date"] or _utc_yesterday().isoformat()

    pending_count, ing_ok, ing_fail, skipped_ft = _ingest_pending_matches(
        conn,
        min_date,
        max_date,
        args.limit,
    )

    stats_after = db_mod.get_discovered_stats(conn)
    conn.close()

    print("[process-pending] done")
    print(f"[process-pending] pending_seen={pending_count}")
    print(
        f"[process-pending] ingested_ok={ing_ok} "
        f"ingested_fail={ing_fail} skipped_ft={skipped_ft}"
    )
    print(
        f"[process-pending] total={stats_after['total']} "
        f"processed={stats_after['processed']} pending={stats_after['pending']} "
        f"pending_with_error={stats_after['pending_with_error']}"
    )


def _empty_eval_stats() -> dict:
    return {
        "samples": 0,
        "bets": 0,
        "hits_all": 0,
        "misses_all": 0,
        "hits_bet": 0,
        "losses_bet": 0,
    }


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _is_ft_complete(data: dict | None) -> bool:
    if not data:
        return False
    quarters = data.get("score", {}).get("quarters", {})
    required = ("Q1", "Q2", "Q3", "Q4")
    return all(q in quarters for q in required)


def _print_progress(
    prefix: str,
    current: int,
    total: int,
    width: int = 28,
) -> None:
    if total <= 0:
        print(f"\r[{prefix}] 0/0", end="", flush=True)
        return
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = ("#" * filled) + ("-" * (width - filled))
    pct = int(ratio * 100)
    print(
        f"\r[{prefix}] [{bar}] {current}/{total} ({pct}%)",
        end="",
        flush=True,
    )


def _finalize_eval(stats: dict, odds: float) -> dict:
    samples = int(stats["samples"])
    bets = int(stats["bets"])
    hits_all = int(stats["hits_all"])
    misses_all = int(stats["misses_all"])
    hits_bet = int(stats["hits_bet"])
    losses_bet = int(stats["losses_bet"])

    if bets:
        profit_units = (hits_bet * (odds - 1.0)) - losses_bet
        roi_units_per_bet = profit_units / bets
    else:
        profit_units = 0.0
        roi_units_per_bet = 0.0

    return {
        "samples": samples,
        "bets": bets,
        "coverage": round(_safe_rate(bets, samples), 6),
        "accuracy_all": round(_safe_rate(hits_all, hits_all + misses_all), 6),
        "hit_rate_bet": round(_safe_rate(hits_bet, bets), 6),
        "roi_units_per_bet": round(roi_units_per_bet, 6),
        "profit_units": round(profit_units, 6),
    }


def _print_eval_row(label: str, row: dict) -> None:
    print(
        f"{label:<8} "
        f"samples={row['samples']:<4} "
        f"bets={row['bets']:<4} "
        f"coverage={row['coverage']:<8} "
        f"acc_all={row['accuracy_all']:<8} "
        f"hit_bet={row['hit_rate_bet']:<8} "
        f"roi/bet={row['roi_units_per_bet']:<8} "
        f"profit={row['profit_units']}"
    )


def cmd_eval_date(args: argparse.Namespace) -> None:
    eval_date = _parse_date(args.date).isoformat()
    conn = _open_db(args.db)

    print(f"[eval-date] date={eval_date}")
    rows = scraper_mod.fetch_finished_match_ids_for_date(eval_date)
    if args.limit_matches is not None:
        rows = rows[:args.limit_matches]
    discovered = db_mod.save_discovered_ft_matches(conn, rows)
    print(f"[eval-date] discovered_finished={discovered}")

    ing_ok = 0
    ing_fail = 0
    already_ft_in_db = 0
    match_ids: list[str] = []
    total_rows = len(rows)
    if total_rows:
        _print_progress("ingest", 0, total_rows)

    for idx, row in enumerate(rows, start=1):
        match_id = str(row.get("match_id", ""))
        if not match_id:
            _print_progress("ingest", idx, total_rows)
            continue
        match_ids.append(match_id)

        existing = db_mod.get_match(conn, match_id)
        if _is_ft_complete(existing):
            already_ft_in_db += 1
            db_mod.mark_discovered_processed(conn, match_id)
            _print_progress("ingest", idx, total_rows)
            continue

        try:
            data = scraper_mod.fetch_match_by_id(match_id)
            db_mod.save_match(conn, match_id, data)
            db_mod.mark_discovered_processed(conn, match_id)
            ing_ok += 1
        except Exception as e:
            err_text = str(e)
            db_mod.mark_discovered_error(conn, match_id, err_text)
            ing_fail += 1
            err_first = (
                err_text.splitlines()[0]
                if err_text
                else "unknown_error"
            )
            print()
            print(f"[eval-date][ingest-error] {match_id}: {err_first}")
        finally:
            _print_progress("ingest", idx, total_rows)

    if total_rows:
        print()

    print(
        f"[eval-date] already_ft_in_db={already_ft_in_db} "
        f"ingested_ok={ing_ok} ingested_fail={ing_fail}"
    )

    infer_mod = importlib.import_module("training.infer_match")
    infer_mod.scraper_mod.fetch_event_snapshot = lambda _mid: None

    stats = {
        "q3": _empty_eval_stats(),
        "q4": _empty_eval_stats(),
        "overall": _empty_eval_stats(),
    }

    total_eval = len(match_ids)
    if total_eval:
        _print_progress("eval", 0, total_eval)

    for idx, match_id in enumerate(match_ids, start=1):
        res = infer_mod.run_inference(
            match_id=match_id,
            metric=args.metric,
            fetch_missing=False,
            force_version=args.force_version,
        )
        if not res.get("ok"):
            _print_progress("eval", idx, total_eval)
            continue

        for target in ("q3", "q4"):
            pred = res.get("predictions", {}).get(target, {})
            if not pred.get("available"):
                continue

            outcome = str(pred.get("result", ""))
            if outcome in ("", "pending", "push"):
                continue

            for bucket in (target, "overall"):
                s = stats[bucket]
                s["samples"] += 1
                if outcome == "hit":
                    s["hits_all"] += 1
                elif outcome == "miss":
                    s["misses_all"] += 1

                if pred.get("final_recommendation") == "BET":
                    s["bets"] += 1
                    if outcome == "hit":
                        s["hits_bet"] += 1
                    elif outcome == "miss":
                        s["losses_bet"] += 1

        _print_progress("eval", idx, total_eval)

    if total_eval:
        print()

    q3_row = _finalize_eval(stats["q3"], odds=args.odds)
    q4_row = _finalize_eval(stats["q4"], odds=args.odds)
    all_row = _finalize_eval(stats["overall"], odds=args.odds)

    print("[eval-date] summary")
    _print_eval_row("Q3", q3_row)
    _print_eval_row("Q4", q4_row)
    _print_eval_row("TOTAL", all_row)

    out = {
        "date": eval_date,
        "metric": args.metric,
        "force_version": args.force_version,
        "odds": args.odds,
        "limit_matches": args.limit_matches,
        "discovered_finished": discovered,
        "already_ft_in_db": already_ft_in_db,
        "ingested_ok": ing_ok,
        "ingested_fail": ing_fail,
        "summary": {
            "q3": q3_row,
            "q4": q4_row,
            "overall": all_row,
        },
    }

    out_dir = Path(__file__).parent / "training" / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = (
        out_dir
        / (
            f"daily_eval_{eval_date}_{args.force_version}_{args.metric}.json"
        )
    )
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[eval-date] saved={out_file}")

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))

    conn.close()


# ── argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SofaScore basketball match scraper & viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        metavar="PATH",
        help=f"SQLite database file (default: {DEFAULT_DB})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # scrape
    p_scrape = sub.add_parser("scrape", help="Scrape a match and save to the DB")
    p_scrape.add_argument("url", help="SofaScore match URL")
    p_scrape.set_defaults(func=cmd_scrape)

    # show
    p_show = sub.add_parser("show", help="Print a stored match as JSON")
    p_show.add_argument("match_id", help="Numeric match ID (e.g. 14442355)")
    p_show.set_defaults(func=cmd_show)

    # list
    p_list = sub.add_parser("list", help="List all stored matches as a JSON array")
    p_list.set_defaults(func=cmd_list)

    # export-features
    p_feat = sub.add_parser(
        "export-features",
        help="Export ML-ready features from all stored matches",
    )
    p_feat.add_argument(
        "--out",
        default=str(Path(__file__).parent / "features.csv"),
        metavar="PATH",
        help="Output file path (default: features.csv in this folder)",
    )
    p_feat.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Output format (default: csv)",
    )
    p_feat.set_defaults(func=cmd_export_features)

    # export-features-quarters
    p_feat_q = sub.add_parser(
        "export-features-quarters",
        help="Export one ML-ready feature row per quarter for all matches",
    )
    p_feat_q.add_argument(
        "--out",
        default=str(Path(__file__).parent / "features_quarters.csv"),
        metavar="PATH",
        help="Output file path (default: features_quarters.csv in this folder)",
    )
    p_feat_q.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Output format (default: csv)",
    )
    p_feat_q.set_defaults(func=cmd_export_features_quarters)

    # backfill-ft
    p_backfill = sub.add_parser(
        "backfill-ft",
        help="Resumable FT discovery by date + per-match ingestion",
    )
    p_backfill.add_argument(
        "--days",
        type=int,
        default=1,
        help="Days to crawl backward in this run (default: 1)",
    )
    p_backfill.add_argument(
        "--start-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for first run (default: yesterday UTC)",
    )
    p_backfill.add_argument(
        "--stop-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Oldest date allowed for discovery (default: 2000-01-01)",
    )
    p_backfill.add_argument(
        "--ingest-limit",
        type=int,
        default=200,
        help="Max pending matches to ingest per run (default: 200)",
    )
    p_backfill.add_argument(
        "--no-ingest",
        action="store_true",
        help="Only discover IDs, skip per-match ingestion",
    )
    p_backfill.add_argument(
        "--resume-key",
        default=DEFAULT_RESUME_KEY,
        help=f"Resume namespace key (default: {DEFAULT_RESUME_KEY})",
    )
    p_backfill.add_argument(
        "--reset-cursor",
        action="store_true",
        help="Reset cursor before run (uses start-date or yesterday)",
    )
    p_backfill.set_defaults(func=cmd_backfill_ft)

    # backfill-status
    p_status = sub.add_parser(
        "backfill-status",
        help="Show resumable FT backfill progress",
    )
    p_status.add_argument(
        "--resume-key",
        default=DEFAULT_RESUME_KEY,
        help=f"Resume namespace key (default: {DEFAULT_RESUME_KEY})",
    )
    p_status.set_defaults(func=cmd_backfill_status)

    # process-pending
    p_pending = sub.add_parser(
        "process-pending",
        help="Ingest only already-discovered pending FT matches",
    )
    p_pending.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max pending matches to ingest in this run (default: 200)",
    )
    p_pending.set_defaults(func=cmd_process_pending)

    # eval-date
    p_eval = sub.add_parser(
        "eval-date",
        help="Discover+ingest+evaluate one date (Q3/Q4 summary)",
    )
    p_eval.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="UTC date to evaluate",
    )
    p_eval.add_argument(
        "--metric",
        choices=["accuracy", "f1", "log_loss"],
        default="f1",
        help="Metric for model selection context (default: f1)",
    )
    p_eval.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Max finished matches to process from that date",
    )
    p_eval.add_argument(
        "--force-version",
        choices=["auto", "v1", "v2", "v4", "hybrid"],
        default="hybrid",
        help="Inference policy (default: hybrid)",
    )
    p_eval.add_argument(
        "--odds",
        type=float,
        default=1.91,
        help="Fixed decimal odds for ROI proxy (default: 1.91)",
    )
    p_eval.add_argument(
        "--json",
        action="store_true",
        help="Also print JSON summary",
    )
    p_eval.set_defaults(func=cmd_eval_date)

    # plot-graph
    p_plot = sub.add_parser(
        "plot-graph",
        help="Reconstruct pressure graph using seaborn",
    )
    p_plot.add_argument("match_id", help="Numeric match ID (e.g. 14442355)")
    p_plot.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="PNG output path (default: graph_<match_id>.png)",
    )
    p_plot.set_defaults(func=cmd_plot_graph)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

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

    backfill-links [--db PATH] [--limit N]
        Refresh stored matches missing SofaScore event slug/customId.

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
    python cli.py backfill-links --limit 200
    python cli.py plot-graph 14442355 --out graph_14442355.png
"""

import argparse
import importlib
import json
import os
import subprocess
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
) -> tuple[int, int, int, int, list[str]]:
    ignored_404 = db_mod.mark_http_404_errors_processed(conn)
    if ignored_404:
        print(f"[ingest] 404_no_retry_marked={ignored_404}")

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
    ingest_error_samples: list[str] = []
    total = len(pending)
    dual_started = False
    if total:
        dual_started = _print_dual_progress(
            "ingest",
            0,
            total,
            0,
            started=dual_started,
        )

    for idx, row in enumerate(pending, start=1):
        match_id = row["match_id"]
        existing = db_mod.get_match(conn, match_id)
        if _is_ft_complete(existing):
            db_mod.mark_discovered_processed(conn, match_id)
            skipped_ft += 1
            dual_started = _print_dual_progress(
                "ingest",
                idx,
                total,
                ing_fail,
                started=dual_started,
            )
            continue

        try:
            data = scraper_mod.fetch_match_by_id(match_id)
            if not _has_usable_data(data):
                db_mod.mark_discovered_processed(conn, match_id)
                skipped_ft += 1
            else:
                db_mod.save_match(conn, match_id, data)
                db_mod.mark_discovered_processed(conn, match_id)
                ing_ok += 1
        except KeyboardInterrupt:
            print()
            print("[ingest] interrupted by user")
            _print_dual_progress(
                "ingest",
                idx - 1,
                total,
                ing_fail,
                started=dual_started,
            )
            break
        except Exception as e:
            db_mod.mark_discovered_error(conn, match_id, str(e))
            ing_fail += 1
            err_text = str(e)
            err_first = (
                err_text.splitlines()[0]
                if err_text
                else "unknown_error"
            )
            if len(ingest_error_samples) < 12:
                ingest_error_samples.append(f"{match_id}: {err_first}")
        finally:
            dual_started = _print_dual_progress(
                "ingest",
                idx,
                total,
                ing_fail,
                started=dual_started,
            )

    if total:
        print()

    return len(pending), ing_ok, ing_fail, skipped_ft, ingest_error_samples


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
        (
            pending_count,
            ing_ok,
            ing_fail,
            skipped_ft,
            ingest_error_samples,
        ) = _ingest_pending_matches(
            conn,
            stop.isoformat(),
            _utc_yesterday().isoformat(),
            args.ingest_limit,
        )
    else:
        ingest_error_samples = []

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
    if ing_fail:
        print(
            f"[backfill-ft] ingest_error_samples="
            f"{len(ingest_error_samples)}/{ing_fail}"
        )
        for sample in ingest_error_samples:
            print(f"[backfill-ft][ingest-error] {sample}")
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

    (
        pending_count,
        ing_ok,
        ing_fail,
        skipped_ft,
        ingest_error_samples,
    ) = _ingest_pending_matches(
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
    if ing_fail:
        print(
            f"[process-pending] ingest_error_samples="
            f"{len(ingest_error_samples)}/{ing_fail}"
        )
        for sample in ingest_error_samples:
            print(f"[process-pending][ingest-error] {sample}")


def cmd_backfill_links(args: argparse.Namespace) -> None:
    conn = _open_db(args.db)
    missing_rows = conn.execute(
        """
        SELECT match_id, home_team, away_team
        FROM matches
        WHERE COALESCE(NULLIF(TRIM(event_slug), ''), 'unknown') = 'unknown'
           OR COALESCE(NULLIF(TRIM(custom_id), ''), '') = ''
        ORDER BY date DESC, time DESC, match_id DESC
        """
    ).fetchall()

    total_missing = len(missing_rows)
    rows_to_process = missing_rows[: args.limit] if args.limit else missing_rows
    print(
        f"[backfill-links] missing_total={total_missing} "
        f"selected={len(rows_to_process)}"
    )

    updated = 0
    failed = 0
    error_samples: list[str] = []

    for index, row in enumerate(rows_to_process, start=1):
        match_id = str(row["match_id"])
        home_team = str(row["home_team"] or "")
        away_team = str(row["away_team"] or "")
        print(
            f"[backfill-links] {index}/{len(rows_to_process)} "
            f"match_id={match_id} {home_team} vs {away_team}"
        )
        try:
            data = scraper_mod.fetch_match_by_id(match_id)
            db_mod.save_match(conn, match_id, data)
            updated += 1
        except KeyboardInterrupt:
            print("[backfill-links] interrupted by user")
            break
        except Exception as exc:
            failed += 1
            err_text = str(exc).splitlines()[0] if str(exc) else "unknown_error"
            if len(error_samples) < 12:
                error_samples.append(f"{match_id}: {err_text}")

    remaining = total_missing - updated
    if remaining < 0:
        remaining = 0

    conn.close()

    print("[backfill-links] done")
    print(
        f"[backfill-links] updated={updated} failed={failed} "
        f"remaining_estimate={remaining}"
    )
    if failed:
        print(
            f"[backfill-links] error_samples="
            f"{len(error_samples)}/{failed}"
        )
        for sample in error_samples:
            print(f"[backfill-links][error] {sample}")


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


def _has_usable_data(data: dict | None) -> bool:
    """Strict completeness check: 4 quarters + PBP with Q1/Q2 events + graph_points.

    Used during backfill to skip matches that lack the data the models need.
    """
    if not _is_ft_complete(data):
        return False
    pbp = data.get("play_by_play") or {}
    if not pbp.get("Q1") and not pbp.get("Q2"):
        return False
    gp = data.get("graph_points") or []
    if not gp:
        return False
    return True


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


def _progress_line(
    prefix: str,
    current: int,
    total: int,
    width: int = 28,
) -> str:
    if total <= 0:
        return f"[{prefix}] 0/0"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = ("#" * filled) + ("-" * (width - filled))
    pct = int(ratio * 100)
    return f"[{prefix}] [{bar}] {current}/{total} ({pct}%)"


def _print_dual_progress(
    prefix: str,
    current: int,
    total: int,
    errors: int,
    *,
    started: bool,
) -> bool:
    done_line = _progress_line(prefix, current, total)
    suffix = f" err={errors}" if errors else ""
    _ = started
    print(f"\r{done_line}{suffix}", end="", flush=True)
    return True


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


def _winner_from_scores(home: int | None, away: int | None) -> str:
    if home is None or away is None:
        return "unknown"
    if home == away:
        return "push"
    return "home" if home > away else "away"


def _quarter_score(data: dict | None, quarter: str) -> tuple[int | None, int | None]:
    if not data:
        return None, None
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


def _preview_target(pred: dict) -> tuple[str, str]:
    if not pred or not pred.get("available"):
        return "unavailable", "NO_BET"
    pick = str(pred.get("predicted_winner", "unavailable") or "unavailable")
    signal = str(
        pred.get("final_recommendation")
        or pred.get("bet_signal")
        or "NO_BET"
    )
    return pick, signal


def cmd_eval_report(args: argparse.Namespace) -> None:
    import datetime
    import sys
    training_dir = Path(__file__).parent / "training"
    script = training_dir / "eval_report_by_model.py"
    
    cmd = [sys.executable, str(script)]
    if args.list:
        cmd.extend(["--list"])
    if args.date:
        cmd.extend(["--date", args.date])
    if args.month:
        cmd.extend(["--month", args.month])
    if args.model:
        cmd.extend(["--model", args.model])
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def cmd_eval_date(args: argparse.Namespace) -> None:
    eval_date = _parse_date(args.date).isoformat()
    conn = _open_db(args.db)
    result_tag = args.result_tag or f"{args.force_version}_{args.metric}"

    print(f"[eval-date] date={eval_date}")
    print(f"[eval-date] result_tag={result_tag}")
    rows = scraper_mod.fetch_finished_match_ids_for_date(eval_date)
    if args.limit_matches is not None:
        rows = rows[:args.limit_matches]
    discovered = db_mod.save_discovered_ft_matches(conn, rows)
    print(f"[eval-date] discovered_finished={discovered}")

    ing_ok = 0
    ing_fail = 0
    already_ft_in_db = 0
    ingest_error_samples: list[str] = []
    match_ids: list[str] = []
    total_rows = len(rows)
    dual_started = False
    if total_rows:
        dual_started = _print_dual_progress(
            "ingest",
            0,
            total_rows,
            0,
            started=dual_started,
        )

    for idx, row in enumerate(rows, start=1):
        match_id = str(row.get("match_id", ""))
        if not match_id:
            if total_rows:
                dual_started = _print_dual_progress(
                    "ingest",
                    idx,
                    total_rows,
                    ing_fail,
                    started=dual_started,
                )
            continue
        match_ids.append(match_id)

        existing = db_mod.get_match(conn, match_id)
        if _is_ft_complete(existing):
            already_ft_in_db += 1
            db_mod.mark_discovered_processed(conn, match_id)
            if total_rows:
                dual_started = _print_dual_progress(
                    "ingest",
                    idx,
                    total_rows,
                    ing_fail,
                    started=dual_started,
                )
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
            if len(ingest_error_samples) < 12:
                ingest_error_samples.append(f"{match_id}: {err_first}")
        finally:
            if total_rows:
                dual_started = _print_dual_progress(
                    "ingest",
                    idx,
                    total_rows,
                    ing_fail,
                    started=dual_started,
                )

    if total_rows:
        print()

    print(
        f"[eval-date] already_ft_in_db={already_ft_in_db} "
        f"ingested_ok={ing_ok} ingested_fail={ing_fail}"
    )
    if ing_fail:
        print(
            f"[eval-date] ingest_error_samples="
            f"{len(ingest_error_samples)}/{ing_fail}"
        )
        for sample in ingest_error_samples:
            print(f"[eval-date][ingest-error] {sample}")

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
        match_blob = db_mod.get_match(conn, match_id)
        match_meta = (match_blob or {}).get("match", {})
        home_team = str(match_meta.get("home_team", ""))
        away_team = str(match_meta.get("away_team", ""))
        q3h, q3a = _quarter_score(match_blob, "Q3")
        q4h, q4a = _quarter_score(match_blob, "Q4")
        q3_winner = _winner_from_scores(q3h, q3a)
        q4_winner = _winner_from_scores(q4h, q4a)

        res = infer_mod.run_inference(
            match_id=match_id,
            metric=args.metric,
            fetch_missing=False,
            force_version=args.force_version,
        )

        predictions = res.get("predictions", {}) if res.get("ok") else {}
        db_mod.save_eval_match_result(
            conn,
            event_date=eval_date,
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            q3_home_score=q3h,
            q3_away_score=q3a,
            q4_home_score=q4h,
            q4_away_score=q4a,
            result_tag=result_tag,
            predictions=predictions,
        )

        q3_pick, q3_signal = _preview_target(predictions.get("q3", {}))
        q4_pick, q4_signal = _preview_target(predictions.get("q4", {}))
        print()
        print(
            f"[eval-live] {idx}/{total_eval} "
            f"local={home_team} visitante={away_team} "
            f"q3={q3_pick} ({q3_signal}) q3_gano={q3_winner} "
            f"q4={q4_pick} ({q4_signal}) q4_gano={q4_winner}"
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
        "result_tag": result_tag,
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


def cmd_run_bot(args: argparse.Namespace) -> None:
    """Run the Telegram bot process from this CLI."""
    if args.db and not os.getenv("MATCH_DB_PATH"):
        os.environ["MATCH_DB_PATH"] = args.db

    bot_mod = importlib.import_module("telegram_bot")
    bot_mod.main()


def cmd_retrain(args: argparse.Namespace) -> None:
    """Interactive sub-menu to retrain models and recalibrate the gate.

    Steps (can be run individually or all at once):
      1. train-v2   — V2 classifier (league/team buckets)
      2. train-v4   — V4 classifier (pressure/comeback features)
      3. train-v6   — V6 classifier (Monte-Carlo + momentum features)
      4. train-v9   — V9 optimised classifier (LogReg + GB ensemble)
      5. train-v10  — V10 O/U regression (Ridge + GB + XGB stacking)
      6. compare    — Rebuild version_comparison.json (picks best live model)
      7. calibrate  — Recalibrate decision-gate thresholds (gate_config.json)
    """
    import subprocess

    training_dir = Path(__file__).parent / "training"

    steps: list[tuple[str, str, list[str]]] = [
        ("train-v2",   "Train V2 classifier",           [sys.executable, str(training_dir / "train_q3_q4_models_v2.py")]),
        ("train-v4",   "Train V4 classifier",           [sys.executable, str(training_dir / "train_q3_q4_models_v4.py")]),
        ("train-v6",   "Train V6 classifier",           [sys.executable, str(training_dir / "train_q3_q4_models_v6.py")]),
        ("train-v9",   "Train V9 classifier",           [sys.executable, str(training_dir / "train_q3_q4_models_v9.py")]),
        ("train-v10",  "Train V10 O/U regression",      [sys.executable, str(training_dir / "train_q3_q4_regression_v10.py")]),
        ("compare",    "Rebuild version_comparison.json", [sys.executable, str(training_dir / "compare_model_versions.py")]),
        ("calibrate",  "Calibrate gate (gate_config.json)", [sys.executable, str(training_dir / "calibrate_gate.py"), "--metric", "f1", "--limit", "3000"]),
    ]

    target = getattr(args, "step", None)

    while True:
        if not target:
            print("\n=== Reentrenamiento de modelos ===")
            for i, (key, label, _) in enumerate(steps, 1):
                print(f"{i}) {label}  [{key}]")
            print(f"{len(steps) + 1}) Ejecutar TODO en orden")
            print("0) volver")
            opt = input("Selecciona paso: ").strip()
            if opt == "0":
                return
            if opt == str(len(steps) + 1):
                target = "all"
            else:
                try:
                    idx = int(opt) - 1
                    if 0 <= idx < len(steps):
                        target = steps[idx][0]
                    else:
                        print("[retrain] opcion invalida")
                        continue
                except ValueError:
                    print("[retrain] opcion invalida")
                    continue

        to_run = steps if target == "all" else [s for s in steps if s[0] == target]
        if not to_run:
            print(f"[retrain] paso desconocido: {target}")
            target = None
            continue

        for key, label, cmd in to_run:
            print(f"\n[retrain] ▶ {label} ...")
            result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
            if result.returncode != 0:
                print(f"[retrain] ✗ {key} falló (exit {result.returncode})")
                if target == "all":
                    print("[retrain] Abortando secuencia.")
                    target = None
                    break
            else:
                print(f"[retrain] ✓ {key} completado")

        target = None  # volver al submenú tras terminar


def cmd_eval_v13(args: argparse.Namespace) -> None:
    """Generate V13 (or multi-model) evaluation Excel report."""
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    # Resolve date range
    date_from = getattr(args, "date_from", None)
    date_to = getattr(args, "date_to", None)
    month = getattr(args, "month", None)

    if month:
        try:
            _dt.strptime(month, "%Y-%m")
        except ValueError:
            print(f"[eval-v13] Formato invalido: '{month}'. Usa YYYY-MM.")
            return
        year, mon = map(int, month.split("-"))
        import calendar
        date_from = f"{month}-01"
        date_to = f"{month}-{calendar.monthrange(year, mon)[1]:02d}"
    elif date_from and date_to:
        pass
    else:
        # Default: current month up to today
        today = _dt.now()
        date_from = today.replace(day=1).strftime("%Y-%m-%d")
        date_to = today.strftime("%Y-%m-%d")
        print(f"[eval-v13] Sin rango especificado, usando {date_from} → {date_to}")

    models_str = getattr(args, "models", "v13")
    odds = getattr(args, "odds", 1.4)
    bet_size = getattr(args, "bet_size", 100.0)
    bank = getattr(args, "bank", 1000.0)
    out = getattr(args, "out", None)

    report_mod = importlib.import_module("training.report_v12_v13")

    if out:
        out_path = _Path(out)
    else:
        ts = _dt.now().strftime("%H%M%S")
        models_tag = models_str.replace(",", "_")
        date_tag = f"{date_from}_{date_to}".replace("-", "")
        out_path = _Path(__file__).parent / "reports" / f"report_{models_tag}_{date_tag}_{ts}.xlsx"

    models_list = [m.strip() for m in models_str.split(",")]

    print(f"[eval-v13] {date_from} → {date_to} | modelos={models_str} | odds={odds} | apuesta=${bet_size:.0f} | banco=${bank:.0f}")
    result = report_mod.generate_report(
        date_from=date_from,
        date_to=date_to,
        odds=odds,
        bet_size=bet_size,
        starting_bank=bank,
        out_path=out_path,
        models=models_list,
    )
    print(f"[eval-v13] Reporte guardado en: {result}")


def cmd_report(args: argparse.Namespace) -> None:
    month = getattr(args, "month", None)
    out = getattr(args, "out", None)

    if not month:
        month = input("Mes (YYYY-MM): ").strip()

    try:
        from datetime import datetime as _dt
        _dt.strptime(month, "%Y-%m")
    except ValueError:
        print(f"[report] Formato invalido: '{month}'. Usa YYYY-MM.")
        return

    report_mod = importlib.import_module("training.report_model_comparison")

    try:
        from tqdm import tqdm as _tqdm  # type: ignore[import]
        _use_tqdm = True
    except ImportError:
        _use_tqdm = False

    print(f"[report] Generando reporte {month}...")

    progress_state: dict = {
        "total": 0, "processed": 0,
        "phase": "starting", "current": "", "output_path": "",
    }

    if _use_tqdm:
        import threading as _threading
        import time as _time

        done = _threading.Event()
        error_holder: list = []
        result_holder: list = []

        def _run() -> None:
            try:
                p = report_mod.generate_report(
                    month,
                    progress_state=progress_state,
                    out_path=Path(out) if out else None,
                )
                result_holder.append(p)
            except Exception as exc:
                error_holder.append(exc)
            finally:
                done.set()

        _threading.Thread(target=_run, daemon=True).start()
        bar = None
        prev = -1
        while not done.is_set():
            total = progress_state.get("total", 0)
            processed = progress_state.get("processed", 0)
            phase = str(progress_state.get("phase", ""))
            current = str(progress_state.get("current", ""))
            if total > 0 and bar is None:
                bar = _tqdm(total=total, desc=f"[report {month}]", unit="partido")
            if bar and processed > prev:
                bar.n = processed
                bar.set_postfix_str(f"{phase} | {current}"[:60])
                bar.refresh()
                prev = processed
            _time.sleep(0.5)
        if bar:
            bar.n = progress_state.get("total", 0)
            bar.set_postfix_str("done")
            bar.close()
        if error_holder:
            print(f"[report] Error: {error_holder[0]}")
            return
        out_path = result_holder[0]
    else:
        try:
            out_path = report_mod.generate_report(
                month,
                out_path=Path(out) if out else None,
            )
        except Exception as exc:
            print(f"[report] Error: {exc}")
            return

    print(f"[report] Guardado en: {out_path}")


def _ask(prompt: str, default: str | None = None) -> str:
    hint = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{hint}: ").strip()
    if not value and default is not None:
        return default
    return value


def _apply_defaults(tokens: list[str], db_path: str | None) -> list[str]:
    if db_path:
        return ["--db", db_path] + tokens
    return tokens


def _run_tokens(parser: argparse.ArgumentParser, tokens: list[str]) -> None:
    try:
        args = parser.parse_args(tokens)
    except SystemExit:
        print("[menu] comando invalido, revisa los datos y prueba de nuevo")
        return
    args.func(args)


def _interactive_menu(parser: argparse.ArgumentParser, db_path: str) -> None:
    while True:
        print("\n=== SofaScore CLI Menu ===")
        print("1) scrape")
        print("2) show")
        print("3) list")
        print("4) export-features")
        print("5) export-features-quarters")
        print("6) backfill-ft")
        print("7) backfill-status")
        print("8) process-pending")
        print("9) eval-date")
        print("10) plot-graph")
        print("11) run-bot")
        print("12) help")
        print("13) retrain models / calibrate gate")
        print("14) reporte comparacion modelos (Excel)")
        print("15) traer fecha nueva (seleccionar dias faltantes)")
        print("0) salir")

        opt = input("Selecciona opcion: ").strip()
        if opt == "0":
            print("[menu] bye")
            return

        if opt == "12":
            parser.print_help()
            continue

        if opt == "1":
            url = _ask("URL SofaScore")
            tokens = _apply_defaults(["scrape", url], db_path)
            _run_tokens(parser, tokens)
            continue

        if opt == "2":
            match_id = _ask("Match ID")
            tokens = _apply_defaults(["show", match_id], db_path)
            _run_tokens(parser, tokens)
            continue

        if opt == "3":
            tokens = _apply_defaults(["list"], db_path)
            _run_tokens(parser, tokens)
            continue

        if opt == "4":
            out = _ask("Output path", "features.csv")
            fmt = _ask("Format (csv/jsonl)", "csv")
            tokens = _apply_defaults(
                ["export-features", "--out", out, "--format", fmt],
                db_path,
            )
            _run_tokens(parser, tokens)
            continue

        if opt == "5":
            out = _ask("Output path", "features_quarters.csv")
            fmt = _ask("Format (csv/jsonl)", "csv")
            tokens = _apply_defaults(
                [
                    "export-features-quarters",
                    "--out",
                    out,
                    "--format",
                    fmt,
                ],
                db_path,
            )
            _run_tokens(parser, tokens)
            continue

        if opt == "6":
            days = _ask("Days", "1")
            ingest_limit = _ask("Ingest limit", "200")
            start_date = _ask("Start date YYYY-MM-DD (opcional)", "")
            stop_date = _ask("Stop date YYYY-MM-DD (opcional)", "")
            no_ingest = _ask("No ingest? (y/n)", "n").lower() == "y"

            tokens = [
                "backfill-ft",
                "--days",
                days,
                "--ingest-limit",
                ingest_limit,
            ]
            if start_date:
                tokens += ["--start-date", start_date]
            if stop_date:
                tokens += ["--stop-date", stop_date]
            if no_ingest:
                tokens += ["--no-ingest"]

            tokens = _apply_defaults(tokens, db_path)
            _run_tokens(parser, tokens)
            continue

        if opt == "7":
            tokens = _apply_defaults(["backfill-status"], db_path)
            _run_tokens(parser, tokens)
            continue

        if opt == "8":
            limit = _ask("Limit", "200")
            tokens = _apply_defaults(
                ["process-pending", "--limit", limit],
                db_path,
            )
            _run_tokens(parser, tokens)
            continue

        if opt == "9":
            date = _ask("Date YYYY-MM-DD")
            metric = _ask("Metric (accuracy/f1/log_loss)", "f1")
            version = _ask("Version (auto/v1/v2/v4/hybrid)", "hybrid")
            odds = _ask("Odds", "1.91")
            limit_matches = _ask("Limit matches (opcional)", "")

            tokens = [
                "eval-date",
                "--date",
                date,
                "--metric",
                metric,
                "--force-version",
                version,
                "--odds",
                odds,
            ]
            if limit_matches:
                tokens += ["--limit-matches", limit_matches]

            tokens = _apply_defaults(tokens, db_path)
            _run_tokens(parser, tokens)
            continue

        if opt == "10":
            match_id = _ask("Match ID")
            out = _ask("Output png path (opcional)", "")
            tokens = ["plot-graph", match_id]
            if out:
                tokens += ["--out", out]
            tokens = _apply_defaults(tokens, db_path)
            _run_tokens(parser, tokens)
            continue

        if opt == "11":
            tokens = _apply_defaults(["run-bot"], db_path)
            _run_tokens(parser, tokens)
            continue

        if opt == "13":
            cmd_retrain(argparse.Namespace(step=None))
            continue

        if opt == "14":
            cmd_report(argparse.Namespace(month=None, out=None))
            continue

        if opt == "15":
            result = _select_fetch_date_interactive(db_path)
            if result is not None:
                event_date, limit = result
                _ingest_date_with_progress(db_path, event_date, limit)
            continue

        print("[menu] opcion invalida")


def cmd_menu(args: argparse.Namespace) -> None:
    parser = _build_parser()
    _interactive_menu(parser, args.db)


# ── fetch-date helpers ────────────────────────────────────────────────────────

UTC_OFFSET_HOURS = -6


def _get_missing_dates_cli(
    db_path: str,
    recent_days: int = 30,
) -> dict:
    """Return missing dates split into recent (last recent_days) and historical.

    Returns a dict with keys:
      - 'recent': list of dates (last recent_days, excl. today) with 0 matches
      - 'historical': list of older dates with 0 matches (from min_date to recent cutoff)
      - 'min_date': earliest date found in the DB (str or None)
    """
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    recent_cutoff = today - timedelta(days=recent_days)

    conn = _open_db(db_path)
    bounds = conn.execute(
        f"""
        SELECT
            MIN(date(datetime(date || ' ' || time,
                '{UTC_OFFSET_HOURS} hours'))) AS min_date,
            MAX(date(datetime(date || ' ' || time,
                '{UTC_OFFSET_HOURS} hours'))) AS max_date
        FROM matches
        """
    ).fetchone()
    min_date_str = bounds["min_date"] if bounds else None

    existing_rows = conn.execute(
        f"""
        SELECT DISTINCT
            date(datetime(date || ' ' || time, '{UTC_OFFSET_HOURS} hours'))
            AS event_date
        FROM matches
        """
    ).fetchall()
    conn.close()

    existing = {row["event_date"] for row in existing_rows}

    # Build missing dates as real gaps in the full [min_date, yesterday] range.
    # This catches holes like: 2026-01-01 present, 2026-01-02 missing, 2026-01-03 present.
    recent_missing: list[str] = []
    historical_missing: list[str] = []
    if min_date_str:
        try:
            min_date = datetime.strptime(min_date_str, "%Y-%m-%d").date()
            cursor = yesterday
            while cursor >= min_date:
                d = cursor.isoformat()
                if d not in existing:
                    if cursor >= recent_cutoff:
                        recent_missing.append(d)
                    else:
                        historical_missing.append(d)
                cursor -= timedelta(days=1)
        except ValueError:
            pass

    return {
        "recent": recent_missing,
        "historical": historical_missing,
        "min_date": min_date_str,
    }


def _select_fetch_date_interactive(db_path: str) -> tuple[str, int | None] | None:
    """Show missing-date selector.  Returns (date_str, limit|None) or None to abort."""
    print("\n=== Traer fecha nueva ===")
    data = _get_missing_dates_cli(db_path)
    recent = data["recent"]
    historical = data["historical"]
    min_date = data["min_date"]

    if min_date:
        print(f"Fecha minima en la base: {min_date}")

    options: list[str] = []

    if min_date:
        print("\nFechas anteriores (o igual) a la minima en base:")
        try:
            min_dt = datetime.strptime(min_date, "%Y-%m-%d").date()
            for i in range(10):
                d = (min_dt - timedelta(days=i)).isoformat()
                options.append(d)
                print(f"  {len(options)}) {d}")
        except ValueError:
            pass

    if recent:
        print("\nFechas recientes sin datos (ultimos 30 dias):")
        for d in recent[:10]:
            options.append(d)
            print(f"  {len(options)}) {d}")

    if historical:
        print(f"\nFechas historicas sin datos ({len(historical)} encontradas):")
        for d in historical[:10]:
            options.append(d)
            print(f"  {len(options)}) {d}")
        if len(historical) > 10:
            print(f"  ... y {len(historical) - 10} mas")

    if not recent and not historical:
        print("\nNo se detectaron huecos de fechas faltantes.")

    print("  m) Ingresar fecha manualmente")
    print("  0) Cancelar")
    choice = input("Selecciona: ").strip()
    if choice == "0":
        return None
    if choice.lower() == "m":
        event_date = _ask("Fecha (YYYY-MM-DD)")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                event_date = options[idx]
            else:
                print("[fetch-date] opcion invalida")
                return None
        except ValueError:
            print("[fetch-date] opcion invalida")
            return None

    # Validate
    try:
        datetime.strptime(event_date, "%Y-%m-%d")
    except ValueError:
        print(f"[fetch-date] Fecha invalida: {event_date}")
        return None

    limit_str = _ask("Limite de partidos (Enter para sin limite)", "")
    limit: int | None = None
    if limit_str.strip():
        try:
            limit = int(limit_str)
            if limit <= 0:
                raise ValueError
        except ValueError:
            print("[fetch-date] Limite invalido, continuando sin limite.")

    return event_date, limit


def _ingest_date_with_progress(
    db_path: str,
    event_date: str,
    limit: int | None,
) -> None:
    """Discover FT match IDs for a date and ingest them with a progress bar."""
    print(f"[fetch-date] Consultando SofaScore para {event_date}...")
    rows_all = scraper_mod.fetch_finished_match_ids_for_date(event_date)
    rows = rows_all[:limit] if limit is not None else rows_all
    print(
        f"[fetch-date] finished_found={len(rows_all)}"
        f"  selected={len(rows)}"
        + (f"  limite={limit}" if limit else "")
    )

    conn = _open_db(db_path)
    discovered = db_mod.save_discovered_ft_matches(conn, rows)
    print(f"[fetch-date] discovered_en_db={discovered}")

    ing_ok = 0
    ing_fail = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}
    total = len(rows)

    dual_started = False
    if total:
        dual_started = _print_dual_progress(
            "fetch-date", 0, total, 0, started=dual_started
        )

    for idx, row in enumerate(rows, start=1):
        match_id = str(row.get("match_id", "") or "")
        if not match_id:
            continue

        existing = db_mod.get_match(conn, match_id)
        if _is_ft_complete(existing):
            db_mod.mark_discovered_processed(conn, match_id)
            reason = "ya_completo_en_db"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            skipped += 1
            dual_started = _print_dual_progress(
                "fetch-date", idx, total, ing_fail, started=dual_started
            )
            continue

        try:
            data = scraper_mod.fetch_match_by_id(match_id)
            if not _has_usable_data(data):
                db_mod.mark_discovered_processed(conn, match_id)
                # Determine skip reason
                status_type = str(
                    (data or {}).get("match", {}).get("status_type", "") or ""
                ).lower()
                if status_type != "finished":
                    reason = f"not_finished({status_type or 'unknown'})"
                else:
                    # check which part is missing
                    quarters = (data or {}).get("score", {}).get("quarters", {})
                    pbp = (data or {}).get("play_by_play", {})
                    gp = (data or {}).get("graph_points", [])
                    missing_q = next(
                        (q for q in ("Q1", "Q2", "Q3", "Q4")
                         if not isinstance(quarters.get(q), dict)),
                        None,
                    )
                    if missing_q:
                        reason = f"sin_cuartos(falta={missing_q})"
                    elif not pbp.get("Q1") and not pbp.get("Q2"):
                        reason = "sin_pbp"
                    elif not gp:
                        reason = "sin_graph"
                    else:
                        reason = "datos_incompletos"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                skipped += 1
            else:
                db_mod.save_match(conn, match_id, data)
                db_mod.mark_discovered_processed(conn, match_id)
                ing_ok += 1
        except KeyboardInterrupt:
            print()
            print("[fetch-date] interrumpido por usuario")
            conn.close()
            return
        except Exception as exc:
            db_mod.mark_discovered_error(conn, match_id, str(exc))
            ing_fail += 1

        dual_started = _print_dual_progress(
            "fetch-date", idx, total, ing_fail, started=dual_started
        )

    if total:
        print()

    conn.close()
    print(
        f"[fetch-date] ok={ing_ok}  skipped={skipped}  fail={ing_fail}"
    )
    if skip_reasons:
        reasons_txt = "  ".join(
            f"{k}={v}" for k, v in sorted(skip_reasons.items())
        )
        print(f"[fetch-date] skip_reasons: {reasons_txt}")


def cmd_fetch_date(args: argparse.Namespace) -> None:
    event_date = args.date
    limit = args.limit
    _ingest_date_with_progress(args.db, event_date, limit)


def cmd_fetch_date_menu(args: argparse.Namespace) -> None:
    """Run the same interactive missing-date flow used by menu option 15."""
    result = _select_fetch_date_interactive(args.db)
    if result is None:
        print("[fetch-date-menu] cancelado")
        return
    event_date, limit = result
    _ingest_date_with_progress(args.db, event_date, limit)


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

    # backfill-links
    p_links = sub.add_parser(
        "backfill-links",
        help="Backfill SofaScore event slug/customId for stored matches",
    )
    p_links.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max stored matches to refresh in this run (default: 200)",
    )
    p_links.set_defaults(func=cmd_backfill_links)

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
        "--result-tag",
        default=None,
        help=(
            "Tag for DB result columns "
            "(default: <force-version>_<metric>)"
        ),
    )
    p_eval.add_argument(
        "--json",
        action="store_true",
        help="Also print JSON summary",
    )
    p_eval.set_defaults(func=cmd_eval_date)

    # eval-report
    p_report = sub.add_parser(
        "eval-report",
        help="Daily/monthly evaluation report by model and quarter",
    )
    p_report.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date (YYYY-MM-DD)",
    )
    p_report.add_argument(
        "--month",
        type=str,
        default=None,
        help="Month (YYYY-MM)",
    )
    p_report.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model name (e.g., v4, v6)",
    )
    p_report.add_argument(
        "--list",
        action="store_true",
        help="List available dates",
    )
    p_report.set_defaults(func=cmd_eval_report)

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

    # run-bot
    p_bot = sub.add_parser(
        "run-bot",
        help="Run Telegram bot (reads TELEGRAM_BOT_TOKEN from .env)",
    )
    p_bot.set_defaults(func=cmd_run_bot)

    # run-retrain
    p_retrain = sub.add_parser(
        "run-retrain",
        help="Retrain models and/or recalibrate gate (interactive or via --step)",
    )
    p_retrain.add_argument(
        "--step",
        choices=["train-v2", "train-v4", "train-v6", "train-v9", "train-v10", "compare", "calibrate", "all"],
        default=None,
        help=(
            "Step to run non-interactively. "
            "Choices: train-v2, train-v4, train-v6, train-v9, train-v10, compare, calibrate, all"
        ),
    )
    p_retrain.set_defaults(func=cmd_retrain)

    # run-report
    p_report = sub.add_parser(
        "run-report",
        help="Genera Excel comparativo de modelos para un mes dado",
    )
    p_report.add_argument(
        "--month",
        default=None,
        help="Mes a reportar (YYYY-MM). Si no se indica, pregunta interactivo.",
    )
    p_report.add_argument(
        "--out",
        default=None,
        help="Ruta de salida .xlsx. Por defecto: reports/model_comparison_<mes>.xlsx",
    )
    p_report.set_defaults(func=cmd_report)

    # eval-v13
    p_ev13 = sub.add_parser(
        "eval-v13",
        help="Genera reporte Excel de evaluacion V13 (out-of-sample)",
    )
    _grp = p_ev13.add_mutually_exclusive_group()
    _grp.add_argument("--month", metavar="YYYY-MM", default=None,
                      help="Mes completo a evaluar")
    _grp.add_argument("--from", dest="date_from", metavar="YYYY-MM-DD",
                      default=None, help="Fecha inicio")
    p_ev13.add_argument("--to", dest="date_to", metavar="YYYY-MM-DD",
                        default=None, help="Fecha fin (requiere --from)")
    p_ev13.add_argument("--models", default="v13",
                        help="Modelos separados por coma (default: v13)")
    p_ev13.add_argument("--odds", type=float, default=1.4,
                        help="Odds simuladas (default: 1.4)")
    p_ev13.add_argument("--bet-size", dest="bet_size", type=float, default=100.0,
                        help="Apuesta por bet (default: 100)")
    p_ev13.add_argument("--bank", type=float, default=1000.0,
                        help="Banco inicial (default: 1000)")
    p_ev13.add_argument("--out", default=None,
                        help="Ruta de salida .xlsx (default: auto con timestamp)")
    p_ev13.set_defaults(func=cmd_eval_v13)

    # menu
    p_menu = sub.add_parser(
        "menu",
        help="Open interactive menu mode",
    )
    p_menu.set_defaults(func=cmd_menu)

    # fetch-date
    p_fetch = sub.add_parser(
        "fetch-date",
        help="Discover and ingest finished matches for a specific date",
    )
    p_fetch.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Date to fetch from SofaScore",
    )
    p_fetch.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Max matches to ingest (default: no limit)",
    )
    p_fetch.set_defaults(func=cmd_fetch_date)

    # fetch-date-menu
    p_fetch_menu = sub.add_parser(
        "fetch-date-menu",
        help=(
            "Interactive missing-date selector (same flow as menu option 15)"
        ),
    )
    p_fetch_menu.set_defaults(func=cmd_fetch_date_menu)

    return parser


def main() -> None:
    parser = _build_parser()
    if len(sys.argv) == 1:
        _interactive_menu(parser, DEFAULT_DB)
        return

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""
bet_monitor.py — Betting window monitor daemon.

Smart alarm system that:
  1. Fetches today's / tomorrow's basketball schedule and stores it in
     `bet_monitor_schedule`.
  2. For each match, wakes up near the Q3 / Q4 window (estimated from live
     graph_points), fetches live data, runs inference, and fires a Telegram
     notification when the model says BET.
  3. Exposes MONITOR_STATUS + helper functions for the Telegram bot UI.
  4. Recovers on restart: reads the schedule table and resumes any pending
     matches.

Usage (inside telegam_bot.py):
    import bet_monitor
    bet_monitor.set_notify_callback(my_async_send_fn)
    bet_monitor.set_model_config({"q3": "v4", "q4": "v4"})
    stop_ev = asyncio.Event()
    task = asyncio.create_task(bet_monitor.run_monitor(DB_PATH, stop_ev))
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Awaitable, Callable

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger(__name__)

# ── Timing constants ──────────────────────────────────────────────────────────
UTC_OFFSET_HOURS = -6        # must match telegram_bot.py

Q3_MINUTE = 24               # game-minute where Q3 data is required
Q4_MINUTE = 36               # game-minute where Q4 data is required
WAKE_BEFORE_MINUTES = 4      # start polling this many game-minutes before cutoff
POLL_NEAR_SECS = 100         # poll interval when within WAKE_BEFORE window
IDLE_POLL_SECS = 300         # poll interval when far from next window
NO_BET_CONFIRM_TICKS = 2     # extra ticks before sending uncertain NO BET notification
SECS_PER_GAME_MIN = 170      # initial estimate: ~2.8 real-min per game-minute
MIN_GP_Q3 = 16               # graph_points with minute ≤ 24 required for Q3 check
MIN_GP_Q4 = 26               # graph_points with minute ≤ 36 required for Q4 check
MAX_FETCH_ERRORS = 4         # consecutive errors before discarding a match
SCHEDULE_REFRESH_HOURS = 8   # re-fetch schedule this often
NO_GRAPH_REAL_SECS = 55 * 60 # discard if no graph_points 55 real-min after start
MAX_CONCURRENT_FETCHES = 6   # max simultaneous Playwright fetches across all watchers
FINAL_FETCH_EXTRA_SECS = 300 # extra real-seconds after estimated end before final save
FINAL_FETCH_MIN_GP = 8       # require at least this many graph_points to attempt save

# ── Global state (read by telegram_bot.py) ────────────────────────────────────
MONITOR_STATUS: dict = {
    "running": False,
    "started_at": None,
    "today_total": 0,
    "tomorrow_total": 0,
    "checked_q3": 0,
    "checked_q4": 0,
    "bets_sent": 0,
    "no_bet": 0,
    "discarded": 0,
    "active_matches": [],
    "last_event": "",
}

# Injected by telegram_bot.py before starting the monitor
_notify_cb: Callable[..., Awaitable[None]] | None = None
_model_config: dict[str, str] = {"q3": "v4", "q4": "v4"}


def _is_bet_signal(signal: str) -> bool:
    """True for BET, BET_HOME, BET_AWAY — but not NO_BET."""
    s = str(signal).upper()
    return "BET" in s and "NO_BET" not in s and "NO BET" not in s


def _q4_timing() -> tuple[int, int, bool, int]:
    """Return (cutoff_minute, min_gp, requires_q3_score, wake_before) for Q4.

    V13/V15/V16 fire Q4 inference at minute ~31 (pre-Q4 or start of Q4
    in 10-min quarter leagues). A large wake_before ensures the monitor
    starts polling long before Q4 so the user has time to place the bet.
    Older models need minute-36 data with Q3 complete.
    """
    model = _model_config.get("q4", "v4")
    if model in ("v13", "v15", "v16"):
        # Start polling at minute 31-12=19 so the bet signal always fires
        # before Q4 even begins, regardless of quarter length (10 or 12 min).
        return 31, 16, False, 12
    # default (v4, v6, v9, v12, ...)
    return Q4_MINUTE, MIN_GP_Q4, True, WAKE_BEFORE_MINUTES


def set_notify_callback(cb: Callable[..., Awaitable[None]]) -> None:
    global _notify_cb
    _notify_cb = cb


def set_model_config(config: dict[str, str]) -> None:
    global _model_config
    _model_config = {**_model_config, **config}


# Lazy semaphore — created inside the monitor thread's event loop
_fetch_sem: asyncio.Semaphore | None = None


def _get_fetch_sem() -> asyncio.Semaphore:
    global _fetch_sem
    if _fetch_sem is None:
        _fetch_sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
    return _fetch_sem


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log(msg: str, level: str = "info") -> None:
    MONITOR_STATUS["last_event"] = msg
    getattr(logger, level)("[MONITOR] %s", msg)


async def _notify(
    msg: str,
    reply_markup: dict | None = None,
    notify_type: str = "bet",
) -> None:
    """notify_type: 'bet' | 'no_bet' | 'result'"""
    if _notify_cb:
        try:
            await _notify_cb(msg, reply_markup, notify_type)
        except Exception as exc:
            logger.error("[MONITOR] notify error: %s", exc)
    else:
        logger.info("[MONITOR][NOTIFY] %s", msg)


# ── DB ────────────────────────────────────────────────────────────────────────

def init_tables(db_path: str) -> None:
    conn = _open_db(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS bet_monitor_schedule (
            match_id          TEXT PRIMARY KEY,
            event_date        TEXT NOT NULL,
            home_team         TEXT,
            away_team         TEXT,
            league            TEXT,
            scheduled_utc_ts  INTEGER,
            scheduled_utc     TEXT,
            status            TEXT DEFAULT 'pending',
            q3_checked        INTEGER DEFAULT 0,
            q4_checked        INTEGER DEFAULT 0,
            q3_signal         TEXT,
            q4_signal         TEXT,
            q3_pick           TEXT,
            q4_pick           TEXT,
            q3_confidence     REAL,
            q4_confidence     REAL,
            q3_model          TEXT,
            q4_model          TEXT,
            q3_notified       INTEGER DEFAULT 0,
            q4_notified       INTEGER DEFAULT 0,
            skip_reason       TEXT,
            created_at        TEXT DEFAULT (datetime('now')),
            updated_at        TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS bet_monitor_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id       TEXT NOT NULL,
            event_date     TEXT,
            home_team      TEXT,
            away_team      TEXT,
            league         TEXT,
            target         TEXT,
            model          TEXT,
            signal         TEXT,
            recommendation TEXT,
            pick           TEXT,
            confidence     REAL,
            scraped_minute INTEGER,
            notified_at    TEXT,
            result         TEXT DEFAULT 'pending',
            result_checked INTEGER DEFAULT 0,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS inference_debug_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id       TEXT NOT NULL,
            target         TEXT NOT NULL,
            model          TEXT,
            scraped_minute INTEGER,
            signal         TEXT,
            confidence     REAL,
            gp_count       INTEGER,
            inference_json TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        -- Add columns if they don't exist yet (safe to run on existing DBs)
        CREATE TABLE IF NOT EXISTS _dummy_q3model (x);
    """)
    # ALTER TABLE is not transactional in SQLite executescript; do it separately
    for col_def in [
        ("bet_monitor_schedule", "q3_model",       "TEXT"),
        ("bet_monitor_schedule", "q4_model",       "TEXT"),
        ("bet_monitor_schedule", "final_fetched",  "INTEGER DEFAULT 0"),
        ("bet_monitor_schedule", "final_fetch_at", "TEXT"),
        ("bet_monitor_log",      "model",           "TEXT"),
        ("bet_monitor_log",      "result_checked",  "INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(
                f"ALTER TABLE {col_def[0]} ADD COLUMN {col_def[1]} {col_def[2]}"
            )
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()


def _open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _upsert_schedule_row(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO bet_monitor_schedule
            (match_id, event_date, home_team, away_team, league,
             scheduled_utc_ts, scheduled_utc, updated_at)
        VALUES (:match_id, :event_date, :home_team, :away_team, :league,
                :scheduled_utc_ts, :scheduled_utc, datetime('now'))
        ON CONFLICT(match_id) DO UPDATE SET
            event_date        = excluded.event_date,
            home_team         = excluded.home_team,
            away_team         = excluded.away_team,
            league            = excluded.league,
            scheduled_utc_ts  = excluded.scheduled_utc_ts,
            scheduled_utc     = excluded.scheduled_utc,
            updated_at        = excluded.updated_at
        WHERE bet_monitor_schedule.status = 'pending'
        """,
        row,
    )


def _get_pending_rows(conn: sqlite3.Connection, local_date: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT * FROM bet_monitor_schedule
        WHERE event_date = ?
          AND status NOT IN ('done', 'discarded')
        ORDER BY scheduled_utc_ts ASC
        """,
        (local_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def _update_row(conn: sqlite3.Connection, match_id: str, **kwargs) -> None:
    if not kwargs:
        return
    kwargs["_mid"] = match_id
    kwargs["_ts"] = datetime.utcnow().isoformat(timespec="seconds")
    sets = ", ".join(f"{k} = :{k}" for k in kwargs if not k.startswith("_"))
    sets += ", updated_at = :_ts"
    conn.execute(
        f"UPDATE bet_monitor_schedule SET {sets} WHERE match_id = :_mid",
        kwargs,
    )
    conn.commit()


def _insert_log(conn: sqlite3.Connection, **kwargs) -> None:
    kwargs.setdefault("notified_at", datetime.utcnow().isoformat(timespec="seconds"))
    cols = ", ".join(kwargs.keys())
    ph = ", ".join(f":{k}" for k in kwargs)
    conn.execute(f"INSERT INTO bet_monitor_log ({cols}) VALUES ({ph})", kwargs)
    conn.commit()


def reconcile_pending_results(db_path: str) -> int:
    """Update pending bet_monitor_log rows by cross-referencing quarter_scores.

    Runs synchronously (DB-only, no scraping).  Safe to call at any time.
    Returns the number of rows resolved.
    """
    conn = _open_db(db_path)
    pending = conn.execute(
        """
        SELECT id, match_id, target, pick
        FROM bet_monitor_log
        WHERE result = 'pending'
          AND signal NOT IN (
              'UNAVAILABLE', 'ERROR', 'NO_BET', 'NO BET',
              'window_missed', 'no_data', 'no_graph'
          )
        """,
    ).fetchall()

    resolved = 0
    for row in pending:
        q_key = str(row["target"] or "").upper()
        pick = str(row["pick"] or "")
        qs = conn.execute(
            "SELECT home, away FROM quarter_scores"
            " WHERE match_id = ? AND quarter = ?",
            (str(row["match_id"]), q_key),
        ).fetchone()
        if not qs:
            continue
        h, a = qs["home"], qs["away"]
        if h is None or a is None:
            continue
        try:
            h, a = int(h), int(a)
        except (TypeError, ValueError):
            continue
        if h == a:
            result_key = "push"
        elif pick == "home":
            result_key = "win" if h > a else "loss"
        else:
            result_key = "win" if a > h else "loss"
        conn.execute(
            "UPDATE bet_monitor_log SET result = ?, result_checked = 1"
            " WHERE id = ?",
            (result_key, row["id"]),
        )
        resolved += 1

    if resolved:
        conn.commit()
        _log(
            f"reconcile: {resolved} resultado(s) actualizado(s)"
        )
    conn.close()
    return resolved


# ── Schedule fetcher ──────────────────────────────────────────────────────────

def _fetch_all_events_for_date_sync(local_date: str) -> list[dict]:
    """Return all basketball events whose LOCAL date equals local_date.

    Queries two UTC dates (local_date and local_date+1) to catch overnight
    games, then filters by startTimestamp converted to local time.
    """
    from playwright.sync_api import sync_playwright

    UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    hdrs = {
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    local_dt = datetime.strptime(local_date, "%Y-%m-%d").date()
    utc_dates = [local_dt.isoformat(), (local_dt + timedelta(days=1)).isoformat()]

    all_events: list = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA)
        page = ctx.new_page()
        try:
            page.goto(
                "https://www.sofascore.com/basketball",
                wait_until="domcontentloaded",
                timeout=45_000,
            )
        except Exception:
            pass
        for utc_date in utc_dates:
            url = (
                "https://api.sofascore.com/api/v1/"
                f"sport/basketball/scheduled-events/{utc_date}"
            )
            try:
                resp = ctx.request.get(url, headers=hdrs, timeout=30_000)
                if resp.ok:
                    body = resp.json() or {}
                    all_events.extend(body.get("events", []))
            except Exception as exc:
                logger.warning("[MONITOR] schedule fetch %s: %s", utc_date, str(exc).split("\n")[0][:160])
        browser.close()

    out: list[dict] = []
    seen: set[str] = set()
    for ev in all_events:
        mid = str(ev.get("id", "") or "")
        if not mid or mid in seen:
            continue
        ts = int(ev.get("startTimestamp", 0) or 0)
        if ts == 0:
            continue
        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        dt_local = dt_utc + timedelta(hours=UTC_OFFSET_HOURS)
        if dt_local.date().isoformat() != local_date:
            continue
        seen.add(mid)
        out.append({
            "match_id": mid,
            "event_date": local_date,
            "home_team": (ev.get("homeTeam") or {}).get("name", ""),
            "away_team": (ev.get("awayTeam") or {}).get("name", ""),
            "league": ((ev.get("tournament") or {}).get("name", "")),
            "scheduled_utc_ts": ts,
            "scheduled_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "status_type": str((ev.get("status") or {}).get("type", "") or ""),
        })

    return sorted(out, key=lambda x: x["scheduled_utc_ts"])


# ── Live data helpers ─────────────────────────────────────────────────────────

def _get_minute(data: dict) -> int | None:
    gp = data.get("graph_points", [])
    if not gp:
        return None
    try:
        return int(gp[-1].get("minute", 0))
    except (TypeError, ValueError):
        return None


def _count_gp_up_to(data: dict, max_min: int) -> int:
    return sum(
        1 for p in data.get("graph_points", [])
        if int(p.get("minute", 0) or 0) <= max_min
    )


def _has_scores(data: dict, *quarters: str) -> bool:
    qs = (data.get("score", {}) or {}).get("quarters", {}) or {}
    return all(
        isinstance(qs.get(q), dict) and qs[q].get("home") is not None
        for q in quarters
    )


def _is_two_half(data: dict) -> bool:
    s = data.get("score", {}) or {}
    qs = s.get("quarters", {}) or {}
    if qs.get("Q3") or qs.get("Q4"):
        return False
    st = str((data.get("match", {}) or {}).get("status_type", "") or "").lower()
    if st == "finished":
        scored = sum(
            1 for k in ("Q1", "Q2", "Q3", "Q4")
            if isinstance(qs.get(k), dict) and qs[k].get("home") is not None
        )
        return scored <= 2
    gp = data.get("graph_points", [])
    if gp:
        try:
            return int(gp[-1].get("minute", 0)) >= 36
        except (TypeError, ValueError):
            pass
    return False


async def _final_fetch_and_save(
    match_id: str,
    db_path: str,
    conn_sched: sqlite3.Connection,
    home: str,
    away: str,
    current_minute: int | None,
    secs_per_gmin: float,
    stop_event: asyncio.Event,
    sleep_fn,
) -> None:
    """Wait for the estimated end of the match, then scrape and persist
    the final result to the matches DB.

    Only attempts the save if the live data already has graph_points
    (FINAL_FETCH_MIN_GP threshold). Matches without graph data are skipped
    to avoid polluting the DB with empty records.
    """
    import scraper as scraper_mod

    # Estimate real seconds until minute 48 (end of Q4)
    GAME_END_MINUTE = 48
    mins_remaining = max(0, GAME_END_MINUTE - (current_minute or GAME_END_MINUTE))
    wait_secs = mins_remaining * secs_per_gmin + FINAL_FETCH_EXTRA_SECS
    _log(
        f"{home} vs {away}: esperando fin estimado "
        f"(~{wait_secs / 60:.0f} min) antes de guardar resultado final"
    )
    if await sleep_fn(wait_secs):
        return  # stop_event fired

    # Poll every 90s until status=finished, then save.  Give up after 40 min.
    MAX_FINISH_WAIT = 40 * 60
    POLL_INTERVAL = 90
    waited_finish = 0
    last_data: dict | None = None

    while waited_finish <= MAX_FINISH_WAIT:
        if stop_event.is_set():
            return
        try:
            async with _get_fetch_sem():
                last_data = await asyncio.to_thread(
                    scraper_mod.fetch_match_by_id, match_id
                )
        except Exception as exc:
            logger.warning(
                "[MONITOR] final_fetch %s error: %s",
                match_id, str(exc).split("\n")[0][:120],
            )
            if await sleep_fn(POLL_INTERVAL):
                return
            waited_finish += POLL_INTERVAL
            continue

        if last_data:
            st = str(
                (last_data.get("match", {}) or {}).get("status_type", "") or ""
            ).lower()
            if st == "finished":
                break
            _log(
                f"{home} vs {away}: final_fetch esperando fin "
                f"(status={st}, espera={waited_finish}s)"
            )

        if await sleep_fn(POLL_INTERVAL):
            return
        waited_finish += POLL_INTERVAL

    data = last_data
    if not data:
        _log(f"{home} vs {away}: final_fetch sin datos al finalizar")
        return

    gp_total = len(data.get("graph_points") or [])
    if gp_total < FINAL_FETCH_MIN_GP:
        _log(
            f"{home} vs {away}: final_fetch omitido "
            f"(sólo {gp_total} graph_points, mínimo {FINAL_FETCH_MIN_GP})"
        )
        return

    # Save to matches DB
    try:
        db_conn = __import__("db").get_conn(db_path)
        __import__("db").init_db(db_conn)
        __import__("db").save_match(db_conn, match_id, data)
        db_conn.close()
        _update_row(
            conn_sched, match_id,
            final_fetched=1,
            final_fetch_at=datetime.utcnow().isoformat(timespec="seconds"),
        )
        _log(f"{home} vs {away}: resultado final guardado (gp={gp_total})")
    except Exception as exc:
        logger.warning(
            "[MONITOR] final_fetch save error %s: %s", match_id, exc
        )


# ── Inference runner ──────────────────────────────────────────────────────────

def _get_v12_mae(target: str, metric_type: str) -> float | None:
    """Load V12 MAE for target/type from all_metrics.csv."""
    try:
        import csv as _csv
        from pathlib import Path as _Path
        csv_path = _Path(__file__).parent / "training" / "model_outputs_v12" / "all_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError
        metrics: dict[str, float] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                key = str(row.get("model") or "")
                metrics[key] = float(row.get("mae") or 0)
        for suffix in ("men_or_open", "women", ""):
            k = f"{target}_{metric_type}{'_' + suffix if suffix else ''}"
            if k in metrics:
                return metrics[k]
    except Exception:
        pass
    fallbacks = {
        "q3_home": 3.63, "q3_away": 3.60, "q3_total": 5.33,
        "q4_home": 3.63, "q4_away": 3.60, "q4_total": 5.33,
    }
    return fallbacks.get(f"{target}_{metric_type}")


def _run_inference_sync(match_id: str, target: str) -> dict:
    """Run inference synchronously (called via asyncio.to_thread)."""
    version = _model_config.get(target, "v4")

    # V12/V13 have their own inference engines — not handled by infer_match.py
    if version == "v12":
        try:
            v12_mod = importlib.import_module("training.v12.infer_match_v12")
            pred = v12_mod.run_inference(
                match_id=match_id,
                target=target,
                fetch_missing=False,
            )
            if isinstance(pred, dict) and not pred.get("ok", True):
                return {"ok": False, "reason": pred.get("reason", "V12 failed")}

            def _attr(obj, name, default=None):
                if isinstance(obj, dict):
                    return obj.get(name, default)
                return getattr(obj, name, default)

            return {
                "ok": True,
                "predictions": {
                    target: {
                        "available": True,
                        "predicted_winner": _attr(pred, "winner_pick"),
                        "confidence": _attr(pred, "winner_confidence"),
                        "bet_signal": _attr(pred, "winner_signal"),
                        "final_recommendation": _attr(pred, "final_signal"),
                        "predicted_total": _attr(pred, "predicted_total"),
                        "predicted_home": _attr(pred, "predicted_home"),
                        "predicted_away": _attr(pred, "predicted_away"),
                        "reasoning": _attr(pred, "reasoning"),
                        "mae": _get_v12_mae(target, "total"),
                        "mae_home": _get_v12_mae(target, "home"),
                        "mae_away": _get_v12_mae(target, "away"),
                        "league_quality": _attr(pred, "league_quality"),
                        "league_bettable": _attr(pred, "league_bettable"),
                        "volatility_index": _attr(pred, "volatility_index"),
                        "data_quality": _attr(pred, "data_quality"),
                    }
                },
            }
        except Exception as exc:
            return {"ok": False, "reason": f"V12 error: {exc}"}

    if version == "v13":
        try:
            v13_mod = importlib.import_module("training.v13.infer_match_v13")
            v13_result = v13_mod.run_inference(match_id=match_id, target=target)
            if not v13_result.get("ok", False):
                return {"ok": False, "reason": v13_result.get("reason", "V13 failed")}
            pred = v13_result.get("prediction")

            def _attr(obj, name, default=None):
                if isinstance(obj, dict):
                    return obj.get(name, default)
                return getattr(obj, name, default)

            return {
                "ok": True,
                "predictions": {
                    target: {
                        "available": True,
                        "predicted_winner": _attr(pred, "winner_pick"),
                        "confidence": _attr(pred, "winner_confidence"),
                        "bet_signal": _attr(pred, "winner_signal"),
                        "final_recommendation": _attr(pred, "final_signal"),
                        "predicted_total": _attr(pred, "predicted_total"),
                        "predicted_home": _attr(pred, "predicted_home"),
                        "predicted_away": _attr(pred, "predicted_away"),
                        "reasoning": _attr(pred, "reasoning"),
                        "mae": _attr(pred, "mae"),
                        "mae_home": _attr(pred, "mae_home"),
                        "mae_away": _attr(pred, "mae_away"),
                        "league_quality": _attr(pred, "league_quality"),
                        "league_bettable": _attr(pred, "league_bettable"),
                        "volatility_index": _attr(pred, "volatility_index"),
                        "data_quality": _attr(pred, "data_quality"),
                        "model_quality": _attr(pred, "model_quality"),
                        "model_samples": _attr(pred, "model_samples"),
                        "model_gap": _attr(pred, "model_gap"),
                        "model_f1": _attr(pred, "model_f1"),
                        "fallback_used": _attr(pred, "fallback_used", False),
                    }
                },
            }
        except Exception as exc:
            return {"ok": False, "reason": f"V13 error: {exc}"}

    infer_mod = importlib.import_module("training.infer_match")
    # Suppress live snapshot fetch — we already have the data
    infer_mod.scraper_mod.fetch_event_snapshot = lambda _mid: None
    return infer_mod.run_inference(
        match_id=match_id,
        metric="f1",
        fetch_missing=False,
        force_version=version,
    )


def _friendly_reason(reason: str) -> str:
    """Translate gate reason codes to Spanish labels."""
    reason_lower = str(reason).lower()
    if "league not bettable" in reason_lower:
        return "liga no apostable"
    if "insufficient graph points" in reason_lower or "missing" in reason_lower and "graph" in reason_lower:
        return "puntos de grafico insuficientes"
    if "insufficient confidence" in reason_lower:
        return "confianza insuficiente para apostar"
    if "too volatile" in reason_lower:
        return "partido muy volatil"
    if "confidence below" in reason_lower:
        return "confianza por debajo del minimo"
    if "missing" in reason_lower and "play" in reason_lower:
        return "faltan datos de jugadas"
    if "match too early" in reason_lower:
        return "partido muy temprano"
    mapping = {
        "league not bettable": "liga no apostable",
        "insufficient_graph_or_pbp_coverage": "cobertura de datos insuficiente",
        "match_too_volatile_for_current_signal": "partido muy volatil",
        "confidence_below_minimum_edge": "confianza por debajo del minimo",
        "missing_q3_score": "Q3 aun no disponible",
        "missing_q1_q2_scores": "sin scores Q1/Q2",
    }
    return mapping.get(reason, reason or "no disponible")


def _league_stats_detail(league: str) -> str | None:
    """Return a one-line stats summary explaining why the league is not bettable."""
    try:
        stats_file = Path(__file__).resolve().parent / "training" / "v12" / "model_outputs" / "league_stats.json"
        if not stats_file.exists():
            return None
        with open(stats_file, "r", encoding="utf-8") as _f:
            _stats = json.load(_f)
        # Exact match first, then base name before comma, then prefix
        s = _stats.get(league, {})
        if not s and "," in league:
            s = _stats.get(league.split(",")[0].strip(), {})
        if not s:
            for key in sorted(_stats.keys(), key=len, reverse=True):
                if league.startswith(key):
                    s = _stats[key]
                    break
        if not s:
            return "📊 Liga sin datos históricos"
        samples = int(s.get("samples", 0))
        home_wr = float(s.get("home_win_rate", 0.5))
        if samples < 30:
            return f"📊 Solo {samples} partidos en histórico (mínimo 30)"
        home_pct = round(home_wr * 100)
        return f"📊 N={samples} partidos | Local gana {home_pct}% — sin ventaja clara"
    except Exception:
        return None


def _is_definitive_no_bet(pred: dict) -> bool:
    """True when the NO BET reason is categorical and won't change with more data.
    These are notified immediately; all other reasons wait for confirmation ticks."""
    reasoning = str(
        pred.get("reasoning") or pred.get("gate_reason") or pred.get("reason") or ""
    ).lower()
    return "league not bettable" in reasoning


def _insert_inference_debug(
    db_path: str,
    match_id: str,
    target: str,
    model: str,
    scraped_minute: int | None,
    signal: str,
    confidence: float,
    gp_count: int,
    inference_json: str,
) -> None:
    """Persist raw inference result to inference_debug_log for later debugging."""
    try:
        conn = _open_db(db_path)
        conn.execute(
            """
            INSERT INTO inference_debug_log
                (match_id, target, model, scraped_minute, signal, confidence,
                 gp_count, inference_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (match_id, target, model, scraped_minute, signal, confidence,
             gp_count, inference_json),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("[MONITOR] inference_debug insert error: %s", exc)


def _bold_num(value) -> str:
    trans = str.maketrans("0123456789", "𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵")
    return str(value).translate(trans)


def _format_bet_notification(
    match_id: str,
    data: dict,
    quarter_label: str,   # "Q3" or "Q4"
    home: str,
    away: str,
    league: str,
    event_date: str,
    current_minute: int | None,
    pick_sym: str,
    pick_name: str,
    confidence: float,
    model: str,
    is_bet: bool = True,
    pred: dict | None = None,
) -> str:
    """Build a match-detail-style BET/NO BET notification message."""
    m_info = data.get("match") or {}
    status_type = str(m_info.get("status_type") or "")
    quarters = (data.get("score") or {}).get("quarters", {})
    q_order = ["Q1", "Q2", "Q3", "Q4"]
    q_end_minute = {"Q1": 12, "Q2": 24, "Q3": 36, "Q4": 48}

    def _q_line(q: str) -> str:
        qd = quarters.get(q)
        if not qd or qd.get("home") is None:
            return f"{q}: - ⏳"
        h, a = int(qd["home"]), int(qd["away"])
        # Score text with bold winner
        if h > a:
            score_txt = f"{_bold_num(h)} - {a}"
        elif a > h:
            score_txt = f"{h} - {_bold_num(a)}"
        else:
            score_txt = f"{h} - {a}"
        # Determine if quarter is final
        is_final = (
            status_type == "finished"
            or (current_minute is not None and current_minute >= q_end_minute.get(q, 99))
            # Quarter is final if a later quarter already has scores
            or any(
                quarters.get(later_q, {}).get("home") is not None
                for later_q in q_order[q_order.index(q) + 1:]
            )
        )
        if not is_final:
            return f"{q}: {score_txt} ⏳"
        if h > a:
            return f"{q}: {score_txt} 🏠 {home}"
        elif a > h:
            return f"{q}: {score_txt} ✈️ {away}"
        return f"{q}: {score_txt}"

    score_lines = "\n".join(_q_line(q) for q in q_order)

    match_time = str(m_info.get("time") or "")
    match_date_str = str(m_info.get("date") or event_date)
    try:
        from datetime import datetime as _dt
        if match_date_str and match_time:
            _utc_dt = _dt.strptime(f"{match_date_str} {match_time}", "%Y-%m-%d %H:%M")
            _local_dt = _utc_dt + timedelta(hours=UTC_OFFSET_HOURS)
            fecha_txt = f"{_local_dt.strftime('%Y-%m-%d %H:%M')} UTC{UTC_OFFSET_HOURS:+d}"
        else:
            fecha_txt = match_date_str
    except Exception:
        fecha_txt = f"{match_date_str} {match_time}" if match_time else match_date_str

    if is_bet:
        header = f"🟢 APUESTA {quarter_label} [{model}]"
        _p = pred or {}
        _footer_parts = [f"Pick: {pick_sym} {pick_name}"]

        # Projection (V12 and other models that expose predicted_home/away)
        def _sf_bet(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        _ph = _sf_bet(_p.get("predicted_home"))
        _pa = _sf_bet(_p.get("predicted_away"))
        _pt = _sf_bet(_p.get("predicted_total"))
        _mae = _sf_bet(_p.get("mae"))
        _mae_h = _sf_bet(_p.get("mae_home"))
        _mae_a = _sf_bet(_p.get("mae_away"))
        if _ph is not None and _pa is not None:
            _mh_txt = f" (±{_mae_h:.1f})" if _mae_h is not None else ""
            _ma_txt = f" (±{_mae_a:.1f})" if _mae_a is not None else ""
            _mt_txt = f" (±{_mae:.0f})" if _mae is not None else ""
            _overlap = ""
            if _mae_h is not None and _mae_a is not None:
                _diff = abs(_ph - _pa)
                _thresh = _mae_h + _mae_a
                if _diff < _thresh:
                    _overlap = f" ⚠️ MAE se superpone (dif={_diff:.1f} < {_thresh:.1f})"
            _pt_val = _pt if _pt is not None else (_ph + _pa)
            _footer_parts.append(
                f"Proyeccion: 🏠~{_ph:.1f}{_mh_txt} | ✈️~{_pa:.1f}{_ma_txt}"
                f" | Total~{_pt_val:.1f}{_mt_txt}{_overlap}"
            )

        _footer_parts.append(f"Confianza: {confidence * 100:.1f}%")
        pick_line = "\n".join(_footer_parts)
    else:
        header = f"🔴 NO APOSTAR {quarter_label} [{model}]"
        _p = pred or {}
        _footer_parts = ["Señal: NO BET"]

        # Tendency
        if pick_sym and pick_name:
            _footer_parts.append(f"Tendencia {pick_sym} {pick_name}")

        # Reason
        _reason_raw = str(_p.get("gate_reason") or _p.get("reasoning") or _p.get("reason") or "").strip()
        if _reason_raw:
            _footer_parts.append(f"Motivo: {_friendly_reason(_reason_raw).capitalize()}")
        if "league not bettable" in _reason_raw.lower():
            _ld = _league_stats_detail(league)
            if _ld:
                _footer_parts.append(_ld)

        # Projection (V12 or any model that sets predicted_home/away)
        def _sf(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        _ph = _sf(_p.get("predicted_home"))
        _pa = _sf(_p.get("predicted_away"))
        _pt = _sf(_p.get("predicted_total"))
        _mae = _sf(_p.get("mae"))
        _mae_h = _sf(_p.get("mae_home"))
        _mae_a = _sf(_p.get("mae_away"))
        if _ph is not None and _pa is not None:
            _mh_txt = f" (±{_mae_h:.1f})" if _mae_h is not None else ""
            _ma_txt = f" (±{_mae_a:.1f})" if _mae_a is not None else ""
            _mt_txt = f" (±{_mae:.0f})" if _mae is not None else ""
            _overlap = ""
            if _mae_h is not None and _mae_a is not None:
                _diff = abs(_ph - _pa)
                _thresh = _mae_h + _mae_a
                if _diff < _thresh:
                    _overlap = f" ⚠️ MAE se superpone (dif={_diff:.1f} < {_thresh:.1f})"
            _pt_val = _pt if _pt is not None else (_ph + _pa)
            _footer_parts.append(
                f"Proyeccion: 🏠~{_ph:.1f}{_mh_txt} | ✈️~{_pa:.1f}{_ma_txt}"
                f" | Total~{_pt_val:.1f}{_mt_txt}{_overlap}"
            )

        # Score (confidence as percentage)
        if confidence > 0:
            _footer_parts.append(f"Score {confidence * 100:.0f}%")

        pick_line = "\n".join(_footer_parts)

    return (
        f"{header}\n\n"
        f"Match ID: {match_id} | Min: {current_minute or '?'}\n"
        f"{home} vs {away}\n"
        f"Fecha: {fecha_txt}\n"
        f"Liga: {league}\n"
        f"{score_lines}\n\n"
        f"{pick_line}"
    )


async def _check_quarter(
    match_id: str,
    data: dict,
    target: str,          # "q3" | "q4"
    db_path: str,
    conn_sched: sqlite3.Connection,
    home: str,
    away: str,
    league: str,
    event_date: str,
    current_minute: int | None,
    suppress_no_bet_notify: bool = False,
) -> tuple[str, bool, dict]:
    """Run inference for one quarter.  Returns (signal, notified, pred)."""
    import db as db_mod

    quarter_label = target.upper()
    _log(f"{home} vs {away}: analizando {quarter_label} (min {current_minute})")

    # Persist scraped data so infer_match can load from DB
    try:
        tmp = _open_db(db_path)
        db_mod.save_match(tmp, match_id, data)
        tmp.close()
    except Exception as exc:
        logger.warning("[MONITOR] save_match error %s: %s", match_id, exc)

    # Run inference in thread
    try:
        result = await asyncio.to_thread(_run_inference_sync, match_id, target)
    except Exception as exc:
        logger.error("[MONITOR] inference error %s %s: %s", match_id, target, exc)
        return "ERROR", False, {}

    pred: dict = {}
    if result.get("ok"):
        pred = result.get("predictions", {}).get(target, {}) or {}

    available = bool(pred.get("available", False))
    signal = str(
        pred.get("final_recommendation") or pred.get("bet_signal") or "NO_BET"
    ).upper().strip()
    pick = str(pred.get("predicted_winner") or "")
    try:
        confidence = float(pred.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    # Update counters
    if target == "q3":
        MONITOR_STATUS["checked_q3"] = MONITOR_STATUS.get("checked_q3", 0) + 1
    else:
        MONITOR_STATUS["checked_q4"] = MONITOR_STATUS.get("checked_q4", 0) + 1

    # Persist raw inference JSON for later debugging
    try:
        _gp_count = len(data.get("graph_points") or [])
        _insert_inference_debug(
            db_path=db_path,
            match_id=match_id,
            target=target,
            model=_model_config.get(target, "-"),
            scraped_minute=current_minute,
            signal=signal if available else "UNAVAILABLE",
            confidence=round(confidence, 4) if available else 0.0,
            gp_count=_gp_count,
            inference_json=json.dumps(result, default=str),
        )
    except Exception as _exc:
        logger.warning("[MONITOR] debug log error: %s", _exc)

    # ── Terminal debug dump ────────────────────────────────────────────────
    _v13_cutoff = 22 if target == "q3" else 31   # matches v13/config.py
    _gp_all = data.get("graph_points") or []
    _gp_used = [p for p in _gp_all if int(p.get("minute", 0)) <= _v13_cutoff]
    print(
        f"\n{'='*60}\n"
        f"[MONITOR DEBUG] {home} vs {away} — {quarter_label} | match_id={match_id}\n"
        f"  Model    : {_model_config.get(target, '-')}\n"
        f"  Signal   : {signal if available else 'UNAVAILABLE'}\n"
        f"  Confidence: {confidence*100:.1f}%\n"
        f"  Min scraped: {current_minute}\n"
        f"  GP total={len(_gp_all)} | GP used (≤min{_v13_cutoff})={len(_gp_used)}\n"
        f"  GP used  : {json.dumps(_gp_used, default=str)}\n"
        f"  Inference JSON:\n{json.dumps(result, ensure_ascii=False, default=str, indent=2)}\n"
        f"{'='*60}",
        flush=True,
    )
    # ── End terminal debug dump ───────────────────────────────────────────

    # Persist log entry
    log_conn = _open_db(db_path)
    _insert_log(
        log_conn,
        match_id=match_id,
        event_date=event_date,
        home_team=home,
        away_team=away,
        league=league,
        target=target,
        model=_model_config.get(target, "-"),
        signal=signal if available else "UNAVAILABLE",
        recommendation=str(pred.get("reason") or signal),
        pick=pick,
        confidence=round(confidence, 4),
        scraped_minute=current_minute,
        result="pending",
    )
    log_conn.close()

    if not available:
        reason = str(pred.get("reason") or "unavailable")
        _log(f"{home} vs {away}: {quarter_label} no disponible ({reason})")
        return "UNAVAILABLE", False, pred

    notified = False
    if _is_bet_signal(signal):
        pick_sym = "🏠" if pick == "home" else ("✈️" if pick == "away" else "?")
        pick_name = home if pick == "home" else (away if pick == "away" else pick)
        model_used = _model_config.get(target, "-")
        msg = _format_bet_notification(
            match_id=match_id,
            data=data,
            quarter_label=quarter_label,
            home=home,
            away=away,
            league=league,
            event_date=event_date,
            current_minute=current_minute,
            pick_sym=pick_sym,
            pick_name=pick_name,
            confidence=confidence,
            model=model_used,
            pred=pred,
        )
        markup = {
            "inline_keyboard": [[
                {"text": "🔍 Ver match", "callback_data": f"notifmatch:{match_id}:{event_date}:0"},
                {"text": "📱 Sofascore", "url": (
                    f"https://www.sofascore.com/{data.get('match', {}).get('event_slug') or ''}/{data.get('match', {}).get('custom_id') or ''}#id:{match_id}"
                    if (data.get("match", {}).get("event_slug") and data.get("match", {}).get("custom_id"))
                    else f"https://www.sofascore.com/basketball/event/{match_id}"
                )},
            ]]
        }
        await _notify(msg, reply_markup=markup, notify_type="bet")
        MONITOR_STATUS["bets_sent"] = MONITOR_STATUS.get("bets_sent", 0) + 1
        notified = True
        _log(f"🟢 BET {quarter_label} [{model_used}]: {home} vs {away} → {pick} ({confidence * 100:.0f}%)")
        # Schedule result check after match finishes
        asyncio.ensure_future(
            _resolve_bet_result(
                match_id, target, pick, pick_name,
                home, away, league, event_date, db_path,
            )
        )
    else:
        model_used = _model_config.get(target, "-")
        # Definitive reasons (e.g. league_not_bettable) fire immediately;
        # uncertain reasons wait for confirmation ticks (suppress_no_bet_notify).
        _send_now = not suppress_no_bet_notify or _is_definitive_no_bet(pred)
        _log(
            f"🔴 NO BET {quarter_label} [{model_used}]: {home} vs {away} | signal={signal}"
            + ("" if _send_now else " [esperando confirmacion]")
        )
        if _send_now:
            MONITOR_STATUS["no_bet"] = MONITOR_STATUS.get("no_bet", 0) + 1
            _nb_pick_sym = "🏠" if pick == "home" else ("✈️" if pick == "away" else "")
            _nb_pick_name = home if pick == "home" else (away if pick == "away" else "")
            no_bet_msg = _format_bet_notification(
                match_id=match_id,
                data=data,
                quarter_label=quarter_label,
                home=home,
                away=away,
                league=league,
                event_date=event_date,
                current_minute=current_minute,
                pick_sym=_nb_pick_sym,
                pick_name=_nb_pick_name,
                confidence=confidence,
                model=model_used,
                is_bet=False,
                pred=pred,
            )
            markup = {
                "inline_keyboard": [[
                    {"text": "🔍 Ver match", "callback_data": f"notifmatch:{match_id}:{event_date}:0"},
                    {"text": "📱 Sofascore", "url": (
                        f"https://www.sofascore.com/{data.get('match', {}).get('event_slug') or ''}/{data.get('match', {}).get('custom_id') or ''}#id:{match_id}"
                        if (data.get("match", {}).get("event_slug") and data.get("match", {}).get("custom_id"))
                        else f"https://www.sofascore.com/basketball/event/{match_id}"
                    )},
                ]]
            }
            await _notify(no_bet_msg, reply_markup=markup, notify_type="no_bet")
            notified = True

    return signal, notified, pred


async def _resolve_bet_result(
    match_id: str,
    target: str,
    pick: str,
    pick_name: str,
    home: str,
    away: str,
    league: str,
    event_date: str,
    db_path: str,
) -> None:
    """Wait for the match to finish then notify win/loss for the bet quarter."""
    import scraper as scraper_mod

    quarter_label = target.upper()
    q_key = {"q3": "Q3", "q4": "Q4"}.get(target, target.upper())
    _log(f"Revision resultado {quarter_label}: {home} vs {away}")

    MAX_WAIT_SECS = 4 * 3600   # give up after 4 h
    POLL_SECS = 120
    waited = 0

    data: dict | None = None
    while waited < MAX_WAIT_SECS:
        await asyncio.sleep(POLL_SECS)
        waited += POLL_SECS
        try:
            async with _get_fetch_sem():
                data = await asyncio.to_thread(scraper_mod.fetch_match_by_id, match_id)
        except Exception as exc:
            logger.warning("[MONITOR] result-poll %s: %s", match_id, str(exc).split("\n")[0][:160])
            continue
        if not data:
            continue
        st = str((data.get("match", {}) or {}).get("status_type", "") or "").lower()
        if st == "finished":
            break
    else:
        _log(f"Revision resultado {quarter_label} {home} vs {away}: timeout sin finalizar")
        return

    # Read the actual quarter score
    qs = ((data or {}).get("score", {}) or {}).get("quarters", {}) or {}
    qd = qs.get(q_key)
    if not isinstance(qd, dict):
        _log(f"Revision resultado {quarter_label}: sin data de cuarto {q_key}")
        return

    q_home = qd.get("home")
    q_away = qd.get("away")
    if q_home is None or q_away is None:
        _log(f"Revision resultado {quarter_label}: scores nulos para {q_key}")
        return

    try:
        q_home = int(q_home)
        q_away = int(q_away)
    except (TypeError, ValueError):
        return

    if q_home == q_away:
        outcome = "empate"
        result_key = "push"
    elif pick == "home":
        outcome = "GANADA ✅" if q_home > q_away else "PERDIDA ❌"
        result_key = "win" if q_home > q_away else "loss"
    else:
        outcome = "GANADA ✅" if q_away > q_home else "PERDIDA ❌"
        result_key = "win" if q_away > q_home else "loss"

    msg = (
        f"{'✅' if result_key == 'win' else ('❌' if result_key == 'loss' else '➖')} "
        f"RESULTADO {quarter_label} — {outcome}\n"
        f"Match: {home} vs {away}\n"
        f"Liga: {league}\n"
        f"Pick: {pick_name}  |  Score {q_key}: {q_home}-{q_away}\n"
        f"Fecha: {event_date}  |  ID: {match_id}"
    )
    _m = (data.get("match") or {}) if data else {}
    _ev_slug = _m.get("event_slug") or ""
    _custom_id = _m.get("custom_id") or ""
    if _ev_slug and _custom_id:
        _sf_url = f"https://www.sofascore.com/{_ev_slug}/{_custom_id}#id:{match_id}"
    else:
        _sf_url = f"https://www.sofascore.com/basketball/event/{match_id}"
    _result_markup = {"inline_keyboard": [
        [
            {"text": "🔍 Ver Match", "callback_data": f"match:{match_id}:_:0"},
            {"text": "📱 Sofascore", "url": _sf_url},
        ],
    ]}
    await _notify(msg, reply_markup=_result_markup, notify_type="result")
    _log(f"Resultado {quarter_label} {home} vs {away}: {outcome} ({q_home}-{q_away})")

    # Update the log row
    try:
        conn = _open_db(db_path)
        row_id = conn.execute(
            """
            SELECT id FROM bet_monitor_log
            WHERE match_id = ? AND target = ? AND result = 'pending'
            ORDER BY id DESC LIMIT 1
            """,
            (match_id, target),
        ).fetchone()
        if row_id:
            conn.execute(
                "UPDATE bet_monitor_log SET result = ?, result_checked = 1 WHERE id = ?",
                (result_key, row_id["id"]),
            )
            conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("[MONITOR] result update error %s: %s", match_id, exc)


# ── Match watcher coroutine ───────────────────────────────────────────────────

async def _watch_match(
    match_id: str,
    row: dict,
    db_path: str,
    stop_event: asyncio.Event,
) -> None:
    """Watch one match from start to finish, checking Q3 and Q4.

    Smart-sleep strategy:
      - Before Q3 window: sleep estimate = (min_to_wake * secs_per_game_min)
      - Inside Q3/Q4 window: poll every POLL_NEAR_SECS until data is ready
      - After Q3+6 min without data: skip Q3 and proceed to Q4
      - If no graph_points 55 min after scheduled start: discard entirely
    """
    import scraper as scraper_mod

    home = str(row.get("home_team") or "?")
    away = str(row.get("away_team") or "?")
    league = str(row.get("league") or "")
    event_date = str(row.get("event_date") or "")
    scheduled_ts = int(row.get("scheduled_utc_ts") or 0)

    _log(f"Vigilando: {home} vs {away} ({match_id})")
    MONITOR_STATUS["active_matches"] = list(
        set(MONITOR_STATUS.get("active_matches", []) + [match_id])
    )

    q3_done = bool(row.get("q3_checked"))
    q4_done = bool(row.get("q4_checked"))
    q3_no_bet_ticks = 0   # confirmation ticks for uncertain Q3 NO BET
    q4_no_bet_ticks = 0   # confirmation ticks for uncertain Q4 NO BET
    errors = 0
    secs_per_gmin = float(SECS_PER_GAME_MIN)
    last_gmin: int | None = None
    last_gmin_wall: float = 0.0

    conn = _open_db(db_path)

    async def _sleep(seconds: float) -> bool:
        """Interruptible sleep.  Returns True if stop_event fired."""
        end = time.monotonic() + max(1.0, seconds)
        while time.monotonic() < end:
            if stop_event.is_set():
                return True
            await asyncio.sleep(min(10.0, end - time.monotonic()))
        return False

    try:
        # Wait until close to match start (2-min buffer)
        if scheduled_ts > 0:
            wait = scheduled_ts - time.time() - 120
            if wait > 120:
                _log(f"{home} vs {away}: inicio en {wait / 60:.0f} min, esperando")
                if await _sleep(wait):
                    return

        # Main watch loop
        while not stop_event.is_set():
            if q3_done and q4_done:
                _update_row(conn, match_id, status="done")
                # Schedule final data save after estimated match end
                gp_total = len(data.get("graph_points") or []) if data else 0
                if gp_total >= FINAL_FETCH_MIN_GP:
                    await _final_fetch_and_save(
                        match_id=match_id,
                        db_path=db_path,
                        conn_sched=conn,
                        home=home,
                        away=away,
                        current_minute=minute,
                        secs_per_gmin=secs_per_gmin,
                        stop_event=stop_event,
                        sleep_fn=_sleep,
                    )
                break

            # Fetch live data
            try:
                async with _get_fetch_sem():
                    data = await asyncio.to_thread(
                        scraper_mod.fetch_match_by_id, match_id
                    )
                errors = 0
            except Exception as exc:
                errors += 1
                # Truncate Playwright call-log to first meaningful line
                exc_short = str(exc).split("\n")[0][:160]
                logger.warning(
                    "[MONITOR] %s fetch error #%d: %s", match_id, errors, exc_short
                )
                if errors >= MAX_FETCH_ERRORS:
                    _update_row(
                        conn, match_id,
                        status="discarded", skip_reason="fetch_errors",
                    )
                    MONITOR_STATUS["discarded"] = (
                        MONITOR_STATUS.get("discarded", 0) + 1
                    )
                    _log(f"{home} vs {away}: descartado ({errors} errores)")
                    break
                if await _sleep(POLL_NEAR_SECS):
                    break
                continue

            if not data:
                if await _sleep(POLL_NEAR_SECS):
                    break
                continue

            # 2-half format → discard (not Q1/Q2/Q3/Q4)
            if _is_two_half(data):
                _update_row(
                    conn, match_id,
                    status="discarded", skip_reason="dos_mitades",
                )
                MONITOR_STATUS["discarded"] = (
                    MONITOR_STATUS.get("discarded", 0) + 1
                )
                _log(f"{home} vs {away}: formato 2 mitades, descartado")
                break

            status_type = str(
                (data.get("match", {}) or {}).get("status_type", "") or ""
            ).lower()
            minute = _get_minute(data)

            # Calibrate secs per game-minute from observed changes
            if minute is not None:
                now_wall = time.monotonic()
                if last_gmin is not None and minute > last_gmin:
                    rate = (now_wall - last_gmin_wall) / (minute - last_gmin)
                    secs_per_gmin = 0.7 * secs_per_gmin + 0.3 * rate
                    secs_per_gmin = max(60.0, min(360.0, secs_per_gmin))
                last_gmin = minute
                last_gmin_wall = now_wall

            # Match finished — do final checks and exit
            if status_type == "finished":
                if not q3_done:
                    gp3 = _count_gp_up_to(data, Q3_MINUTE)
                    if _has_scores(data, "Q1", "Q2") and gp3 >= MIN_GP_Q3:
                        sig, notified, _ = await _check_quarter(
                            match_id, data, "q3", db_path, conn,
                            home, away, league, event_date, minute,
                        )
                    else:
                        sig, notified = "no_data", False
                    _update_row(
                        conn, match_id,
                        q3_checked=1, q3_signal=sig, q3_notified=int(notified),
                        q3_model=_model_config.get("q3", "-"),
                    )
                    q3_done = True

                if not q4_done:
                    q4_cut, q4_mgp, q4_need_q3, _q4wb = _q4_timing()
                    gp4 = _count_gp_up_to(data, q4_cut)
                    score_ok = _has_scores(data, "Q1", "Q2", "Q3") if q4_need_q3 else _has_scores(data, "Q1", "Q2")
                    if score_ok and gp4 >= q4_mgp:
                        sig, notified, _ = await _check_quarter(
                            match_id, data, "q4", db_path, conn,
                            home, away, league, event_date, minute,
                        )
                    else:
                        sig, notified = "no_data", False
                    _update_row(
                        conn, match_id,
                        q4_checked=1, q4_signal=sig, q4_notified=int(notified),
                        q4_model=_model_config.get("q4", "-"),
                    )
                    q4_done = True

                _update_row(conn, match_id, status="done")

                # Save final data immediately (match is already finished)
                gp_total = len(data.get("graph_points") or [])
                if gp_total >= FINAL_FETCH_MIN_GP:
                    try:
                        db_conn = __import__("db").get_conn(db_path)
                        __import__("db").init_db(db_conn)
                        __import__("db").save_match(db_conn, match_id, data)
                        db_conn.close()
                        _update_row(
                            conn, match_id,
                            final_fetched=1,
                            final_fetch_at=datetime.utcnow().isoformat(timespec="seconds"),
                        )
                        _log(f"{home} vs {away}: resultado final guardado al detectar fin (gp={gp_total})")
                    except Exception as _exc:
                        logger.warning("[MONITOR] final_fetch save (finished branch) %s: %s", match_id, _exc)
                break

            # ── Q3 logic ─────────────────────────────────────────────────────
            if not q3_done:
                if minute is None:
                    # No graph data yet
                    elapsed_real = time.time() - scheduled_ts if scheduled_ts > 0 else 0
                    if elapsed_real > NO_GRAPH_REAL_SECS:
                        _update_row(
                            conn, match_id,
                            q3_checked=1, q3_signal="no_graph",
                            q4_checked=1, q4_signal="no_graph",
                            status="discarded", skip_reason="no_graph_by_ht",
                        )
                        MONITOR_STATUS["discarded"] = (
                            MONITOR_STATUS.get("discarded", 0) + 1
                        )
                        _log(f"{home} vs {away}: sin gráfica tras {elapsed_real/60:.0f}min reales, descartado")
                        break
                    if await _sleep(POLL_NEAR_SECS):
                        break
                    continue

                gp3 = _count_gp_up_to(data, Q3_MINUTE)

                if minute > Q3_MINUTE + 6:
                    # Q3 window has passed
                    _update_row(conn, match_id, q3_checked=1, q3_signal="window_missed")
                    q3_done = True
                    _log(f"{home} vs {away}: ventana Q3 pasada (min {minute})")

                elif minute >= Q3_MINUTE - WAKE_BEFORE_MINUTES:
                    # In Q3 window — check if data is ready
                    if _has_scores(data, "Q1", "Q2") and gp3 >= MIN_GP_Q3:
                        _is_last_q3 = q3_no_bet_ticks >= NO_BET_CONFIRM_TICKS
                        sig, notified, _ = await _check_quarter(
                            match_id, data, "q3", db_path, conn,
                            home, away, league, event_date, minute,
                            suppress_no_bet_notify=not _is_last_q3,
                        )
                        if notified or _is_last_q3 or sig in ("UNAVAILABLE", "ERROR"):
                            _update_row(
                                conn, match_id,
                                q3_checked=1, q3_signal=sig, q3_notified=int(notified),
                                q3_model=_model_config.get("q3", "-"),
                            )
                            q3_done = True
                        else:
                            q3_no_bet_ticks += 1
                            _log(
                                f"{home} vs {away}: Q3 NO BET incierto "
                                f"(tick {q3_no_bet_ticks}/{NO_BET_CONFIRM_TICKS}), "
                                f"re-evaluando en {POLL_NEAR_SECS}s"
                            )
                            if await _sleep(POLL_NEAR_SECS):
                                break
                            continue
                    else:
                        _log(
                            f"{home} vs {away}: Q3 ventana abierta pero datos insuficientes "
                            f"(min={minute} gp={gp3} Q1Q2={_has_scores(data,'Q1','Q2')})"
                        )
                        if await _sleep(POLL_NEAR_SECS):
                            break
                        continue

                else:
                    # Still before Q3 window — smart sleep
                    mins_to_wake = (Q3_MINUTE - WAKE_BEFORE_MINUTES) - minute
                    sleep_secs = max(30.0, min(mins_to_wake * secs_per_gmin, IDLE_POLL_SECS))
                    _log(
                        f"{home} vs {away}: Q3 en ~{mins_to_wake:.0f} game-min, "
                        f"durmiendo {sleep_secs:.0f}s"
                    )
                    if await _sleep(sleep_secs):
                        break
                    continue

            # ── Q4 logic ─────────────────────────────────────────────────────
            if not q4_done:
                if minute is None:
                    if await _sleep(POLL_NEAR_SECS):
                        break
                    continue

                q4_cut, q4_mgp, q4_need_q3, q4_wake_before = _q4_timing()
                gp4 = _count_gp_up_to(data, q4_cut)

                if minute > q4_cut + 6:
                    _update_row(conn, match_id, q4_checked=1, q4_signal="window_missed")
                    q4_done = True
                    _log(f"{home} vs {away}: ventana Q4 pasada (min {minute}, cutoff={q4_cut})")

                elif minute >= q4_cut - q4_wake_before:
                    score_ok = (_has_scores(data, "Q1", "Q2", "Q3") if q4_need_q3
                                else _has_scores(data, "Q1", "Q2"))
                    if score_ok and gp4 >= q4_mgp:
                        _is_last_q4 = q4_no_bet_ticks >= NO_BET_CONFIRM_TICKS
                        sig, notified, _ = await _check_quarter(
                            match_id, data, "q4", db_path, conn,
                            home, away, league, event_date, minute,
                            suppress_no_bet_notify=not _is_last_q4,
                        )
                        if notified or _is_last_q4 or sig in ("UNAVAILABLE", "ERROR"):
                            _update_row(
                                conn, match_id,
                                q4_checked=1, q4_signal=sig, q4_notified=int(notified),
                                q4_model=_model_config.get("q4", "-"),
                            )
                            q4_done = True
                        else:
                            q4_no_bet_ticks += 1
                            _log(
                                f"{home} vs {away}: Q4 NO BET incierto "
                                f"(tick {q4_no_bet_ticks}/{NO_BET_CONFIRM_TICKS}), "
                                f"re-evaluando en {POLL_NEAR_SECS}s"
                            )
                            if await _sleep(POLL_NEAR_SECS):
                                break
                            continue
                    else:
                        need_q3_str = f" Q3={_has_scores(data,'Q3')}" if q4_need_q3 else ""
                        _log(
                            f"{home} vs {away}: Q4 ventana abierta pero datos insuficientes "
                            f"(min={minute} gp={gp4}/{q4_mgp}{need_q3_str})"
                        )
                        if await _sleep(POLL_NEAR_SECS):
                            break
                        continue

                else:
                    mins_to_wake = (q4_cut - q4_wake_before) - minute
                    sleep_secs = max(30.0, min(mins_to_wake * secs_per_gmin, IDLE_POLL_SECS))
                    _log(
                        f"{home} vs {away}: Q4 en ~{mins_to_wake:.0f} game-min "
                        f"(cutoff={q4_cut}), durmiendo {sleep_secs:.0f}s"
                    )
                    if await _sleep(sleep_secs):
                        break
                    continue

    except asyncio.CancelledError:
        _log(f"{home} vs {away}: watcher cancelado")
    except Exception as exc:
        logger.error(
            "[MONITOR] watcher exception %s: %s", match_id, exc, exc_info=True
        )
    finally:
        conn.close()
        MONITOR_STATUS["active_matches"] = [
            m for m in MONITOR_STATUS.get("active_matches", []) if m != match_id
        ]
        _log(f"Fin watcher: {home} vs {away}")


# ── Main monitor loop ─────────────────────────────────────────────────────────

async def run_monitor(db_path: str, stop_event: asyncio.Event) -> None:
    """Main monitor coroutine.  Run as asyncio.create_task(run_monitor(...))."""
    MONITOR_STATUS.update({
        "running": True,
        "started_at": datetime.utcnow().isoformat(timespec="seconds"),
        "stop_requested": False,
        "checked_q3": 0,
        "checked_q4": 0,
        "bets_sent": 0,
        "no_bet": 0,
        "discarded": 0,
        "active_matches": [],
    })

    init_tables(db_path)
    reconcile_pending_results(db_path)
    _log("Monitor iniciado")

    active_tasks: dict[str, asyncio.Task] = {}
    last_refresh_wall: float = 0.0
    last_date: str = ""

    # ── Resume: launch watchers for rows already in DB before fetching schedule
    _resume_conn = _open_db(db_path)
    _today_str = datetime.now().date().isoformat()
    _tomorrow_str = (datetime.now().date() + timedelta(days=1)).isoformat()
    _existing = _get_pending_rows(_resume_conn, _today_str) + _get_pending_rows(_resume_conn, _tomorrow_str)
    _resume_conn.close()
    if _existing:
        _log(f"Retomando {len(_existing)} partido(s) pendientes de la base")
        for _row in _existing:
            _mid = _row["match_id"]
            _sched_ts = int(_row.get("scheduled_utc_ts") or 0)
            if _sched_ts > 0 and (_sched_ts - time.time()) > 20 * 3600:
                continue
            _task = asyncio.create_task(
                _watch_match(_mid, _row, db_path, stop_event),
                name=f"watch_{_mid}",
            )
            active_tasks[_mid] = _task
            _log(f"Retomado: {_row.get('home_team')} vs {_row.get('away_team')} ({_mid})")

    try:
        while not stop_event.is_set():
            today_str = datetime.now().date().isoformat()
            tomorrow_str = (datetime.now().date() + timedelta(days=1)).isoformat()
            now_wall = time.monotonic()

            # Refresh schedule if stale or new day
            if (
                today_str != last_date
                or now_wall - last_refresh_wall > SCHEDULE_REFRESH_HOURS * 3600
            ):
                for d in (today_str, tomorrow_str):
                    try:
                        rows = await asyncio.to_thread(
                            _fetch_all_events_for_date_sync, d
                        )
                        conn = _open_db(db_path)
                        for row in rows:
                            rdata = {k: v for k, v in row.items() if k != "status_type"}
                            _upsert_schedule_row(conn, rdata)
                        conn.commit()
                        conn.close()
                        count = len(rows)
                        if d == today_str:
                            MONITOR_STATUS["today_total"] = count
                        else:
                            MONITOR_STATUS["tomorrow_total"] = count
                        _log(f"Itinerario {d}: {count} partidos")
                    except Exception as exc:
                        logger.error("[MONITOR] schedule error %s: %s", d, exc)

                last_refresh_wall = now_wall
                last_date = today_str

            # Launch watcher tasks for pending matches
            conn = _open_db(db_path)
            pending_today = _get_pending_rows(conn, today_str)
            pending_tomorrow = _get_pending_rows(conn, tomorrow_str)
            conn.close()

            for row in pending_today + pending_tomorrow:
                mid = row["match_id"]

                # Already being watched and still running
                existing = active_tasks.get(mid)
                if existing and not existing.done():
                    continue
                # Clean up finished task slot
                if existing and existing.done():
                    del active_tasks[mid]

                # Don't start tomorrow's matches more than 20h from now
                sched_ts = int(row.get("scheduled_utc_ts") or 0)
                if sched_ts > 0 and (sched_ts - time.time()) > 20 * 3600:
                    continue

                task = asyncio.create_task(
                    _watch_match(mid, row, db_path, stop_event),
                    name=f"watch_{mid}",
                )
                active_tasks[mid] = task
                _log(
                    f"Tarea lanzada: {row.get('home_team')} vs "
                    f"{row.get('away_team')} ({mid})"
                )

            # Clean up done tasks
            for k in [k for k, t in active_tasks.items() if t.done()]:
                del active_tasks[k]

            await asyncio.sleep(60)

    except asyncio.CancelledError:
        _log("Monitor cancelado")
    except Exception as exc:
        logger.error("[MONITOR] main loop exception: %s", exc, exc_info=True)
    finally:
        for task in active_tasks.values():
            if not task.done():
                task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks.values(), return_exceptions=True)
        MONITOR_STATUS["running"] = False
        MONITOR_STATUS["active_matches"] = []
        _log("Monitor detenido")


# ── Status / display helpers (called by telegram_bot.py) ─────────────────────

def status_text() -> str:
    """One-shot status string for the Telegram bot menu."""
    running = MONITOR_STATUS.get("running", False)
    estado = "🟢 ACTIVO" if running else "🔴 DETENIDO"
    started = MONITOR_STATUS.get("started_at") or "-"
    today = MONITOR_STATUS.get("today_total", 0)
    tmrw = MONITOR_STATUS.get("tomorrow_total", 0)
    q3 = MONITOR_STATUS.get("checked_q3", 0)
    q4 = MONITOR_STATUS.get("checked_q4", 0)
    bets = MONITOR_STATUS.get("bets_sent", 0)
    no_bet = MONITOR_STATUS.get("no_bet", 0)
    discarded = MONITOR_STATUS.get("discarded", 0)
    active = MONITOR_STATUS.get("active_matches", [])
    last = MONITOR_STATUS.get("last_event") or "-"

    lines = [
        f"Monitor Apuestas: {estado}",
        f"Inicio UTC: {started}",
        f"Hoy: {today} partidos  |  Manana: {tmrw}",
        f"Chequeados: Q3={q3}  Q4={q4}",
        f"Apuestas: {bets}  Sin apuesta: {no_bet}  Descartados: {discarded}",
    ]
    if active:
        lines.append(f"Vigilando ahora: {len(active)} partido(s)")
    lines.append(f"Ultima actividad: {last}")
    return "\n".join(lines)


def schedule_text(db_path: str, local_date: str) -> str:
    """Return a formatted schedule for local_date, capped at ~3800 chars.

    Rows with a BET signal are shown first; the rest are shown after,
    trimmed if the total would exceed Telegram's 4096-char message limit.
    """
    conn = _open_db(db_path)
    rows = conn.execute(
        """
        SELECT * FROM bet_monitor_schedule
        WHERE event_date = ?
        ORDER BY scheduled_utc_ts ASC
        """,
        (local_date,),
    ).fetchall()
    conn.close()

    if not rows:
        return f"Sin itinerario guardado para {local_date}."

    def _fmt_row(row: sqlite3.Row) -> str:
        ts = int(row["scheduled_utc_ts"] or 0)
        if ts:
            dt_l = (
                datetime.fromtimestamp(ts, tz=timezone.utc)
                + timedelta(hours=UTC_OFFSET_HOURS)
            )
            hora = dt_l.strftime("%H:%M")
        else:
            hora = "--:--"
        status_short = {"pending": "⏳", "done": "✔", "discarded": "✗"}.get(
            str(row["status"] or "pending"), "?"
        )
        q3s = str(row["q3_signal"] or "-")[:6]
        q4s = str(row["q4_signal"] or "-")[:6]
        n3 = "✉" if int(row["q3_notified"] or 0) else ""
        n4 = "✉" if int(row["q4_notified"] or 0) else ""
        home_s = str(row["home_team"] or "?")[:11]
        away_s = str(row["away_team"] or "?")[:11]
        return (
            f"{hora} {home_s} vs {away_s} {status_short} "
            f"Q3:{q3s}{n3} Q4:{q4s}{n4}"
        )

    # Rows with a BET come first so they are never truncated
    bet_rows = [
        r for r in rows
        if _is_bet_signal(str(r["q3_signal"] or "")) or _is_bet_signal(str(r["q4_signal"] or ""))
    ]
    other_rows = [
        r for r in rows
        if r not in bet_rows
    ]

    header = f"Itinerario {local_date} ({len(rows)} partidos):"
    parts = [header]
    if bet_rows:
        parts.append("— Apuestas —")
        for r in bet_rows:
            parts.append(_fmt_row(r))
        parts.append("— Resto —")

    LIMIT = 3800
    used = sum(len(p) + 1 for p in parts)
    omitted = 0
    for r in other_rows:
        line = _fmt_row(r)
        if used + len(line) + 1 > LIMIT:
            omitted += 1
        else:
            parts.append(line)
            used += len(line) + 1

    if omitted:
        parts.append(f"... y {omitted} partido(s) mas")

    return "\n".join(parts)


def log_text(db_path: str, limit: int = 20) -> str:
    """Return recent bet_monitor_log entries, capped at Telegram's limit."""
    conn = _open_db(db_path)
    rows = conn.execute(
        "SELECT * FROM bet_monitor_log ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    if not rows:
        return "Sin registros en la bitacora del monitor."

    lines = [f"Bitacora monitor (ultimos {len(rows)}):"]
    for row in rows:
        ts = str(row["created_at"] or "-")[:16]
        tgt = str(row["target"] or "?").upper()
        sig = str(row["signal"] or "-")
        pick = str(row["pick"] or "-")
        conf = row["confidence"]
        conf_txt = f" {float(conf) * 100:.0f}%" if conf else ""
        result = str(row["result"] or "pending")
        res_sym = {"win": "✅", "loss": "❌", "push": "➖", "pending": "⏳"}.get(result, "")
        home_s = str(row["home_team"] or "?")[:10]
        away_s = str(row["away_team"] or "?")[:10]
        lines.append(
            f"{ts} | {tgt} {sig}{conf_txt} {res_sym} | {home_s} vs {away_s} → {pick}"
        )
    text = "\n".join(lines)
    if len(text) > 3800:
        text = text[:3750] + "\n... (truncado)"
    return text


def schedule_keyboard(db_path: str, local_date: str):
    """Return (header_text, InlineKeyboardMarkup) — one button per match."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    conn = _open_db(db_path)
    rows = conn.execute(
        "SELECT * FROM bet_monitor_schedule WHERE event_date = ?"
        " ORDER BY scheduled_utc_ts ASC",
        (local_date,),
    ).fetchall()
    conn.close()

    nav = [
        [InlineKeyboardButton("⬅️ Monitor", callback_data="menu:monitor")],
        [InlineKeyboardButton("Menu principal", callback_data="nav:main")],
    ]

    if not rows:
        return (f"Sin itinerario para {local_date}.", InlineKeyboardMarkup(nav))

    def _sig_emoji(row) -> str:
        q3 = str(row["q3_signal"] or "")
        q4 = str(row["q4_signal"] or "")
        if _is_bet_signal(q3) or _is_bet_signal(q4):
            return "🟢"
        if q3 in ("NO_BET", "NO BET") or q4 in ("NO_BET", "NO BET"):
            return "🔴"
        if str(row["status"] or "") in ("done", "discarded"):
            return "✗"
        return "⏳"

    bet_rows = [r for r in rows if _is_bet_signal(str(r["q3_signal"] or "")) or _is_bet_signal(str(r["q4_signal"] or ""))]
    other_rows = [r for r in rows if r not in bet_rows]

    buttons = []
    for r in (bet_rows + other_rows)[:80]:
        emoji = _sig_emoji(r)
        ts = int(r["scheduled_utc_ts"] or 0)
        hora = (
            (datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=UTC_OFFSET_HOURS)).strftime("%H:%M")
            if ts else "--:--"
        )
        home_s = str(r["home_team"] or "?")[:12]
        away_s = str(r["away_team"] or "?")[:12]
        q3s = str(r["q3_signal"] or "-")[:6]
        q4s = str(r["q4_signal"] or "-")[:6]
        q3m = str(r["q3_model"] or "")
        q4m = str(r["q4_model"] or "")
        model_txt = f"[{q3m}/{q4m}]" if (q3m or q4m) else ""
        label = f"{emoji} {hora} {home_s} vs {away_s} Q3:{q3s} Q4:{q4s} {model_txt}"[:64]
        cd = f"match:{r['match_id']}:{r['event_date']}:0"
        buttons.append([InlineKeyboardButton(label, callback_data=cd)])

    buttons.extend(nav)
    header = f"Itinerario {local_date} — {len(rows)} partidos ({len(bet_rows)} apuestas):"
    return header, InlineKeyboardMarkup(buttons)


def log_keyboard(db_path: str, limit: int = 20):
    """Return (header_text, InlineKeyboardMarkup) — one button per log row."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    conn = _open_db(db_path)
    rows = conn.execute(
        "SELECT * FROM bet_monitor_log ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    nav = [
        [InlineKeyboardButton("⬅️ Monitor", callback_data="menu:monitor")],
        [InlineKeyboardButton("Menu principal", callback_data="nav:main")],
    ]

    if not rows:
        return ("Sin registros en la bitacora.", InlineKeyboardMarkup(nav))

    buttons = []
    for r in rows:
        result = str(r["result"] or "pending")
        res_emoji = {"win": "✅", "loss": "❌", "push": "➖", "pending": "⏳"}.get(result, "⏳")
        sig = str(r["signal"] or "-")
        sig_emoji = "🟢" if _is_bet_signal(sig) else ("🔴" if "NO" in sig.upper() else "⚪")
        tgt = str(r["target"] or "?").upper()
        model = str(r["model"] or "-")
        conf = r["confidence"]
        conf_txt = f" {float(conf) * 100:.0f}%" if conf else ""
        home_s = str(r["home_team"] or "?")[:10]
        away_s = str(r["away_team"] or "?")[:10]
        label = f"{res_emoji}{sig_emoji} {tgt}{conf_txt} [{model}] {home_s} vs {away_s}"[:64]
        ev_date = str(r["event_date"] or "_")
        cd = f"match:{r['match_id']}:{ev_date}:0"
        buttons.append([InlineKeyboardButton(label, callback_data=cd)])

    buttons.extend(nav)
    header = f"Bitácora monitor (últimos {len(rows)} registros):"
    return header, InlineKeyboardMarkup(buttons)


def signals_text_today(
    db_path: str,
    local_date: str,
    pref: str = "all",
) -> str:
    """Return a formatted list of today's signals (BET and/or NO_BET).

    pref='bet_only' hides NO_BET-only matches and NO_BET lines inside
    matches that have at least one BET signal.
    pref='all' (default) shows everything.
    """
    conn = _open_db(db_path)

    # Latest log entry per (match_id, target) for today, ordered by schedule time
    rows = conn.execute(
        """
        SELECT l.match_id, l.target, l.signal, l.pick, l.confidence,
               l.recommendation, l.result, l.created_at,
               l.home_team, l.away_team, l.league, l.model,
               s.scheduled_utc_ts
        FROM bet_monitor_log l
        INNER JOIN (
            SELECT match_id, target, MAX(id) AS max_id
            FROM bet_monitor_log
            WHERE event_date = ?
            GROUP BY match_id, target
        ) latest ON l.match_id = latest.match_id
                 AND l.target  = latest.target
                 AND l.id      = latest.max_id
        LEFT JOIN bet_monitor_schedule s ON l.match_id = s.match_id
        ORDER BY COALESCE(s.scheduled_utc_ts, 0) ASC, l.created_at ASC
        """,
        (local_date,),
    ).fetchall()

    # Scheduled matches for today (to show pending ones)
    sched_rows = conn.execute(
        """
        SELECT match_id, home_team, away_team, league,
               scheduled_utc_ts, status
        FROM bet_monitor_schedule
        WHERE event_date = ?
        ORDER BY scheduled_utc_ts ASC
        """,
        (local_date,),
    ).fetchall()

    # Quarter scores stored after match finishes
    _qs_rows = conn.execute(
        """
        SELECT qs.match_id, qs.quarter, qs.home, qs.away
        FROM quarter_scores qs
        INNER JOIN bet_monitor_schedule s ON qs.match_id = s.match_id
        WHERE s.event_date = ?
        """,
        (local_date,),
    ).fetchall()
    conn.close()

    # Build: quarter_actual[(match_id, "Q3")] = "home" | "away" | "push"
    quarter_actual: dict[tuple, str] = {}
    for qr in _qs_rows:
        h = qr["home"]
        a = qr["away"]
        if h is None or a is None:
            continue
        try:
            h, a = int(h), int(a)
        except (TypeError, ValueError):
            continue
        if h > a:
            winner = "home"
        elif a > h:
            winner = "away"
        else:
            winner = "push"
        quarter_actual[(str(qr["match_id"]), str(qr["quarter"]))] = winner

    # Spanish date header
    _DAY_ES = [
        "Lunes", "Martes", "Miércoles", "Jueves",
        "Viernes", "Sábado", "Domingo",
    ]
    _MON_ES = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ]
    try:
        _d = datetime.strptime(local_date, "%Y-%m-%d")
        _date_hdr = (
            f"{_DAY_ES[_d.weekday()]} {_d.day}"
            f" {_MON_ES[_d.month - 1]} {_d.year}"
        )
    except ValueError:
        _date_hdr = local_date

    def _hora(ts) -> str:
        ts_int = int(ts or 0)
        if not ts_int:
            return "--:--"
        return (
            datetime.fromtimestamp(ts_int, tz=timezone.utc)
            + timedelta(hours=UTC_OFFSET_HOURS)
        ).strftime("%H:%M")

    def _pick_sym(pick: str) -> str:
        if pick == "home":
            return "🏠"
        if pick == "away":
            return "✈️"
        return pick

    def _sig_line(
        tgt: str,
        s: dict,
        show_no_bet: bool = True,
        mid: str = "",
    ) -> str | None:
        sig    = str(s.get("signal") or "-").upper()
        pick   = str(s.get("pick") or "")
        conf   = s.get("confidence") or 0.0
        result = str(s.get("result") or "pending")
        model  = str(s.get("model") or "")
        conf_txt  = f" {float(conf) * 100:.0f}%" if conf else ""
        res_sym   = {"win": "✅", "loss": "❌", "push": "➖", "pending": "⏳"}.get(result, "⏳")
        model_txt = f" [{model}]" if model else ""
        pick_sym  = _pick_sym(pick)
        if _is_bet_signal(sig):
            return (
                f"  🟢 {tgt.upper()}{model_txt}"
                f" → {pick_sym}{conf_txt} {res_sym}"
            ).rstrip()
        # NO_BET line
        if not show_no_bet:
            return None
        tendency = f" {pick_sym}{conf_txt}" if pick_sym else ""
        # Real quarter result (if available)
        _q_key = tgt.upper()   # "Q3" or "Q4"
        actual = quarter_actual.get((mid, _q_key))
        if actual == "home":
            real_txt = " (🏠)"
        elif actual == "away":
            real_txt = " (✈️)"
        elif actual == "push":
            real_txt = " (=)"
        else:
            real_txt = ""
        return f"  🔴 {tgt.upper()}{model_txt}{tendency}{real_txt}"

    # Group signals by match_id, preserving insertion order
    match_signals: dict[str, dict] = {}
    match_meta: dict[str, dict] = {}
    for row in rows:
        mid = str(row["match_id"])
        tgt = str(row["target"] or "").lower()
        if mid not in match_signals:
            match_signals[mid] = {}
            match_meta[mid] = {
                "home":   str(row["home_team"] or "?"),
                "away":   str(row["away_team"] or "?"),
                "league": str(row["league"] or "?"),
                "ts":     row["scheduled_utc_ts"],
            }
        match_signals[mid][tgt] = dict(row)

    # Scheduled matches that have no signals at all yet
    signaled_ids = set(match_signals.keys())
    pending_sched = [
        dict(r) for r in sched_rows
        if r["match_id"] not in signaled_ids
    ]

    show_no_bet = pref == "all"

    bet_mids = [
        m for m in match_signals
        if any(
            _is_bet_signal(str(match_signals[m].get(t, {}).get("signal", "")))
            for t in ("q3", "q4")
        )
    ]
    nobet_mids = [m for m in match_signals if m not in bet_mids]

    def _render_match(mid: str, force_show_no_bet: bool) -> list[str]:
        meta = match_meta[mid]
        hora = _hora(meta["ts"])
        lines_m = [f"{hora}  {meta['home']} vs {meta['away']} ({meta['league']})"]
        for tgt in ("q3", "q4"):
            if tgt in match_signals[mid]:
                line = _sig_line(
                    tgt,
                    match_signals[mid][tgt],
                    show_no_bet=force_show_no_bet,
                    mid=mid,
                )
                if line is not None:
                    lines_m.append(line)
        # Skip match block if no lines were added
        if len(lines_m) == 1:
            return []
        return lines_m

    lines: list[str] = [f"📊 Señales {_date_hdr}:"]

    if bet_mids:
        lines.append("— 🟢 Apuestas —")
        for mid in bet_mids:
            # In bet_only mode, suppress NO_BET lines within the match
            lines.extend(_render_match(mid, force_show_no_bet=show_no_bet))

    if show_no_bet and nobet_mids:
        lines.append("— 🔴 Sin apuesta —")
        for mid in nobet_mids:
            lines.extend(_render_match(mid, force_show_no_bet=True))

    if pending_sched:
        lines.append(f"— ⏳ Sin evaluar ({len(pending_sched)}) —")
        for r in pending_sched[:20]:
            hora = _hora(r.get("scheduled_utc_ts"))
            home_s = str(r.get("home_team") or "?")[:14]
            away_s = str(r.get("away_team") or "?")[:14]
            status = str(r.get("status") or "pending")
            disc = " [descartado]" if status == "discarded" else ""
            lines.append(f"  {hora} {home_s} vs {away_s}{disc}")
        if len(pending_sched) > 20:
            lines.append(f"  ... y {len(pending_sched) - 20} más")

    if not match_signals and not pending_sched:
        lines.append("Sin datos para hoy. ¿El monitor está corriendo?")

    # ── Stats + simulated bank ────────────────────────────────────────────
    # Read sim config from settings
    cfg_conn = _open_db(db_path)
    try:
        _bank0  = float(cfg_conn.execute(
            "SELECT value FROM settings WHERE key='sig_bank'"
        ).fetchone()["value"] or 1000)
    except Exception:
        _bank0 = 1000.0
    try:
        _bet_sz = float(cfg_conn.execute(
            "SELECT value FROM settings WHERE key='sig_bet_size'"
        ).fetchone()["value"] or 100)
    except Exception:
        _bet_sz = 100.0
    try:
        _odds   = float(cfg_conn.execute(
            "SELECT value FROM settings WHERE key='sig_odds'"
        ).fetchone()["value"] or 1.4)
    except Exception:
        _odds = 1.4
    cfg_conn.close()

    def _calc_stats(mids: list) -> dict:
        """Count wins/losses/push/pending for BET signals in mids."""
        w = l = p = pending = 0
        for mid in mids:
            for tgt in ("q3", "q4"):
                s = match_signals[mid].get(tgt)
                if not s:
                    continue
                if not _is_bet_signal(str(s.get("signal") or "")):
                    continue
                r = str(s.get("result") or "pending")
                if r == "win":
                    w += 1
                elif r in ("loss", "push"):
                    l += 1
                elif r == "pending":
                    pending += 1
        return {"w": w, "l": l, "pending": pending}

    def _calc_nobet_stats(mids: list) -> dict:
        """For NO_BET signals: what would outcome have been?"""
        w = l = p = pending = 0
        for mid in mids:
            for tgt in ("q3", "q4"):
                s = match_signals[mid].get(tgt)
                if not s:
                    continue
                if _is_bet_signal(str(s.get("signal") or "")):
                    continue
                pick = str(s.get("pick") or "")
                actual = quarter_actual.get((mid, tgt.upper()))
                if actual is None:
                    pending += 1
                    continue
                if actual == "push":
                    l += 1
                elif actual == pick:
                    w += 1
                else:
                    l += 1
        return {"w": w, "l": l, "pending": pending}

    def _roi_line(st: dict, label: str, bank: float, bet: float, odds: float) -> str:
        played = st["w"] + st["l"]
        if played == 0 and st["pending"] == 0:
            return ""
        profit = st["w"] * bet * (odds - 1) - st["l"] * bet
        bank_end = bank + profit
        roi = (profit / (played * bet) * 100) if played > 0 else 0.0
        hit = (st["w"] / played * 100) if played > 0 else 0.0
        pend_txt = f"+{st['pending']}⏳\n" if st["pending"] else ""
        sign = "+" if profit >= 0 else ""
        return (
            f"{label}\n"
            f"{st['w']}✅{hit:.0f}%\n"
            f"{st['l']}❌\n"
            f"{pend_txt}"
            f"ROI {sign}{roi:.1f}%\n"
            f"Bank ${bank:.0f}→${bank_end:.0f}"
        )

    bet_st    = _calc_stats(bet_mids)
    nobet_st  = _calc_nobet_stats(nobet_mids) if show_no_bet else None
    # Also compute nobet lines inside bet matches
    nobet_in_bet = _calc_nobet_stats(bet_mids) if show_no_bet else None

    stat_lines = []
    bet_line = _roi_line(bet_st, "💰 Apostado", _bank0, _bet_sz, _odds)
    if bet_line:
        stat_lines.append(bet_line)
    if show_no_bet:
        # Merge nobet_mids + nobet quarters inside bet matches
        nb_all = {
            "w": (nobet_st["w"] if nobet_st else 0)
                 + (nobet_in_bet["w"] if nobet_in_bet else 0),
            "l": (nobet_st["l"] if nobet_st else 0)
                 + (nobet_in_bet["l"] if nobet_in_bet else 0),
            "pending": (nobet_st["pending"] if nobet_st else 0)
                       + (nobet_in_bet["pending"] if nobet_in_bet else 0),
        }
        nb_line = _roi_line(
            nb_all, "📉 No apostado",
            _bank0, _bet_sz, _odds,
        )
        if nb_line:
            stat_lines.append(nb_line)
    if stat_lines:
        lines.append("")
        cfg_txt = (
            f"(odds {_odds} · apuesta ${int(_bet_sz)}"
            f" · Bank ${int(_bank0)})"
        )
        lines.append(f"── Resumen del día {cfg_txt} ──")
        lines.extend(stat_lines)

    text = "\n".join(lines)
    if len(text) > 3800:
        text = text[:3750] + "\n... (truncado)"
    return text

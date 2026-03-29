import asyncio
import csv
from datetime import datetime, timedelta
import html
import importlib
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, MenuButtonCommands, ReplyKeyboardMarkup, Update
from telegram.error import BadRequest, NetworkError, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import db as db_mod
import ml_tools as ml_mod
import scraper as scraper_mod


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DB_PATH = os.getenv("MATCH_DB_PATH", str(BASE_DIR / "matches.db")).strip()
ALLOWED_CHAT_IDS_RAW = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DATE_PAGE_SIZE = 8
MATCH_PAGE_SIZE = 8
LIVE_PAGE_SIZE = 8
AWAITING_MATCH_ID_KEY = "awaiting_match_id"
AWAITING_FETCH_DATE_KEY = "awaiting_fetch_date"
LIVE_RESULT_KEY = "live_result"

RETRAIN_LOCK = asyncio.Lock()
TRAIN_STATUS: dict[str, object] = {
    "running": False,
    "owner_chat_id": None,
    "last_exit_code": None,
    "last_finished_utc": None,
    "last_error_tail": "",
    "current_task": "",
    "progress_done": 0,
    "progress_total": 0,
}
MENU_BUTTON_TEXT = "Menu"
UTC_OFFSET_HOURS = -6
DATE_INGEST_PROGRESS_EVERY = 10
DATE_INGEST_STATUS_INTERVAL_SECONDS = 4
REFRESH_DATE_STATUS_INTERVAL_SECONDS = 4
DATE_INGEST_JOBS: dict[int, dict[str, object]] = {}
REFRESH_JOBS: dict[int, dict[str, object]] = {}
FOLLOW_JOBS: dict[int, dict[str, object]] = {}
MODEL_OUTPUTS_V4_DIR = BASE_DIR / "training" / "model_outputs_v4"
MODEL_OUTPUTS_V2_DIR = BASE_DIR / "training" / "model_outputs_v2"
FOLLOW_REFRESH_SECONDS = 45
FOLLOW_STALE_CYCLES_LIMIT = 8


def _log_raw_inference_result(match_id: str, source: str, infer_result: object) -> None:
    """Log raw inference payload to terminal for full model-output visibility."""
    try:
        payload = json.dumps(infer_result, ensure_ascii=False, default=str)
    except Exception:
        payload = str(infer_result)
    logger.info("[MODEL_RAW] source=%s match_id=%s", source, match_id)
    logger.info("[MODEL_RAW] payload=%s", payload)


def _parse_allowed_chat_ids(raw: str) -> set[int]:
    if not raw:
        return set()
    out: set[int] = set()
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        try:
            out.add(int(value))
        except ValueError:
            continue
    return out


ALLOWED_CHAT_IDS = _parse_allowed_chat_ids(ALLOWED_CHAT_IDS_RAW)


def _open_conn() -> sqlite3.Connection:
    return db_mod.get_conn(DB_PATH)


def _winner_from_scores(home: int | None, away: int | None) -> str:
    if home is None or away is None:
        return "-"
    if home == away:
        return "push"
    return "home" if home > away else "away"


def _winner_short(home: int | None, away: int | None) -> str:
    winner = _winner_from_scores(home, away)
    if winner == "home":
        return "H"
    if winner == "away":
        return "A"
    if winner == "push":
        return "P"
    return "-"


def _outcome_emoji(outcome: str | None) -> str:
    value = (outcome or "").lower()
    if value == "hit":
        return "✅"
    if value == "miss":
        return "❌"
    if value == "push":
        return "➖"
    return ""


def _abbr_team(name: str, max_len: int = 14) -> str:
    cleaned = " ".join((name or "").strip().split())
    if len(cleaned) <= max_len:
        return cleaned

    parts = [p for p in cleaned.split(" ") if p]
    if len(parts) >= 2:
        initials = "".join(p[0].upper() for p in parts[:4])
        if 2 <= len(initials) <= max_len:
            return initials

    return cleaned[: max_len - 1] + "~"


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_local_datetime(date_str: str, time_str: str) -> datetime | None:
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    return dt + timedelta(hours=UTC_OFFSET_HOURS)


def _fetch_date_summaries() -> list[dict]:
    conn = _open_conn()
    rows = conn.execute(
                f"""
        SELECT
                        date(datetime(m.date || ' ' || m.time, '{UTC_OFFSET_HOURS} hours')) AS event_date,
            COUNT(*) AS total_matches,
            SUM(CASE WHEN q3.home IS NOT NULL AND q3.away IS NOT NULL AND q3.home <> q3.away THEN 1 ELSE 0 END) AS q3_won,
            SUM(CASE WHEN q4.home IS NOT NULL AND q4.away IS NOT NULL AND q4.home <> q4.away THEN 1 ELSE 0 END) AS q4_won
        FROM matches m
        LEFT JOIN quarter_scores q3
          ON q3.match_id = m.match_id AND q3.quarter = 'Q3'
        LEFT JOIN quarter_scores q4
          ON q4.match_id = m.match_id AND q4.quarter = 'Q4'
                GROUP BY date(datetime(m.date || ' ' || m.time, '{UTC_OFFSET_HOURS} hours'))
                ORDER BY event_date DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _fetch_dates_pred_stats() -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for row in _fetch_date_summaries():
        event_date = str(row.get("event_date", "") or "")
        if not event_date:
            continue
        pred_map = _fetch_date_pred_outcomes(event_date)
        counts = {
            "q3_hit": 0,
            "q3_miss": 0,
            "q3_push": 0,
            "q4_hit": 0,
            "q4_miss": 0,
            "q4_push": 0,
        }
        for pred in pred_map.values():
            q3o = str(pred.get("q3_outcome") or "").lower()
            q4o = str(pred.get("q4_outcome") or "").lower()
            if q3o == "hit":
                counts["q3_hit"] += 1
            elif q3o == "miss":
                counts["q3_miss"] += 1
            elif q3o == "push":
                counts["q3_push"] += 1
            if q4o == "hit":
                counts["q4_hit"] += 1
            elif q4o == "miss":
                counts["q4_miss"] += 1
            elif q4o == "push":
                counts["q4_push"] += 1
        stats[event_date] = counts
    return stats


def _short_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %d")
    except ValueError:
        return date_str


def _fetch_matches_for_date(event_date: str) -> list[dict]:
    conn = _open_conn()
    rows = conn.execute(
        f"""
        SELECT
            m.match_id,
            m.date,
            m.time,
            strftime('%H:%M', datetime(m.date || ' ' || m.time, '{UTC_OFFSET_HOURS} hours')) AS display_time,
            m.status_type,
            m.home_team,
            m.away_team,
            m.league,
            m.home_score,
            m.away_score,
            q3.home AS q3_home,
            q3.away AS q3_away,
            q4.home AS q4_home,
            q4.away AS q4_away
        FROM matches m
        LEFT JOIN quarter_scores q3
          ON q3.match_id = m.match_id AND q3.quarter = 'Q3'
        LEFT JOIN quarter_scores q4
          ON q4.match_id = m.match_id AND q4.quarter = 'Q4'
                WHERE date(datetime(m.date || ' ' || m.time, '{UTC_OFFSET_HOURS} hours')) = ?
                ORDER BY datetime(m.date || ' ' || m.time, '{UTC_OFFSET_HOURS} hours') DESC, m.match_id DESC
        """,
        (event_date,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _fetch_date_pred_outcomes(event_date: str) -> dict[str, dict]:
    match_rows = _fetch_matches_for_date(event_date)
    match_ids = [str(row.get("match_id", "") or "") for row in match_rows]
    match_ids = [mid for mid in match_ids if mid]
    if not match_ids:
        return {}

    conn = _open_conn()
    tags = _all_result_tags(conn)
    if not tags:
        conn.close()
        return {}
    placeholders = ",".join("?" for _ in match_ids)
    rows = conn.execute(
        f"""
        SELECT *
        FROM eval_match_results
        WHERE match_id IN ({placeholders})
        ORDER BY match_id, updated_at DESC
        """,
        tuple(match_ids),
    ).fetchall()
    conn.close()

    result: dict[str, dict] = {}
    seen: set[str] = set()
    for row in rows:
        mid = str(row["match_id"])
        if mid in seen:
            continue
        best_candidate: dict | None = None
        fallback_candidate: dict | None = None

        for tag in tags:
            q3_av_key = f"q3_available__{tag}"
            if q3_av_key not in row.keys():
                continue
            q3_av = int(row[q3_av_key] or 0)
            q4_av_key = f"q4_available__{tag}"
            q4_av = int(row[q4_av_key] or 0)
            candidate = {
                "q3_available": bool(q3_av),
                "q4_available": bool(q4_av),
                "q3_signal": _row_value(row, f"q3_signal__{tag}"),
                "q4_signal": _row_value(row, f"q4_signal__{tag}"),
                "q3_outcome": _row_value(row, f"q3_outcome__{tag}") if q3_av else None,
                "q4_outcome": _row_value(row, f"q4_outcome__{tag}") if q4_av else None,
                "q3_pick": _row_value(row, f"q3_pick__{tag}") if q3_av else None,
                "q4_pick": _row_value(row, f"q4_pick__{tag}") if q4_av else None,
            }

            if q3_av or q4_av:
                best_candidate = candidate
                break
            if fallback_candidate is None:
                fallback_candidate = candidate

        selected = best_candidate or fallback_candidate
        if selected is not None:
            result[mid] = selected
            seen.add(mid)
    return result


def _pred_stats_text(pred_map: dict, total_matches: int) -> str:
    def _pct(count: int, total: int) -> str:
        if total <= 0:
            return "0.0%"
        return f"{(count * 100.0 / total):.1f}%"

    def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        head = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        sep = "-+-".join("-" * w for w in widths)
        out = [head, sep]
        for row in rows:
            out.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        return out

    def _profit_text(hit: int, miss: int, push: int) -> str:
        stake = 20.0
        odds = 1.91
        # 1X2 simulation: tie/push counts as lost ticket when betting side 1 or 2.
        net = hit * (stake * (odds - 1.0)) - (miss + push) * stake
        sign = "+" if net >= 0 else ""
        return f"{sign}${net:.1f}"

    stats = {
        "q3": {"hit": 0, "miss": 0, "push": 0, "pending": 0, "no_bet": 0},
        "q4": {"hit": 0, "miss": 0, "push": 0, "pending": 0, "no_bet": 0},
    }

    for match_id, pred in pred_map.items():
        _ = match_id
        for quarter in ("q3", "q4"):
            available = bool(pred.get(f"{quarter}_available"))
            if not available:
                stats[quarter]["no_bet"] += 1
                continue

            outcome = str(pred.get(f"{quarter}_outcome") or "pending").lower()
            if outcome == "hit":
                stats[quarter]["hit"] += 1
            elif outcome == "miss":
                stats[quarter]["miss"] += 1
            elif outcome == "push":
                stats[quarter]["push"] += 1
            else:
                stats[quarter]["pending"] += 1

    missing = max(total_matches - len(pred_map), 0)
    stats["q3"]["no_bet"] += missing
    stats["q4"]["no_bet"] += missing

    total_hit = stats["q3"]["hit"] + stats["q4"]["hit"]
    total_miss = stats["q3"]["miss"] + stats["q4"]["miss"]
    total_push = stats["q3"]["push"] + stats["q4"]["push"]
    total_pending = stats["q3"]["pending"] + stats["q4"]["pending"]
    resolved_total = total_hit + total_miss + total_push

    headers = ["Filtro", "W", "L", "P", "NB", "%W", "Ganancia"]
    rows: list[list[str]] = []
    for quarter in ("q3", "q4"):
        s = stats[quarter]
        resolved = max(s["hit"] + s["miss"] + s["push"], 0)
        no_bet_total = s["no_bet"] + s["pending"]
        rows.append(
            [
                quarter.upper(),
                str(s["hit"]),
                str(s["miss"]),
                str(s["push"]),
                str(no_bet_total),
                _pct(s["hit"], resolved),
                _profit_text(s["hit"], s["miss"], s["push"]),
            ]
        )

    rows.append(
        [
            "GLOBAL",
            str(total_hit),
            str(total_miss),
            str(total_push),
            str(total_pending),
            _pct(total_hit, resolved_total),
            _profit_text(total_hit, total_miss, total_push),
        ]
    )

    return "\n".join(_table(headers, rows))


def _event_date_title_es(event_date: str, total_matches: int) -> str:
    try:
        dt = datetime.strptime(event_date, "%Y-%m-%d")
    except ValueError:
        return f"Matches del {event_date} [{total_matches}]"

    weekdays = [
        "Lunes",
        "Martes",
        "Miercoles",
        "Jueves",
        "Viernes",
        "Sabado",
        "Domingo",
    ]
    months = [
        "Enero",
        "Febrero",
        "Marzo",
        "Abril",
        "Mayo",
        "Junio",
        "Julio",
        "Agosto",
        "Septiembre",
        "Octubre",
        "Noviembre",
        "Diciembre",
    ]

    wd = weekdays[dt.weekday()]
    month = months[dt.month - 1]
    return f"Matches de {wd} {dt.day:02d} {month} {dt.year} [{total_matches}]"


def _refresh_recent_dates(
    days: int,
    progress_state: dict[str, object] | None = None,
) -> dict[str, object]:
    summaries = _fetch_date_summaries()
    event_dates = [
        str(row.get("event_date", "") or "")
        for row in summaries
        if str(row.get("event_date", "") or "")
    ][:days]

    if progress_state is not None:
        progress_state.update(
            {
                "phase7d": "running",
                "total_dates": len(event_dates),
                "processed_dates": 0,
                "current_date": "",
                "sum_total": 0,
                "sum_updated_ok": 0,
                "sum_still_incomplete": 0,
                "sum_skipped_complete": 0,
                "sum_failed": 0,
            }
        )

    per_date: list[dict[str, object]] = []
    sum_total = 0
    sum_updated_ok = 0
    sum_still_incomplete = 0
    sum_skipped_complete = 0
    sum_failed = 0

    for index, event_date in enumerate(event_dates, start=1):
        if progress_state is not None:
            progress_state.update(
                {
                    "current_date": event_date,
                    "processed_dates": index - 1,
                }
            )

        date_result = _refresh_incomplete_matches_for_date(
            event_date,
            progress_state,
        )
        per_date.append(date_result)

        sum_total += int(date_result.get("total") or 0)
        sum_updated_ok += int(date_result.get("updated_ok") or 0)
        sum_still_incomplete += int(date_result.get("still_incomplete") or 0)
        sum_skipped_complete += int(date_result.get("skipped_complete") or 0)
        sum_failed += int(date_result.get("failed") or 0)

        if progress_state is not None:
            progress_state.update(
                {
                    "processed_dates": index,
                    "sum_total": sum_total,
                    "sum_updated_ok": sum_updated_ok,
                    "sum_still_incomplete": sum_still_incomplete,
                    "sum_skipped_complete": sum_skipped_complete,
                    "sum_failed": sum_failed,
                }
            )

    if progress_state is not None:
        progress_state.update(
            {
                "phase7d": "done",
                "processed_dates": len(event_dates),
                "current_date": "",
            }
        )

    return {
        "days": days,
        "dates": event_dates,
        "per_date": per_date,
        "sum_total": sum_total,
        "sum_updated_ok": sum_updated_ok,
        "sum_still_incomplete": sum_still_incomplete,
        "sum_skipped_complete": sum_skipped_complete,
        "sum_failed": sum_failed,
    }


def _refresh_7d_progress_text(progress_state: dict[str, object]) -> str:
    def _fmt_duration(seconds: float | int) -> str:
        total = max(int(seconds), 0)
        mm, ss = divmod(total, 60)
        hh, mm = divmod(mm, 60)
        if hh:
            return f"{hh}h {mm:02d}m {ss:02d}s"
        return f"{mm:02d}m {ss:02d}s"

    phase = str(progress_state.get("phase7d") or "starting")
    total_dates = int(progress_state.get("total_dates") or 0)
    processed_dates = int(progress_state.get("processed_dates") or 0)
    current_date = str(progress_state.get("current_date") or "")
    sum_total = int(progress_state.get("sum_total") or 0)
    sum_updated_ok = int(progress_state.get("sum_updated_ok") or 0)
    sum_still_incomplete = int(progress_state.get("sum_still_incomplete") or 0)
    sum_skipped_complete = int(progress_state.get("sum_skipped_complete") or 0)
    sum_failed = int(progress_state.get("sum_failed") or 0)
    current_total_matches = int(progress_state.get("total_matches") or 0)
    current_pending_total = int(progress_state.get("pending_total") or 0)
    current_processed = int(progress_state.get("processed") or 0)
    current_updated_ok = int(progress_state.get("updated_ok") or 0)
    current_still_incomplete = int(progress_state.get("still_incomplete") or 0)
    current_failed = int(progress_state.get("failed") or 0)
    current_skipped_complete = int(progress_state.get("skipped_complete") or 0)

    started_monotonic = progress_state.get("started_monotonic")
    elapsed_txt = None
    eta_txt = None
    if isinstance(started_monotonic, (int, float)):
        elapsed = max(0.0, time.monotonic() - float(started_monotonic))
        elapsed_txt = _fmt_duration(elapsed)
        if processed_dates > 0 and total_dates > processed_dates:
            speed = elapsed / float(processed_dates)
            eta = speed * float(total_dates - processed_dates)
            eta_txt = _fmt_duration(eta)

    lines = [
        "[refresh-7d] rango=ultimos 7 dias",
        f"[refresh-7d] progreso_fechas={processed_dates}/{total_dates}",
    ]
    if current_date:
        lines.append(f"[refresh-7d] fecha_actual={current_date}")
        lines.append(
            "[refresh-7d] "
            f"progreso_fecha={current_processed}/{current_pending_total} "
            f"actualizados={current_updated_ok} "
            f"aun_incompletos={current_still_incomplete} "
            f"ya_completos={current_skipped_complete} "
            f"fallidos={current_failed} total_matches={current_total_matches}"
        )
    lines.append(
        "[refresh-7d] "
        f"total={sum_total} actualizados={sum_updated_ok} "
        f"aun_incompletos={sum_still_incomplete} "
        f"ya_completos={sum_skipped_complete} fallidos={sum_failed}"
    )
    if elapsed_txt is not None:
        if eta_txt is not None and phase != "done":
            lines.append(f"[refresh-7d] tiempo elapsed={elapsed_txt} eta={eta_txt}")
        else:
            lines.append(f"[refresh-7d] tiempo elapsed={elapsed_txt}")
    lines.append(
        "[refresh-7d] estado=finalizado" if phase == "done"
        else "[refresh-7d] estado=procesando"
    )
    return "\n".join(lines)


def _pred_outcome_emoji_for_row(pred: dict, quarter: str) -> str:
    available = bool(pred.get(f"{quarter}_available"))
    if not pred:
        return ""
    if not available:
        return "🚫"
    signal = str(pred.get(f"{quarter}_signal") or "").strip().upper().replace("_", " ")
    if signal == "NO BET":
        return "🚫"
    outcome = str(pred.get(f"{quarter}_outcome") or "").lower()
    if outcome == "hit":
        return "✅"
    if outcome == "miss":
        return "❌"
    if outcome == "push":
        return "➖"
    return "⏳"


def _get_match_detail(match_id: str) -> dict | None:
    conn = _open_conn()
    data = db_mod.get_match(conn, match_id)
    conn.close()
    return data


def _is_ft_complete(data: dict | None) -> bool:
    if not data:
        return False
    status_type = str(data.get("match", {}).get("status_type", "") or "").lower()
    if status_type != "finished":
        return False

    quarters = data.get("score", {}).get("quarters", {})
    required = ("Q1", "Q2", "Q3", "Q4")
    for q in required:
        q_score = quarters.get(q)
        if not isinstance(q_score, dict):
            return False
        if q_score.get("home") is None or q_score.get("away") is None:
            return False

    pbp = data.get("play_by_play", {})
    if not isinstance(pbp, dict):
        return False
    for q in required:
        q_plays = pbp.get(q)
        if not isinstance(q_plays, list) or len(q_plays) == 0:
            return False

    graph_points = data.get("graph_points", [])
    if not isinstance(graph_points, list) or len(graph_points) == 0:
        return False

    max_minute = None
    for point in graph_points:
        try:
            minute = int((point or {}).get("minute"))
        except (TypeError, ValueError, AttributeError):
            continue
        max_minute = minute if max_minute is None else max(max_minute, minute)

    if max_minute is None or max_minute < 48:
        return False

    return True


def _refresh_incomplete_matches_for_date(
    event_date: str,
    progress_state: dict[str, object] | None = None,
) -> dict:
    logger.info(f"[REFRESH_DATE] Starting refresh for date={event_date}")
    rows = _fetch_matches_for_date(event_date)
    total = len(rows)
    updated_ok = 0
    still_incomplete = 0
    failed = 0
    skipped_complete = 0

    if progress_state is not None:
        progress_state.update(
            {
                "phase": "scanning",
                "total_matches": total,
                "pending_total": 0,
                "processed": 0,
                "updated_ok": 0,
                "still_incomplete": 0,
                "failed": 0,
                "skipped_complete": 0,
            }
        )

    conn = _open_conn()
    pending_ids: list[str] = []
    for row in rows:
        match_id = str(row.get("match_id", "") or "")
        if not match_id:
            continue

        current = db_mod.get_match(conn, match_id)
        if _is_ft_complete(current):
            skipped_complete += 1
            continue

        pending_ids.append(match_id)

    pending_total = len(pending_ids)
    if progress_state is not None:
        progress_state.update(
            {
                "phase": "refreshing",
                "pending_total": pending_total,
                "processed": 0,
                "updated_ok": 0,
                "still_incomplete": 0,
                "failed": 0,
                "skipped_complete": skipped_complete,
            }
        )

    for index, match_id in enumerate(pending_ids, start=1):
        try:
            fresh = scraper_mod.fetch_match_by_id(match_id)
            db_mod.save_match(conn, match_id, fresh)
            if _is_ft_complete(fresh):
                updated_ok += 1
            else:
                still_incomplete += 1
        except Exception as exc:
            logger.warning(f"[REFRESH_DATE] Failed match={match_id}: {exc}")
            failed += 1

        if progress_state is not None:
            progress_state.update(
                {
                    "phase": "refreshing",
                    "processed": index,
                    "updated_ok": updated_ok,
                    "still_incomplete": still_incomplete,
                    "failed": failed,
                    "skipped_complete": skipped_complete,
                }
            )

    conn.close()
    if progress_state is not None:
        progress_state.update(
            {
                "phase": "done",
                "processed": pending_total,
                "updated_ok": updated_ok,
                "still_incomplete": still_incomplete,
                "failed": failed,
                "skipped_complete": skipped_complete,
            }
        )
    logger.info(
        "[REFRESH_DATE] Done date=%s total=%s updated_ok=%s still_incomplete=%s skipped_complete=%s failed=%s",
        event_date,
        total,
        updated_ok,
        still_incomplete,
        skipped_complete,
        failed,
    )
    return {
        "date": event_date,
        "total": total,
        "updated_ok": updated_ok,
        "still_incomplete": still_incomplete,
        "failed": failed,
        "skipped_complete": skipped_complete,
    }


def _refresh_date_progress_text(event_date: str, progress_state: dict[str, object]) -> str:
    def _fmt_duration(seconds: float | int) -> str:
        total = max(int(seconds), 0)
        mm, ss = divmod(total, 60)
        hh, mm = divmod(mm, 60)
        if hh:
            return f"{hh}h {mm:02d}m {ss:02d}s"
        return f"{mm:02d}m {ss:02d}s"

    phase = str(progress_state.get("phase") or "starting")
    total_matches = int(progress_state.get("total_matches") or 0)
    pending_total = int(progress_state.get("pending_total") or 0)
    processed = int(progress_state.get("processed") or 0)
    updated_ok = int(progress_state.get("updated_ok") or 0)
    still_incomplete = int(progress_state.get("still_incomplete") or 0)
    failed = int(progress_state.get("failed") or 0)
    skipped_complete = int(progress_state.get("skipped_complete") or 0)

    started_monotonic = progress_state.get("started_monotonic")
    elapsed_txt = None
    eta_txt = None
    if isinstance(started_monotonic, (int, float)):
        elapsed = max(0.0, time.monotonic() - float(started_monotonic))
        elapsed_txt = _fmt_duration(elapsed)
        if processed > 0 and pending_total > processed:
            speed = elapsed / float(processed)
            eta = speed * float(pending_total - processed)
            eta_txt = _fmt_duration(eta)

    lines = [
        f"[refresh-date] date={event_date}",
        f"[refresh-date] total_matches={total_matches}",
    ]
    if phase in ("starting", "scanning"):
        lines.append("[refresh-date] estado=detectando pendientes...")
        lines.append(f"[refresh-date] ya_completos={skipped_complete}")
    else:
        lines.append(f"[refresh-date] pendientes={pending_total}")
        lines.append(
            "[refresh-date] "
            f"progreso={processed}/{pending_total} "
            f"actualizados={updated_ok} "
            f"aun_incompletos={still_incomplete} "
            f"fallidos={failed} "
            f"ya_completos={skipped_complete}"
        )

    if elapsed_txt is not None:
        if eta_txt is not None and phase not in ("done", "cancelled"):
            lines.append(f"[refresh-date] tiempo elapsed={elapsed_txt} eta={eta_txt}")
        else:
            lines.append(f"[refresh-date] tiempo elapsed={elapsed_txt}")

    if phase == "done":
        lines.append("[refresh-date] estado=finalizado")

    return "\n".join(lines)


def _ingest_new_date(event_date: str, limit: int | None = None) -> dict:
    rows_all = scraper_mod.fetch_finished_match_ids_for_date(event_date)
    rows = rows_all[:limit] if limit is not None else rows_all
    conn = _open_conn()
    discovered = db_mod.save_discovered_ft_matches(conn, rows)

    ing_ok = 0
    ing_fail = 0
    skipped_ft = 0

    for row in rows:
        match_id = str(row.get("match_id", "") or "")
        if not match_id:
            continue

        existing = db_mod.get_match(conn, match_id)
        if _is_ft_complete(existing):
            db_mod.mark_discovered_processed(conn, match_id)
            skipped_ft += 1
            continue

        try:
            data = scraper_mod.fetch_match_by_id(match_id)
            if not _is_ft_complete(data):
                db_mod.mark_discovered_processed(conn, match_id)
                skipped_ft += 1
                continue
            db_mod.save_match(conn, match_id, data)
            db_mod.mark_discovered_processed(conn, match_id)
            ing_ok += 1
        except Exception as exc:
            db_mod.mark_discovered_error(conn, match_id, str(exc))
            ing_fail += 1

    pending_row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM discovered_ft_matches
        WHERE event_date = ? AND processed = 0
        """,
        (event_date,),
    ).fetchone()
    conn.close()

    pending = int(pending_row["n"] if pending_row else 0)
    return {
        "date": event_date,
        "finished_found_total": len(rows_all),
        "finished_selected": len(rows),
        "limit_applied": limit,
        "discovered_rows": discovered,
        "ingested_ok": ing_ok,
        "ingested_fail": ing_fail,
        "skipped_ft": skipped_ft,
        "pending_after": pending,
    }


def _get_discovered_date_state(event_date: str) -> dict:
    conn = _open_conn()
    total_row = conn.execute(
        "SELECT COUNT(*) AS n FROM discovered_ft_matches WHERE event_date = ?",
        (event_date,),
    ).fetchone()
    pending_row = conn.execute(
        "SELECT COUNT(*) AS n FROM discovered_ft_matches WHERE event_date = ? AND processed = 0",
        (event_date,),
    ).fetchone()
    processed_row = conn.execute(
        "SELECT COUNT(*) AS n FROM discovered_ft_matches WHERE event_date = ? AND processed = 1",
        (event_date,),
    ).fetchone()
    conn.close()
    return {
        "discovered_total": int(total_row["n"] if total_row else 0),
        "pending_total": int(pending_row["n"] if pending_row else 0),
        "processed_total": int(processed_row["n"] if processed_row else 0),
    }


def _load_resume_rows(event_date: str, limit: int | None) -> list[dict]:
    conn = _open_conn()
    rows = db_mod.list_pending_discovered_ft(
        conn,
        event_date,
        event_date,
        limit=limit,
    )
    conn.close()
    return rows


def _ingest_new_date_with_progress(
    event_date: str,
    limit: int | None,
    progress_state: dict,
    cancel_event,
) -> dict:
    print(
        f"[date-ingest] start event_date={event_date} limit={limit}",
        flush=True,
    )
    resume_state = _get_discovered_date_state(event_date)
    rows_all: list[dict] | None = None
    if resume_state["pending_total"] > 0 and resume_state["discovered_total"] > 0:
        rows = _load_resume_rows(event_date, limit=limit)
        selected_total = (
            min(resume_state["discovered_total"], limit)
            if limit is not None
            else resume_state["discovered_total"]
        )
        processed_initial = max(0, selected_total - len(rows))
        progress_state["phase"] = "resuming"
        progress_state["finished_found_total"] = resume_state["discovered_total"]
        progress_state["finished_selected"] = selected_total
        progress_state["processed"] = processed_initial
        progress_state["resume_used"] = True
        print(
            (
                "[date-ingest] resume mode "
                f"discovered_total={resume_state['discovered_total']} "
                f"pending_total={resume_state['pending_total']} "
                f"selected_total={selected_total}"
            ),
            flush=True,
        )
    else:
        rows_all = scraper_mod.fetch_finished_match_ids_for_date(event_date)
        rows = rows_all[:limit] if limit is not None else rows_all
        progress_state["phase"] = "ingesting"
        progress_state["finished_found_total"] = len(rows_all)
        progress_state["finished_selected"] = len(rows)
        progress_state["processed"] = 0
        progress_state["resume_used"] = False
        print(
            (
                "[date-ingest] fresh discover mode "
                f"finished_found_total={len(rows_all)} "
                f"finished_selected={len(rows)}"
            ),
            flush=True,
        )
    progress_state["total"] = int(progress_state.get("finished_selected") or len(rows))

    conn = _open_conn()
    if rows_all is not None:
        discovered = db_mod.save_discovered_ft_matches(conn, rows)
    else:
        discovered = len(rows)
    progress_state["discovered_rows"] = discovered

    ing_ok = 0
    ing_fail = 0
    skipped_ft = 0

    for index, row in enumerate(rows, start=1):
        if cancel_event.is_set():
            progress_state["phase"] = "cancelled"
            break
        match_id = str(row.get("match_id", "") or "")
        if not match_id:
            progress_state["processed"] = int(progress_state.get("processed") or 0) + 1
            continue

        existing = db_mod.get_match(conn, match_id)
        if _is_ft_complete(existing):
            db_mod.mark_discovered_processed(conn, match_id)
            skipped_ft += 1
            progress_state["processed"] = int(progress_state.get("processed") or 0) + 1
            progress_state["ingested_ok"] = ing_ok
            progress_state["ingested_fail"] = ing_fail
            progress_state["skipped_ft"] = skipped_ft
            continue

        try:
            data = scraper_mod.fetch_match_by_id(match_id)
            if not _is_ft_complete(data):
                db_mod.mark_discovered_processed(conn, match_id)
                skipped_ft += 1
                progress_state["processed"] = int(progress_state.get("processed") or 0) + 1
                progress_state["ingested_ok"] = ing_ok
                progress_state["ingested_fail"] = ing_fail
                progress_state["skipped_ft"] = skipped_ft
                if index % DATE_INGEST_PROGRESS_EVERY == 0:
                    progress_state["last_progress_at"] = index
                continue
            db_mod.save_match(conn, match_id, data)
            db_mod.mark_discovered_processed(conn, match_id)
            ing_ok += 1
        except Exception as exc:
            db_mod.mark_discovered_error(conn, match_id, str(exc))
            ing_fail += 1

        progress_state["processed"] = int(progress_state.get("processed") or 0) + 1
        progress_state["ingested_ok"] = ing_ok
        progress_state["ingested_fail"] = ing_fail
        progress_state["skipped_ft"] = skipped_ft
        if index % DATE_INGEST_PROGRESS_EVERY == 0:
            progress_state["last_progress_at"] = index

    pending_row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM discovered_ft_matches
        WHERE event_date = ? AND processed = 0
        """,
        (event_date,),
    ).fetchone()
    conn.close()

    pending = int(pending_row["n"] if pending_row else 0)
    if cancel_event.is_set():
        progress_state["phase"] = "cancelled"
    else:
        progress_state["phase"] = "done"
    progress_state["pending_after"] = pending
    return {
        "date": event_date,
        "finished_found_total": progress_state["finished_found_total"],
        "finished_selected": progress_state["finished_selected"],
        "limit_applied": limit,
        "discovered_rows": discovered,
        "ingested_ok": ing_ok,
        "ingested_fail": ing_fail,
        "skipped_ft": skipped_ft,
        "pending_after": pending,
        "cancelled": bool(cancel_event.is_set()),
        "resume_used": bool(progress_state.get("resume_used")),
    }


def _is_message_not_modified_error(exc: Exception) -> bool:
    return isinstance(exc, BadRequest) and "Message is not modified" in str(exc)


async def _safe_edit_message(
    app: Application,
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> bool:
    try:
        await app.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
        )
        return True
    except Exception as exc:
        if _is_message_not_modified_error(exc):
            print("[date-ingest] telegram edit skipped (message not modified)", flush=True)
            return False
        if isinstance(exc, (TimedOut, NetworkError)):
            print(
                f"[date-ingest] telegram edit timeout/network error: {type(exc).__name__}",
                flush=True,
            )
            return False

        if isinstance(exc, BadRequest):
            text = str(exc)
            if (
                "Message to edit not found" in text
                or "message can't be edited" in text
                or "chat not found" in text
            ):
                print(
                    f"[date-ingest] telegram edit skipped (bad request): {text}",
                    flush=True,
                )
                return False

        print(
            f"[date-ingest] telegram edit unexpected error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False


def _date_ingest_progress_text(event_date: str, limit: int | None, progress_state: dict) -> str:
    def _fmt_duration(seconds: float | int) -> str:
        total = max(int(seconds), 0)
        mm, ss = divmod(total, 60)
        hh, mm = divmod(mm, 60)
        if hh:
            return f"{hh}h {mm:02d}m {ss:02d}s"
        return f"{mm:02d}m {ss:02d}s"

    phase = str(progress_state.get("phase") or "starting")
    found_total = int(progress_state.get("finished_found_total") or 0)
    selected = int(progress_state.get("finished_selected") or 0)
    processed = int(progress_state.get("processed") or 0)
    total = int(progress_state.get("total") or 0)
    ing_ok = int(progress_state.get("ingested_ok") or 0)
    ing_fail = int(progress_state.get("ingested_fail") or 0)
    skipped_ft = int(progress_state.get("skipped_ft") or 0)
    discovered = int(progress_state.get("discovered_rows") or 0)
    resume_used = bool(progress_state.get("resume_used"))
    started_monotonic = progress_state.get("started_monotonic")

    elapsed_txt = None
    eta_txt = None
    if isinstance(started_monotonic, (int, float)):
        elapsed = max(0.0, time.monotonic() - float(started_monotonic))
        elapsed_txt = _fmt_duration(elapsed)
        if processed > 0 and total > processed:
            speed = elapsed / float(processed)
            eta = speed * float(total - processed)
            eta_txt = _fmt_duration(eta)

    lines = [f"[date-ingest] date={event_date}"]
    if phase == "starting":
        lines.append("[date-ingest] estado=iniciando consulta a SofaScore")
    else:
        if resume_used:
            lines.append("[date-ingest] modo=reanudando desde descubiertos guardados")
        lines.append(f"[date-ingest] finished_found_total={found_total}")
        lines.append(f"[date-ingest] finished_selected={selected}")
        lines.append(f"[date-ingest] limit_applied={limit}")
        lines.append(f"[date-ingest] discovered_rows={discovered}")
        lines.append(
            f"[date-ingest] progreso={processed}/{total} ok={ing_ok} fail={ing_fail} skipped_ft={skipped_ft}"
        )
        if elapsed_txt is not None:
            if eta_txt is not None and phase not in ("done", "cancelled"):
                lines.append(f"[date-ingest] tiempo elapsed={elapsed_txt} eta={eta_txt}")
            else:
                lines.append(f"[date-ingest] tiempo elapsed={elapsed_txt}")
        if phase == "done":
            pending_after = int(progress_state.get("pending_after") or 0)
            lines.append(f"[date-ingest] pending_after={pending_after}")
            lines.append("[date-ingest] estado=finalizado")
        elif phase == "cancelled":
            pending_after = int(progress_state.get("pending_after") or 0)
            lines.append(f"[date-ingest] pending_after={pending_after}")
            lines.append("[date-ingest] estado=cancelado")
        else:
            lines.append("[date-ingest] estado=procesando")
    return "\n".join(lines)


def _date_ingest_progress_keyboard(running: bool) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if running:
        rows.append(
            [InlineKeyboardButton("Cancelar", callback_data="fetchdate:cancel")]
        )
    rows.append([InlineKeyboardButton("Menu principal", callback_data="nav:main")])
    return InlineKeyboardMarkup(rows)


def _refresh_job_progress_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
    )


def _refresh_job_status_text(job: dict[str, object]) -> str:
    mode = str(job.get("mode") or "date")
    event_date = str(job.get("event_date") or "")
    progress_state = job.get("progress_state")
    if not isinstance(progress_state, dict):
        progress_state = {"phase": "starting"}

    if mode == "7d":
        return _refresh_7d_progress_text(progress_state)
    return _refresh_date_progress_text(event_date, progress_state)


def _follow_status_text(job: dict[str, object]) -> str:
    match_id = str(job.get("match_id") or "-")
    phase = str(job.get("phase") or "running")
    reason = str(job.get("stop_reason") or "")
    cycles = int(job.get("cycles") or 0)
    unchanged = int(job.get("unchanged_cycles") or 0)
    last_tick = str(job.get("last_tick_utc") or "-")
    score = str(job.get("last_score") or "-")
    status = str(job.get("last_status") or "-")
    minute_est = job.get("minute_est")
    minute_txt = f"{minute_est}" if minute_est is not None else "-"

    lines = [
        f"Seguimiento match_id={match_id}",
        f"estado={phase}",
        f"status={status} minuto={minute_txt} score={score}",
        f"ciclos={cycles} sin_cambios={unchanged}",
        f"ultimo_tick_utc={last_tick}",
    ]
    if reason:
        lines.append(f"motivo={reason}")
    return "\n".join(lines)


def _follow_status_keyboard(job: dict[str, object]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    phase = str(job.get("phase") or "running")
    if phase == "running":
        rows.append([InlineKeyboardButton("Detener seguimiento", callback_data="follow:stop")])

    match_id = str(job.get("match_id") or "")
    event_token = str(job.get("event_token") or "_")
    page = int(job.get("page") or 0)
    if match_id:
        rows.append([InlineKeyboardButton("Abrir match seguido", callback_data=f"follow:open:{match_id}:{event_token}:{page}")])

    rows.append([InlineKeyboardButton("Menu principal", callback_data="nav:main")])
    return InlineKeyboardMarkup(rows)


async def _publish_follow_status(
    app: Application,
    chat_id: int,
    job: dict[str, object],
    *,
    force: bool = False,
) -> None:
    match_id = str(job.get("match_id") or "")
    data = job.get("follow_data")
    pred_row = job.get("follow_pred")
    is_detail = bool(data is not None and isinstance(data, dict))
    
    if is_detail:
        text = _detail_text(match_id, data, pred_row, following=True)
    else:
        text = _follow_status_text(job)

    event_token = str(job.get("event_token") or "_")
    event_date = None if event_token == "_" else event_token
    keyboard = _detail_keyboard(
        match_id,
        event_date=event_date,
        page=int(job.get("page") or 0),
        match_data=data,
        chat_id=chat_id,
    ) if is_detail else _follow_status_keyboard(job)

    last_published = str(job.get("status_last_text") or "")
    if not force and text == last_published:
        return

    if is_detail:
        image_path: Path | None = None
        try:
            image_path = _build_graph_image(match_id, data)
        except Exception:
            image_path = None

        if image_path is not None and image_path.exists():
            prev_message_id = job.get("status_message_id")
            sent = None
            try:
                with image_path.open("rb") as f:
                    sent = await app.bot.send_photo(
                        chat_id=chat_id,
                        photo=f,
                        caption=text,
                        reply_markup=keyboard,
                    )
            except Exception:
                sent = None
            finally:
                try:
                    image_path.unlink()
                except OSError:
                    pass

            if sent is not None:
                if isinstance(prev_message_id, int) and prev_message_id != sent.message_id:
                    try:
                        await app.bot.delete_message(chat_id=chat_id, message_id=prev_message_id)
                    except Exception:
                        pass
                job["status_message_id"] = sent.message_id
                job["status_last_text"] = text
                return

    message_id = job.get("status_message_id")
    if isinstance(message_id, int):
        edited = await _safe_edit_message(
            app,
            chat_id,
            message_id,
            text,
            reply_markup=keyboard,
        )
        if edited:
            job["status_last_text"] = text
            return

    try:
        sent = await app.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=keyboard,
        )
    except Exception:
        return

    job["status_message_id"] = sent.message_id
    job["status_last_text"] = text


async def _run_follow_match_job(app: Application, chat_id: int) -> None:
    job = FOLLOW_JOBS.get(chat_id)
    if not isinstance(job, dict):
        return

    match_id = str(job.get("match_id") or "")
    interval = int(job.get("interval_seconds") or FOLLOW_REFRESH_SECONDS)
    stale_limit = int(job.get("stale_cycles_limit") or FOLLOW_STALE_CYCLES_LIMIT)

    job["phase"] = "running"
    job["started_utc"] = datetime.utcnow().isoformat(timespec="seconds")
    job["cycles"] = 0
    job["unchanged_cycles"] = 0
    job["stop_reason"] = ""
    last_fp: tuple | None = None
    consecutive_errors = 0

    try:
        while True:
            current = FOLLOW_JOBS.get(chat_id)
            if current is not job:
                return

            if bool(job.get("stop_requested")):
                job["phase"] = "stopped"
                job["stop_reason"] = "manual"
                break

            data = None
            try:
                data = await asyncio.to_thread(_refresh_match_data, match_id)
                if data is None:
                    data = _get_match_detail(match_id)
                if data is None:
                    raise RuntimeError("match no encontrado al refrescar")

                await asyncio.to_thread(_get_or_compute_predictions, match_id, data)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                job["last_error"] = str(exc)
                if consecutive_errors >= 5:
                    job["phase"] = "stopped"
                    job["stop_reason"] = "errores_consecutivos"
                    break
                await asyncio.sleep(interval)
                continue

            m = data.get("match", {}) if isinstance(data, dict) else {}
            s = data.get("score", {}) if isinstance(data, dict) else {}
            graph_points = data.get("graph_points", []) if isinstance(data, dict) else []
            minute_est = None
            if graph_points:
                try:
                    minute_est = int(graph_points[-1].get("minute", 0))
                except (TypeError, ValueError):
                    minute_est = None

            status_type = str(m.get("status_type", "") or "").lower()
            status_desc = str(m.get("status_description", "") or "")
            home_score = _safe_int(s.get("home"))
            away_score = _safe_int(s.get("away"))
            score_txt = f"{home_score if home_score is not None else '-'}-{away_score if away_score is not None else '-'}"

            fp = (
                status_type,
                status_desc,
                home_score,
                away_score,
                minute_est,
                len(graph_points),
            )
            if fp == last_fp:
                job["unchanged_cycles"] = int(job.get("unchanged_cycles") or 0) + 1
            else:
                job["unchanged_cycles"] = 0
                last_fp = fp
                job["last_change_utc"] = datetime.utcnow().isoformat(timespec="seconds")

            job["cycles"] = int(job.get("cycles") or 0) + 1
            job["last_tick_utc"] = datetime.utcnow().isoformat(timespec="seconds")
            job["last_status"] = status_type or status_desc or "live"
            job["last_score"] = score_txt
            job["minute_est"] = minute_est
            job["follow_data"] = data
            pred_row = _get_or_compute_predictions(match_id, data)
            job["follow_pred"] = pred_row
            await _publish_follow_status(app, chat_id, job)

            if status_type == "finished":
                job["phase"] = "done"
                job["stop_reason"] = "match_finished"
                break

            if int(job.get("unchanged_cycles") or 0) >= stale_limit:
                job["phase"] = "stopped"
                job["stop_reason"] = "sin_cambios"
                break

            await asyncio.sleep(interval)
    finally:
        final_job = FOLLOW_JOBS.get(chat_id)
        if final_job is job:
            await _publish_follow_status(app, chat_id, job, force=True)
            FOLLOW_JOBS.pop(chat_id, None)


async def _run_refresh_date_job(
    app: Application,
    chat_id: int,
    progress_message_id: int,
    event_date: str,
) -> None:
    progress_state: dict[str, object] = {
        "phase": "starting",
        "total_matches": 0,
        "pending_total": 0,
        "processed": 0,
        "updated_ok": 0,
        "still_incomplete": 0,
        "failed": 0,
        "skipped_complete": 0,
        "started_monotonic": time.monotonic(),
    }
    REFRESH_JOBS[chat_id] = {
        "mode": "date",
        "event_date": event_date,
        "message_id": progress_message_id,
        "progress_state": progress_state,
        "suppress_edits": False,
        "force_next_edit": False,
    }

    task = asyncio.create_task(
        asyncio.to_thread(
            _refresh_incomplete_matches_for_date,
            event_date,
            progress_state,
        )
    )

    last_text = ""
    try:
        while not task.done():
            text = _refresh_date_progress_text(event_date, progress_state)
            if text != last_text:
                job = REFRESH_JOBS.get(chat_id, {})
                suppress_edits = bool(job.get("suppress_edits"))
                force_next_edit = bool(job.get("force_next_edit"))
                if force_next_edit and isinstance(job, dict):
                    job["force_next_edit"] = False

                if not suppress_edits or force_next_edit:
                    await _safe_edit_message(
                        app,
                        chat_id=chat_id,
                        message_id=progress_message_id,
                        text=text,
                        reply_markup=_refresh_job_progress_keyboard(),
                    )
                last_text = text

            await asyncio.sleep(REFRESH_DATE_STATUS_INTERVAL_SECONDS)

        result = await task
    except Exception as exc:
        await _safe_edit_message(
            app,
            chat_id=chat_id,
            message_id=progress_message_id,
            text=f"Error refrescando pendientes del dia: {exc}",
            reply_markup=_refresh_job_progress_keyboard(),
        )
        REFRESH_JOBS.pop(chat_id, None)
        return

    rows = _fetch_matches_for_date(event_date)
    pred_map = _fetch_date_pred_outcomes(event_date)
    stats = _pred_stats_text(pred_map, len(rows))
    summary = (
        f"Descarga pendientes completada para {event_date}:\n"
        f"Total: {result['total']} | Actualizados: {result['updated_ok']} | "
        f"Aun incompletos: {result['still_incomplete']} | "
        f"Ya completos: {result['skipped_complete']} | Fallidos: {result['failed']}"
    )
    header = f"{summary}\n\nMatches del {event_date}:"
    if stats:
        header += f"\n{stats}"
    await _safe_edit_message(
        app,
        chat_id=chat_id,
        message_id=progress_message_id,
        text=header,
        reply_markup=_matches_keyboard(rows, event_date, 0, pred_map),
    )
    REFRESH_JOBS.pop(chat_id, None)


async def _run_refresh_7d_job(
    app: Application,
    chat_id: int,
    progress_message_id: int,
) -> None:
    progress_state: dict[str, object] = {
        "phase7d": "starting",
        "total_dates": 0,
        "processed_dates": 0,
        "current_date": "",
        "sum_total": 0,
        "sum_updated_ok": 0,
        "sum_still_incomplete": 0,
        "sum_skipped_complete": 0,
        "sum_failed": 0,
        "started_monotonic": time.monotonic(),
    }
    REFRESH_JOBS[chat_id] = {
        "mode": "7d",
        "event_date": "",
        "message_id": progress_message_id,
        "progress_state": progress_state,
        "suppress_edits": False,
        "force_next_edit": False,
    }

    task = asyncio.create_task(
        asyncio.to_thread(
            _refresh_recent_dates,
            7,
            progress_state,
        )
    )

    last_text = ""
    try:
        while not task.done():
            text = _refresh_7d_progress_text(progress_state)
            if text != last_text:
                job = REFRESH_JOBS.get(chat_id, {})
                suppress_edits = bool(job.get("suppress_edits"))
                force_next_edit = bool(job.get("force_next_edit"))
                if force_next_edit and isinstance(job, dict):
                    job["force_next_edit"] = False

                if not suppress_edits or force_next_edit:
                    await _safe_edit_message(
                        app,
                        chat_id=chat_id,
                        message_id=progress_message_id,
                        text=text,
                        reply_markup=_refresh_job_progress_keyboard(),
                    )
                last_text = text

            await asyncio.sleep(REFRESH_DATE_STATUS_INTERVAL_SECONDS)

        result7d = await task
    except Exception as exc:
        await _safe_edit_message(
            app,
            chat_id=chat_id,
            message_id=progress_message_id,
            text=f"Error refrescando pendientes de 7 dias: {exc}",
            reply_markup=_refresh_job_progress_keyboard(),
        )
        REFRESH_JOBS.pop(chat_id, None)
        return

    dates = result7d.get("dates") or []
    dates_text = ", ".join(str(d) for d in dates) if dates else "(sin fechas)"
    summary7d = (
        "Descarga pendientes (ultimos 7 dias) completada:\n"
        f"Fechas: {len(dates)}\n"
        f"Total: {int(result7d.get('sum_total') or 0)} | "
        f"Actualizados: {int(result7d.get('sum_updated_ok') or 0)} | "
        f"Aun incompletos: {int(result7d.get('sum_still_incomplete') or 0)} | "
        f"Ya completos: {int(result7d.get('sum_skipped_complete') or 0)} | "
        f"Fallidos: {int(result7d.get('sum_failed') or 0)}\n"
        f"Rango: {dates_text}"
    )
    await _safe_edit_message(
        app,
        chat_id=chat_id,
        message_id=progress_message_id,
        text=summary7d,
        reply_markup=_refresh_job_progress_keyboard(),
    )
    REFRESH_JOBS.pop(chat_id, None)


async def _run_date_ingest_job(
    app: Application,
    chat_id: int,
    progress_message_id: int,
    event_date: str,
    limit: int | None,
) -> None:
    progress_state: dict[str, object] = {
        "phase": "starting",
        "processed": 0,
        "total": 0,
        "ingested_ok": 0,
        "ingested_fail": 0,
        "skipped_ft": 0,
        "discovered_rows": 0,
        "started_monotonic": time.monotonic(),
    }
    cancel_event = threading.Event()
    DATE_INGEST_JOBS[chat_id] = {
        "cancel_event": cancel_event,
        "message_id": progress_message_id,
        "event_date": event_date,
        "limit": limit,
        "progress_state": progress_state,
        "suppress_edits": False,
        "force_next_edit": False,
    }
    task = asyncio.create_task(
        asyncio.to_thread(
            _ingest_new_date_with_progress,
            event_date,
            limit,
            progress_state,
            cancel_event,
        )
    )

    last_text = _date_ingest_progress_text(event_date, limit, progress_state)
    print("[date-ingest] background job started", flush=True)
    try:
        while not task.done():
            text = _date_ingest_progress_text(event_date, limit, progress_state)
            if text != last_text:
                print(text.replace("\n", " | "), flush=True)
                job = DATE_INGEST_JOBS.get(chat_id, {})
                suppress_edits = bool(job.get("suppress_edits"))
                force_next_edit = bool(job.get("force_next_edit"))
                if force_next_edit and isinstance(job, dict):
                    job["force_next_edit"] = False

                if not suppress_edits or force_next_edit:
                    await _safe_edit_message(
                        app,
                        chat_id=chat_id,
                        message_id=progress_message_id,
                        text=text,
                        reply_markup=_date_ingest_progress_keyboard(True),
                    )
                last_text = text
            await asyncio.sleep(DATE_INGEST_STATUS_INTERVAL_SECONDS)

        result = await task
    except Exception as exc:
        print(f"[date-ingest] background job error: {exc}", flush=True)
        await _safe_edit_message(
            app,
            chat_id=chat_id,
            message_id=progress_message_id,
            text=f"No pude traer fecha {event_date}: {exc}",
            reply_markup=_date_ingest_progress_keyboard(False),
        )
        DATE_INGEST_JOBS.pop(chat_id, None)
        return

    progress_state.update(result)
    final_text = _date_ingest_progress_text(event_date, limit, progress_state)
    print(final_text.replace("\n", " | "), flush=True)
    await _safe_edit_message(
        app,
        chat_id=chat_id,
        message_id=progress_message_id,
        text=final_text,
        reply_markup=_date_ingest_progress_keyboard(False),
    )
    print("[date-ingest] background job finished", flush=True)
    DATE_INGEST_JOBS.pop(chat_id, None)


def _main_menu_keyboard(chat_id: int | None = None) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("1) Matches en vivo", callback_data="menu:live")],
        [InlineKeyboardButton("2) Matches por fecha", callback_data="menu:dates")],
        [InlineKeyboardButton("3) Match por ID", callback_data="menu:id")],
        [InlineKeyboardButton("4) Re entrenar base", callback_data="menu:train")],
        [InlineKeyboardButton("5) Traer fecha nueva", callback_data="menu:fetchdate")],
        [
            InlineKeyboardButton(
                "6) Descargar pendientes 7 dias",
                callback_data="menu:refresh7d",
            )
        ],
    ]
    if chat_id is not None and chat_id in DATE_INGEST_JOBS:
        rows.append(
            [
                InlineKeyboardButton(
                    "7) Status descarga",
                    callback_data="menu:fetchdate:status",
                )
            ]
        )
    if chat_id is not None and chat_id in REFRESH_JOBS:
        rows.append(
            [
                InlineKeyboardButton(
                    "8) Status refresh pendientes",
                    callback_data="menu:refresh:status",
                )
            ]
        )
    if chat_id is not None and chat_id in FOLLOW_JOBS:
        rows.append(
            [
                InlineKeyboardButton(
                    "9) Status seguimiento",
                    callback_data="menu:follow:status",
                )
            ]
        )
    return InlineKeyboardMarkup(rows)


def _live_minute_text(row: dict) -> str:
    def _fmt_mmss(total_seconds: int) -> str:
        minutes, seconds = divmod(max(total_seconds, 0), 60)
        return f"{minutes}:{seconds:02d}"

    def _bet_badge(match_id: str, quarter: str) -> str:
        if not match_id:
            return ""
        try:
            pred_row = _read_prediction_row(match_id)
            pred = (pred_row or {}).get(quarter, {}) if isinstance(pred_row, dict) else {}
            if not isinstance(pred, dict) or not pred.get("available"):
                return ""
            signal = str(pred.get("signal") or "").strip().upper().replace("_", " ")
            if signal in {"BET", "APUESTA"}:
                return "🟢"
        except Exception:
            return ""
        return ""

    def _period_short(status_desc: str) -> str:
        value = status_desc.strip().lower()
        mapping = {
            "1st quarter": "Q1",
            "2nd quarter": "Q2",
            "3rd quarter": "Q3",
            "4th quarter": "Q4",
            "halftime": "HT",
        }
        if value in mapping:
            return mapping[value]
        if value.startswith("pause"):
            return "P"
        return status_desc or "live"

    played_seconds = _safe_int(row.get("played_seconds"))
    status_desc = str(row.get("status_description", "") or "")
    match_id = str(row.get("match_id", "") or "")
    if played_seconds is None:
        return _period_short(status_desc)

    now_txt = f"{_period_short(status_desc)} {_fmt_mmss(played_seconds)}"

    q3_diff = 24 * 60 - played_seconds
    q4_diff = 36 * 60 - played_seconds
    q3_badge = _bet_badge(match_id, "q3")
    q4_badge = _bet_badge(match_id, "q4")
    q3_txt = f"Q3{q3_badge} ok" if q3_diff <= 0 else f"aQ3{q3_badge} {_fmt_mmss(q3_diff)}"
    q4_txt = f"Q4{q4_badge} ok" if q4_diff <= 0 else f"aQ4{q4_badge} {_fmt_mmss(q4_diff)}"

    # First segment: nearest betting quarter ETA (Q3, then Q4)
    if q3_diff > 0:
        nearest_txt = f"aQ3{q3_badge} {_fmt_mmss(q3_diff)}"
    elif q4_diff > 0:
        nearest_txt = f"aQ4{q4_badge} {_fmt_mmss(q4_diff)}"
    else:
        nearest_txt = f"Q4{q4_badge} ok"

    return f"{nearest_txt} | {now_txt} | {q3_txt} | {q4_txt}"


def _live_sort_key(row: dict) -> tuple[int, int, int]:
    played_seconds = _safe_int(row.get("played_seconds"))
    if played_seconds is None:
        return (3, 10**9, 10**9)

    q3_diff = 24 * 60 - played_seconds
    q4_diff = 36 * 60 - played_seconds

    if q3_diff > 0:
        return (0, q3_diff, played_seconds)
    if q4_diff > 0:
        return (1, q4_diff, played_seconds)
    return (2, played_seconds, played_seconds)


def _live_summary_text(result: dict) -> str:
    return (
        f"Live ahora: {len(result.get('live_rows', []))} | "
        f"guardados_ok={result.get('ingested_ok', 0)} "
        f"fallidos={result.get('ingested_fail', 0)}"
    )


def _live_keyboard(result: dict, page: int) -> InlineKeyboardMarkup:
    live_rows = sorted(
        result.get("live_rows", []),
        key=_live_sort_key,
    )
    total_pages = max((len(live_rows) - 1) // LIVE_PAGE_SIZE + 1, 1)
    page = max(0, min(page, total_pages - 1))

    start = page * LIVE_PAGE_SIZE
    end = start + LIVE_PAGE_SIZE
    page_rows = live_rows[start:end]

    keyboard: list[list[InlineKeyboardButton]] = []
    for row in page_rows:
        match_id = str(row.get("match_id", "") or "")
        if not match_id:
            continue
        league = str(row.get("league", "") or "-")[:12]
        home = _abbr_team(str(row.get("home_team", "")), max_len=14)
        away = _abbr_team(str(row.get("away_team", "")), max_len=14)
        minute = _live_minute_text(row)
        label = f"{league} | {home} vs {away} | {minute}"
        keyboard.append(
            [
                InlineKeyboardButton(
                    label,
                    callback_data=f"livematch:{match_id}:{page}",
                )
            ]
        )

    nav_row: list[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(
            InlineKeyboardButton("<<", callback_data=f"livepage:{page - 1}")
        )
    nav_row.append(
        InlineKeyboardButton(
            f"Pag {page + 1}/{total_pages}",
            callback_data="noop",
        )
    )
    if page < total_pages - 1:
        nav_row.append(
            InlineKeyboardButton(">>", callback_data=f"livepage:{page + 1}")
        )
    keyboard.append(nav_row)
    keyboard.append([InlineKeyboardButton("Refrescar en vivo", callback_data="menu:live")])
    keyboard.append([InlineKeyboardButton("Menu principal", callback_data="nav:main")])
    return InlineKeyboardMarkup(keyboard)


def _ingest_live_matches() -> dict:
    print("[live] consultando lista live de SofaScore...", flush=True)
    live_rows = scraper_mod.fetch_live_match_ids()
    total = len(live_rows)
    print(f"[live] encontrados={total}", flush=True)

    match_ids = [str(row.get("match_id", "") or "") for row in live_rows]
    match_ids = [mid for mid in match_ids if mid]

    conn = _open_conn()

    ing_ok = 0
    ing_fail = 0
    results = scraper_mod.fetch_matches_by_ids(match_ids)
    for index, (match_id, data, error) in enumerate(results, start=1):
        print(
            f"[live] {index}/{total} match_id={match_id}...",
            flush=True,
        )
        if data is None:
            ing_fail += 1
            print(
                f"[live] {index}/{total} fail match_id={match_id} error={error}",
                flush=True,
            )
            continue
        try:
            db_mod.save_match(conn, match_id, data)
            ing_ok += 1
            print(
                f"[live] {index}/{total} ok match_id={match_id}",
                flush=True,
            )
        except Exception as exc:
            ing_fail += 1
            print(
                f"[live] {index}/{total} fail-save match_id={match_id} error={exc}",
                flush=True,
            )
    conn.close()

    print(
        f"[live] terminado ok={ing_ok} fail={ing_fail}",
        flush=True,
    )

    return {
        "live_rows": live_rows,
        "ingested_ok": ing_ok,
        "ingested_fail": ing_fail,
    }


async def _render_live(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    page: int = 0,
    *,
    refetch: bool = False,
) -> None:
    if refetch or LIVE_RESULT_KEY not in context.user_data:
        await _set_waiting_state(update, text="Espere... trayendo live")
        try:
            result = await asyncio.to_thread(_ingest_live_matches)
        except Exception as exc:
            await _replace_callback_message(
                update,
                text=f"No pude traer matches live: {exc}",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return
        context.user_data[LIVE_RESULT_KEY] = result
    else:
        result = context.user_data[LIVE_RESULT_KEY]

    await _replace_callback_message(
        update,
        text=_live_summary_text(result),
        reply_markup=_live_keyboard(result, page),
    )


def _follow_toggle_button(
    chat_id: int | None,
    match_id: str,
    event_token: str,
    page: int,
    match_data: dict | None = None,
) -> InlineKeyboardButton | None:
    status_type = str(
        (match_data or {}).get("match", {}).get("status_type", "") or ""
    ).lower()
    if status_type == "finished":
        return None

    if chat_id is not None:
        job = FOLLOW_JOBS.get(chat_id)
        if isinstance(job, dict):
            followed_match_id = str(job.get("match_id") or "")
            phase = str(job.get("phase") or "")
            stop_requested = bool(job.get("stop_requested"))
            if (
                followed_match_id == str(match_id)
                and phase in {"starting", "running"}
                and not stop_requested
            ):
                return InlineKeyboardButton(
                    "Detener seguimiento",
                    callback_data="follow:stop",
                )

    return InlineKeyboardButton(
        "Iniciar seguimiento",
        callback_data=f"follow:start:{match_id}:{event_token}:{page}",
    )


def _live_detail_keyboard(
    match_id: str,
    page: int,
    match_data: dict | None = None,
    chat_id: int | None = None,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    rows.append(
        [InlineKeyboardButton("Ver SofaScore", url=_sofascore_match_url(match_id, match_data))]
    )
    rows.append(
        [
            InlineKeyboardButton(
                "Refresh datos + pred",
                callback_data=f"refreshlive:all:{match_id}:{page}",
            ),
            InlineKeyboardButton(
                "Refresh pred",
                callback_data=f"refreshlive:pred:{match_id}:{page}",
            ),
        ]
    )
    _ftb = _follow_toggle_button(chat_id, match_id, "_", page, match_data)
    if _ftb is not None:
        rows.append([_ftb])
    rows.append(
        [
            InlineKeyboardButton(
                "Borrar match de la base",
                callback_data=f"dellive:confirm:{match_id}:{page}",
            )
        ]
    )
    rows.append([InlineKeyboardButton("Volver a live", callback_data=f"livepage:{page}")])
    rows.append([InlineKeyboardButton("Menu principal", callback_data="nav:main")])
    return InlineKeyboardMarkup(rows)


async def _send_live_detail_message(
    update: Update,
    app: Application,
    match_id: str,
    data: dict,
    pred_row: dict | None,
    page: int,
) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    detail_text = _detail_text(match_id, data, pred_row)
    keyboard = _live_detail_keyboard(
        match_id,
        page,
        match_data=data,
        chat_id=chat_id,
    )

    image_path: Path | None = None
    try:
        image_path = _build_graph_image(match_id, data)
    except Exception:
        image_path = None

    if image_path is not None and image_path.exists():
        try:
            with image_path.open("rb") as f:
                await app.bot.send_photo(
                    chat_id=chat_id,
                    photo=f,
                    caption=detail_text,
                    reply_markup=keyboard,
                )
        finally:
            try:
                image_path.unlink()
            except OSError:
                pass
        return

    await app.bot.send_message(
        chat_id=chat_id,
        text=detail_text,
        reply_markup=keyboard,
    )


async def _render_live_detail(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    match_id: str,
    page: int,
    *,
    pred_row: dict | None = None,
) -> None:
    await _set_waiting_state(update)
    try:
        data = _get_match_detail(match_id)
        if data is None:
            data = await asyncio.to_thread(_refresh_match_data, match_id)
    except Exception as exc:
        await _replace_callback_message(
            update,
            text=f"No se pudo abrir match live {match_id}: {exc}",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Volver a live", callback_data=f"livepage:{page}")]]
            ),
        )
        return

    if pred_row is None:
        pred_row = _get_or_compute_predictions(match_id, data)
    await _send_live_detail_message(
        update,
        context.application,
        match_id,
        data,
        pred_row,
        page,
    )
    if update.callback_query and update.callback_query.message:
        try:
            await update.callback_query.message.delete()
        except Exception:
            pass


async def _refresh_live_detail(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    mode: str,
    match_id: str,
    page: int,
) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else None
    cached = _get_match_detail(match_id)
    await _set_waiting_state(update, text=_refresh_waiting_text(match_id, cached))

    if mode == "all":
        try:
            data = await asyncio.to_thread(_refresh_match_data, match_id)
        except Exception as exc:
            await _replace_callback_message(
                update,
                text=f"No pude refrescar datos de {match_id}: {exc}",
                reply_markup=_live_detail_keyboard(match_id, page, chat_id=chat_id),
            )
            return
        pred_row = _refresh_predictions(match_id, data)
        await _render_live_detail(
            update,
            context,
            match_id,
            page,
            pred_row=pred_row,
        )
        return

    data = _get_match_detail(match_id)
    if not data:
        await _replace_callback_message(
            update,
            text=f"No se encontro el match {match_id}.",
            reply_markup=_live_detail_keyboard(match_id, page, chat_id=chat_id),
        )
        return

    pred_row = _refresh_predictions(match_id, data)
    await _render_live_detail(
        update,
        context,
        match_id,
        page,
        pred_row=pred_row,
    )


def _menu_reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[MENU_BUTTON_TEXT]],
        resize_keyboard=True,
        is_persistent=True,
    )


def _train_submenu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Entrenar (v2 + v4 + calibrar)",
                    callback_data="train:run",
                )
            ],
            [
                InlineKeyboardButton(
                    "Vaciar Resultados",
                    callback_data="train:clear:confirm",
                )
            ],
            [
                InlineKeyboardButton(
                    "Recalcular universo",
                    callback_data="train:recalc:confirm",
                )
            ],
            [InlineKeyboardButton("Stats Modelo", callback_data="train:stats")],
            [InlineKeyboardButton("Estado entrenamiento", callback_data="train:status")],
            [InlineKeyboardButton("Menu principal", callback_data="nav:main")],
        ]
    )


def _train_recalc_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Confirmar recalcular universo",
                    callback_data="train:recalc:run",
                )
            ],
            [InlineKeyboardButton("Volver", callback_data="menu:train")],
            [InlineKeyboardButton("Menu principal", callback_data="nav:main")],
        ]
    )


def _train_clear_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Confirmar vaciar resultados",
                    callback_data="train:clear:run",
                )
            ],
            [InlineKeyboardButton("Volver", callback_data="menu:train")],
            [InlineKeyboardButton("Menu principal", callback_data="nav:main")],
        ]
    )


def _dates_keyboard(
    rows: list[dict],
    page: int,
    pred_stats: dict | None = None,
) -> InlineKeyboardMarkup:
    total_pages = max((len(rows) - 1) // DATE_PAGE_SIZE + 1, 1)
    page = max(0, min(page, total_pages - 1))

    start = page * DATE_PAGE_SIZE
    end = start + DATE_PAGE_SIZE
    page_rows = rows[start:end]

    keyboard: list[list[InlineKeyboardButton]] = []
    for row in page_rows:
        date_str = str(row.get("event_date", ""))
        short = _short_date(date_str)
        ps = (pred_stats or {}).get(date_str, {})
        q3_hit  = ps.get("q3_hit", 0)
        q3_miss = ps.get("q3_miss", 0)
        q3_push = ps.get("q3_push", 0)
        q4_hit  = ps.get("q4_hit", 0)
        q4_miss = ps.get("q4_miss", 0)
        q4_push = ps.get("q4_push", 0)
        if q3_hit or q3_miss or q3_push or q4_hit or q4_miss or q4_push:
            q3_str = f"Q3:{q3_hit}\u2705{q3_miss}\u274c"
            if q3_push:
                q3_str += f"{q3_push}\u2796"
            q4_str = f"Q4:{q4_hit}\u2705{q4_miss}\u274c"
            if q4_push:
                q4_str += f"{q4_push}\u2796"
            text = f"{short} | {q3_str} {q4_str}"
        else:
            total = int(row.get("total_matches", 0) or 0)
            text = f"{short} | M:{total}"
        keyboard.append(
            [
                InlineKeyboardButton(
                    text=text,
                    callback_data=f"date:{date_str}:0",
                )
            ]
        )

    nav_row: list[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(
            InlineKeyboardButton("<<", callback_data=f"dates:{page - 1}")
        )
    nav_row.append(
        InlineKeyboardButton(
            f"Pag {page + 1}/{total_pages}",
            callback_data="noop",
        )
    )
    if page < total_pages - 1:
        nav_row.append(
            InlineKeyboardButton(">>", callback_data=f"dates:{page + 1}")
        )
    keyboard.append(nav_row)

    keyboard.append([InlineKeyboardButton("Menu principal", callback_data="nav:main")])
    return InlineKeyboardMarkup(keyboard)


def _matches_keyboard(
    rows: list[dict],
    event_date: str,
    page: int,
    pred_map: dict | None = None,
) -> InlineKeyboardMarkup:
    total_pages = max((len(rows) - 1) // MATCH_PAGE_SIZE + 1, 1)
    page = max(0, min(page, total_pages - 1))

    start = page * MATCH_PAGE_SIZE
    end = start + MATCH_PAGE_SIZE
    page_rows = rows[start:end]

    keyboard: list[list[InlineKeyboardButton]] = []
    for row in page_rows:
        match_id = str(row.get("match_id", ""))
        home = _abbr_team(str(row.get("home_team", "")))
        away = _abbr_team(str(row.get("away_team", "")))
        tipoff = str(row.get("display_time", "") or "--:--")
        league = str(row.get("league", "") or "-")
        league_short = league[:10]
        q3 = _winner_short(_safe_int(row.get("q3_home")), _safe_int(row.get("q3_away")))
        q4 = _winner_short(_safe_int(row.get("q4_home")), _safe_int(row.get("q4_away")))
        status_type = str(row.get("status_type", "") or "").strip().lower()
        is_finished = status_type == "finished"
        q4_started = (
            _safe_int(row.get("q4_home")) is not None
            and _safe_int(row.get("q4_away")) is not None
        )
        if not is_finished and not q4_started:
            q3 = "⏳"
        if not is_finished:
            q4 = "⏳"
        pred = (pred_map or {}).get(match_id, {})
        q3e = _pred_outcome_emoji_for_row(pred, "q3")
        q4e = _pred_outcome_emoji_for_row(pred, "q4")
        if not is_finished and not q4_started:
            q3e = ""
        if not is_finished:
            q4e = ""
        # Resolve ⏳ from actual scores when available
        if q3e == "⏳" and pred.get("q3_available") and q4_started:
            _q3h_r = _safe_int(row.get("q3_home"))
            _q3a_r = _safe_int(row.get("q3_away"))
            _q3p_r = str(pred.get("q3_pick") or "").lower()
            if _q3h_r is not None and _q3a_r is not None and _q3p_r in ("home", "away"):
                _q3w_r = _winner_from_scores(_q3h_r, _q3a_r)
                if _q3w_r == "push":
                    q3e = "➖"
                elif _q3w_r in ("home", "away"):
                    q3e = "✅" if _q3w_r == _q3p_r else "❌"
        if q4e == "⏳" and pred.get("q4_available") and is_finished:
            _q4h_r = _safe_int(row.get("q4_home"))
            _q4a_r = _safe_int(row.get("q4_away"))
            _q4p_r = str(pred.get("q4_pick") or "").lower()
            if _q4h_r is not None and _q4a_r is not None and _q4p_r in ("home", "away"):
                _q4w_r = _winner_from_scores(_q4h_r, _q4a_r)
                if _q4w_r == "push":
                    q4e = "➖"
                elif _q4w_r in ("home", "away"):
                    q4e = "✅" if _q4w_r == _q4p_r else "❌"
        label = f"{tipoff} {league_short} | {home} vs {away} | Q3:{q3}{q3e} Q4:{q4}{q4e}"
        callback = f"match:{match_id}:{event_date}:{page}"
        keyboard.append([InlineKeyboardButton(label, callback_data=callback)])

    nav_row: list[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(
            InlineKeyboardButton(
                "<<",
                callback_data=f"date:{event_date}:{page - 1}",
            )
        )
    nav_row.append(
        InlineKeyboardButton(
            f"Pag {page + 1}/{total_pages}",
            callback_data="noop",
        )
    )
    if page < total_pages - 1:
        nav_row.append(
            InlineKeyboardButton(
                ">>",
                callback_data=f"date:{event_date}:{page + 1}",
            )
        )
    keyboard.append(nav_row)

    keyboard.append(
        [InlineKeyboardButton("Volver a fechas", callback_data="dates:0")]
    )
    keyboard.append(
        [InlineKeyboardButton("🧠 Recalcular apuestas del dia", callback_data=f"calc:date:{event_date}")]
    )
    keyboard.append(
        [InlineKeyboardButton("⬇️ Descargar pendientes del dia", callback_data=f"refresh:date:{event_date}")]
    )
    keyboard.append([InlineKeyboardButton("Menu principal", callback_data="nav:main")])
    return InlineKeyboardMarkup(keyboard)


def _calc_date_predictions(event_date: str, *, force_recalc: bool = True) -> dict:
    rows = _fetch_matches_for_date(event_date)
    existing = _fetch_date_pred_outcomes(event_date)
    ok = fail = skipped = 0
    for row in rows:
        match_id = str(row.get("match_id", ""))
        if not match_id:
            continue
        if (not force_recalc) and (match_id in existing):
            skipped += 1
            continue
        data = _get_match_detail(match_id)
        if not data:
            fail += 1
            continue
        try:
            result = _compute_and_store_predictions(match_id, data)
            if result is not None:
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
    return {
        "total": len(rows),
        "recalculated": ok + fail,
        "skipped": skipped,
        "ok": ok,
        "fail": fail,
    }


def _quarter_line(data: dict, quarter: str) -> str:
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return f"{quarter}: -"

    def _bold_num(value: int) -> str:
        trans = str.maketrans("0123456789-", "𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵-")
        return str(value).translate(trans)

    home = _safe_int(q.get("home"))
    away = _safe_int(q.get("away"))
    if home is None or away is None:
        return f"{quarter}: {q.get('home', '-')} - {q.get('away', '-')}"

    home_txt = str(home)
    away_txt = str(away)
    if home > away:
        home_txt = _bold_num(home)
    elif away > home:
        away_txt = _bold_num(away)
    return f"{quarter}: {home_txt} - {away_txt}"


def _graph_summary(graph_points: list[dict]) -> tuple[int, int, int, int, int]:
    if not graph_points:
        return 0, 0, 0, 0, 0

    values: list[int] = []
    for p in graph_points:
        try:
            values.append(int(p.get("value", 0)))
        except (TypeError, ValueError):
            continue

    if not values:
        return 0, 0, 0, 0, 0

    last = values[-1]
    peak_home = max(values)
    peak_away = min(values)
    spread = peak_home - peak_away
    return len(values), peak_home, peak_away, last, spread


def _match_detail_text(match_id: str, data: dict) -> str:
    m = data.get("match", {})
    s = data.get("score", {})
    home_team_name = str(m.get("home_team", "home") or "home")
    away_team_name = str(m.get("away_team", "away") or "away")
    q3 = s.get("quarters", {}).get("Q3")
    q4 = s.get("quarters", {}).get("Q4")
    q2 = s.get("quarters", {}).get("Q2")

    q2_home = _safe_int((q2 or {}).get("home"))
    q2_away = _safe_int((q2 or {}).get("away"))
    q3_home = _safe_int((q3 or {}).get("home"))
    q3_away = _safe_int((q3 or {}).get("away"))
    q4_home = _safe_int((q4 or {}).get("home"))
    q4_away = _safe_int((q4 or {}).get("away"))

    utc_date = str(m.get("date", "") or "")
    utc_time = str(m.get("time", "") or "")
    local_dt = _to_local_datetime(utc_date, utc_time)
    if local_dt is not None:
        fecha_line = (
            f"Fecha: {local_dt.strftime('%Y-%m-%d %H:%M')} UTC{UTC_OFFSET_HOURS:+d} "
            f"(UTC: {utc_date} {utc_time})"
        )
    else:
        fecha_line = f"Fecha: {utc_date} {utc_time} UTC"

    status_type = str(m.get("status_type", "") or "")
    status_description = str(m.get("status_description", "") or "")
    minute_est = None
    graph_points = data.get("graph_points", [])
    if graph_points:
        try:
            minute_est = int(graph_points[-1].get("minute", 0))
        except (TypeError, ValueError):
            minute_est = None
    minute_suffix = f" | Min: {minute_est}" if minute_est is not None else ""

    def _quarter_winner_text(quarter: str, home: int | None, away: int | None) -> str:
        if home is None or away is None:
            return "⏳"
        if quarter == "Q4" and status_type != "finished":
            return "⏳"
        if quarter == "Q3" and status_type != "finished":
            q4 = s.get("quarters", {}).get("Q4")
            if not q4 and minute_est is not None and minute_est < 36 and "Q4" not in status_description:
                return "⏳"
        winner = _winner_from_scores(home, away)
        if winner == "home":
            return f"🏠 {home_team_name}"
        if winner == "away":
            return f"🛫 {away_team_name}"
        if winner == "push":
            return "➖ push"
        return "-"

    def _quarter_line_with_winner(quarter: str, home: int | None, away: int | None) -> str:
        def _bold_num(value: int) -> str:
            trans = str.maketrans("0123456789-", "𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵-")
            return str(value).translate(trans)

        if home is None or away is None:
            score_txt = "-"
        else:
            home_txt = str(home)
            away_txt = str(away)
            if home > away:
                home_txt = _bold_num(home)
            elif away > home:
                away_txt = _bold_num(away)
            score_txt = f"{home_txt} - {away_txt}"
        winner_txt = _quarter_winner_text(quarter, home, away)
        if winner_txt == "-":
            return f"{quarter}: {score_txt}"
        return f"{quarter}: {score_txt} {winner_txt}"

    def _quarter2_line() -> str:
        if q2_home is None or q2_away is None:
            if status_type != "finished":
                return "Q2: - ⏳"
            return "Q2: -"
        return _quarter_line(data, "Q2")

    lines = [
        f"Match ID: {match_id}{minute_suffix}",
        f"{m.get('home_team', '')} vs {m.get('away_team', '')}",
        fecha_line,
        f"Liga: {m.get('league', '')}",
        _quarter_line(data, "Q1"),
        _quarter2_line(),
        _quarter_line_with_winner("Q3", q3_home, q3_away),
        _quarter_line_with_winner("Q4", q4_home, q4_away),
    ]
    return "\n".join(lines)


def _quarter_score(data: dict | None, quarter: str) -> tuple[int | None, int | None]:
    if not data:
        return None, None
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return _safe_int(q.get("home")), _safe_int(q.get("away"))


def _all_result_tags(conn: sqlite3.Connection) -> list[str]:
    cols = [
        row["name"]
        for row in conn.execute("PRAGMA table_info(eval_match_results)").fetchall()
    ]
    tags: set[str] = set()
    for col in cols:
        if col.startswith("q3_pick__"):
            tags.add(col.split("__", maxsplit=1)[1])
    return sorted(tags)


def _row_value(row: sqlite3.Row, key: str) -> object:
    return row[key] if key in row.keys() else None


def _read_prediction_row(match_id: str) -> dict | None:
    conn = _open_conn()
    tags = _all_result_tags(conn)
    if not tags:
        conn.close()
        return None

    rows = conn.execute(
        """
        SELECT *
        FROM eval_match_results
        WHERE match_id = ?
        ORDER BY updated_at DESC, event_date DESC
        LIMIT 25
        """,
        (match_id,),
    ).fetchall()
    conn.close()

    fallback: dict | None = None
    for row in rows:
        keys = set(row.keys())
        for tag in tags:
            q3_av_key = f"q3_available__{tag}"
            q4_av_key = f"q4_available__{tag}"
            if q3_av_key not in keys or q4_av_key not in keys:
                continue

            q3_av = int(row[q3_av_key] or 0)
            q4_av = int(row[q4_av_key] or 0)

            candidate = {
                "result_tag": tag,
                "event_date": row["event_date"],
                "updated_at": row["updated_at"],
                "q3": {
                    "available": bool(q3_av),
                    "pick": _row_value(row, f"q3_pick__{tag}"),
                    "signal": _row_value(row, f"q3_signal__{tag}"),
                    "outcome": _row_value(row, f"q3_outcome__{tag}"),
                    "confidence": _row_value(row, f"q3_confidence__{tag}"),
                    "threshold_lean": _row_value(row, f"q3_threshold_lean__{tag}"),
                    "threshold_bet": _row_value(row, f"q3_threshold_bet__{tag}"),
                },
                "q4": {
                    "available": bool(q4_av),
                    "pick": _row_value(row, f"q4_pick__{tag}"),
                    "signal": _row_value(row, f"q4_signal__{tag}"),
                    "outcome": _row_value(row, f"q4_outcome__{tag}"),
                    "confidence": _row_value(row, f"q4_confidence__{tag}"),
                    "threshold_lean": _row_value(row, f"q4_threshold_lean__{tag}"),
                    "threshold_bet": _row_value(row, f"q4_threshold_bet__{tag}"),
                },
            }

            if q3_av or q4_av:
                return candidate
            if fallback is None:
                fallback = candidate

    return fallback


def _friendly_reason(reason: str) -> str:
    mapping = {
        "match_not_found": "match no encontrado en base/scraper",
        "missing_q3_score": "Q3 aun no disponible",
        "missing_q4_score": "Q4 aun no disponible",
        "missing_graph_points": "sin graph_points suficientes",
        "missing_play_by_play": "sin play-by-play suficiente",
        "no_best_model_entry": "sin modelo seleccionado en comparacion",
        "v3_q3_available_from_minute_24": "Q3 disponible desde minuto 24",
        "v3_q4_available_from_minute_24": "Q4 disponible desde minuto 24",
        "insufficient_graph_or_pbp_coverage": "cobertura de datos insuficiente",
        "match_too_volatile_for_current_signal": "bloqueado por alta volatilidad",
        "confidence_below_minimum_edge": "confianza por debajo del minimo",
        "passed_all_gates": "paso todos los filtros",
    }
    return mapping.get(reason, reason or "unavailable")


def _enrich_prediction_row_from_infer(
    pred_row: dict | None,
    infer_result: dict,
) -> dict | None:
    if pred_row is None:
        return None

    predictions = infer_result.get("predictions", {}) if isinstance(infer_result, dict) else {}
    minute_estimate = (
        (infer_result.get("match") or {}).get("minute_estimate")
        if isinstance(infer_result, dict)
        else None
    )

    for target in ("q3", "q4"):
        pred = pred_row.get(target)
        src = predictions.get(target)
        if not isinstance(pred, dict) or not isinstance(src, dict):
            continue
        pred["available"] = bool(src.get("available", pred.get("available")))
        pred["pick"] = src.get("predicted_winner") or pred.get("pick")
        pred["signal"] = src.get("final_recommendation") or src.get("bet_signal") or pred.get("signal")
        pred["outcome"] = src.get("result") or pred.get("outcome")
        pred["confidence"] = src.get("confidence", pred.get("confidence"))
        pred["p_home_win"] = src.get("p_home_win", pred.get("p_home_win"))
        pred["p_away_win"] = src.get("p_away_win", pred.get("p_away_win"))
        pred["threshold_lean"] = src.get("threshold_lean", pred.get("threshold_lean"))
        pred["threshold_bet"] = src.get("threshold_bet", pred.get("threshold_bet"))
        pred["reason"] = src.get("reason") or src.get("gate_reason")
        pred["decision_gate"] = src.get("decision_gate")
        pred["gate_reason"] = src.get("gate_reason")
        pred["volatility_index"] = src.get("volatility_index")
        pred["volatility_swings"] = src.get("volatility_swings")
        pred["volatility_lead_changes"] = src.get("volatility_lead_changes")
        pred["gp_count"] = src.get("gate_gp_count")
        pred["pbp_count"] = src.get("gate_pbp_count")
        pred["minute_estimate"] = minute_estimate

    return pred_row


def _needs_unavailable_reason(pred_row: dict | None) -> bool:
    if not isinstance(pred_row, dict):
        return False
    for target in ("q3", "q4"):
        pred = pred_row.get(target, {})
        if isinstance(pred, dict) and (not pred.get("available")) and (not pred.get("reason")):
            return True
    return False


def _missing_prediction_confidence(pred_row: dict | None) -> bool:
    if not isinstance(pred_row, dict):
        return True
    for target in ("q3", "q4"):
        pred = pred_row.get(target, {})
        if not isinstance(pred, dict) or not pred.get("available"):
            continue
        if pred.get("confidence") is None:
            return True
    return False


def _compute_and_store_predictions(match_id: str, data: dict) -> dict | None:
    infer_mod = importlib.import_module("training.infer_match")
    infer_mod.scraper_mod.fetch_event_snapshot = lambda _mid: None
    
    logger.info(f"[COMPUTE_PRED] Starting predictions for {match_id}")
    gp_count = len(data.get("graph_points", []))
    pbp_count = sum(len(q) for q in data.get("play_by_play", {}).values())
    quarters = data.get("score", {}).get("quarters", {})
    logger.info(
        f"[COMPUTE_PRED] Data: score={data.get('score', {}).get('home')}-{data.get('score', {}).get('away')} "
        f"gp={gp_count} pbp={pbp_count} quarters={list(quarters.keys())}"
    )
    
    result = infer_mod.run_inference(
        match_id=match_id,
        metric="f1",
        fetch_missing=False,
        force_version="hybrid",
    )
    _log_raw_inference_result(match_id, source="compute", infer_result=result)
    
    logger.info(f"[COMPUTE_PRED] Inference result ok={result.get('ok')}")
    if result.get("ok"):
        preds = result.get("predictions", {})
        logger.info(
            f"[COMPUTE_PRED] Q3 available={preds.get('q3', {}).get('available')} "
            f"| Q4 available={preds.get('q4', {}).get('available')}"
        )
    
    if not result.get("ok"):
        logger.warning(f"[COMPUTE_PRED] Inference failed for {match_id}")
        return None

    q3h, q3a = _quarter_score(data, "Q3")
    q4h, q4a = _quarter_score(data, "Q4")
    event_date = str(data.get("match", {}).get("date", "") or "")
    if not event_date:
        event_date = datetime.utcnow().date().isoformat()

    conn = _open_conn()
    db_mod.save_eval_match_result(
        conn,
        event_date=event_date,
        match_id=match_id,
        home_team=str(data.get("match", {}).get("home_team", "")),
        away_team=str(data.get("match", {}).get("away_team", "")),
        q3_home_score=q3h,
        q3_away_score=q3a,
        q4_home_score=q4h,
        q4_away_score=q4a,
        result_tag="bot_hybrid_f1",
        predictions=result.get("predictions", {}),
    )
    conn.close()
    logger.info(f"[COMPUTE_PRED] Saved to database for {match_id}")
    base_row = _read_prediction_row(match_id)
    return _enrich_prediction_row_from_infer(base_row, result)


def _get_or_compute_predictions(match_id: str, data: dict) -> dict | None:
    from_db = _read_prediction_row(match_id)
    if (
        from_db is not None
        and not _needs_unavailable_reason(from_db)
        and not _missing_prediction_confidence(from_db)
    ):
        # If the match is finished and outcomes are still pending, recompute to resolve them
        is_ft = str((data or {}).get("match", {}).get("status_type", "") or "").lower() == "finished"
        if is_ft:
            q3_out = str((from_db.get("q3") or {}).get("outcome") or "").lower()
            q4_out = str((from_db.get("q4") or {}).get("outcome") or "").lower()
            has_pending = q3_out in ("pending", "") or q4_out in ("pending", "")
            if has_pending:
                try:
                    refreshed = _compute_and_store_predictions(match_id, data)
                    if refreshed is not None:
                        return refreshed
                except Exception:
                    pass
        return from_db
    if from_db is not None:
        try:
            infer_mod = importlib.import_module("training.infer_match")
            infer_mod.scraper_mod.fetch_event_snapshot = lambda _mid: None
            infer_result = infer_mod.run_inference(
                match_id=match_id,
                metric="f1",
                fetch_missing=False,
                force_version="hybrid",
            )
            _log_raw_inference_result(match_id, source="enrich-from-db", infer_result=infer_result)
            return _enrich_prediction_row_from_infer(from_db, infer_result)
        except Exception:
            return from_db
    try:
        return _compute_and_store_predictions(match_id, data)
    except Exception:
        return None


def _refresh_predictions(match_id: str, data: dict) -> dict | None:
    logger.info(f"[REFRESH_PRED] Starting for match {match_id}")
    try:
        result = _compute_and_store_predictions(match_id, data)
        logger.info(f"[REFRESH_PRED] Success: computed predictions for {match_id}")
        return result
    except Exception as exc:
        logger.warning(f"[REFRESH_PRED] Compute failed for {match_id}: {exc}. Falling back to cached row.")
        fallback = _read_prediction_row(match_id)
        logger.info(f"[REFRESH_PRED] Fallback result: available={fallback.get('available') if fallback else None}")
        return fallback


def _refresh_match_data(match_id: str) -> dict | None:
    logger.info(f"[REFRESH_DATA] Starting for match {match_id}")
    try:
        logger.info(f"[REFRESH_DATA] Calling fetch_match_by_id({match_id}) from scraper")
        fresh = scraper_mod.fetch_match_by_id(match_id)
        if not fresh:
            logger.error(f"[REFRESH_DATA] Scraper returned None for {match_id}")
            return None
        
        home = fresh.get("match", {}).get("home_team", "?")
        away = fresh.get("match", {}).get("away_team", "?")
        score_home = fresh.get("score", {}).get("home", "?")
        score_away = fresh.get("score", {}).get("away", "?")
        gp_count = len(fresh.get("graph_points", []))
        pbp_count = sum(len(q) for q in fresh.get("play_by_play", {}).values())
        logger.info(
            f"[REFRESH_DATA] Fetched {home} vs {away} {score_home}-{score_away} | "
            f"graph_points={gp_count} play_by_play={pbp_count}"
        )
        
        logger.info(f"[REFRESH_DATA] Saving to database")
        conn = _open_conn()
        db_mod.save_match(conn, match_id, fresh)
        conn.close()
        logger.info(f"[REFRESH_DATA] Saved match {match_id} to database")
        
        return fresh
    except Exception as exc:
        logger.error(f"[REFRESH_DATA] Error during refresh for {match_id}: {exc}", exc_info=True)
        return None


def _prediction_text(pred_row: dict | None, data: dict | None = None) -> str:
    if pred_row is None:
        return (
            "Predicciones:\n"
            "No disponibles (no se pudieron calcular)."
        )

    rendered_row = pred_row
    if isinstance(rendered_row, dict):
        rendered_row = {
            **rendered_row,
            "q3": dict(rendered_row.get("q3", {})),
            "q4": dict(rendered_row.get("q4", {})),
        }

    home_team_name = "home"
    away_team_name = "away"
    q4_waiting_preverdict = False
    if isinstance(data, dict):
        match_info_for_name = data.get("match", {})
        home_team_name = str(match_info_for_name.get("home_team") or "home")
        away_team_name = str(match_info_for_name.get("away_team") or "away")

    if isinstance(rendered_row, dict) and isinstance(data, dict):
        match_info = data.get("match", {})
        status_type = str(match_info.get("status_type", "") or "")
        status_description = str(match_info.get("status_description", "") or "")
        minute_est = None
        graph_points = data.get("graph_points", [])
        if graph_points:
            try:
                minute_est = int(graph_points[-1].get("minute", 0))
            except (TypeError, ValueError):
                minute_est = None

        q4_pred = rendered_row.get("q4", {})
        if isinstance(q4_pred, dict) and q4_pred.get("available") and status_type != "finished":
            q4_pred["outcome"] = "pending"

        q3_pred = rendered_row.get("q3", {})
        q4_score = (data.get("score", {}).get("quarters", {}) or {}).get("Q4")
        q4_started = bool(q4_score) or ("Q4" in status_description)
        if minute_est is not None and minute_est >= 36:
            q4_started = True
        if status_type != "finished" and not q4_started:
            q4_waiting_preverdict = True
        if isinstance(q3_pred, dict) and q3_pred.get("available") and status_type != "finished" and not q4_started:
            q3_pred["outcome"] = "pending"

        # Resolve Q3 outcome from actual scores once Q3 is over
        if isinstance(q3_pred, dict) and q3_pred.get("available"):
            q3_done = status_type == "finished" or q4_started
            if q3_done:
                _q3_sc = (data.get("score", {}).get("quarters", {}) or {}).get("Q3", {})
                _q3h = _safe_int((_q3_sc or {}).get("home"))
                _q3a = _safe_int((_q3_sc or {}).get("away"))
                if _q3h is not None and _q3a is not None:
                    _q3_pick = str(q3_pred.get("pick") or "").lower()
                    _q3_winner = _winner_from_scores(_q3h, _q3a)
                    if _q3_winner == "push":
                        q3_pred["outcome"] = "push"
                    elif _q3_winner in ("home", "away") and _q3_pick in ("home", "away"):
                        q3_pred["outcome"] = "hit" if _q3_winner == _q3_pick else "miss"

        # Resolve Q4 outcome from actual scores once game is finished
        if isinstance(q4_pred, dict) and q4_pred.get("available") and status_type == "finished":
            _q4_sc = (data.get("score", {}).get("quarters", {}) or {}).get("Q4", {})
            _q4h = _safe_int((_q4_sc or {}).get("home"))
            _q4a = _safe_int((_q4_sc or {}).get("away"))
            if _q4h is not None and _q4a is not None:
                _q4_pick = str(q4_pred.get("pick") or "").lower()
                _q4_winner = _winner_from_scores(_q4h, _q4a)
                if _q4_winner == "push":
                    q4_pred["outcome"] = "push"
                elif _q4_winner in ("home", "away") and _q4_pick in ("home", "away"):
                    q4_pred["outcome"] = "hit" if _q4_winner == _q4_pick else "miss"

    def _emoji(outcome: str) -> str:
        e = _outcome_emoji(outcome)
        return e if e else "⏳"

    def _confidence_pct(pred: dict, pick_norm: str = "") -> str:
        value = pred.get("confidence")
        if value is None:
            return ""
        try:
            pct = round(float(value) * 100)
        except (TypeError, ValueError):
            return ""
        bet_thr = pred.get("threshold_bet")
        try:
            bet_thr_pct = round(float(bet_thr) * 100) if bet_thr is not None else None
        except (TypeError, ValueError):
            bet_thr_pct = None
        p_home = pred.get("p_home_win")
        p_away = pred.get("p_away_win")
        prob_pct = None
        try:
            if pick_norm == "home" and p_home is not None:
                prob_pct = round(float(p_home) * 100)
            elif pick_norm == "away" and p_away is not None:
                prob_pct = round(float(p_away) * 100)
        except (TypeError, ValueError):
            prob_pct = None

        parts = [f"Score {pct}%"]
        if prob_pct is not None:
            parts.append(f"Probabilidad {prob_pct}%")
        return ", ".join(parts)

    def _line(label: str, pred: dict, waiting_preverdict: bool = False) -> str:
        if label == "Q4" and waiting_preverdict:
            return "Q4: Esperando datos Q3"
        if not pred.get("available"):
            reason_code = str(pred.get("reason") or pred.get("gate_reason") or pred.get("outcome") or "unavailable")
            reason_code_norm = reason_code.strip().lower().replace(" ", "_")
            reason_code_simple = reason_code.strip().lower().replace(" ", "_").replace("/", "_")
            if label == "Q4" and reason_code_norm == "missing_q3_score":
                return "Q4: Esperando datos Q3"
            if reason_code_simple == "missing_q1_q2_scores":
                return f"{label}: Esperando datos Q1|Q2"
            reason_txt = _friendly_reason(reason_code)
            minute_est = pred.get("minute_estimate")
            gp_count = pred.get("gp_count")
            pbp_count = pred.get("pbp_count")
            extra = []
            if minute_est is not None:
                extra.append(f"minuto_actual={minute_est}")
            if gp_count is not None and pbp_count is not None:
                extra.append(
                    "datos: "
                    f"grafica={gp_count} eventos, "
                    f"play-by-play={pbp_count} eventos"
                )
            extra_txt = f" | {' '.join(extra)}" if extra else ""
            return f"{label}: no disponible ({reason_txt}){extra_txt}"
        pick = str(pred.get("pick") or "-")
        pick_side_emoji = ""
        pick_norm = pick.strip().lower()
        if pick_norm == "home":
            pick = home_team_name
            pick_side_emoji = "🏠"
        elif pick_norm == "away":
            pick = away_team_name
            pick_side_emoji = "🛫"
        signal = pred.get("signal") or "-"
        signal_up = str(signal).upper()
        signal_map = {
            "BET": "APUESTA",
            "LEAN": "NO APOSTAR",
            "NO BET": "NO APOSTAR",
            "NO_BET": "NO APOSTAR",
        }
        signal_txt = signal_map.get(signal_up, signal_up)
        outcome = str(pred.get("outcome") or "pending")
        confidence_txt = _confidence_pct(pred, pick_norm)
        pick_txt = f"{pick_side_emoji} {pick}" if pick_side_emoji else pick
        line_emoji = "🚫" if signal_up in {"LEAN", "NO BET", "NO_BET"} else _emoji(outcome)
        if signal_up in {"LEAN", "NO BET", "NO_BET"}:
            first_line = f"{line_emoji}{label}: {signal_txt}"
        else:
            first_line = f"{line_emoji}{label}: {signal_txt} {pick_txt}"
        trend_txt = ""
        if signal_up in {"LEAN", "NO BET", "NO_BET"}:
            if pick_txt != "-":
                trend_txt = f"Tendencia {pick_txt}"
        gate_reason_norm = str(pred.get("gate_reason") or pred.get("reason") or "").strip().lower()
        decision_gate_norm = str(pred.get("decision_gate") or "").strip().upper()
        volatility_txt = ""
        if (
            gate_reason_norm == "match_too_volatile_for_current_signal"
            or decision_gate_norm == "BLOCK_HIGH_VOLATILITY"
        ):
            volatility_txt = "⚠️ Alta volatilidad"
        extra_lines = [line for line in [trend_txt, volatility_txt, confidence_txt] if line]
        if extra_lines:
            return f"{first_line}\n" + "\n".join(extra_lines)
        return first_line

    return "\n".join(
        [
            "Predicciones:",
            _line("Q3", rendered_row.get("q3", {})),
            _line("Q4", rendered_row.get("q4", {}), waiting_preverdict=q4_waiting_preverdict),
        ]
    )


def _build_graph_image(match_id: str, data: dict) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "pulpa_match_graphs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"graph_{match_id}.png"
    ml_mod.plot_graph(data, str(out_path))
    return out_path


async def _send_match_graph(
    app: Application,
    chat_id: int,
    match_id: str,
    data: dict,
) -> None:
    try:
        image_path = _build_graph_image(match_id, data)
    except Exception as exc:
        await app.bot.send_message(
            chat_id=chat_id,
            text=(
                f"No pude generar grafica para {match_id}: {exc}"
            ),
        )
        return

    try:
        with image_path.open("rb") as f:
            await app.bot.send_photo(
                chat_id=chat_id,
                photo=f,
                caption=f"Grafica de presion match {match_id}",
            )
    finally:
        try:
            image_path.unlink()
        except OSError:
            pass


def _detail_text(match_id: str, data: dict, pred_row: dict | None, following: bool = False) -> str:
    detail = _match_detail_text(match_id, data) + "\n\n" + _prediction_text(
        pred_row,
        data,
    )
    if following:
        detail = "🔴 LIVE SEGUIMIENTO\n\n" + detail
    return detail


def _refresh_waiting_text(match_id: str, data: dict | None) -> str:
    if not data:
        return "Actualizando..."

    m = data.get("match", {})
    utc_date = str(m.get("date", "") or "")
    utc_time = str(m.get("time", "") or "")
    local_dt = _to_local_datetime(utc_date, utc_time)
    if local_dt is not None:
        fecha_line = (
            f"Fecha: {local_dt.strftime('%Y-%m-%d %H:%M')} UTC{UTC_OFFSET_HOURS:+d} "
            f"(UTC: {utc_date} {utc_time})"
        )
    else:
        fecha_line = f"Fecha: {utc_date} {utc_time} UTC"

    minute_est = None
    graph_points = data.get("graph_points", [])
    if graph_points:
        try:
            minute_est = int(graph_points[-1].get("minute", 0))
        except (TypeError, ValueError):
            minute_est = None
    minute_suffix = f" | Min: {minute_est}" if minute_est is not None else ""

    text = "\n".join(
        [
            f"Match ID: {match_id}{minute_suffix}",
            f"{m.get('home_team', '')} vs {m.get('away_team', '')}",
            fecha_line,
            f"Liga: {m.get('league', '')}",
            "",
            "Actualizando...",
        ]
    )
    if len(text) > 1000:
        return text[:997] + "..."
    return text


async def _set_waiting_state(update: Update, text: str = "Espere...") -> None:
    query = update.callback_query
    if not query:
        return

    message = query.message
    try:
        if message and getattr(message, "photo", None):
            await query.edit_message_caption(caption=text)
            return

        await query.edit_message_text(text=text)
    except Exception as exc:
        if _is_message_not_modified_error(exc):
            print("[callback] waiting-state edit skipped (message not modified)", flush=True)
            return
        if isinstance(exc, (TimedOut, NetworkError)):
            print(
                f"[callback] waiting-state edit timeout/network error: {type(exc).__name__}",
                flush=True,
            )
            return
        if isinstance(exc, BadRequest):
            text_exc = str(exc)
            if (
                "Message to edit not found" in text_exc
                or "message can't be edited" in text_exc
                or "chat not found" in text_exc
            ):
                print(
                    f"[callback] waiting-state edit skipped (bad request): {text_exc}",
                    flush=True,
                )
                return
        raise


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    if isinstance(err, (TimedOut, NetworkError)):
        logger.warning("[telegram] network timeout error: %s", type(err).__name__)
        return
    logger.exception("[telegram] unhandled exception while processing update")


async def _replace_callback_message(
    update: Update,
    text: str,
    reply_markup: InlineKeyboardMarkup,
    parse_mode: str | None = None,
) -> None:
    query = update.callback_query
    if not query:
        return

    message = query.message
    if message and getattr(message, "photo", None):
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is None:
            return
        await query.get_bot().send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
        )
        try:
            await message.delete()
        except Exception:
            pass
        return

    await query.edit_message_text(
        text=text,
        reply_markup=reply_markup,
        parse_mode=parse_mode,
    )


async def _send_detail_message(
    update: Update,
    app: Application,
    match_id: str,
    data: dict,
    pred_row: dict | None,
    event_date: str | None,
    page: int,
) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    detail_text = _detail_text(match_id, data, pred_row)
    keyboard = _detail_keyboard(
        match_id,
        event_date=event_date,
        page=page,
        match_data=data,
        chat_id=chat_id,
    )

    image_path: Path | None = None
    try:
        image_path = _build_graph_image(match_id, data)
    except Exception:
        image_path = None

    if image_path is not None and image_path.exists():
        try:
            with image_path.open("rb") as f:
                await app.bot.send_photo(
                    chat_id=chat_id,
                    photo=f,
                    caption=detail_text,
                    reply_markup=keyboard,
                )
        finally:
            try:
                image_path.unlink()
            except OSError:
                pass
        return

    await app.bot.send_message(
        chat_id=chat_id,
        text=detail_text,
        reply_markup=keyboard,
    )


def _detail_token(event_date: str | None) -> str:
    return event_date if event_date else "_"


def _normalize_sofascore_slug(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "unknown"
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text or "unknown"


def _sofascore_match_url(match_id: str, match_data: dict | None = None) -> str:
    event_slug = "unknown"
    custom_id = ""
    home_slug = "unknown"
    away_slug = "unknown"

    if match_data and "match" in match_data:
        match_info = match_data["match"]
        event_slug = _normalize_sofascore_slug(match_info.get("event_slug"))
        custom_id = str(match_info.get("custom_id") or "").strip()
        home_slug = _normalize_sofascore_slug(match_info.get("home_slug"))
        away_slug = _normalize_sofascore_slug(match_info.get("away_slug"))
        if home_slug == "unknown":
            home_slug = _normalize_sofascore_slug(match_info.get("home_team"))
        if away_slug == "unknown":
            away_slug = _normalize_sofascore_slug(match_info.get("away_team"))

    if event_slug != "unknown" and custom_id:
        return (
            "https://www.sofascore.com/basketball/match/"
            f"{event_slug}/{custom_id}#id:{match_id}"
        )

    return (
        "https://www.sofascore.com/basketball/match/"
        f"{home_slug}/{away_slug}#id:{match_id}"
    )


def _detail_keyboard(
    match_id: str,
    event_date: str | None = None,
    page: int = 0,
    match_data: dict | None = None,
    chat_id: int | None = None,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    token = _detail_token(event_date)
    rows.append(
        [
            InlineKeyboardButton(
                "Ver SofaScore",
                url=_sofascore_match_url(match_id, match_data),
            )
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                "Refresh datos + pred",
                callback_data=f"refresh:all:{match_id}:{token}:{page}",
            ),
            InlineKeyboardButton(
                "Refresh pred",
                callback_data=f"refresh:pred:{match_id}:{token}:{page}",
            ),
        ]
    )
    _ftb = _follow_toggle_button(chat_id, match_id, token, page, match_data)
    if _ftb is not None:
        rows.append([_ftb])
    rows.append(
        [
            InlineKeyboardButton(
                "Borrar match de la base",
                callback_data=f"delmatch:confirm:{match_id}:{token}:{page}",
            )
        ]
    )
    if event_date:
        rows.append(
            [
                InlineKeyboardButton(
                    "Volver al dia",
                    callback_data=f"date:{event_date}:{page}",
                )
            ]
        )
    rows.append([InlineKeyboardButton("Menu principal", callback_data="nav:main")])
    return InlineKeyboardMarkup(rows)


async def _render_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data[AWAITING_MATCH_ID_KEY] = False
    context.user_data[AWAITING_FETCH_DATE_KEY] = False
    chat_id = update.effective_chat.id if update.effective_chat else None
    job = DATE_INGEST_JOBS.get(chat_id) if chat_id is not None else None
    text = "Menu principal\n\nSelecciona una opcion:"
    if job:
        text += (
            "\n\nDescarga en curso en segundo plano. "
            "Usa 'Status descarga' para ver el avance."
        )

    if update.callback_query:
        await _replace_callback_message(
            update,
            text=text,
            reply_markup=_main_menu_keyboard(chat_id),
        )
    elif update.message:
        await update.message.reply_text(
            text=text,
            reply_markup=_main_menu_keyboard(chat_id),
        )


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_main_menu(update, context)


async def _render_dates(update: Update, page: int) -> None:
    rows = _fetch_date_summaries()
    if not rows:
        await _replace_callback_message(
            update,
            text="No hay matches en la base.",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
            ),
        )
        return

    pred_stats = _fetch_dates_pred_stats()
    await _replace_callback_message(
        update,
        text="Fechas disponibles:",
        reply_markup=_dates_keyboard(rows, page, pred_stats),
    )


async def _render_matches_for_date(update: Update, event_date: str, page: int) -> None:
    rows = _fetch_matches_for_date(event_date)
    if not rows:
        await _replace_callback_message(
            update,
            text=f"No hay matches para {event_date}.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Volver a fechas", callback_data="dates:0")],
                    [InlineKeyboardButton("Menu principal", callback_data="nav:main")],
                ]
            ),
        )
        return

    pred_map = _fetch_date_pred_outcomes(event_date)
    stats = _pred_stats_text(pred_map, len(rows))
    header = _event_date_title_es(event_date, len(rows))
    if stats:
        header += f"\n<pre>{html.escape(stats)}</pre>"
    await _replace_callback_message(
        update,
        text=header,
        reply_markup=_matches_keyboard(rows, event_date, page, pred_map),
        parse_mode="HTML",
    )


async def _render_match_detail(
    update: Update,
    app: Application,
    match_id: str,
    event_date: str | None,
    page: int,
    *,
    pred_row: dict | None = None,
    send_graph: bool = True,
) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else None
    await _set_waiting_state(update)
    data = _get_match_detail(match_id)
    if not data:
        await _replace_callback_message(
            update,
            text=f"No se encontro el match {match_id}.",
            reply_markup=_detail_keyboard(
                match_id,
                event_date=event_date,
                page=page,
                chat_id=chat_id,
            ),
        )
        return

    if pred_row is None:
        pred_row = _get_or_compute_predictions(match_id, data)
    await _send_detail_message(
        update,
        app,
        match_id,
        data,
        pred_row,
        event_date,
        page,
    )
    if update.callback_query and update.callback_query.message:
        try:
            await update.callback_query.message.delete()
        except Exception:
            pass


async def _refresh_detail(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    mode: str,
    match_id: str,
    event_date: str | None,
    page: int,
) -> None:
    query = update.callback_query
    if not query:
        return
    chat_id = update.effective_chat.id if update.effective_chat else None

    logger.info(f"[REFRESH_DETAIL] Callback for mode={mode} match={match_id} event_date={event_date}")
    cached = _get_match_detail(match_id)
    await _set_waiting_state(update, text=_refresh_waiting_text(match_id, cached))

    if mode == "all":
        logger.info(f"[REFRESH_DETAIL] Mode='all': refreshing both data and predictions")
        try:
            logger.info(f"[REFRESH_DETAIL] Running _refresh_match_data in thread")
            data = await asyncio.to_thread(_refresh_match_data, match_id)
            if data is None:
                logger.error(f"[REFRESH_DETAIL] Data refresh returned None for {match_id}")
                await _replace_callback_message(
                    update,
                    text=f"No pude refrescar datos de {match_id}: datos nulos del scraper",
                    reply_markup=_detail_keyboard(
                        match_id,
                        event_date=event_date,
                        page=page,
                        chat_id=chat_id,
                    ),
                )
                return
            logger.info(f"[REFRESH_DETAIL] Data refresh succeeded, now computing predictions")
        except Exception as exc:
            logger.error(f"[REFRESH_DETAIL] Exception in _refresh_match_data: {exc}", exc_info=True)
            await _replace_callback_message(
                update,
                text=f"No pude refrescar datos de {match_id}: {exc}",
                reply_markup=_detail_keyboard(
                    match_id,
                    event_date=event_date,
                    page=page,
                    chat_id=chat_id,
                ),
            )
            return
        pred_row = _refresh_predictions(match_id, data)
        logger.info(f"[REFRESH_DETAIL] Rendering match detail after refresh")
        await _render_match_detail(
            update,
            context.application,
            match_id,
            event_date,
            page,
            pred_row=pred_row,
            send_graph=True,
        )
        return

    logger.info(f"[REFRESH_DETAIL] Mode='pred': refreshing predictions only")
    logger.info(f"[REFRESH_DETAIL] Loading from cache for {match_id}")
    data = _get_match_detail(match_id)
    if not data:
        logger.error(f"[REFRESH_DETAIL] No cached data found for {match_id}")
        await _replace_callback_message(
            update,
            text=f"No se encontro el match {match_id}.",
            reply_markup=_detail_keyboard(
                match_id,
                event_date=event_date,
                page=page,
                chat_id=chat_id,
            ),
        )
        return

    logger.info(f"[REFRESH_DETAIL] Computing predictions from cached data")
    pred_row = _refresh_predictions(match_id, data)
    await _render_match_detail(
        update,
        context.application,
        match_id,
        event_date,
        page,
        pred_row=pred_row,
        send_graph=True,
    )


def _tail_lines(text: str, max_lines: int = 16, max_chars: int = 3000) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail or "(sin salida)"


def _safe_pct_from_csv(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _read_train_v4_effectiveness_summary() -> str:
    lines: list[str] = []
    for target in ("q3", "q4"):
        csv_path = MODEL_OUTPUTS_V4_DIR / f"{target}_metrics.csv"
        if not csv_path.exists():
            lines.append(f"- {target.upper()}: metrics no encontrado")
            continue

        rows: list[dict[str, str]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            lines.append(f"- {target.upper()}: sin filas de metrics")
            continue

        best = max(rows, key=lambda r: float(r.get("f1") or 0.0))
        ensemble = next(
            (r for r in rows if str(r.get("model") or "") == "ensemble_avg_prob"),
            None,
        )
        best_txt = (
            f"best_f1 model={best.get('model')} "
            f"f1={_safe_pct_from_csv(best.get('f1'))} "
            f"acc={_safe_pct_from_csv(best.get('accuracy'))}"
        )
        if ensemble is not None:
            ens_txt = (
                f"ensemble f1={_safe_pct_from_csv(ensemble.get('f1'))} "
                f"acc={_safe_pct_from_csv(ensemble.get('accuracy'))}"
            )
            lines.append(f"- {target.upper()}: {best_txt} | {ens_txt}")
        else:
            lines.append(f"- {target.upper()}: {best_txt}")

    return "\n".join(lines) if lines else "(sin metrics)"


def _clear_eval_results_table() -> int:
    conn = _open_conn()
    count_row = conn.execute("SELECT COUNT(*) AS n FROM eval_match_results").fetchone()
    before = int(count_row["n"] if count_row else 0)
    conn.execute("DELETE FROM eval_match_results")
    conn.commit()
    conn.close()
    return before


def _delete_match_cascade(match_id: str) -> dict[str, int]:
    conn = _open_conn()

    def _del(sql: str, params: tuple[object, ...]) -> int:
        cur = conn.execute(sql, params)
        affected = int(cur.rowcount or 0)
        return affected if affected >= 0 else 0

    counts = {
        "eval_match_results": _del(
            "DELETE FROM eval_match_results WHERE match_id = ?",
            (match_id,),
        ),
        "discovered_ft_matches": _del(
            "DELETE FROM discovered_ft_matches WHERE match_id = ?",
            (match_id,),
        ),
        "play_by_play": _del(
            "DELETE FROM play_by_play WHERE match_id = ?",
            (match_id,),
        ),
        "graph_points": _del(
            "DELETE FROM graph_points WHERE match_id = ?",
            (match_id,),
        ),
        "quarter_scores": _del(
            "DELETE FROM quarter_scores WHERE match_id = ?",
            (match_id,),
        ),
        "matches": _del(
            "DELETE FROM matches WHERE match_id = ?",
            (match_id,),
        ),
    }
    conn.commit()
    conn.close()
    return counts


def _list_candidate_match_ids_for_universe() -> list[str]:
    conn = _open_conn()
    rows = conn.execute(
        """
        SELECT match_id
        FROM matches
        ORDER BY date DESC, time DESC, match_id DESC
        """
    ).fetchall()
    conn.close()
    return [str(r["match_id"]) for r in rows if r["match_id"] is not None]


def _build_universe_stats_text() -> str:
    conn = _open_conn()
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"]
        finished = conn.execute(
            "SELECT COUNT(*) AS n FROM matches WHERE status_type='finished'"
        ).fetchone()["n"]
        with_q4 = conn.execute(
            """
            SELECT COUNT(DISTINCT match_id) AS n FROM quarter_scores
            WHERE quarter='Q4'
            """
        ).fetchone()["n"]
        full_qs = conn.execute(
            """
            SELECT COUNT(*) AS n FROM (
                SELECT match_id
                FROM quarter_scores
                WHERE quarter IN ('Q1','Q2','Q3','Q4') AND home IS NOT NULL AND away IS NOT NULL
                GROUP BY match_id
                HAVING COUNT(DISTINCT quarter) = 4
            )
            """
        ).fetchone()["n"]
        with_pbp = conn.execute(
            "SELECT COUNT(DISTINCT match_id) AS n FROM play_by_play"
        ).fetchone()["n"]
        with_gp = conn.execute(
            "SELECT COUNT(DISTINCT match_id) AS n FROM graph_points"
        ).fetchone()["n"]
        eval_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM eval_match_results"
        ).fetchone()["n"]
        date_range = conn.execute(
            "SELECT MIN(date) AS d_min, MAX(date) AS d_max FROM matches"
        ).fetchone()
        d_min = date_range["d_min"] or "?"
        d_max = date_range["d_max"] or "?"
    finally:
        conn.close()

    lines = [
        "Universo DB:",
        f"  Total matches: {total}",
        f"  Status finished: {finished}",
        f"  Con Q1-Q4 completo: {full_qs}",
        f"  Con play-by-play: {with_pbp}",
        f"  Con graph_points: {with_gp}",
        f"  Eval calculados: {eval_rows}",
        f"  Rango fechas: {d_min} a {d_max}",
    ]
    return "\n".join(lines)


def _build_model_stats_text() -> str:
    def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        head = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        sep = "-+-".join("-" * w for w in widths)
        out = [head, sep]
        for row in rows:
            out.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        return out

    # Model metrics from CSVs (v2 and v4)
    metrics_sections: list[str] = []
    version_dirs = [
        ("v2", MODEL_OUTPUTS_V2_DIR),
        ("v4", MODEL_OUTPUTS_V4_DIR),
    ]
    headers = ["Filtro", "F1", "ACC", "ROC", "Ntrain", "Ntest", "Ntotal"]
    for version_name, version_dir in version_dirs:
        metrics_sections.append(f"Modelos {version_name}:")
        for target in ("q3", "q4"):
            csv_path = version_dir / f"{target}_metrics.csv"
            if not csv_path.exists():
                metrics_sections.append(f"{target.upper()}: metrics no encontrado")
                continue

            parsed_rows: list[dict[str, str]] = []
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    parsed_rows.append(row)
            if not parsed_rows:
                metrics_sections.append(f"{target.upper()}: sin filas")
                continue

            table_rows: list[list[str]] = []
            for row in parsed_rows:
                table_rows.append(
                    [
                        str(row.get("model", "?")),
                        _safe_pct_from_csv(row.get("f1")),
                        _safe_pct_from_csv(row.get("accuracy")),
                        _safe_pct_from_csv(row.get("roc_auc")),
                        str(row.get("samples_train", "?")),
                        str(row.get("samples_test", "?")),
                        str(row.get("samples_total", "?")),
                    ]
                )

            metrics_sections.append(f"{target.upper()} (test):")
            metrics_sections.extend(_table(headers, table_rows))
            metrics_sections.append("")

    # Gate config
    gate_lines: list[str] = []
    gate_path = BASE_DIR / "training" / "model_outputs_v2" / "gate_config.json"
    if gate_path.exists():
        try:
            with gate_path.open("r", encoding="utf-8") as f:
                gcfg = json.load(f)
            gen_at = str(gcfg.get("generated_at_utc", "?"))[:19]
            thresholds = gcfg.get("thresholds", {})
            q3_thr = (thresholds.get("q3") or {}).get("default") or {}
            q4_thr = (thresholds.get("q4") or {}).get("36") or {}
            gate_lines.append(f"Gate calibrado: {gen_at}")
            if q3_thr:
                gate_lines.append(
                    f"  Q3: edge>={_safe_pct_from_csv(q3_thr.get('min_edge'))}"
                    f" vol<{q3_thr.get('volatility_block_at')}"
                    f" gp>={q3_thr.get('min_graph_points')}"
                    f" pbp>={q3_thr.get('min_pbp_events')}"
                )
            if q4_thr:
                gate_lines.append(
                    f"  Q4: edge>={_safe_pct_from_csv(q4_thr.get('min_edge'))}"
                    f" vol<{q4_thr.get('volatility_block_at')}"
                    f" gp>={q4_thr.get('min_graph_points')}"
                    f" pbp>={q4_thr.get('min_pbp_events')}"
                )
        except Exception:
            gate_lines.append("Gate config: error leyendo archivo")
    else:
        gate_lines.append("Gate config: no encontrado")

    sections = metrics_sections + gate_lines
    return "\n".join(sections)


def _train_status_text() -> str:
    running = bool(TRAIN_STATUS.get("running"))
    owner = TRAIN_STATUS.get("owner_chat_id")
    last_exit = TRAIN_STATUS.get("last_exit_code")
    last_finished = TRAIN_STATUS.get("last_finished_utc")
    last_error = str(TRAIN_STATUS.get("last_error_tail") or "")

    if running:
        task = str(TRAIN_STATUS.get("current_task") or "train-v4")
        done = int(TRAIN_STATUS.get("progress_done") or 0)
        total = int(TRAIN_STATUS.get("progress_total") or 0)
        progress_txt = f"\nprogreso={done}/{total}" if total > 0 else ""
        return (
            "Estado entrenamiento: RUNNING\n"
            f"owner_chat_id={owner}\n"
            f"tarea={task}{progress_txt}"
        )

    if last_exit is None:
        return "Estado entrenamiento: IDLE (sin ejecuciones previas)"

    base = (
        "Estado entrenamiento: IDLE\n"
        f"ultimo_exit={last_exit}\n"
        f"ultimo_fin_utc={last_finished}"
    )
    if last_exit != 0 and last_error:
        return base + "\n" + "ultimo_error_tail:\n" + last_error
    return base


def _is_train_allowed(chat_id: int | None) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    if chat_id is None:
        return False
    return chat_id in ALLOWED_CHAT_IDS


async def _stream_process_output(
    stream: asyncio.StreamReader | None,
    task_label: str,
    is_stderr: bool = False,
) -> str:
    if stream is None:
        return ""

    chunks: list[str] = []
    while True:
        raw = await stream.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            continue
        chunks.append(line)
        if is_stderr:
            logger.warning("[TRAIN_PIPE][%s][stderr] %s", task_label, line)
        else:
            logger.info("[TRAIN_PIPE][%s] %s", task_label, line)
    return "\n".join(chunks)


async def _run_train_pipeline(chat_id: int, app: Application) -> None:
    """Run full training pipeline: train-v2 → train-v4 → compare → calibrate."""
    pipeline_steps = [
        ("1/4 train-v2", ["training/model_cli.py", "train-v2"]),
        ("2/4 train-v4", ["training/model_cli.py", "train-v4"]),
        ("3/4 compare",  ["training/model_cli.py", "compare"]),
        ("4/4 calibrate", ["training/calibrate_gate.py"]),
    ]
    async with RETRAIN_LOCK:
        TRAIN_STATUS["running"] = True
        TRAIN_STATUS["owner_chat_id"] = chat_id
        TRAIN_STATUS["last_error_tail"] = ""
        TRAIN_STATUS["progress_done"] = 0
        TRAIN_STATUS["progress_total"] = len(pipeline_steps)

        failed_step: tuple[str, int, str] | None = None

        for i, (task_label, script_args) in enumerate(pipeline_steps, start=1):
            TRAIN_STATUS["current_task"] = task_label
            TRAIN_STATUS["progress_done"] = i - 1
            cmd_display = " ".join([sys.executable, "-u", *script_args])
            logger.info("[TRAIN_PIPE] start step=%s cmd=%s", task_label, cmd_display)
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                *script_args,
                cwd=str(BASE_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_task = asyncio.create_task(
                _stream_process_output(proc.stdout, task_label, is_stderr=False)
            )
            stderr_task = asyncio.create_task(
                _stream_process_output(proc.stderr, task_label, is_stderr=True)
            )
            stdout_text, stderr_text = await asyncio.gather(stdout_task, stderr_task)
            return_code = await proc.wait()
            logger.info(
                "[TRAIN_PIPE] end step=%s exit=%s",
                task_label,
                return_code,
            )

            if return_code != 0:
                failed_step = (task_label, return_code, stderr_text or stdout_text)
                break

        TRAIN_STATUS["progress_done"] = len(pipeline_steps)

        if failed_step is None:
            metrics_summary = _read_train_v4_effectiveness_summary()
            msg = (
                "Pipeline entrenamiento finalizado OK.\n"
                "Pasos: train-v2, train-v4, compare, calibrate.\n"
                "Efectividad final (test):\n"
                f"{metrics_summary}"
            )
            TRAIN_STATUS["last_error_tail"] = ""
            TRAIN_STATUS["last_exit_code"] = 0
        else:
            label, code, stderr_text = failed_step
            err_tail = _tail_lines(stderr_text)
            msg = (
                f"Pipeline fallo en: {label} (exit={code}).\n"
                "STDERR:\n"
                f"{err_tail}"
            )
            TRAIN_STATUS["last_error_tail"] = err_tail
            TRAIN_STATUS["last_exit_code"] = code

        TRAIN_STATUS["running"] = False
        TRAIN_STATUS["last_finished_utc"] = datetime.utcnow().isoformat(
            timespec="seconds"
        )
        TRAIN_STATUS["current_task"] = ""
        TRAIN_STATUS["progress_done"] = 0
        TRAIN_STATUS["progress_total"] = 0

        await app.bot.send_message(chat_id=chat_id, text=msg)


async def _run_recalc_universe(chat_id: int, app: Application) -> None:
    async with RETRAIN_LOCK:
        TRAIN_STATUS["running"] = True
        TRAIN_STATUS["owner_chat_id"] = chat_id
        TRAIN_STATUS["last_error_tail"] = ""
        TRAIN_STATUS["current_task"] = "recalculo-universo"
        TRAIN_STATUS["progress_done"] = 0
        TRAIN_STATUS["progress_total"] = 0

        try:
            match_ids = await asyncio.to_thread(_list_candidate_match_ids_for_universe)
            total = len(match_ids)
            TRAIN_STATUS["progress_total"] = total

            ok = 0
            fail = 0
            skipped_not_ft = 0

            for index, match_id in enumerate(match_ids, start=1):
                data = _get_match_detail(match_id)
                if not _is_ft_complete(data):
                    skipped_not_ft += 1
                    TRAIN_STATUS["progress_done"] = index
                    continue

                try:
                    result = await asyncio.to_thread(
                        _compute_and_store_predictions,
                        match_id,
                        data,
                    )
                    if result is None:
                        fail += 1
                    else:
                        ok += 1
                except Exception:
                    fail += 1

                TRAIN_STATUS["progress_done"] = index

            msg = (
                "Recalculo universo finalizado.\n"
                f"candidatos={total}\n"
                f"recalculados_ok={ok}\n"
                f"fallidos={fail}\n"
                f"omitidos_no_ft={skipped_not_ft}"
            )
            TRAIN_STATUS["last_error_tail"] = ""
            TRAIN_STATUS["last_exit_code"] = 0
        except Exception as exc:
            err_tail = _tail_lines(str(exc), max_lines=12, max_chars=1500)
            msg = (
                "Recalculo universo fallo.\n"
                f"error={err_tail}"
            )
            TRAIN_STATUS["last_error_tail"] = err_tail
            TRAIN_STATUS["last_exit_code"] = 1
        finally:
            TRAIN_STATUS["running"] = False
            TRAIN_STATUS["last_finished_utc"] = datetime.utcnow().isoformat(
                timespec="seconds"
            )
            TRAIN_STATUS["current_task"] = ""
            TRAIN_STATUS["progress_done"] = 0
            TRAIN_STATUS["progress_total"] = 0

        await app.bot.send_message(chat_id=chat_id, text=msg)


async def myid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _ = context
    chat_id = update.effective_chat.id if update.effective_chat else None
    user_id = update.effective_user.id if update.effective_user else None
    await update.effective_message.reply_text(
        "Tus IDs:\n"
        f"chat_id={chat_id}\n"
        f"user_id={user_id}"
    )


async def train_status_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    _ = context
    await update.effective_message.reply_text(_train_status_text())


async def _post_init(app: Application) -> None:
    await app.bot.set_my_commands(
        [
            BotCommand("start", "Abrir menu principal"),
            BotCommand("myid", "Ver chat_id y user_id"),
            BotCommand("trainstatus", "Estado del ultimo reentreno"),
        ]
    )
    await app.bot.set_chat_menu_button(menu_button=MenuButtonCommands())


async def _handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return

    await query.answer()
    data = query.data or ""

    if data == "noop":
        return

    if data == "nav:main":
        chat_id = update.effective_chat.id if update.effective_chat else None
        job = DATE_INGEST_JOBS.get(chat_id) if chat_id is not None else None
        if isinstance(job, dict):
            job["suppress_edits"] = True
        refresh_job = REFRESH_JOBS.get(chat_id) if chat_id is not None else None
        if isinstance(refresh_job, dict):
            refresh_job["suppress_edits"] = True
        await _render_main_menu(update, context)
        return

    if data == "menu:fetchdate:status":
        chat_id = update.effective_chat.id if update.effective_chat else None
        job = DATE_INGEST_JOBS.get(chat_id) if chat_id is not None else None
        if not isinstance(job, dict):
            await query.edit_message_text(
                text="No hay una descarga en curso.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return

        event_date = str(job.get("event_date") or "-")
        limit = job.get("limit")
        progress_state = job.get("progress_state")
        if not isinstance(progress_state, dict):
            progress_state = {"phase": "starting"}
            job["progress_state"] = progress_state

        job["suppress_edits"] = False
        job["force_next_edit"] = True
        text = _date_ingest_progress_text(event_date, limit, progress_state)
        await query.edit_message_text(
            text=text,
            reply_markup=_date_ingest_progress_keyboard(True),
        )
        return

    if data == "menu:refresh:status":
        chat_id = update.effective_chat.id if update.effective_chat else None
        job = REFRESH_JOBS.get(chat_id) if chat_id is not None else None
        if not isinstance(job, dict):
            await query.edit_message_text(
                text="No hay una descarga de pendientes en curso.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return

        job["suppress_edits"] = False
        job["force_next_edit"] = True
        text = _refresh_job_status_text(job)
        await query.edit_message_text(
            text=text,
            reply_markup=_refresh_job_progress_keyboard(),
        )
        return

    if data == "menu:follow:status":
        chat_id = update.effective_chat.id if update.effective_chat else None
        job = FOLLOW_JOBS.get(chat_id) if chat_id is not None else None
        if not isinstance(job, dict):
            await query.edit_message_text(
                text="No hay seguimiento activo.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return
        await query.edit_message_text(
            text=_follow_status_text(job),
            reply_markup=_follow_status_keyboard(job),
        )
        if query.message and isinstance(job, dict):
            job["status_message_id"] = query.message.message_id
            job["status_last_text"] = _follow_status_text(job)
        return

    if data == "menu:refresh7d":
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is None:
            await _replace_callback_message(
                update,
                text="No pude identificar el chat para ejecutar la descarga.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return
        if chat_id in REFRESH_JOBS:
            await query.edit_message_text(
                text="Ya hay una descarga de pendientes en curso.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Ver status", callback_data="menu:refresh:status")]]
                ),
            )
            return

        await _set_waiting_state(
            update,
            text="Descargando pendientes de los ultimos 7 dias... en background",
        )
        message = query.message
        message_id = message.message_id if message else None
        if message_id is None:
            return
        context.application.create_task(
            _run_refresh_7d_job(
                context.application,
                chat_id,
                message_id,
            )
        )
        return

    if data == "fetchdate:cancel":
        chat_id = update.effective_chat.id if update.effective_chat else None
        job = DATE_INGEST_JOBS.get(chat_id) if chat_id is not None else None
        if not job:
            await query.edit_message_text(
                text="No hay una ingesta de fecha en curso.",
                reply_markup=_date_ingest_progress_keyboard(False),
            )
            return
        cancel_event = job.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        await query.edit_message_text(
            text="Cancelacion solicitada. Esperando cierre del match en curso...",
            reply_markup=_date_ingest_progress_keyboard(False),
        )
        return

    if data == "menu:dates":
        await _render_dates(update, 0)
        return

    if data == "menu:live":
        context.user_data[AWAITING_FETCH_DATE_KEY] = False
        context.user_data[AWAITING_MATCH_ID_KEY] = False
        await _render_live(update, context, 0, refetch=True)
        return

    if data.startswith("dates:"):
        try:
            _, page_text = data.split(":", maxsplit=1)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_dates(update, 0)
            return
        await _render_dates(update, page)
        return

    if data.startswith("livepage:"):
        try:
            _, page_text = data.split(":", maxsplit=1)
            page = int(page_text)
        except (ValueError, TypeError):
            page = 0
        await _render_live(update, context, page, refetch=False)
        return

    if data == "menu:id":
        context.user_data[AWAITING_MATCH_ID_KEY] = True
        context.user_data[AWAITING_FETCH_DATE_KEY] = False
        await query.edit_message_text(
            text=(
                "Enviar Match ID numerico.\n"
                "Ejemplo: 14442355"
            ),
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
            ),
        )
        return

    if data == "menu:fetchdate":
        context.user_data[AWAITING_FETCH_DATE_KEY] = True
        context.user_data[AWAITING_MATCH_ID_KEY] = False
        await query.edit_message_text(
            text=(
                "Enviar fecha en formato YYYY-MM-DD.\n"
                "Opcional limite: YYYY-MM-DD 100\n"
                "Ejemplo: 2026-03-26 200"
            ),
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
            ),
        )
        return

    if data == "menu:train":
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not _is_train_allowed(chat_id):
            await query.edit_message_text(
                text=(
                    "No autorizado para reentrenar.\n"
                    f"Tu chat_id es {chat_id}.\n"
                    "Agregalo en TELEGRAM_ALLOWED_CHAT_IDS del .env"
                ),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return
        await query.edit_message_text(
            text=(
                "Re-entrenar base.\n"
                "Elige una opcion:"
            ),
            reply_markup=_train_submenu_keyboard(),
        )
        return

    if data == "train:status":
        await query.edit_message_text(
            text=_train_status_text(),
            reply_markup=_train_submenu_keyboard(),
        )
        return

    if data == "train:stats":
        try:
            model_txt = await asyncio.to_thread(_build_model_stats_text)
            universe_txt = await asyncio.to_thread(_build_universe_stats_text)
            stats_txt = model_txt + "\n\n" + universe_txt
            stats_render = f"<pre>{html.escape(stats_txt)}</pre>"
        except Exception as exc:
            stats_render = f"Error al leer stats: {exc}"
        await query.edit_message_text(
            text=stats_render,
            parse_mode="HTML",
            reply_markup=_train_submenu_keyboard(),
        )
        return

    if data == "train:run":
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not _is_train_allowed(chat_id):
            await query.edit_message_text(
                text=(
                    "No autorizado para reentrenar.\n"
                    f"Tu chat_id es {chat_id}.\n"
                    "Agregalo en TELEGRAM_ALLOWED_CHAT_IDS del .env"
                ),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return

        if RETRAIN_LOCK.locked():
            await query.edit_message_text(
                text="Ya hay un reentreno en ejecucion. Espera a que termine.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return

        await query.edit_message_text(
            text=(
                "Pipeline entrenamiento iniciado en background.\n"
                "Pasos: train-v2 → train-v4 → compare → calibrate.\n"
                "Te aviso al terminar."
            ),
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
            ),
        )
        safe_chat_id = int(chat_id) if chat_id is not None else 0
        context.application.create_task(
            _run_train_pipeline(safe_chat_id, context.application)
        )
        return

    if data == "train:clear:confirm":
        await query.edit_message_text(
            text=(
                "Vas a vaciar la tabla de eval_match_results.\n"
                "Esta accion elimina todos los resultados calculados.\n"
                "No se puede deshacer."
            ),
            reply_markup=_train_clear_confirm_keyboard(),
        )
        return

    if data == "train:clear:run":
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not _is_train_allowed(chat_id):
            await query.edit_message_text(
                text=(
                    "No autorizado.\n"
                    f"Tu chat_id es {chat_id}.\n"
                    "Agregalo en TELEGRAM_ALLOWED_CHAT_IDS del .env"
                ),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return
        try:
            deleted_rows = await asyncio.to_thread(_clear_eval_results_table)
            await query.edit_message_text(
                text=f"Tabla vaciada. Filas eliminadas: {deleted_rows}.",
                reply_markup=_train_submenu_keyboard(),
            )
        except Exception as exc:
            await query.edit_message_text(
                text=f"Error al vaciar tabla: {exc}",
                reply_markup=_train_submenu_keyboard(),
            )
        return

    if data == "train:recalc:confirm":
        await query.edit_message_text(
            text=(
                "Vas a recalcular predicciones del universo.\n"
                "El proceso corre en background y puede tardar bastante.\n"
                "Los resultados existentes se sobreescriben si el match ya tiene calculo."
            ),
            reply_markup=_train_recalc_confirm_keyboard(),
        )
        return

    if data == "train:recalc:run":
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not _is_train_allowed(chat_id):
            await query.edit_message_text(
                text=(
                    "No autorizado para reentrenar/recalcular.\n"
                    f"Tu chat_id es {chat_id}.\n"
                    "Agregalo en TELEGRAM_ALLOWED_CHAT_IDS del .env"
                ),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return

        if RETRAIN_LOCK.locked():
            await query.edit_message_text(
                text="Ya hay un proceso de entrenamiento/recalculo en ejecucion.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Estado entrenamiento", callback_data="train:status")]]
                ),
            )
            return

        await query.edit_message_text(
            text=(
                "Recalculo de universo iniciado en background.\n"
                "Puedes seguir usando el bot y revisar en Estado entrenamiento."
            ),
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
            ),
        )
        safe_chat_id = int(chat_id) if chat_id is not None else 0
        context.application.create_task(
            _run_recalc_universe(safe_chat_id, context.application)
        )
        return

    if data.startswith("date:"):
        try:
            _, event_date, page_text = data.split(":", maxsplit=2)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_dates(update, 0)
            return
        await _render_matches_for_date(update, event_date, page)
        return

    if data.startswith("refresh:date:"):
        event_date = data[len("refresh:date:"):]
        logger.info(f"[REFRESH_DATE] Callback received date={event_date}")
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is None:
            await _replace_callback_message(
                update,
                text="No pude identificar el chat para ejecutar la descarga.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Volver", callback_data=f"date:{event_date}:0")]]
                ),
            )
            return
        if chat_id in REFRESH_JOBS:
            await query.edit_message_text(
                text="Ya hay una descarga de pendientes en curso.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Ver status", callback_data="menu:refresh:status")]]
                ),
            )
            return

        await _set_waiting_state(
            update,
            text="Descargando pendientes del dia... en background",
        )
        message = query.message
        message_id = message.message_id if message else None
        if message_id is None:
            return
        context.application.create_task(
            _run_refresh_date_job(
                context.application,
                chat_id,
                message_id,
                event_date,
            )
        )
        return

    if data.startswith("refresh:"):
        try:
            _, mode, match_id, event_token, page_text = data.split(
                ":",
                maxsplit=4,
            )
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_dates(update, 0)
            return
        event_date = None if event_token == "_" else event_token
        await _refresh_detail(
            update,
            context,
            mode,
            match_id,
            event_date,
            page,
        )
        return

    if data.startswith("follow:start:"):
        try:
            _, _, match_id, event_token, page_text = data.split(":", maxsplit=4)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_main_menu(update, context)
            return

        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is None:
            await _replace_callback_message(
                update,
                text="No pude identificar el chat para iniciar seguimiento.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return

        existing = FOLLOW_JOBS.get(chat_id)
        if isinstance(existing, dict) and str(existing.get("phase") or "") == "running":
            existing_mid = str(existing.get("match_id") or "")
            if existing_mid != match_id:
                await _replace_callback_message(
                    update,
                    text=(
                        f"Ya hay seguimiento activo para {existing_mid}.\n"
                        "Detenlo antes de iniciar otro."
                    ),
                    reply_markup=_follow_status_keyboard(existing),
                )
                return

        data_row = _get_match_detail(match_id)
        status_type = str((data_row or {}).get("match", {}).get("status_type", "") or "").lower()
        if status_type == "finished":
            await _replace_callback_message(
                update,
                text=f"El match {match_id} ya esta finalizado. No requiere seguimiento.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return

        event_date = None if event_token == "_" else event_token
        FOLLOW_JOBS[chat_id] = {
            "match_id": match_id,
            "event_token": event_token,
            "page": page,
            "phase": "starting",
            "stop_requested": False,
            "interval_seconds": FOLLOW_REFRESH_SECONDS,
            "stale_cycles_limit": FOLLOW_STALE_CYCLES_LIMIT,
        }
        if query.message:
            FOLLOW_JOBS[chat_id]["status_message_id"] = query.message.message_id
        
        try:
            pred_row = _get_or_compute_predictions(match_id, data_row)
            FOLLOW_JOBS[chat_id]["follow_data"] = data_row
            FOLLOW_JOBS[chat_id]["follow_pred"] = pred_row
        except Exception:
            pred_row = None

        await _publish_follow_status(
            context.application,
            chat_id,
            FOLLOW_JOBS[chat_id],
            force=True,
        )
        context.application.create_task(_run_follow_match_job(context.application, chat_id))

        job = FOLLOW_JOBS.get(chat_id)
        if not isinstance(job, dict):
            await _render_main_menu(update, context)
            return

        return

    if data == "follow:stop":
        chat_id = update.effective_chat.id if update.effective_chat else None
        job = FOLLOW_JOBS.get(chat_id) if chat_id is not None else None
        if not isinstance(job, dict):
            await _replace_callback_message(
                update,
                text="No hay seguimiento activo para detener.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return

        job["stop_requested"] = True
        await _replace_callback_message(
            update,
            text="Deteniendo seguimiento...",
            reply_markup=_follow_status_keyboard(job),
        )
        return

    if data.startswith("follow:open:"):
        try:
            _, _, match_id, event_token, page_text = data.split(":", maxsplit=4)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_main_menu(update, context)
            return

        event_date = None if event_token == "_" else event_token
        if event_date is None:
            await _render_live_detail(update, context, match_id, page)
            return
        await _render_match_detail(
            update,
            context.application,
            match_id,
            event_date,
            page,
        )
        return

    if data.startswith("delmatch:confirm:"):
        try:
            _, _, match_id, event_token, page_text = data.split(":", maxsplit=4)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_dates(update, 0)
            return
        event_date = None if event_token == "_" else event_token
        await _replace_callback_message(
            update,
            text=(
                f"Vas a borrar el match {match_id} de la base y sus dependencias.\n"
                "Incluye: quarter_scores, play_by_play, graph_points, eval_match_results.\n"
                "Esta accion no se puede deshacer."
            ),
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Confirmar borrado",
                            callback_data=f"delmatch:run:{match_id}:{event_token}:{page}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Cancelar",
                            callback_data=f"match:{match_id}:{event_token}:{page}",
                        )
                    ],
                ]
            ),
        )
        return

    if data.startswith("delmatch:run:"):
        try:
            _, _, match_id, event_token, page_text = data.split(":", maxsplit=4)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_dates(update, 0)
            return
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not _is_train_allowed(chat_id):
            await _replace_callback_message(
                update,
                text="No autorizado para borrar matches.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return
        counts = await asyncio.to_thread(_delete_match_cascade, match_id)
        event_date = None if event_token == "_" else event_token
        if event_date:
            rows_after = _fetch_matches_for_date(event_date)
            if rows_after:
                await _render_matches_for_date(update, event_date, page)
            else:
                summary = (
                    f"Match {match_id} borrado.\n"
                    f"matches={counts['matches']} quarter_scores={counts['quarter_scores']}\n"
                    f"play_by_play={counts['play_by_play']} graph_points={counts['graph_points']}\n"
                    f"eval={counts['eval_match_results']} discovered={counts['discovered_ft_matches']}\n\n"
                    f"Ya no quedan matches en {event_date}."
                )
                await _replace_callback_message(
                    update,
                    text=summary,
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [InlineKeyboardButton("Volver a fechas", callback_data="dates:0")],
                            [InlineKeyboardButton("Menu principal", callback_data="nav:main")],
                        ]
                    ),
                )
        else:
            summary = (
                f"Match {match_id} borrado.\n"
                f"matches={counts['matches']} quarter_scores={counts['quarter_scores']}\n"
                f"play_by_play={counts['play_by_play']} graph_points={counts['graph_points']}\n"
                f"eval={counts['eval_match_results']} discovered={counts['discovered_ft_matches']}"
            )
            await _replace_callback_message(
                update,
                text=summary,
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
        return

    if data.startswith("calc:date:"):
        event_date = data[len("calc:date:"):]
        await _set_waiting_state(update, text="Recalculando apuestas... espere")
        try:
            result = await asyncio.to_thread(
                _calc_date_predictions,
                event_date,
                force_recalc=True,
            )
        except Exception as exc:
            await _replace_callback_message(
                update,
                text=f"Error calculando apuestas: {exc}",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Volver", callback_data=f"date:{event_date}:0")]]
                ),
            )
            return
        summary = (
            f"Recalculo completado para {event_date}:\n"
            f"Total: {result['total']} | Recalculadas: {result['recalculated']} | "
            f"OK: {result['ok']} | Fallidas: {result['fail']}"
        )
        rows = _fetch_matches_for_date(event_date)
        pred_map = _fetch_date_pred_outcomes(event_date)
        stats = _pred_stats_text(pred_map, len(rows))
        title = _event_date_title_es(event_date, len(rows))
        header = f"{summary}\n\n{title}"
        if stats:
            header += f"\n<pre>{html.escape(stats)}</pre>"
        await _replace_callback_message(
            update,
            text=header,
            reply_markup=_matches_keyboard(rows, event_date, 0, pred_map),
            parse_mode="HTML",
        )
        return

    if data.startswith("refreshlive:"):
        try:
            _, mode, match_id, page_text = data.split(":", maxsplit=3)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_live(update, context, 0, refetch=False)
            return
        await _refresh_live_detail(update, context, mode, match_id, page)
        return

    if data.startswith("dellive:confirm:"):
        try:
            _, _, match_id, page_text = data.split(":", maxsplit=3)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_live(update, context, 0, refetch=False)
            return
        await _replace_callback_message(
            update,
            text=(
                f"Vas a borrar el match {match_id} de la base y sus dependencias.\n"
                "Incluye: quarter_scores, play_by_play, graph_points, eval_match_results.\n"
                "Esta accion no se puede deshacer."
            ),
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Confirmar borrado",
                            callback_data=f"dellive:run:{match_id}:{page}",
                        )
                    ],
                    [InlineKeyboardButton("Cancelar", callback_data=f"livematch:{match_id}:{page}")],
                ]
            ),
        )
        return

    if data.startswith("dellive:run:"):
        try:
            _, _, match_id, page_text = data.split(":", maxsplit=3)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_live(update, context, 0, refetch=False)
            return
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not _is_train_allowed(chat_id):
            await _replace_callback_message(
                update,
                text="No autorizado para borrar matches.",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Menu principal", callback_data="nav:main")]]
                ),
            )
            return
        await asyncio.to_thread(_delete_match_cascade, match_id)
        await _render_live(update, context, page, refetch=True)
        return

    if data.startswith("livematch:"):
        try:
            _, match_id, page_text = data.split(":", maxsplit=2)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_live(update, context, 0, refetch=False)
            return
        await _render_live_detail(update, context, match_id, page)
        return

    if data.startswith("match:"):
        try:
            _, match_id, event_date, page_text = data.split(":", maxsplit=3)
            page = int(page_text)
        except (ValueError, TypeError):
            await _render_dates(update, 0)
            return
        event_date = None if event_date == "_" else event_date
        await _render_match_detail(
            update,
            context.application,
            match_id,
            event_date,
            page,
        )
        return


async def _handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    raw = (update.message.text or "").strip()
    if raw.lower() == MENU_BUTTON_TEXT.lower():
        await _render_main_menu(update, context)
        return

    if context.user_data.get(AWAITING_FETCH_DATE_KEY):
        parts = [p for p in raw.replace(",", " ").split() if p]
        if not parts:
            await update.message.reply_text(
                "Fecha invalida. Usa YYYY-MM-DD o YYYY-MM-DD 100"
            )
            return

        try:
            date_obj = datetime.strptime(parts[0], "%Y-%m-%d")
            event_date = date_obj.date().isoformat()
        except ValueError:
            await update.message.reply_text(
                "Fecha invalida. Usa formato YYYY-MM-DD."
            )
            return

        limit: int | None = None
        if len(parts) >= 2:
            try:
                limit = int(parts[1])
            except ValueError:
                await update.message.reply_text(
                    "Limite invalido. Usa entero positivo."
                )
                return
            if limit <= 0:
                await update.message.reply_text(
                    "Limite invalido. Usa entero positivo."
                )
                return

        context.user_data[AWAITING_FETCH_DATE_KEY] = False
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is not None and chat_id in DATE_INGEST_JOBS:
            await update.message.reply_text(
                "Ya hay una ingesta de fecha en curso en este chat. Puedes cancelarla desde su mensaje de progreso.",
                reply_markup=_menu_reply_keyboard(),
            )
            return
        status_msg = await update.message.reply_text(
            _date_ingest_progress_text(
                event_date,
                limit,
                {"phase": "starting"},
            ),
            reply_markup=_date_ingest_progress_keyboard(True),
        )
        if chat_id is not None:
            context.application.create_task(
                _run_date_ingest_job(
                    context.application,
                    chat_id,
                    status_msg.message_id,
                    event_date,
                    limit,
                )
            )
        await update.message.reply_text(
            "Proceso iniciado. Te voy avisando el progreso aqui.",
            reply_markup=_menu_reply_keyboard(),
        )
        return

    if not context.user_data.get(AWAITING_MATCH_ID_KEY):
        return

    match_id = raw
    if not re.fullmatch(r"\d{6,}", match_id):
        await update.message.reply_text(
            "ID invalido. Debe ser numerico. Ejemplo: 14442355"
        )
        return

    context.user_data[AWAITING_MATCH_ID_KEY] = False

    data = _get_match_detail(match_id)
    if not data:
        await update.message.reply_text(
            text=f"No se encontro el match {match_id}.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Buscar otro ID", callback_data="menu:id")],
                    [InlineKeyboardButton("Menu principal", callback_data="nav:main")],
                ]
            ),
        )
        return

    wait_msg = await update.message.reply_text(
        text="Espere...",
        reply_markup=_menu_reply_keyboard(),
    )
    await _send_detail_message(
        update,
        context.application,
        match_id,
        data,
        _get_or_compute_predictions(match_id, data),
        None,
        0,
    )
    try:
        await wait_msg.delete()
    except Exception:
        pass


def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit(
            "Falta TELEGRAM_BOT_TOKEN. Define la variable en .env"
        )

    app = Application.builder().token(BOT_TOKEN).post_init(_post_init).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("myid", myid_cmd))
    app.add_handler(CommandHandler("trainstatus", train_status_cmd))
    app.add_handler(CallbackQueryHandler(_handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text))
    app.add_error_handler(_on_error)

    print("[telegram-bot] iniciado")
    if ALLOWED_CHAT_IDS:
        print(f"[telegram-bot] allowed_chat_ids={sorted(ALLOWED_CHAT_IDS)}")
    else:
        print("[telegram-bot] allowed_chat_ids=ALL (sin restriccion)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

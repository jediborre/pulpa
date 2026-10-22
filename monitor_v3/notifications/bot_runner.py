# =====================================================================
# BOT DE TELEGRAM V3 (PROCESO APARTE, INTERACTIVO)
# =====================================================================
# Propósito: Bot interactivo para consultar el estado del monitor v3.
#   - Corre como PROCESO APARTE del daemon (monitor_v3/main.py).
#   - El daemon solo ENVÍA notificaciones (sendMessage); este bot es el
#     ÚNICO que hace getUpdates -> no hay conflicto 409 aunque compartan token.
#   - Lee exclusivamente las tablas con sufijo '_v3'.
#
# Comandos:
#   /start   -> menú (teclado persistente debajo del input)
#   /signals -> señales de hoy (apuestas + resultados reales)
#   /status  -> estadísticas de la BD v3
#
# Lanzamiento: python -m monitor_v3.notifications.bot_runner
# =====================================================================

import asyncio
import html
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.error import NetworkError, RetryAfter, TelegramError, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from monitor_v3.config.constants import (
    TELEGRAM_BOT_TOKEN,
    ACTIVE_MODELS,
    UTC_OFFSET_HOURS,
)
from monitor_v3.database.connection import get_real_db_path
from monitor_v3.utils.logger import log_info, log_warning, log_error
# Textos de los botones persistentes (debajo del input de Telegram)
SIGNALS_BUTTON_TEXT = "🎯 Apuestas"
STATUS_BUTTON_TEXT = "📊 Status"
RESUMEN_BUTTON_TEXT = "📈 Resumen"

# ─────────────────────────────────────────────────────────────────────
# Timeouts y resiliencia de red
# ─────────────────────────────────────────────────────────────────────
# El bot corre sobre una red doméstica con microcaídas (~10 s) que provocan
# httpx.ReadError en el long polling de Telegram. PTB ya reintenta el polling
# indefinidamente (max_retries=-1), pero sus envíos salientes NO se reintentan:
# de ahí los timeouts holgados + reintentos manuales con backoff.
CONNECT_TIMEOUT_SECS = 10.0
READ_TIMEOUT_SECS = 20.0     # al polling se le suma el long-poll (10 s) → 30 s reales
WRITE_TIMEOUT_SECS = 20.0
POOL_TIMEOUT_SECS = 5.0
SEND_MAX_ATTEMPTS = 3
SEND_BACKOFF_BASE_SECS = 1.0
POLL_ERROR_LOG_INTERVAL_SECS = 60.0

_last_poll_error_log = 0.0
_poll_error_count = 0


# ─────────────────────────────────────────────────────────────────────
# Resiliencia de red (polling + envíos salientes)
# ─────────────────────────────────────────────────────────────────────

def _on_poll_error(exc: TelegramError) -> None:
    """
    Callback de errores del bucle de polling de PTB.

    PTB reintenta el polling indefinidamente, así que un corte de red NO detiene
    el bot. Aquí solo se registra en el log diario, con throttle para no inundar
    la consola con el traceback completo en cada reconexión.
    """
    global _last_poll_error_log, _poll_error_count
    _poll_error_count += 1
    now = time.monotonic()
    if now - _last_poll_error_log < POLL_ERROR_LOG_INTERVAL_SECS:
        return
    _last_poll_error_log = now
    log_warning(
        "TELEGRAM",
        f"[BOT_V3] Corte de red en polling ({type(exc).__name__}: {exc or 'sin detalle'}). "
        f"Reintentando... (fallos acumulados: {_poll_error_count})",
    )


async def _send_with_retry(send, *, description: str, attempts: int = SEND_MAX_ATTEMPTS):
    """
    Ejecuta un envío con reintentos y backoff exponencial.

    `send` debe ser una función sin argumentos que devuelva una corrutina nueva en
    cada llamada (las corrutinas no se pueden reutilizar tras un fallo).
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await send()
        except RetryAfter as exc:
            delay = float(exc.retry_after) + 0.5
            log_warning(
                "TELEGRAM",
                f"[BOT_V3] Rate limit en {description}; esperando {delay:.1f}s "
                f"(intento {attempt}/{attempts})",
            )
        except (NetworkError, TimedOut) as exc:
            last_exc = exc
            delay = SEND_BACKOFF_BASE_SECS * (2 ** (attempt - 1))
            log_warning(
                "TELEGRAM",
                f"[BOT_V3] Fallo de red en {description} "
                f"(intento {attempt}/{attempts}): {exc or 'sin detalle'}. "
                f"Reintento en {delay:.1f}s",
            )
        if attempt < attempts:
            await asyncio.sleep(delay)

    log_error(
        "TELEGRAM",
        f"[BOT_V3] No se pudo completar {description} tras {attempts} intentos: {last_exc}",
    )
    return None


# ─────────────────────────────────────────────────────────────────────
# Apuestas: selector por fecha y partido (botones inline)
# ─────────────────────────────────────────────────────────────────────

DIAS_ES = ["LU", "MA", "MI", "JU", "VI", "SA", "DO"]
MESES_ABR = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"]


def _fmt_date_button(date_str: str, count: int) -> str:
    """'LU 19 OCT 2026 [80]' a partir de 'YYYY-MM-DD'."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        return f"{date_str} [{count}]"
    return f"{DIAS_ES[d.weekday()]} {d.day:02d} {MESES_ABR[d.month - 1]} {d.year} [{count}]"


def fetch_signal_dates(limit: int = 14) -> list[dict]:
    """Fechas (UTC-6) con señales, más recientes primero, con nº de partidos."""
    db = _db_path()
    if not db.exists():
        return []
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            """
            SELECT date(created_at) AS d, COUNT(DISTINCT match_id) AS n
            FROM bet_monitor_log_v3
            GROUP BY d
            ORDER BY d DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [{"date": r[0], "count": int(r[1])} for r in rows if r[0]]


def fetch_day_matches(date_str: str) -> list[dict]:
    """Partidos con señales de una fecha, en orden de su número del día."""
    db = _db_path()
    if not db.exists():
        return []
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT l.match_id,
                   MAX(s.daily_seq)        AS seq,
                   MAX(s.home_team)        AS home_team,
                   MAX(s.away_team)        AS away_team,
                   MAX(s.league)           AS league,
                   MAX(s.scheduled_utc_ts) AS ts,
                   SUM(CASE WHEN l.signal_type LIKE '%BET%'
                             AND l.signal_type NOT LIKE '%NO_BET%' THEN 1 ELSE 0 END) AS bets,
                   SUM(CASE WHEN l.result IN ('win','hit')   THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN l.result IN ('loss','miss') THEN 1 ELSE 0 END) AS losses
            FROM bet_monitor_log_v3 l
            LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
            WHERE date(l.created_at) = ?
            GROUP BY l.match_id
            ORDER BY (seq IS NULL), seq, ts, l.match_id
            """,
            (date_str,),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_match_logs(match_id: str) -> list[dict]:
    """Logs de todos los modelos para un partido (para la tarjeta de detalle)."""
    db = _db_path()
    if not db.exists():
        return []
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT l.*, s.home_team, s.away_team, s.league, s.scheduled_utc_ts, s.daily_seq,
                   s.final_fetched, s.event_date
            FROM bet_monitor_log_v3 l
            LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
            WHERE l.match_id = ?
            ORDER BY l.id
            """,
            (str(match_id),),
        ).fetchall()
    return [dict(r) for r in rows]


def _match_button_label(row: dict) -> str:
    """Etiqueta corta del botón de un partido."""
    seq = f"#{row['seq']}" if row.get("seq") else "#?"
    ts = row.get("ts")
    hhmm = ""
    if ts:
        tz = timezone(timedelta(hours=UTC_OFFSET_HOURS))
        hhmm = datetime.fromtimestamp(int(ts), tz=tz).strftime("%H:%M")
    home = str(row.get("home_team") or "?")
    away = str(row.get("away_team") or "?")
    names = f"{home} vs {away}"
    if len(names) > 34:
        names = names[:31] + "..."
    record = ""
    wins, losses = int(row.get("wins") or 0), int(row.get("losses") or 0)
    if wins or losses:
        record = f" {wins}-{losses}"
    return f"{seq} {hhmm} {names}{record}".strip()


def _day_stats_image(stats: dict, date_str: str) -> bytes | None:
    """Imagen del resumen del día (una sola sección, estilo /status)."""
    try:
        from monitor_v3.notifications.stats_image import build_stats_image
        from monitor_v3.notifications.stats_image import _section_from_stats

        section = _section_from_stats(date_str, stats, ACTIVE_MODELS)
        return build_stats_image(
            title="Apuestas del día",
            sections=[section],
            footer="Bet Monitor V3 · datos de bet_monitor_log_v3",
        )
    except Exception as e:  # noqa: BLE001
        log_warning("TELEGRAM", f"[BOT_V3] Sin imagen del día ({type(e).__name__}: {e})")
        return None


def _day_caption(date_str: str, stats: dict) -> str:
    """Caption con el resumen del día (mismos números que la imagen)."""
    t = stats.get("total") or {}
    resolved = int(t.get("win") or 0) + int(t.get("loss") or 0)
    acc = f"{int(round(int(t.get('win') or 0) * 100 / resolved))}%" if resolved else "—"
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        pretty = f"{DIAS_ES[d.weekday()]} {d.day:02d} {MESES_ABR[d.month - 1]} {d.year}"
    except (TypeError, ValueError):
        pretty = date_str
    lines = [
        f"🎯 <b>Apuestas del día</b> · {pretty}",
        f"Apuestas: <b>{t.get('bets', 0)}</b> | ✅ {t.get('win', 0)}  ❌ {t.get('loss', 0)}  "
        f"➖ {t.get('push', 0)}  ⏳ {t.get('pending', 0)}",
        f"Acierto: <b>{acc}</b>",
        "",
        "<i>Elige un partido para ver su detalle.</i>",
    ]
    return "\n".join(lines)


def _match_caption(logs: list[dict], data: dict | None = None) -> str:
    """
    Caption del detalle de un partido, con el mismo formato de los avisos FT:

        #9 REG BBC vs Foxes Divas
        FIBA ... Group A
        Lu Oct 19, 04:30 am | Q4 19 - 18

        ➖ m27_v5_2 72% [NO BET (<72%)] → 🏠REG BBC
        ✅ m27_v5_3 75% → 🏠REG BBC
    """
    if not logs:
        return "Sin datos del partido."

    from monitor_v3.notifications.telegram_bot import _format_utc6_datetime, _threshold_pct

    first = logs[0]
    seq = f"#{first['daily_seq']} " if first.get("daily_seq") else ""
    home = html.escape(str(first.get("home_team") or "?"), quote=False)
    away = html.escape(str(first.get("away_team") or "?"), quote=False)
    league = html.escape(str(first.get("league") or ""), quote=False)

    lines = [f"{seq}{home} vs {away}"]
    if league:
        lines.append(league)

    # Fecha/hora + parcial del cuarto objetivo (mercado de la apuesta)
    when = ""
    ts = first.get("scheduled_utc_ts")
    if ts:
        when = _format_utc6_datetime(int(ts))
    quarter = f"Q{first.get('target_quarter') or 4}"
    q_data = ((data or {}).get("score") or {}).get("quarters", {}).get(quarter) or {}
    partial = ""
    if q_data.get("home") is not None and q_data.get("away") is not None:
        partial = f"{quarter} {q_data['home']} - {q_data['away']}"
    detail = " | ".join(part for part in (when, partial) if part)
    if detail:
        lines.append(detail)
    lines.append("")

    for log in logs:
        sig = str(log.get("signal_type") or "")
        is_real = "BET" in sig and "NO_BET" not in sig
        model = html.escape(str(log.get("model_version") or "?"), quote=False)
        conf = log.get("confidence") or 0.0
        if conf > 1.0:
            conf = conf / 100.0
        is_home = str(log.get("picked_side") or "").upper() == "HOME"
        side = "🏠" if is_home else "✈️"
        team = home if is_home else away

        res = str(log.get("result") or "").lower()
        if res in ("win", "hit"):
            result_emoji = "✅"
        elif res in ("loss", "miss"):
            result_emoji = "❌"
        elif res in ("push", "void", ""):
            result_emoji = "➖"
        else:
            result_emoji = "⏳"

        if is_real:
            lines.append(f"{result_emoji} {model} {int(round(conf * 100))}% → {side}{team}")
        else:
            threshold = _threshold_pct(model, log)
            lines.append(
                f"{result_emoji} {model} {int(round(conf * 100))}% "
                f"[NO BET (&lt;{threshold}%)] → {side}{team}"
            )
    return "\n".join(lines)


def _build_detail_image(data: dict | None, logs: list[dict]) -> bytes | None:
    """
    Construye la imagen del partido con los logs y el payload ya descargado.

    - Partido terminado (final_fetched o status finished) → tarjeta + gráfica.
    - En vivo / sin gráfica → solo tarjeta.
    """
    if not logs:
        return None
    try:
        from monitor_v3.notifications.bet_image import build_q4_alert_image

        first = logs[0]
        finished = bool(first.get("final_fetched")) or str(
            ((data or {}).get("match") or {}).get("status_type") or ""
        ).lower() == "finished"

        bets = []
        for log in logs:
            is_home = str(log.get("picked_side") or "").upper() == "HOME"
            signal = str(log.get("signal_type") or "")
            is_real = "BET" in signal and "NO_BET" not in signal
            result = log.get("result")
            if not is_real and finished:
                # Lo que HUBIERA pasado con el pick, usando el parcial del cuarto
                # objetivo (mismo criterio que la liquidación real).
                try:
                    from monitor_v3.notifications.telegram_bot import _hypothetical_result

                    quarter = f"Q{log.get('target_quarter') or 4}"
                    q_data = ((data or {}).get("score") or {}).get("quarters", {}).get(quarter) or {}
                    result = _hypothetical_result(
                        is_home,
                        data,
                        q_home=q_data.get("home"),
                        q_away=q_data.get("away"),
                        quarter=quarter,
                    ) or result
                except Exception:
                    pass
            bets.append(
                {
                    "model": log.get("model_version"),
                    "signal": signal,
                    "picked_team": first.get("home_team") if is_home else first.get("away_team"),
                    "is_home": is_home,
                    "confidence": log.get("confidence"),
                    "result": result,
                    "inference_json": _inference_json_dict(log.get("inference_json")),
                }
            )

        return build_q4_alert_image(
            data,
            bets,
            minute=None,
            q_key=f"Q{first.get('target_quarter') or 4}",
            scheduled_ts=first.get("scheduled_utc_ts"),
            is_result=finished,
            include_chart=finished,
            seq=first.get("daily_seq"),
        )
    except Exception as e:  # noqa: BLE001
        match_id = str((logs[0] or {}).get("match_id") or "?")
        log_warning("TELEGRAM", f"[BOT_V3] Sin imagen de detalle para {match_id}: {type(e).__name__}: {e}")
        return None


def _inference_json_dict(value) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            import json as _json

            parsed = _json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


async def _show_date_picker(message) -> None:
    """Muestra los días con señales como botones inline."""
    dates = await asyncio.to_thread(fetch_signal_dates)
    if not dates:
        await _send_with_retry(
            lambda: message.reply_text("Sin señales registradas todavía."),
            description="/signals (sin datos)",
        )
        return
    keyboard = [
        [InlineKeyboardButton(_fmt_date_button(d["date"], d["count"]), callback_data=f"d:{d['date']}")]
        for d in dates
    ]
    await _send_with_retry(
        lambda: message.reply_text(
            "🎯 <b>Apuestas</b>\nElige el día:",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard),
        ),
        description="/signals (fechas)",
    )


async def _send_day_view(message, date_str: str) -> None:
    """Resumen del día (imagen estilo status) + botones con cada partido."""
    stats = await asyncio.to_thread(fetch_v3_stats, date_str, None)
    matches = await asyncio.to_thread(fetch_day_matches, date_str)
    photo = await asyncio.to_thread(_day_stats_image, stats, date_str)
    caption = _day_caption(date_str, stats)

    rows = [
        [InlineKeyboardButton(_match_button_label(m), callback_data=f"m:{m['match_id']}")]
        for m in matches
    ]
    rows.append([InlineKeyboardButton("◀ Días", callback_data="x:dates")])
    keyboard = InlineKeyboardMarkup(rows)

    if photo:
        await _send_with_retry(
            lambda: message.reply_photo(photo, caption=caption, parse_mode="HTML", reply_markup=keyboard),
            description=f"día {date_str}",
        )
    else:
        await _send_with_retry(
            lambda: message.reply_text(caption, parse_mode="HTML", reply_markup=keyboard),
            description=f"día {date_str} (texto)",
        )


async def _send_match_view(message, match_id: str) -> None:
    """Detalle del partido: imagen generada (FT con gráfica) + botón de volver."""
    logs = await asyncio.to_thread(fetch_match_logs, match_id)

    # El payload se descarga en el loop del bot (el cliente móvil y su pool de
    # tokens usan primitivas asyncio y no deben moverse a otro event loop).
    data = None
    if logs:
        try:
            from monitor_v3.scrapers.match_detail_scraper import fetch_match_by_id

            data = await fetch_match_by_id(match_id, is_ft=False)
        except Exception as e:  # noqa: BLE001
            log_warning("TELEGRAM", f"[BOT_V3] Sin payload de {match_id}: {type(e).__name__}: {e}")

    caption = _match_caption(logs, data)
    photo = await asyncio.to_thread(_build_detail_image, data, logs) if logs else None

    date_str = str((logs[0].get("event_date") if logs else "") or "")[:10]
    back_row = []
    if date_str:
        back_row.append(InlineKeyboardButton("◀ Día", callback_data=f"d:{date_str}"))
    back_row.append(InlineKeyboardButton("📱 Sofascore", url=f"https://www.sofascore.com/event/{match_id}"))
    keyboard = InlineKeyboardMarkup([back_row])

    if photo:
        await _send_with_retry(
            lambda: message.reply_photo(
                photo, caption=caption[:1024], parse_mode="HTML", reply_markup=keyboard
            ),
            description=f"partido {match_id}",
        )
    else:
        await _send_with_retry(
            lambda: message.reply_text(caption, parse_mode="HTML", reply_markup=keyboard),
            description=f"partido {match_id} (texto)",
        )


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Router de los botones inline del flujo de apuestas."""
    query = update.callback_query
    if query is None:
        return
    if not _is_allowed(update):
        await query.answer("No autorizado", show_alert=True)
        return

    data = query.data or ""
    if data.startswith("d:"):
        await query.answer("Cargando día...")
        await _send_day_view(query.message, data[2:])
    elif data.startswith("m:"):
        await query.answer("Generando imagen...")
        await _send_match_view(query.message, data[2:])
    elif data == "x:dates":
        await query.answer()
        await _show_date_picker(query.message)
    else:
        await query.answer()




def _db_path() -> Path:
    return Path(get_real_db_path())


def _esc(value) -> str:
    return html.escape(str(value or ""), quote=False)


def _today_str() -> str:
    tz = timezone(timedelta(hours=UTC_OFFSET_HOURS))
    return datetime.now(tz).strftime("%Y-%m-%d")


def _signal_emoji(signal_type: str, confidence: float | None) -> str:
    sig = signal_type or ""
    if "NO_BET" in sig or "BET" not in sig:
        return "🔴"
    conf = confidence or 0.0
    if conf > 1.0:
        conf = conf / 100.0
    is_low = round(conf, 4) < 0.30
    is_late = "LATE" in sig
    if is_late:
        return "🟡⚪️" if is_low else "⚪️"
    return "🟡" if is_low else "🟢"


def _result_emoji(result: str) -> str:
    res = (result or "").lower()
    if res in ("win", "hit"):
        return "✅"
    if res in ("loss", "miss"):
        return "❌"
    if res == "push":
        return "➖"
    return "⏳"


def _day_summary_lines(rows: list[dict]) -> list[str]:
    """Resumen del día: ganadas/perdidas/push/pendientes, no-bet y por categoría."""
    win = loss = push = pending = nobet = 0
    cats = {"🟢": [0, 0], "🟡": [0, 0], "⚪️": [0, 0], "🟡⚪️": [0, 0]}
    for r in rows:
        sig = r.get("signal_type") or ""
        if "NO_BET" in sig or "BET" not in sig:
            nobet += 1
            continue
        cat = _signal_emoji(sig, r.get("confidence"))
        res = (r.get("result") or "").lower()
        if res in ("win", "hit"):
            win += 1
            cats.setdefault(cat, [0, 0])[0] += 1
        elif res in ("loss", "miss"):
            loss += 1
            cats.setdefault(cat, [0, 0])[1] += 1
        elif res == "push":
            push += 1
        else:
            pending += 1

    bets = win + loss + push + pending
    resolved = win + loss
    acc = f"{int(round(win * 100 / resolved))}%" if resolved else "—"

    out = ["", "📊 <b>Resumen del día</b>"]
    out.append(f"Apuestas: <b>{bets}</b> | ✅ {win}  ❌ {loss}  ➖ {push}  ⏳ {pending}")
    out.append(f"No bet: <b>{nobet}</b> | Acierto (resueltas): <b>{acc}</b>")
    out.append("Por categoría:")
    for cat in ("🟢", "🟡", "⚪️", "🟡⚪️"):
        w, l = cats.get(cat, [0, 0])
        if w + l:
            out.append(f"  {cat}: ✅ {w}  ❌ {l}")
    return out


def fetch_today_signals() -> list[dict]:
    """Señales de hoy desde bet_monitor_log_v3 + datos del partido."""
    db = _db_path()
    if not db.exists():
        return []
    today = _today_str()
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT l.match_id, l.model_version, l.signal_type, l.picked_side,
                   l.confidence, l.result, l.inference_minute,
                   s.home_team, s.away_team, s.league, s.scheduled_utc_ts, s.daily_seq
            FROM bet_monitor_log_v3 l
            LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
            WHERE date(l.created_at) = ?
            ORDER BY s.daily_seq, s.scheduled_utc_ts, l.match_id, l.model_version
            """,
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]


def fetch_v3_stats(created_date: str | None = None, created_month: str | None = None) -> dict:
    """
    Estadísticas (globales, de un día o de un mes) y por modelo desde bet_monitor_log_v3.

    - `created_date`: 'YYYY-MM-DD' → solo ese día.
    - `created_month`: 'YYYY-MM' → solo ese mes.
    - Sin filtros: histórico completo.
    """
    db = _db_path()
    stats = {
        "per_model": {},
        "total": {"signals": 0, "bets": 0, "win": 0, "loss": 0, "push": 0, "pending": 0, "void": 0},
    }
    if not db.exists():
        return stats
    query = "SELECT model_version, signal_type, confidence, result FROM bet_monitor_log_v3"
    params: list[str] = []
    if created_date:
        query += " WHERE date(created_at) = ?"
        params.append(created_date)
    elif created_month:
        query += " WHERE strftime('%Y-%m', created_at) = ?"
        params.append(created_month)

    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    for r in rows:
        model = r["model_version"] or "?"
        m = stats["per_model"].setdefault(
            model, {"signals": 0, "bets": 0, "win": 0, "loss": 0, "push": 0, "pending": 0, "void": 0}
        )
        m["signals"] += 1
        stats["total"]["signals"] += 1
        sig = r["signal_type"] or ""
        is_bet = "BET" in sig and "NO_BET" not in sig
        res = (r["result"] or "").lower()
        if res == "void":
            # Anuladas por falta de datos: no cuentan como apuesta ni en acierto.
            m["void"] += 1
            stats["total"]["void"] += 1
            continue
        if is_bet:
            m["bets"] += 1
            stats["total"]["bets"] += 1
            if res in ("win", "hit"):
                m["win"] += 1
                stats["total"]["win"] += 1
            elif res in ("loss", "miss"):
                m["loss"] += 1
                stats["total"]["loss"] += 1
            elif res == "push":
                m["push"] += 1
                stats["total"]["push"] += 1
            elif res in ("", "pending"):
                m["pending"] += 1
                stats["total"]["pending"] += 1
    return stats


# ─────────────────────────────────────────────────────────────────────
# Formateo de mensajes
# ─────────────────────────────────────────────────────────────────────

def build_signals_text() -> str:
    rows = fetch_today_signals()
    today = _today_str()
    if not rows:
        return f"🎯 <b>Señales de hoy</b> ({today})\n\nSin señales registradas aún."

    by_match: dict[str, list[dict]] = {}
    for r in rows:
        by_match.setdefault(str(r["match_id"]), []).append(r)

    n_matches = len(by_match)
    n_bets = sum(
        1 for r in rows if "BET" in (r["signal_type"] or "") and "NO_BET" not in (r["signal_type"] or "")
    )
    lines = [f"🎯 <b>Señales de hoy</b> ({today})", f"{n_matches} partidos | {n_bets} apuestas", ""]

    for mid, items in by_match.items():
        s0 = items[0]
        home = s0.get("home_team") or "?"
        away = s0.get("away_team") or "?"
        league = s0.get("league") or ""
        ts = s0.get("scheduled_utc_ts")
        hhmm = ""
        if ts:
            tz = timezone(timedelta(hours=UTC_OFFSET_HOURS))
            hhmm = datetime.fromtimestamp(int(ts), tz=tz).strftime("%H:%M")
        seq = s0.get("daily_seq")
        badge = f"<b>#{seq}</b> · " if seq else ""
        lines.append(f"{badge}{hhmm} <b>{_esc(home)} vs {_esc(away)}</b>")
        if league:
            lines.append(f"  <i>{_esc(league)}</i>")
        for r in items:
            model = r["model_version"] or "?"
            conf = r["confidence"] or 0.0
            if conf > 1.0:
                conf = conf / 100.0
            conf_pct = int(round(conf * 100))
            side = "🏠" if (r["picked_side"] or "").upper() == "HOME" else "✈️"
            team = home if (r["picked_side"] or "").upper() == "HOME" else away
            sig = _signal_emoji(r["signal_type"], r["confidence"])
            res = _result_emoji(r["result"])
            lines.append(f"  {sig} {res} {_esc(model)} {conf_pct}% → {side} {_esc(team)}")
        lines.append("")

    lines.extend(_day_summary_lines(rows))
    return "\n".join(lines).rstrip()


def _range_block(title: str, stats: dict, emoji: str = "") -> list[str]:
    """Bloque de texto con las cifras de un rango (hoy/mes)."""
    t = stats.get("total") or {}
    models = list((stats.get("per_model") or {}).keys()) or list(ACTIVE_MODELS)
    resolved = int(t.get("win") or 0) + int(t.get("loss") or 0)
    acc = f"{int(round(int(t.get('win') or 0) * 100 / resolved))}%" if resolved else "—"

    lines = [f"{emoji} <b>{title}</b>".strip()]
    lines.append(
        f"Apuestas: <b>{t.get('bets', 0)}</b> | ✅ {t.get('win', 0)}  ❌ {t.get('loss', 0)}  "
        f"➖ {t.get('push', 0)}  ⏳ {t.get('pending', 0)}"
    )
    extra = f" | Anuladas: <b>{t.get('void', 0)}</b>" if t.get("void") else ""
    lines.append(f"Señales: <b>{t.get('signals', 0)}</b> | Acierto: <b>{acc}</b>{extra}")

    # Modelos ordenados por efectividad (mejor % de acierto primero)
    rows = []
    for m in models:
        s = (stats.get("per_model") or {}).get(m)
        if not s:
            continue
        win = int(s.get("win") or 0)
        loss = int(s.get("loss") or 0)
        resolved = win + loss
        rate = (win / resolved) if resolved else -1.0
        rows.append((rate, resolved, m, s, win, loss))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)

    for _rate, _resolved, m, s, win, loss in rows:
        r = win + loss
        a = f"{int(round(win * 100 / r))}%" if r else "—"
        lines.append(
            f"  {m}: BET {s.get('bets', 0)} | ✅ {win}  ❌ {loss}  "
            f"⏳ {s.get('pending', 0)} | {a}"
        )
    return lines


_DIAS_ES = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
_MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
             "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _month_label(dt: datetime) -> str:
    return f"{_MESES_ES[dt.month - 1]} {dt.year}".capitalize()


def _day_label(dt: datetime) -> str:
    return f"{_DIAS_ES[dt.weekday()]} {dt.day} {_MESES_ES[dt.month - 1]} {dt.year}"


def build_status_text(day_stats: dict | None = None, month_stats: dict | None = None) -> str:
    """Texto del /status con el resumen de HOY y del MES."""
    if day_stats is None:
        day_stats = fetch_v3_stats(created_date=_today_str())
    if month_stats is None:
        month_stats = fetch_v3_stats(created_month=_today_str()[:7])

    tz = timezone(timedelta(hours=UTC_OFFSET_HOURS))
    now = datetime.now(tz)

    lines = ["📊 <b>Status BD (v3)</b>", ""]
    lines.extend(_range_block("Hoy", day_stats, "📅"))
    lines.append("")
    lines.extend(_range_block(f"Mes ({_month_label(now)})", month_stats, "🗓"))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Teclado persistente (debajo del input)
# ─────────────────────────────────────────────────────────────────────

def _reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[SIGNALS_BUTTON_TEXT, STATUS_BUTTON_TEXT], [RESUMEN_BUTTON_TEXT]],
        resize_keyboard=True,
        is_persistent=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Allow-list
# ─────────────────────────────────────────────────────────────────────

def _allowed_chat_ids() -> set[int]:
    ids: set[int] = set()
    raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()
    for part in re.split(r"[,\s]+", raw):
        part = part.strip()
        if part.lstrip("-").isdigit():
            ids.add(int(part))
    try:
        from monitor_v3.notifications.telegram_bot import _get_subscribers_from_db

        ids.update(_get_subscribers_from_db().keys())
    except Exception:
        pass
    return ids


def _is_allowed(update: Update) -> bool:
    allowed = _allowed_chat_ids()
    if not allowed:
        return True  # sin restricción configurada
    chat_id = update.effective_chat.id if update.effective_chat else None
    return chat_id in allowed


# ─────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    text = (
        "🤖 <b>Bot Monitor V3</b>\n\n"
        "Consulta el estado del monitor:\n"
        f"• {SIGNALS_BUTTON_TEXT} — elige un día y luego un partido (imagen de detalle)\n"
        f"• {STATUS_BUTTON_TEXT} — estadísticas de hoy y del mes\n"
        f"• {RESUMEN_BUTTON_TEXT} — resumen del modelo en vivo (WR por día, bandas, "
        "curva de umbral)"
    )
    await _send_with_retry(
        lambda: update.effective_message.reply_text(
            text, parse_mode="HTML", reply_markup=_reply_keyboard()
        ),
        description="/start",
    )


async def _reply_long(message, text: str) -> None:
    """Envía un texto largo troceado respetando el límite de Telegram (~4096)."""
    MAX = 3800
    chunks: list[str] = []
    remaining = text
    while remaining:
        chunk = remaining[:MAX]
        if len(remaining) > MAX:
            nl = chunk.rfind("\n")
            if nl > 0:
                chunk = remaining[:nl]
        chunks.append(chunk)
        remaining = remaining[len(chunk):].lstrip("\n")
    for idx, c in enumerate(chunks, start=1):
        await _send_with_retry(
            lambda c=c: message.reply_text(c, parse_mode="HTML"),
            description=f"mensaje largo ({idx}/{len(chunks)})",
        )


async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Botón/comando de apuestas: selector de día con botones inline."""
    if not _is_allowed(update):
        return
    msg = await _send_with_retry(
        lambda: update.effective_message.reply_text("⏳ Cargando días..."),
        description="/signals (aviso)",
    )
    if msg:
        await _send_with_retry(msg.delete, description="/signals (borrar aviso)")
    await _show_date_picker(update.effective_message)


async def resumen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Botón/comando de resumen: el estado del modelo en vivo (WR por día,
    bandas de confianza, serie por minuto y curva de umbral)."""
    if not _is_allowed(update):
        return
    msg = await _send_with_retry(
        lambda: update.effective_message.reply_text("⏳ Calculando el resumen..."),
        description="/resumen (aviso)",
    )
    try:
        from monitor_v3.models.live_summary import compute_live_summary, render_live_summary_text
        with sqlite3.connect(str(_db_path())) as conn:
            conn.row_factory = sqlite3.Row
            data = await asyncio.to_thread(compute_live_summary, conn, "m27_v5_3")
        text = render_live_summary_text(data)
        if msg:
            await _send_with_retry(msg.delete, description="/resumen (borrar aviso)")
        await _reply_long(update.effective_message, text)
    except Exception as exc:  # noqa: BLE001
        log_error("TELEGRAM", f"[BOT_V3] Error en /resumen: {type(exc).__name__}: {exc}")
        await _send_with_retry(
            lambda: update.effective_message.reply_text(
                f"❌ No pude calcular el resumen: {type(exc).__name__}"),
            description="/resumen (error)",
        )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    msg = await _send_with_retry(
        lambda: update.effective_message.reply_text("⏳ Consultando estadísticas..."),
        description="/status (aviso)",
    )
    try:
        day_stats = await asyncio.to_thread(fetch_v3_stats, _today_str(), None)
        month_stats = await asyncio.to_thread(fetch_v3_stats, None, _today_str()[:7])
        text = build_status_text(day_stats, month_stats)

        # Imagen de estadísticas (HOY + MES); si falla, cae a texto.
        photo = await asyncio.to_thread(_status_image, day_stats, month_stats)
        if photo:
            if msg:
                await _send_with_retry(msg.delete, description="/status (borrar aviso)")
            if len(text) <= 1024:
                await _send_with_retry(
                    lambda: update.effective_message.reply_photo(
                        photo, caption=text, parse_mode="HTML"
                    ),
                    description="/status (imagen)",
                )
            else:
                await _send_with_retry(
                    lambda: update.effective_message.reply_photo(photo),
                    description="/status (imagen)",
                )
                await _reply_long(update.effective_message, text)
            return

        if msg:
            await _send_with_retry(
                lambda: msg.edit_text(text=text, parse_mode="HTML"),
                description="/status",
            )
    except Exception as e:
        log_error("TELEGRAM", f"[BOT_V3] Error en /status: {e}")
        if msg:
            await _send_with_retry(
                lambda: msg.edit_text("❌ Error al leer estadísticas"),
                description="/status (error)",
            )


def _status_image(day_stats: dict, month_stats: dict) -> bytes | None:
    """Imagen del status HOY + MES (importa Pillow solo cuando se usa)."""
    try:
        from monitor_v3.notifications.stats_image import status_image

        tz = timezone(timedelta(hours=UTC_OFFSET_HOURS))
        now = datetime.now(tz)
        subtitle = f"{_day_label(now)}  ·  {_month_label(now)}"
        return status_image(
            day_stats,
            month_stats,
            ACTIVE_MODELS,
            subtitle=subtitle,
            footer="Bet Monitor V3 · datos de bet_monitor_log_v3",
        )
    except Exception as e:  # noqa: BLE001
        log_warning("TELEGRAM", f"[BOT_V3] Sin imagen de status ({type(e).__name__}: {e})")
        return None


async def _handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    raw = (update.message.text or "").strip()
    if raw.lower() == SIGNALS_BUTTON_TEXT.lower():
        await signals_cmd(update, context)
    elif raw.lower() == STATUS_BUTTON_TEXT.lower():
        await status_cmd(update, context)
    elif raw.lower() == RESUMEN_BUTTON_TEXT.lower():
        await resumen_cmd(update, context)


async def _on_handler_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Registra excepciones no controladas sin tumbar el bot.

    En PTB 21.x los fallos del bucle de polling (httpx.ReadError, timeouts)
    llegan aquí con `update=None` vía Application.process_error, así que se
    enrutan al log con throttle en lugar de imprimir tracebacks completos.
    """
    exc = context.error
    if update is None and isinstance(exc, TelegramError):
        _on_poll_error(exc)
        return
    if isinstance(exc, (NetworkError, TimedOut)):
        log_warning(
            "TELEGRAM",
            f"[BOT_V3] Fallo de red en handler: {type(exc).__name__}: {exc or 'sin detalle'}",
        )
    else:
        log_error(
            "TELEGRAM",
            f"[BOT_V3] Excepción no controlada en handler: {type(exc).__name__}: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Falta TELEGRAM_BOT_TOKEN. Defínelo en .env")

    # Timeouts holgados: el long-poll (10 s) se suma al read_timeout.
    httpx_opts = dict(
        connect_timeout=CONNECT_TIMEOUT_SECS,
        read_timeout=READ_TIMEOUT_SECS,
        write_timeout=WRITE_TIMEOUT_SECS,
        pool_timeout=POOL_TIMEOUT_SECS,
    )
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(HTTPXRequest(**httpx_opts))
        .get_updates_request(HTTPXRequest(**httpx_opts))
        .build()
    )
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("resumen", resumen_cmd))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text))
    app.add_error_handler(_on_handler_error)

    log_info("TELEGRAM", "[BOT_V3] Bot interactivo V3 iniciado (polling).")
    print("[bot-v3] iniciado. Ctrl+C para detener.")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        poll_interval=1.0,
        timeout=10,
    )


if __name__ == "__main__":
    main()

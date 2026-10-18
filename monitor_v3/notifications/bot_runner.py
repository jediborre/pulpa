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
    ReplyKeyboardMarkup,
    Update,
)
from telegram.error import NetworkError, RetryAfter, TelegramError, TimedOut
from telegram.ext import (
    Application,
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
# Acceso a datos (_v3)
# ─────────────────────────────────────────────────────────────────────

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
                   s.home_team, s.away_team, s.league, s.scheduled_utc_ts
            FROM bet_monitor_log_v3 l
            LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
            WHERE date(l.created_at) = ?
            ORDER BY s.scheduled_utc_ts, l.match_id, l.model_version
            """,
            (today,),
        ).fetchall()
        return [dict(r) for r in rows]


def fetch_v3_stats() -> dict:
    """Estadísticas globales y por modelo desde bet_monitor_log_v3."""
    db = _db_path()
    stats = {"per_model": {}, "total": {"signals": 0, "bets": 0, "win": 0, "loss": 0, "pending": 0}}
    if not db.exists():
        return stats
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT model_version, signal_type, confidence, result FROM bet_monitor_log_v3"
        ).fetchall()

    for r in rows:
        model = r["model_version"] or "?"
        m = stats["per_model"].setdefault(
            model, {"signals": 0, "bets": 0, "win": 0, "loss": 0, "pending": 0}
        )
        m["signals"] += 1
        stats["total"]["signals"] += 1
        sig = r["signal_type"] or ""
        is_bet = "BET" in sig and "NO_BET" not in sig
        res = (r["result"] or "").lower()
        if is_bet:
            m["bets"] += 1
            stats["total"]["bets"] += 1
            if res in ("win", "hit"):
                m["win"] += 1
                stats["total"]["win"] += 1
            elif res in ("loss", "miss"):
                m["loss"] += 1
                stats["total"]["loss"] += 1
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
        lines.append(f"{hhmm} <b>{_esc(home)} vs {_esc(away)}</b>")
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


def build_status_text() -> str:
    stats = fetch_v3_stats()
    t = stats["total"]
    models = list(stats["per_model"].keys()) or list(ACTIVE_MODELS)

    lines = ["📊 <b>Status BD (v3)</b>", ""]
    lines.append(f"Señales totales: <b>{t['signals']}</b>")
    lines.append(
        f"Apuestas (BET): <b>{t['bets']}</b> | ✅ {t['win']}  ❌ {t['loss']}  ⏳ {t['pending']}"
    )
    resolved = t["win"] + t["loss"]
    acc = f"{int(round(t['win'] * 100 / resolved))}%" if resolved else "—"
    lines.append(f"Acierto: <b>{acc}</b>")
    lines.append("")
    lines.append("<b>Por modelo</b>")
    for m in models:
        s = stats["per_model"].get(m)
        if not s:
            lines.append(f"  {m}: sin datos")
            continue
        r = s["win"] + s["loss"]
        a = f"{int(round(s['win'] * 100 / r))}%" if r else "—"
        lines.append(
            f"  {m}: BET {s['bets']} | ✅ {s['win']}  ❌ {s['loss']}  ⏳ {s['pending']} | {a}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Teclado persistente (debajo del input)
# ─────────────────────────────────────────────────────────────────────

def _reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[SIGNALS_BUTTON_TEXT, STATUS_BUTTON_TEXT]],
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
        f"• {SIGNALS_BUTTON_TEXT} — señales de hoy (apuestas + resultados)\n"
        f"• {STATUS_BUTTON_TEXT} — estadísticas de la BD v3"
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
    if not _is_allowed(update):
        return
    msg = await _send_with_retry(
        lambda: update.effective_message.reply_text("⏳ Procesando señales de hoy..."),
        description="/signals (aviso)",
    )
    try:
        text = await asyncio.to_thread(build_signals_text)
        if msg:
            await _send_with_retry(msg.delete, description="/signals (borrar aviso)")
        await _reply_long(update.effective_message, text)
    except Exception as e:
        log_error("TELEGRAM", f"[BOT_V3] Error en /signals: {e}")
        if msg:
            await _send_with_retry(
                lambda: msg.edit_text("❌ Error al procesar señales"),
                description="/signals (error)",
            )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return
    msg = await _send_with_retry(
        lambda: update.effective_message.reply_text("⏳ Consultando estadísticas..."),
        description="/status (aviso)",
    )
    try:
        text = await asyncio.to_thread(build_status_text)
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


async def _handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    raw = (update.message.text or "").strip()
    if raw.lower() == SIGNALS_BUTTON_TEXT.lower():
        await signals_cmd(update, context)
    elif raw.lower() == STATUS_BUTTON_TEXT.lower():
        await status_cmd(update, context)


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

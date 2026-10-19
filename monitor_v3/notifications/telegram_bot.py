# =====================================================================
# REGLAS DE ARQUITECTURA, FORMATEO Y LOGGING DE PRODUCCIÓN (MANTENER):
# 1. ESTRUCTURA: Estrictamente modularizado (config, database, scrapers, 
#    models, notifications, utils) coordinados asíncronamente por main.py.
#    Cualquier aproximación monolítica de archivo único viola esta especificación.
# 2. INFRAESTRUCTURA DB: El archivo base SQLite se localiza exclusivamente en 
#    /matches.db (en la raíz del proyecto) y todas las tablas sin excepción finalizan con el sufijo '_v2'.
# 3. TABLA DE LOGS: 'bet_monitor_log_v2' se particiona por modelo y contiene 
#    obligatoriamente los campos 'raw_json' (TEXT), 'inference_minute' (INT), 
#    y 'graph_points_count' (INT) junto con marcadores reales del juego.
# 4. CONFIGURACIÓN DE LIGAS: Prohibido hardcodear filtros o patrones de texto 
#    en las consultas SQL o lógica directa. Debe consumirse declarativamente 
#    desde config/leagues.yaml o cargarse dinámicamente desde la BD SQLite.
# 5. Formato Log: {Fecha Hora} [INFO/WARNING/ERROR] [COMPONENTE]
#    - Colores ANSI: INFO=Azul, WARNING=Amarillo, ERROR=Rojo.
# 6. Formato Matches en Log: {horario_match} {match_id} {home} vs {away}
#    - horario_match en Amarillo, match_id en Azul (sin texto UTC-6).
# 7. Errores de red críticos: Imprimir explícitamente "HTTP 403/404" en ROJO.
# 8. Monitoreo avanzado en progreso de cuarto final: Usar obligatoriamente "Q4 🟠".
# 9. Telegram Prefijos de Apuestas: 🟢 (Bettable), 🟡 (No Bettable), ⚪ (Tardía).
# 10. Telegram Resultados FT: Prefijar con ✅ (Ganada) o ❌ (Perdida) manteniendo emoji base.
# 11. Conversión de tiempos siempre legibles en formato humano (ej. 1 dia 2h 15min / 45s).
# =====================================================================

import json
import re
import unicodedata
import sqlite3
import urllib.request
import urllib.error
import html
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta

from monitor_v3.config.constants import TELEGRAM_BOT_TOKEN, UTC_OFFSET_HOURS
from monitor_v3.database.connection import get_real_db_path
from monitor_v3.utils.logger import log_info, log_warning, log_error, COLOR_GREEN, COLOR_RESET

# Clave donde el bot de Telegram guarda los suscriptores en la tabla settings
_SUBSCRIBERS_SETTING_KEY = "monitor_subscribers"


def _get_subscribers_from_db() -> dict[int, dict]:
    """
    Lee los chat IDs suscritos desde la tabla `settings` de matches.db en raíz.
    Retorna un dict {chat_id: {"signal_type": ..., "quarters": [...]}} o vacío si no hay.
    """
    try:
        db_path = Path(get_real_db_path())
        if not db_path.exists():
            return {}
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (_SUBSCRIBERS_SETTING_KEY,)
            ).fetchone()
        finally:
            conn.close()

        if not row or not row["value"]:
            return {}

        raw = json.loads(row["value"])
        if not isinstance(raw, dict):
            return {}

        result: dict[int, dict] = {}
        for k, v in raw.items():
            try:
                chat_id = int(k)
            except (ValueError, TypeError):
                continue
            if isinstance(v, str):
                # Formato antiguo: solo string "all"/"bet_only"
                result[chat_id] = {"signal_type": v, "quarters": ["q3", "q4"]}
            elif isinstance(v, dict):
                result[chat_id] = {
                    "signal_type": v.get("signal_type", "all"),
                    "quarters": v.get("quarters", ["q3", "q4"]),
                }
        return result
    except Exception as e:
        log_error("TELEGRAM", f"[DB] Error leyendo subscribers: {e}")
        return {}


def _send_request(url: str, payload: dict) -> dict:
    """Ejecuta una petición síncrona a la API de Telegram."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = response.read().decode("utf-8")
            return json.loads(res_body)
    except urllib.error.HTTPError as err:
        try:
            err_body = err.read().decode("utf-8")
            err_json = json.loads(err_body)
            desc = err_json.get("description", str(err))
            return {"ok": False, "description": f"{err}: {desc}"}
        except Exception:
            return {"ok": False, "description": str(err)}
    except Exception as e:
        return {"ok": False, "description": str(e)}


async def _send_to_chat(chat_id: int, text: str, match_url: str | None = None) -> dict:
    """Envía un mensaje a un chat_id específico."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    if match_url:
        payload["reply_markup"] = {
            "inline_keyboard": [
                [
                    {
                        "text": "📱 Sofascore",
                        "url": match_url
                    }
                ]
            ]
        }
    res = await asyncio.to_thread(_send_request, url, payload)
    if not res.get("ok"):
        log_error("TELEGRAM", f"[ERROR] chat_id={chat_id}: {res.get('description', '?')}")
    return res


def _resolve_recipients(signal_type_filter: str = "all") -> list[int] | None:
    """
    Chat IDs que deben recibir el aviso según sus preferencias.
    Con 'bet' se excluye a los suscriptores 'info_only'.
    Devuelve None si no hay suscriptores configurados.
    """
    subscribers = _get_subscribers_from_db()
    if not subscribers:
        return None

    if signal_type_filter == "bet":
        return [cid for cid, prefs in subscribers.items() if prefs.get("signal_type") != "info_only"]

    active: list[int] = []
    for chat_id, prefs in subscribers.items():
        sub_signal = prefs.get("signal_type", "all")
        if sub_signal == "bet_only" and signal_type_filter == "info":
            continue
        active.append(chat_id)
    return active


async def broadcast_message(text: str, signal_type_filter: str = "all", match_url: str | None = None) -> list[dict]:
    """
    Envía `text` a todos los chat IDs suscritos en matches.db en raíz.
    - signal_type_filter: 'all' envía a todos; 'bet' solo a quienes tienen señal operable.
    Retorna lista de respuestas (una por suscriptor).
    """
    if not TELEGRAM_BOT_TOKEN:
        log_warning("TELEGRAM", "[SKIP] TELEGRAM_BOT_TOKEN no configurado")
        return [{"ok": False, "description": "Sin token"}]

    active_chat_ids = await asyncio.to_thread(_resolve_recipients, signal_type_filter)
    if active_chat_ids is None:
        log_warning("TELEGRAM", "[SKIP] Sin suscriptores en monitor_subscribers (DB settings)")
        return [{"ok": False, "description": "Sin suscriptores"}]
    if not active_chat_ids:
        return []

    # Formatear chat IDs en verde
    chats_str = ", ".join(f"{COLOR_GREEN}{cid}{COLOR_RESET}" for cid in active_chat_ids)
    url_line = f"\n   {match_url}" if match_url else ""
    log_info("TELEGRAM", f"{COLOR_GREEN}[BROADCAST]{COLOR_RESET} Enviando {chats_str}{url_line}")

    results = []
    for chat_id in active_chat_ids:
        res = await _send_to_chat(chat_id, text, match_url=match_url)
        if not res.get("ok"):
            log_error("TELEGRAM", f"[FAIL] chat_id={chat_id} | {res.get('description', '?')}")
        results.append(res)

    return results


def _send_photo_request(
    chat_id: int, photo: bytes, caption: str, match_url: str | None = None
) -> dict:
    """
    Sube una imagen a Telegram (sendPhoto) vía multipart/form-data.

    Detalle: la Bot API NO acepta sendPhoto con cuerpo JSON; la imagen debe ir
    como multipart. Se usa `requests` (ya presente en el entorno) por ser la vía
    más robusta para el upload binario.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data: dict = {
        "chat_id": str(chat_id),
        "caption": caption or "",
        "parse_mode": "HTML",
    }
    if match_url:
        data["reply_markup"] = json.dumps(
            {"inline_keyboard": [[{"text": "📱 Sofascore", "url": match_url}]]}
        )
    try:
        import requests

        with requests.post(
            url,
            data=data,
            files={"photo": ("apuesta_q4.jpg", photo, "image/jpeg")},
            timeout=45,
        ) as resp:
            try:
                return resp.json()
            except Exception:
                return {"ok": False, "description": f"HTTP {resp.status_code}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "description": str(e)}


async def broadcast_photo(
    photo: bytes, caption: str, signal_type_filter: str = "bet", match_url: str | None = None
) -> list[dict]:
    """
    Envía una imagen con caption a los suscriptores, con la misma política de
    destinatarios que `broadcast_message`. Devuelve una respuesta por suscriptor.
    """
    if not TELEGRAM_BOT_TOKEN:
        log_warning("TELEGRAM", "[SKIP] TELEGRAM_BOT_TOKEN no configurado")
        return [{"ok": False, "description": "Sin token"}]

    active_chat_ids = await asyncio.to_thread(_resolve_recipients, signal_type_filter)
    if active_chat_ids is None:
        log_warning("TELEGRAM", "[SKIP] Sin suscriptores en monitor_subscribers (DB settings)")
        return [{"ok": False, "description": "Sin suscriptores"}]
    if not active_chat_ids:
        return []

    chats_str = ", ".join(f"{COLOR_GREEN}{cid}{COLOR_RESET}" for cid in active_chat_ids)
    log_info(
        "TELEGRAM",
        f"{COLOR_GREEN}[BROADCAST-IMG]{COLOR_RESET} Enviando imagen ({len(photo)} bytes) a {chats_str}",
    )

    results = []
    for chat_id in active_chat_ids:
        res = await asyncio.to_thread(_send_photo_request, chat_id, photo, caption, match_url)
        if not res.get("ok"):
            log_error("TELEGRAM", f"[FAIL] chat_id={chat_id} (imagen) | {res.get('description', '?')}")
        results.append(res)
    return results


_CAPTION_MAX = 1024  # límite de Telegram para el caption de una foto


def _photo_caption(text: str) -> str:
    """Recorta el texto del aviso para usarlo como caption de la foto."""
    if len(text) <= _CAPTION_MAX:
        return text
    return text[: _CAPTION_MAX - 3].rstrip() + "..."


def _is_real_bet_signal(signal: str | None) -> bool:
    """True si la señal es una apuesta operable (BET y no NO_BET)."""
    value = str(signal or "")
    return "BET" in value and "NO_BET" not in value


def _hypothetical_result(is_home: bool, match_data: dict | None) -> str | None:
    """
    Resultado que HUBIERA tenido el pick de un modelo que no apostó, según el
    marcador final del partido. None si hay empate o no hay marcador.
    """
    score = (match_data or {}).get("score") or {}
    try:
        home = float(score.get("home"))
        away = float(score.get("away"))
    except (TypeError, ValueError):
        return None
    if home == away:
        return None
    home_won = home > away
    won = home_won if is_home else not home_won
    return "would_win" if won else "would_loss"


def _inference_json(log: dict) -> dict | None:
    """Normaliza `inference_json` (str JSON o dict) para leer el umbral del modelo."""
    value = log.get("inference_json")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    return value if isinstance(value, dict) else None


async def _send_bet_photo(
    text: str,
    bets: list[dict],
    match_data: dict | None,
    match_url: str | None,
    *,
    minute: int | None = None,
    q_key: str | None = None,
    scheduled_ts: int | None = None,
    is_result: bool = False,
    include_chart: bool = True,
) -> list[dict] | None:
    """
    Intenta enviar el aviso como imagen (tarjeta + gráfica opcional).
    Devuelve None si no se pudo generar la imagen (el caller cae a texto plano).
    """
    try:
        from monitor_v3.notifications.bet_image import build_q4_alert_image

        photo = await asyncio.to_thread(
            build_q4_alert_image,
            match_data,
            bets,
            minute=minute,
            q_key=q_key,
            scheduled_ts=scheduled_ts,
            is_result=is_result,
            include_chart=include_chart,
        )
    except Exception as e:  # noqa: BLE001
        log_warning("TELEGRAM", f"[IMG] Fallo generando tarjeta Q4 ({type(e).__name__}: {e})")
        return None

    if not photo:
        return None

    results = await broadcast_photo(photo, _photo_caption(text), match_url=match_url)
    # Si TODOS los envíos fallaron, se degrada a texto plano desde el caller.
    if results and any(r.get("ok") for r in results):
        return results
    return None


def _decimal_odds_for_side(match_data: dict | None, is_home: bool) -> float | None:
    """Cuota decimal in-play (ML) del lado apostado, si el payload la trae."""
    odds_live = (match_data or {}).get("odds_live") or {}
    key = "ml_home" if is_home else "ml_away"
    value = odds_live.get(key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 1.0 else None



async def send_telegram_message(text: str) -> dict:
    """
    Compatibilidad: envía a todos los suscriptores y retorna el primer resultado.
    """
    results = await broadcast_message(text)
    return results[0] if results else {"ok": False, "description": "Sin suscriptores"}


async def edit_telegram_message(message_id: int, new_text: str) -> dict:
    """
    Edita un mensaje existente. Solo aplica al primer suscriptor (el que originó el msg).
    Para alertas de monitoreo automático se usa send en lugar de edit.
    """
    if not TELEGRAM_BOT_TOKEN:
        return {"ok": False, "description": "Sin token"}

    subscribers = await asyncio.to_thread(_get_subscribers_from_db)
    if not subscribers:
        return {"ok": False, "description": "Sin suscriptores"}

    # Editar solo en el primer chat (el mensaje_id es específico por chat)
    first_chat_id = next(iter(subscribers))
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"
    payload = {
        "chat_id": first_chat_id,
        "message_id": message_id,
        "text": new_text,
        "parse_mode": "HTML"
    }
    return await asyncio.to_thread(_send_request, url, payload)


def format_bet_message(model: str, picked_team: str, is_home: bool, signal: str, confidence: float | None = None) -> str:
    """
    Formatea el mensaje base de Telegram de acuerdo con las especificaciones.
    """
    side_emoji = "🏠" if is_home else "✈️"
    clean_team = html.escape(str(picked_team), quote=False)
    
    if confidence is not None and confidence > 0:
        # Normalize in case confidence is expressed as a percentage > 1 (e.g. 29.0 instead of 0.29)
        if confidence > 1.0:
            confidence = confidence / 100.0
            
        conf_pct = int(round(confidence * 100))
        model_str = f"[{model} {conf_pct}%]"
        is_low_conf = (round(confidence, 4) < 0.30)
    else:
        model_str = f"[{model}]"
        is_low_conf = False

    if "LATE" in signal:
        prefix = "🟡⚪️" if is_low_conf else "⚪️"
        return f"{prefix} APUESTA TARDIA Q4 {model_str} → {side_emoji} {clean_team}"
    elif "BET" in signal and "NO_BET" not in signal:
        prefix = "🟡" if is_low_conf else "🟢"
        return f"{prefix} APUESTA Q4 {model_str} → {side_emoji} {clean_team}"
    elif "NO_BET" in signal:
        return f"⚪️ NO BET {model_str} → {side_emoji} {clean_team}"
    else:
        return f"⚪️ NO BET {model_str} → {side_emoji} {clean_team}"



def _format_telegram_details(
    text: str,
    home_team: str | None,
    away_team: str | None,
    league: str | None,
    scheduled_ts: int | None,
    minute: int | None,
    q_key: str | None,
    q_home: int | None,
    q_away: int | None,
    is_final: bool = False
) -> str:
    """Format the multiline detailed match info block for Telegram."""
    details = []
    
    # Line 2: {home} vs {away}
    if home_team and away_team:
        clean_home = html.escape(str(home_team), quote=False)
        clean_away = html.escape(str(away_team), quote=False)
        details.append(f"{clean_home} vs {clean_away}")
        
    # Line 3: {league} (raw string, no prefix)
    if league:
        clean_league = html.escape(str(league), quote=False)
        details.append(clean_league)
        
    # Line 4: {time_str} | MIN {min_actual} | {q_key}: {q_home} - {q_away}
    if scheduled_ts:
        time_str = _format_utc6_datetime(scheduled_ts)
        if is_final:
            q_home_val = q_home if q_home is not None else 0
            q_away_val = q_away if q_away is not None else 0
            q_disp = f"{q_key} {q_home_val} - {q_away_val}" if q_key else "Q4 0 - 0"
            details.append(f"{time_str} | {q_disp}")
        else:
            min_disp = f"MIN {minute}" if minute is not None else "MIN -"
            q_home_val = q_home if q_home is not None else 0
            q_away_val = q_away if q_away is not None else 0
            q_disp = f"{q_key}: {q_home_val} - {q_away_val}" if q_key else "Q4: 0 - 0"
            details.append(f"{time_str} | {min_disp} | {q_disp}")
        
    if details:
        return f"{text}\n" + "\n".join(details)
    return text


async def send_bet_alert(
    model: str, picked_team: str, is_home: bool, signal: str,
    match_id: str | None = None,
    match_data: dict | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    minute: int | None = None,
    q_key: str | None = None,
    q_home: int | None = None,
    q_away: int | None = None,
    league: str | None = None,
    scheduled_ts: int | None = None,
    confidence: float | None = None
) -> dict:
    """
    Despacha una alerta de apuesta en vivo a todos los suscriptores.
    Retorna el primer resultado de la lista para compatibilidad con el caller.
    """
    base_text = format_bet_message(model, picked_team, is_home, signal, confidence=confidence)
    text = _format_telegram_details(
        base_text, home_team, away_team, league, scheduled_ts, minute, q_key, q_home, q_away
    )

    match_url = None
    if match_id:
        match_url = _sofascore_match_url(match_id, match_data=match_data, home_team=home_team, away_team=away_team)

    # Las apuestas solo van a suscriptores que quieren señales operable (todos excepto info_only)
    bets = [
        {
            "model": model,
            "signal": signal,
            "picked_team": picked_team,
            "is_home": is_home,
            "confidence": confidence,
            "decimal_odds": _decimal_odds_for_side(match_data, is_home),
            "inference_json": None,
        }
    ]
    results = await _send_bet_photo(
        text,
        bets,
        match_data,
        match_url,
        minute=minute,
        q_key=q_key,
        scheduled_ts=scheduled_ts,
        include_chart=False,
    )
    if results is None:
        # Sin imagen (o todos los envíos de imagen fallaron): aviso en texto plano
        results = await broadcast_message(text, signal_type_filter="bet", match_url=match_url)

    res = dict(results[0]) if results else {"ok": False, "description": "Sin suscriptores"}
    res["match_url"] = match_url
    return res


def model_display_sort_key(model_name: str) -> int:
    """
    Prioridad de visualización para que m27_v5_3 siempre aparezca hasta arriba
    tanto en alertas de apuesta como en resultados finales FT.
    """
    order = ["m27_v5_3", "m27_v5_2", "v6_2", "m27_v3", "m27_v5_1", "m27_v5", "m27_v4"]
    m_clean = str(model_name).strip().lower()
    for idx, target in enumerate(order):
        if target in m_clean:
            return idx
    return 99


def format_combined_bet_message(predictions: dict, home_team: str, away_team: str) -> str:
    """
    Formatea el mensaje base de Telegram para señales múltiples combinadas.
    Distingue explícitamente qué modelos están en APUESTA (BET) y cuáles están en [NO BET].
    m27_v5_3 se posiciona siempre hasta arriba en el reporte.
    """
    is_late = any("LATE" in pred.get("signal", "") for pred in predictions.values())
    has_real_bet = any("BET" in pred.get("signal", "") and "NO_BET" not in pred.get("signal", "") for pred in predictions.values())
    
    if is_late:
        title = "⚪️ APUESTA TARDIA Q4"
    elif has_real_bet:
        title = "🟢 APUESTA Q4"
    else:
        title = "⚪️ NO BET Q4"
        
    lines = [title]
    sorted_models = sorted(predictions.keys(), key=model_display_sort_key)
    for model in sorted_models:
        pred = predictions[model]
        sig = str(pred.get("signal", ""))
        raw_sig = str(pred.get("raw_model_signal", sig))
        is_bet = ("BET" in sig and "NO_BET" not in sig) or ("BET" in raw_sig and "NO_BET" not in raw_sig)

        conf = pred.get("confidence")
        if conf is not None:
            if conf > 1.0:
                conf = conf / 100.0
            conf_pct = int(round(conf * 100))
            model_conf_str = f"{model} {conf_pct}%"
        else:
            model_conf_str = f"{model}"
            
        picked_side = pred.get("pick")
        if not picked_side or picked_side == "NONE":
            p_home = pred.get("p_home_win")
            if p_home is not None:
                picked_side = "HOME" if float(p_home) >= 0.5 else "AWAY"
        is_home_pick = (str(picked_side).upper() == "HOME")
        side_emoji = "🏠" if is_home_pick else "✈️"
        picked_team_name = home_team if is_home_pick else away_team
        clean_team = html.escape(str(picked_team_name), quote=False)
        
        # Umbral específico del modelo (usar &lt; para parse_mode HTML)
        threshold_note = "&lt;75%" if ("v5_2" in model or "v5_3" in model) else ("&lt;68%" if "v5" in model else "&lt;65%")

        if is_bet:
            lines.append(f"{model_conf_str} → {side_emoji} {clean_team}")
        else:
            lines.append(f"{model_conf_str} [NO BET ({threshold_note})] → {side_emoji} {clean_team}")
        
    return "\n".join(lines)


async def send_combined_bet_alert(
    predictions: dict,
    match_id: str | None = None,
    match_data: dict | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    minute: int | None = None,
    q_key: str | None = None,
    q_home: int | None = None,
    q_away: int | None = None,
    league: str | None = None,
    scheduled_ts: int | None = None
) -> dict:
    """
    Despacha una alerta de apuesta combinada en vivo a todos los suscriptores.
    """
    base_text = format_combined_bet_message(predictions, home_team, away_team)
    text = _format_telegram_details(
        base_text, home_team, away_team, league, scheduled_ts, minute, q_key, q_home, q_away
    )
    
    match_url = None
    if match_id:
        match_url = _sofascore_match_url(match_id, match_data=match_data, home_team=home_team, away_team=away_team)
        
    bets = []
    for model_name, pred in predictions.items():
        is_home = str(pred.get("pick") or "").upper() == "HOME"
        p_home = pred.get("p_home_win")
        try:
            p_home = float(p_home) if p_home is not None else None
        except (TypeError, ValueError):
            p_home = None
        prob = None
        if p_home is not None:
            prob = p_home if is_home else (1.0 - p_home)
        bets.append(
            {
                "model": model_name,
                "signal": pred.get("signal"),
                "picked_team": home_team if is_home else away_team,
                "is_home": is_home,
                "confidence": pred.get("confidence"),
                "probability": prob,
                "decimal_odds": _decimal_odds_for_side(match_data, is_home),
                "inference_json": pred.get("inference_json"),
            }
        )

    results = await _send_bet_photo(
        text,
        bets,
        match_data,
        match_url,
        minute=minute,
        q_key=q_key,
        scheduled_ts=scheduled_ts,
        include_chart=False,
    )
    if results is None:
        # Sin imagen (o todos los envíos de imagen fallaron): aviso en texto plano
        results = await broadcast_message(text, signal_type_filter="bet", match_url=match_url)

    res = dict(results[0]) if results else {"ok": False, "description": "Sin suscriptores"}
    res["match_url"] = match_url
    return res


def format_combined_final_message(logs: list, home_team: str, away_team: str) -> str:
    """
    Formatea el mensaje base de Telegram para confirmaciones finales de múltiples modelos,
    homologándolo al formato de las apuestas combinadas (sin prefijar ✅/❌ al título general,
    sino colocándolo al inicio de la línea de cada modelo, y mostrando '🔴 model NO BET' para no operables).
    """
    # Determinar si alguna apuesta es tardía o de alta confianza para el prefijo
    is_late = any("LATE" in (log.get("signal_type") or "") for log in logs)
    
    has_high_conf = False
    for log in logs:
        # Solo consideramos la confianza de apuestas reales
        if "BET" in (log.get("signal_type") or ""):
            conf = log.get("confidence")
            if conf is not None:
                if conf > 1.0:
                    conf = conf / 100.0
                if conf >= 0.30:
                    has_high_conf = True
                    break
                
    is_low_conf = not has_high_conf
    
    if is_late:
        prefix = "🟡⚪️" if is_low_conf else "⚪️"
        title = f"{prefix} RESULTADO TARDIO Q4"
    else:
        prefix = "🟡" if is_low_conf else "🟢"
        title = f"{prefix} RESULTADO Q4"
        
    lines = [title]
    sorted_logs = sorted(logs, key=lambda l: model_display_sort_key(l.get("model_version", "")))
    for log in sorted_logs:
        model = log.get("model_version")
        sig = log.get("signal_type") or ""
        
        if "BET" not in sig or "NO_BET" in sig:
            # Es NO_BET!
            conf = log.get("confidence")
            picked_side = log.get("picked_side")
            
            # Intentar recuperar de inference_json si no viene en las columnas directas o si vino NONE
            if (not conf or conf == 0.0 or not picked_side or picked_side == "NONE") and log.get("inference_json"):
                try:
                    inf = json.loads(log["inference_json"]) if isinstance(log["inference_json"], str) else log["inference_json"]
                    conf = conf or inf.get("confidence")
                    p_home = inf.get("p_home_win")
                    inf_pick = inf.get("pick")
                    if inf_pick and inf_pick != "NONE":
                        picked_side = inf_pick
                    elif p_home is not None:
                        picked_side = "HOME" if float(p_home) >= 0.5 else "AWAY"
                        conf = conf or max(float(p_home), 1.0 - float(p_home))
                except Exception:
                    pass

            if (not picked_side or picked_side == "NONE") and log.get("p_home_win") is not None:
                p_h = float(log["p_home_win"])
                picked_side = "HOME" if p_h >= 0.5 else "AWAY"
                conf = conf or max(p_h, 1.0 - p_h)

            if conf and conf > 0 and picked_side and picked_side != "NONE":
                if conf > 1.0:
                    conf = conf / 100.0
                conf_pct = int(round(conf * 100))
                is_home_pick = (str(picked_side).upper() == "HOME")
                side_emoji = "🏠" if is_home_pick else "✈️"
                picked_team_name = home_team if is_home_pick else away_team
                clean_team = html.escape(str(picked_team_name), quote=False)
                threshold_note = "&lt;75%" if ("v5_2" in model or "v5_3" in model) else ("&lt;68%" if "v5" in model else "&lt;65%")
                
                lines.append(f"🔴 {model} {conf_pct}% [NO BET ({threshold_note})] → {side_emoji} {clean_team}")
            else:
                lines.append(f"🔴 {model} NO BET")
        else:
            conf = log.get("confidence")
            if conf is not None:
                if conf > 1.0:
                    conf = conf / 100.0
                conf_pct = int(round(conf * 100))
                model_conf_str = f"{model} {conf_pct}%"
            else:
                model_conf_str = f"{model}"
                
            picked_side = log.get("picked_side")
            is_home_pick = (str(picked_side).upper() == "HOME")
            side_emoji = "🏠" if is_home_pick else "✈️"
            picked_team_name = home_team if is_home_pick else away_team
            clean_team = html.escape(str(picked_team_name), quote=False)
            
            result = log.get("result")
            res_emoji = "✅" if result == "win" else "❌"
            
            lines.append(f"{res_emoji} {model_conf_str} → {side_emoji} {clean_team}")
        
    return "\n".join(lines)


async def send_combined_final_confirmation(
    logs: list,
    match_id: str | None = None,
    match_data: dict | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    minute: int | None = None,
    q_key: str | None = None,
    q_home: int | None = None,
    q_away: int | None = None,
    league: str | None = None,
    scheduled_ts: int | None = None
) -> dict:
    """
    Consolida la verificación final del partido para múltiples modelos en un solo mensaje.
    """
    base_text = format_combined_final_message(logs, home_team, away_team)
    text = _format_telegram_details(
        base_text, home_team, away_team, league, scheduled_ts, minute, q_key, q_home, q_away, is_final=True
    )
            
    match_url = None
    if match_id:
        match_url = _sofascore_match_url(match_id, match_data=match_data, home_team=home_team, away_team=away_team)
        
    bets = []
    for log in logs:
        signal = str(log.get("signal_type") or "")
        is_real = _is_real_bet_signal(signal)
        is_home = str(log.get("picked_side") or "").upper() == "HOME"
        result = log.get("result")
        if not is_real:
            # El modelo no apostó: se muestra qué HUBIERA pasado con su pick.
            result = _hypothetical_result(is_home, match_data) or result
        bets.append(
            {
                "model": log.get("model_version"),
                "signal": signal,
                "picked_team": home_team if is_home else away_team,
                "is_home": is_home,
                "confidence": log.get("confidence"),
                "result": result,
                "decimal_odds": _decimal_odds_for_side(match_data, is_home),
                "inference_json": _inference_json(log),
            }
        )
    results = await _send_bet_photo(
        text,
        bets,
        match_data,
        match_url,
        minute=minute,
        q_key=q_key,
        scheduled_ts=scheduled_ts,
        is_result=True,
        include_chart=True,
    )
    if results is None:
        results = await broadcast_message(text, signal_type_filter="bet", match_url=match_url)
    res = dict(results[0]) if results else {"ok": False, "description": "Sin suscriptores"}
    res["match_url"] = match_url
    return res


async def send_final_confirmation(
    model: str, picked_team: str, is_home: bool, signal: str,
    outcome: str, original_message_id: int = None,
    match_id: str | None = None,
    match_data: dict | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    minute: int | None = None,
    q_key: str | None = None,
    q_home: int | None = None,
    q_away: int | None = None,
    league: str | None = None,
    scheduled_ts: int | None = None,
    confidence: float | None = None
) -> dict:
    """
    Consolida la verificación final del partido, homologando el formato
    con el de las apuestas combinadas (sin prefijar ✅/❌ al título general,
    sino colocándolo al inicio de la línea de cada modelo, y sin MIN).
    """
    is_low_conf = False
    if confidence is not None:
        conf_val = confidence / 100.0 if confidence > 1.0 else confidence
        is_low_conf = (round(conf_val, 4) < 0.30)
        
    is_late = "LATE" in signal
    if is_late:
        prefix = "🟡⚪️" if is_low_conf else "⚪️"
        title = f"{prefix} RESULTADO TARDIO Q4"
    else:
        prefix = "🟡" if is_low_conf else "🟢"
        title = f"{prefix} RESULTADO Q4"
        
    conf_pct_str = ""
    if confidence is not None:
        conf_val = confidence / 100.0 if confidence > 1.0 else confidence
        conf_pct_str = f" {int(round(conf_val * 100))}%"
        
    side_emoji = "🏠" if is_home else "✈️"
    res_emoji = "✅" if outcome == "win" else "❌"
    model_line = f"{res_emoji} {model}{conf_pct_str} → {side_emoji} {picked_team}"
    
    base_text = f"{title}\n{model_line}"
    
    text = _format_telegram_details(
        base_text, home_team, away_team, league, scheduled_ts, minute, q_key, q_home, q_away, is_final=True
    )

    match_url = None
    if match_id:
        match_url = _sofascore_match_url(match_id, match_data=match_data, home_team=home_team, away_team=away_team)

    bets = [
        {
            "model": model,
            "signal": signal,
            "picked_team": picked_team,
            "is_home": is_home,
            "confidence": confidence,
            "result": (
                outcome
                if _is_real_bet_signal(signal)
                else (_hypothetical_result(is_home, match_data) or outcome)
            ),
            "decimal_odds": _decimal_odds_for_side(match_data, is_home),
        }
    ]
    results = await _send_bet_photo(
        text,
        bets,
        match_data,
        match_url,
        minute=minute,
        q_key=q_key,
        scheduled_ts=scheduled_ts,
        is_result=True,
        include_chart=True,
    )
    if results is None:
        results = await broadcast_message(text, signal_type_filter="bet", match_url=match_url)
    res = dict(results[0]) if results else {"ok": False, "description": "Sin suscriptores"}
    res["match_url"] = match_url
    return res


def _get_model_stats_dict(model: str) -> dict:
    """
    Consulta bet_monitor_log_v2 y retorna estadísticas crudas del modelo como dict.
    Estructura retornada:
    {
      "🟢":   {"win": N, "loss": N},
      "🟡":   {"win": N, "loss": N},
      "⚪️":  {"win": N, "loss": N},
      "🟡⚪️":{"win": N, "loss": N},
      "🔴":   {"count": N},          # NO_BET
    }
    """
    empty = {
        "🟢":    {"win": 0, "loss": 0},
        "🟡":    {"win": 0, "loss": 0},
        "⚪️":   {"win": 0, "loss": 0},
        "🟡⚪️": {"win": 0, "loss": 0},
        "🔴":    {"count": 0},
    }
    db_path = Path(get_real_db_path())
    if not db_path.exists():
        return empty

    try:
        tz = timezone(timedelta(hours=UTC_OFFSET_HOURS))
        today_str = datetime.now(tz).strftime("%Y-%m-%d")
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT signal_type, confidence, result
                FROM bet_monitor_log_v2
                WHERE model_version = ? AND created_at >= ?
            """, (model, today_str))
            rows = cursor.fetchall()

        for r in rows:
            sig  = r["signal_type"] or ""
            conf = r["confidence"]  or 0.0
            res  = r["result"]      or ""

            # Normalizar confianza
            if conf > 1.0:
                conf = conf / 100.0

            # ── NO_BET ──────────────────────────────────
            if "NO_BET" in sig or "BET" not in sig:
                empty["🔴"]["count"] += 1
                continue

            is_late     = "LATE" in sig
            is_low_conf = round(conf, 4) < 0.30

            cat = ("🟡⚪️" if is_low_conf else "⚪️") if is_late else ("🟡" if is_low_conf else "🟢")

            # win/loss: acepta tanto 'win/loss' como 'hit/miss'
            if res in ("win", "hit"):
                empty[cat]["win"]  += 1
            elif res in ("loss", "miss"):
                empty[cat]["loss"] += 1

        return empty

    except Exception as e:
        log_error("TELEGRAM", f"[DB] Error leyendo stats {e}: {e}")
        return empty


def _visual_len(s: str) -> int:
    """Calcula la longitud visual considerando emojis como 2 espacios de ancho."""
    emojis = ["✅", "❌", "🟢", "🟡", "⚪", "🔴"]
    clean_s = s.replace("\ufe0f", "")
    length = 0
    for char in clean_s:
        if char in emojis:
            length += 2
        else:
            length += 1
    return length


def _pad_cell(text: str, target_width: int) -> str:
    """Rellena la celda con espacios basándose en su longitud visual."""
    v_len = _visual_len(text)
    padding_needed = max(0, target_width - v_len)
    return text + (" " * padding_needed)


def build_all_models_stats_message(active_models: list | None = None) -> str:
    """
    Construye un mensaje de texto con las estadísticas lado-a-lado de todos los modelos activos.
    Se formatea usando <pre>...</pre> para garantizar alineación de columnas perfecta.
    """
    from monitor_v3.config.constants import ACTIVE_MODELS as _DEFAULT_MODELS
    models = list(active_models or _DEFAULT_MODELS)

    # Recopilar stats de cada modelo
    all_stats = {m: _get_model_stats_dict(m) for m in models}

    BET_CATS = ["🟢", "🟡", "⚪️", "🟡⚪️"]
    COL_W = 24   # ancho visual de cada columna

    # Encabezados de columnas (ej. "v6_2 Stats            m27_v3 Stats")
    header_parts = [_pad_cell(f"{m} Stats", COL_W) for m in models]
    header = "    ".join(header_parts).rstrip()

    table_lines = [header]

    for cat in BET_CATS:
        for outcome, emoji in (("win", "✅"), ("loss", "❌")):
            cells = []
            for m in models:
                st  = all_stats[m][cat]
                n   = st["win"] if outcome == "win" else st["loss"]
                tot = st["win"] + st["loss"]
                
                # Ajustar espacio tras los emojis para alinear correctamente los números
                clean_cat = cat.replace("\ufe0f", "")
                emoji_spacing = " " if len(clean_cat) > 1 else "  "
                
                if tot > 0:
                    pct  = int(round(n * 100.0 / tot))
                    cell = f"{emoji}{cat}{emoji_spacing}{n}   {pct}%"
                else:
                    cell = f"{emoji}{cat}{emoji_spacing}0   0%"
                cells.append(_pad_cell(cell, COL_W))
            table_lines.append("    ".join(cells).rstrip())

    # ── fila NO_BET ────────────────────────────────────────────────────────
    nobet_cells = []
    for m in models:
        count = all_stats[m]["🔴"]["count"]
        # Calcular total de todos los logs del modelo
        total_all = count
        for cat in BET_CATS:
            total_all += all_stats[m][cat]["win"] + all_stats[m][cat]["loss"]
        
        if total_all > 0:
            pct = int(round(count * 100.0 / total_all))
            cell = f"🔴 {count}   {pct}%"
        else:
            cell = f"🔴 0   0%"
        nobet_cells.append(_pad_cell(cell, COL_W))
    table_lines.append("    ".join(nobet_cells).rstrip())

    table_content = "\n".join(table_lines)
    return f"📊 <b>Estadísticas de Modelos</b>\n<pre>{table_content}</pre>"


def _get_model_stats_text(model: str) -> str:
    """Retorna un bloque de texto de stats para un único modelo (compatibilidad legada)."""
    stats = _get_model_stats_dict(model)
    BET_CATS = ["🟢", "🟡", "⚪️", "🟡⚪️"]
    lines = [f"\n{model} Stats"]

    for cat in BET_CATS:
        hit  = stats[cat]["win"]
        miss = stats[cat]["loss"]
        tot  = hit + miss
        if tot > 0:
            lines.append(f"✅{cat}  {hit}   {int(round(hit * 100.0 / tot))}%")
            lines.append(f"❌{cat}  {miss}   {int(round(miss * 100.0 / tot))}%")

    nobet = stats["🔴"]["count"]
    if nobet > 0:
        lines.append(f"🔴 {nobet}")

    return "\n".join(lines) if len(lines) > 1 else ""


async def send_stats_message(active_models: list | None = None) -> dict:
    """
    Construye y transmite el mensaje de estadísticas de todos los modelos activos
    a todos los suscriptores configurados. Se envía como mensaje independiente.
    """
    text = await asyncio.to_thread(build_all_models_stats_message, active_models)
    if not text:
        return {"ok": False, "description": "Sin datos de stats aún"}
    results = await broadcast_message(text, signal_type_filter="bet")
    return dict(results[0]) if results else {"ok": False, "description": "Sin suscriptores"}


def build_daily_summary_text() -> str:
    """
    Resumen del día desde las tablas _v3: apuestas ganadas/perdidas/push/pendientes,
    no-bet y acierto, con desglose por modelo.
    """
    db_path = Path(get_real_db_path())
    if not db_path.exists():
        return ""
    tz = timezone(timedelta(hours=UTC_OFFSET_HOURS))
    today = datetime.now(tz).strftime("%Y-%m-%d")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT model_version, signal_type, result FROM bet_monitor_log_v3 "
                "WHERE date(created_at) = ?",
                (today,),
            ).fetchall()
    except Exception as e:
        log_error("TELEGRAM", f"[DB] Error en resumen diario: {e}")
        return ""

    if not rows:
        return ""

    win = loss = push = pending = nobet = 0
    per_model: dict[str, dict] = {}
    for r in rows:
        model = r["model_version"] or "?"
        m = per_model.setdefault(model, {"bets": 0, "win": 0, "loss": 0, "pending": 0})
        sig = r["signal_type"] or ""
        if "NO_BET" in sig or "BET" not in sig:
            nobet += 1
            continue
        m["bets"] += 1
        res = (r["result"] or "").lower()
        if res in ("win", "hit"):
            win += 1
            m["win"] += 1
        elif res in ("loss", "miss"):
            loss += 1
            m["loss"] += 1
        elif res == "push":
            push += 1
        else:
            pending += 1
            m["pending"] += 1

    bets = win + loss + push + pending
    resolved = win + loss
    acc = f"{int(round(win * 100 / resolved))}%" if resolved else "—"

    lines = [f"📊 <b>Resumen del día</b> ({today})", ""]
    lines.append(f"Apuestas: <b>{bets}</b> | ✅ {win}  ❌ {loss}  ➖ {push}  ⏳ {pending}")
    lines.append(f"No bet: <b>{nobet}</b> | Acierto (resueltas): <b>{acc}</b>")
    lines.append("")
    lines.append("<b>Por modelo</b>")
    for model, s in per_model.items():
        r = s["win"] + s["loss"]
        a = f"{int(round(s['win'] * 100 / r))}%" if r else "—"
        lines.append(f"  {model}: BET {s['bets']} | ✅ {s['win']}  ❌ {s['loss']}  ⏳ {s['pending']} | {a}")
    return "\n".join(lines)


async def send_daily_summary() -> dict:
    """Construye y transmite el resumen del día a los suscriptores."""
    text = await asyncio.to_thread(build_daily_summary_text)
    if not text:
        return {"ok": False, "description": "Sin datos para hoy"}
    return await broadcast_message(text, signal_type_filter="bet")


def _normalize_sofascore_slug(value: str | None) -> str:
    """Normalize team/event names for SofaScore URL."""
    text = (value or "").strip().lower()
    if not text:
        return "unknown"
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text or "unknown"


def _sofascore_match_url(match_id: str, match_data: dict | None = None, home_team: str | None = None, away_team: str | None = None) -> str:
    """
    Construye la URL canónica de un partido en SofaScore.

    Estructura real:
        https://www.sofascore.com/{deporte}/match/{slug}/{custom_id}
    donde `slug` es el `event.slug` del API (p. ej. "al-ula-al-kuwait"; NO siempre
    es `home-away`, a veces SofaScore lo devuelve en orden inverso) y `custom_id`
    es el `event.customId` corto (p. ej. "fEicsGRpi"). Sin `custom_id` la ruta
    resuelve a 404. El fragmento `#id:{match_id}` es solo informativo/legacy.
    """
    event_slug = "unknown"
    custom_id = ""
    home_slug = _normalize_sofascore_slug(home_team) if home_team else "unknown"
    away_slug = _normalize_sofascore_slug(away_team) if away_team else "unknown"

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

    # El slug canónico es el event_slug del API; si falta, combinar home-away.
    slug = event_slug if event_slug != "unknown" else f"{home_slug}-{away_slug}"
    if custom_id:
        return f"https://www.sofascore.com/basketball/match/{slug}/{custom_id}#id:{match_id}"
    return f"https://www.sofascore.com/basketball/match/{slug}#id:{match_id}"


def _format_utc6_datetime(scheduled_ts: int) -> str:
    """Format an epoch timestamp into Spanish human date/time in UTC-6."""
    try:
        tz = timezone(timedelta(hours=-6))
        dt = datetime.fromtimestamp(scheduled_ts, tz=tz)
        
        days_es = ["Lu", "Ma", "Mi", "Ju", "Vi", "Sa", "Do"]
        months_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        
        day_abbrev = days_es[dt.weekday()]
        month_abbrev = months_es[dt.month - 1]
        day_num = dt.day
        
        hour_12 = dt.hour % 12
        if hour_12 == 0:
            hour_12 = 12
        am_pm = "am" if dt.hour < 12 else "pm"
        
        return f"{day_abbrev} {month_abbrev} {day_num}, {hour_12}:{dt.minute:02d} {am_pm}"
    except Exception:
        return "Desconocida"


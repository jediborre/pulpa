# =====================================================================
# REGLAS DE ARQUITECTURA, FORMATEO Y LOGGING DE PRODUCCIÓN (MANTENER):
# 1. ESTRUCTURA: Estrictamente modularizado (config, database, scrapers, 
#    models, notifications, utils) coordinados asíncronamente por main.py.
#    Cualquier aproximación monolítica de archivo único viola esta especificación.
# 2. INFRAESTRUCTURA DB: El archivo base SQLite se localiza exclusivamente en 
#    /match/matches.db y todas las tablas sin excepción finalizan con el sufijo '_v2'.
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
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bet_monitor_v2.config.constants import TELEGRAM_BOT_TOKEN
from bet_monitor_v2.database.connection import get_real_db_path
from bet_monitor_v2.utils.logger import log_info, log_warning, log_error, COLOR_GREEN, COLOR_RESET

# Clave donde el bot de Telegram guarda los suscriptores en la tabla settings
_SUBSCRIBERS_SETTING_KEY = "monitor_subscribers"


def _get_subscribers_from_db() -> dict[int, dict]:
    """
    Lee los chat IDs suscritos desde la tabla `settings` de match/matches.db.
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


async def broadcast_message(text: str, signal_type_filter: str = "all", match_url: str | None = None) -> list[dict]:
    """
    Envía `text` a todos los chat IDs suscritos en match/matches.db.
    - signal_type_filter: 'all' envía a todos; 'bet_only' solo a quienes tienen señal operable.
    Retorna lista de respuestas (una por suscriptor).
    """
    if not TELEGRAM_BOT_TOKEN:
        log_warning("TELEGRAM", "[SKIP] TELEGRAM_BOT_TOKEN no configurado")
        return [{"ok": False, "description": "Sin token"}]

    subscribers = await asyncio.to_thread(_get_subscribers_from_db)

    if not subscribers:
        log_warning("TELEGRAM", "[SKIP] Sin suscriptores en monitor_subscribers (DB settings)")
        return [{"ok": False, "description": "Sin suscriptores"}]

    # Filtrar destinatarios reales según las preferencias de señal
    active_chat_ids = []
    for chat_id, prefs in subscribers.items():
        sub_signal = prefs.get("signal_type", "all")
        if sub_signal == "bet_only" and signal_type_filter == "info":
            continue
        active_chat_ids.append(chat_id)

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
        return f"{prefix} APUESTA TARDIA Q4 {model_str} → {side_emoji} {picked_team}"
    elif "BET" in signal:
        prefix = "🟡" if is_low_conf else "🟢"
        return f"{prefix} APUESTA Q4 {model_str} → {side_emoji} {picked_team}"
    else:
        prefix = "🟡" if is_low_conf else "🟢"
        return f"{prefix} APUESTA Q4 {model_str} → {side_emoji} {picked_team}"



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
        details.append(f"{home_team} vs {away_team}")
        
    # Line 3: {league} (raw string, no prefix)
    if league:
        details.append(league)
        
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
    results = await broadcast_message(text, signal_type_filter="bet", match_url=match_url)
    res = dict(results[0]) if results else {"ok": False, "description": "Sin suscriptores"}
    res["match_url"] = match_url
    return res


def format_combined_bet_message(predictions: dict, home_team: str, away_team: str) -> str:
    """
    Formatea el mensaje base de Telegram para señales múltiples combinadas.
    """
    is_late = any("LATE" in pred.get("signal", "") for pred in predictions.values())
    
    has_high_conf = False
    for pred in predictions.values():
        conf = pred.get("confidence")
        if conf is not None:
            if conf > 1.0:
                conf = conf / 100.0
            if conf >= 0.30:
                has_high_conf = True
                break
                
    is_low_conf = not has_high_conf
    
    if is_late:
        prefix = "🟡⚪️" if is_low_conf else "⚪️"
        title = f"{prefix} APUESTA TARDIA Q4"
    else:
        prefix = "🟡" if is_low_conf else "🟢"
        title = f"{prefix} APUESTA Q4"
        
    lines = [title]
    for model, pred in predictions.items():
        conf = pred.get("confidence")
        if conf is not None:
            if conf > 1.0:
                conf = conf / 100.0
            conf_pct = int(round(conf * 100))
            model_conf_str = f"{model} {conf_pct}%"
        else:
            model_conf_str = f"{model}"
            
        picked_side = pred.get("pick")
        is_home_pick = (picked_side == "HOME")
        side_emoji = "🏠" if is_home_pick else "✈️"
        picked_team_name = home_team if is_home_pick else away_team
        
        lines.append(f"{model_conf_str} → {side_emoji} {picked_team_name}")
        
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
        title = f"{prefix} APUESTA TARDIA Q4"
    else:
        prefix = "🟡" if is_low_conf else "🟢"
        title = f"{prefix} APUESTA Q4"
        
    lines = [title]
    for log in logs:
        model = log.get("model_version")
        sig = log.get("signal_type") or ""
        
        if "BET" not in sig or "NO_BET" in sig:
            # Es NO_BET!
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
            is_home_pick = (picked_side == "HOME")
            side_emoji = "🏠" if is_home_pick else "✈️"
            picked_team_name = home_team if is_home_pick else away_team
            
            result = log.get("result")
            res_emoji = "✅" if result == "win" else "❌"
            
            lines.append(f"{res_emoji} {model_conf_str} → {side_emoji} {picked_team_name}")
        
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
        title = f"{prefix} APUESTA TARDIA Q4"
    else:
        prefix = "🟡" if is_low_conf else "🟢"
        title = f"{prefix} APUESTA Q4"
        
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
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT signal_type, confidence, result
                FROM bet_monitor_log_v2
                WHERE model_version = ?
            """, (model,))
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
    from bet_monitor_v2.config.constants import ACTIVE_MODELS as _DEFAULT_MODELS
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
    """Generate SofaScore URL for a match."""
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

    if event_slug != "unknown" and custom_id:
        return (
            "https://www.sofascore.com/basketball/match/"
            f"{event_slug}/{custom_id}#id:{match_id}"
        )

    return (
        "https://www.sofascore.com/basketball/match/"
        f"{home_slug}/{away_slug}#id:{match_id}"
    )


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

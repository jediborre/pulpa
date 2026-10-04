# =====================================================================
# MONITOR V3: DAEMON ASÍNCRONO DE MONITOREO Y APUESTAS DE BALONCESTO
# Basado en Extracción Nativa de API Móvil (Zero-Browser) + Pool Multi-JWT
# =====================================================================

import asyncio
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Configurar encoding en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor_v3.config.constants import (
    ACTIVE_MODELS,
    FINAL_FETCH_MIN_GP,
    POLL_INTERVAL_LIVE_SECS,
    POLL_INTERVAL_IDLE_SECS,
    POLL_NEAR_SECS,
    PRESTART_PROBE_MIN_SECS,
    PRESTART_PROBE_MAX_SECS,
    PRESTART_PROBE_BACKOFF,
    Q4_ONLY_EARLY_WAKE_MINUTE,
    SCHEDULE_REFRESH_HOURS,
    PENDING_RECHECK_SECS,
    SECS_PER_GAME_MIN,
    UTC_OFFSET_HOURS,
)
from monitor_v3.core.token_manager import get_token_pool
from monitor_v3.core.mobile_client import get_mobile_client
from monitor_v3.database.connection import get_db_connection
from monitor_v3.database.repository import (
    init_tables,
    sync_leagues_config,
    get_leagues_config,
    get_league_mode,
    save_schedule_matches,
    update_schedule_status,
    update_schedule_q4,
    get_pending_schedule_matches,
    save_quarter_scores,
    save_bet_log,
    save_eval_match_results,
    reconcile_pending_results,
)
from monitor_v3.models.evaluator import (
    GameWatcherState,
    evaluate_match_q4,
    load_models_to_cache,
)
from monitor_v3.notifications.telegram_bot import (
    send_combined_bet_alert,
    send_bet_alert,
    send_combined_final_confirmation,
    send_final_confirmation,
    send_stats_message,
)
from monitor_v3.scrapers.schedule_scraper import fetch_schedule_matches_for_date
from monitor_v3.scrapers.live_scraper import fetch_event_snapshot, fetch_live_events
from monitor_v3.scrapers.match_detail_scraper import fetch_match_by_id
from monitor_v3.utils.helpers import (
    format_human_time,
    calculate_ema_secs_per_gmin,
    calculate_jitter_sleep_secs,
    is_proxy_accessible,
)
from monitor_v3.utils.logger import (
    get_utc6_now,
    format_match_log,
    log_info,
    log_warning,
    log_error,
    COLOR_GREEN,
    COLOR_WARNING,
    COLOR_ERROR,
    COLOR_BRIGHT_RED,
    COLOR_RESET,
)

# Diccionario global de tareas watcher activas: match_id -> asyncio.Task
_watcher_tasks: dict[str, asyncio.Task] = {}
_total_watchers_spawned = 0
_probe_completed_count = 0

def colorize_scores(h, a) -> tuple[str, str]:
    """Colorea el marcador del ganador en verde y el perdedor en rojo."""
    try:
        h_val = int(h)
        a_val = int(a)
    except (ValueError, TypeError):
        return str(h), str(a)
    
    if h_val > a_val:
        return f"{COLOR_GREEN}{h}{COLOR_RESET}", f"{COLOR_BRIGHT_RED}{a}{COLOR_RESET}"
    elif a_val > h_val:
        return f"{COLOR_BRIGHT_RED}{h}{COLOR_RESET}", f"{COLOR_GREEN}{a}{COLOR_RESET}"
    else:
        return f"{h}", f"{a}"

def _infer_minute_from_pbp(data: dict) -> int | None:
    try:
        from models.common.pbp_utils import infer_minute_from_pbp
        return infer_minute_from_pbp(data)
    except Exception:
        gp = data.get("graph_points", [])
        return len(gp) if gp else None

async def _final_fetch_and_save(match_id: str, home: str, away: str) -> None:
    """
    Descarga el payload final FT vía API móvil y persiste scores y apuestas.
    """
    # Deduplicación: si ya se finalizó/notificó este partido, no repetir el FT
    # ni reenviar la confirmación a Telegram (evita spam si se relanza el watcher).
    try:
        with get_db_connection() as conn:
            _row = conn.execute(
                "SELECT final_fetched FROM bet_monitor_schedule_v3 WHERE match_id = ?",
                (match_id,)
            ).fetchone()
        if _row and _row["final_fetched"]:
            log_info("DESCARGA", f"[FT] Ya finalizado/notificado previamente, omitiendo | {home} vs {away}")
            return
    except Exception:
        pass

    try:
        data = await fetch_match_by_id(match_id, is_ft=True)
        gp_total = len(data.get("graph_points") or [])
        
        if gp_total < FINAL_FETCH_MIN_GP:
            if gp_total == 0:
                log_info("DESCARGA", f"[FT] Descartado (sin gráfica) | {home} vs {away}")
                update_schedule_status(match_id, status="done", skip_reason="no_graph")
                return
            log_warning("DESCARGA", f"[FT] Cierre con gráfica corta (gp={gp_total}) | {home} vs {away}")
            
        # 1. Guardar marcadores por cuarto
        score_data = data.get("score", {})
        quarters = score_data.get("quarters", {})
        save_quarter_scores(match_id, quarters)
        
        # 2. Guardar en matches.db histórica
        try:
            import match.db as db_mod
            legacy_conn = db_mod.get_conn(str(ROOT / "matches.db"))
            db_mod.init_db(legacy_conn)
            db_mod.save_match(legacy_conn, match_id, data)
            legacy_conn.close()
        except Exception:
            pass
            
        # 3. Liquidar apuestas pendientes
        reconcile_pending_results()
        
        # 4. Notificaciones finales de Telegram
        q4_ft_data = quarters.get("Q4") or {}
        q4_ft_home = q4_ft_data.get("home")
        q4_ft_away = q4_ft_data.get("away")
        
        with get_db_connection() as conn:
            sched_row = conn.execute(
                "SELECT league, scheduled_utc_ts FROM bet_monitor_schedule_v3 WHERE match_id = ?",
                (match_id,)
            ).fetchone()
            league = sched_row["league"] if sched_row else None
            scheduled_ts = sched_row["scheduled_utc_ts"] if sched_row else None

            cursor = conn.execute("""
                SELECT id, model_version, picked_side, signal_type, result,
                       inference_minute, confidence, actual_home_score, actual_away_score, target_quarter
                FROM bet_monitor_log_v3
                WHERE match_id = ?
            """, (match_id,))
            logs = [dict(r) for r in cursor.fetchall()]
            
            # Deduplicar por modelo
            deduped_logs = {}
            for log in logs:
                deduped_logs[log["model_version"]] = log
            logs = list(deduped_logs.values())
            
            # Comprobar antigüedad
            is_too_old = False
            if scheduled_ts and (time.time() - scheduled_ts > 21600):
                is_too_old = True
            
            has_operable_bet = any("BET" in (log.get("signal_type") or "") for log in logs)
            
            if has_operable_bet and not is_too_old:
                if len(logs) > 1:
                    log0 = logs[0]
                    target_quarter = log0["target_quarter"] or 4
                    q_key = f"Q{target_quarter}"
                    ft_q_data = quarters.get(q_key) or {}
                    ft_q_home = ft_q_data.get("home") if ft_q_data else q4_ft_home
                    ft_q_away = ft_q_data.get("away") if ft_q_data else q4_ft_away
                    
                    await send_combined_final_confirmation(
                        logs=logs,
                        match_id=match_id,
                        match_data=data,
                        home_team=home,
                        away_team=away,
                        minute=log0["inference_minute"],
                        q_key=q_key,
                        q_home=ft_q_home,
                        q_away=ft_q_away,
                        league=league,
                        scheduled_ts=scheduled_ts
                    )
                elif len(logs) == 1:
                    log = logs[0]
                    picked_side = log["picked_side"]
                    target_quarter = log["target_quarter"] or 4
                    q_key = f"Q{target_quarter}"
                    ft_q_data = quarters.get(q_key) or {}
                    ft_q_home = ft_q_data.get("home") if ft_q_data else log["actual_home_score"]
                    ft_q_away = ft_q_data.get("away") if ft_q_data else log["actual_away_score"]
                    
                    await send_final_confirmation(
                        model=log["model_version"],
                        picked_team=home if picked_side == "HOME" else away,
                        is_home=(picked_side == "HOME"),
                        signal=log["signal_type"],
                        outcome=log["result"],
                        match_id=match_id,
                        match_data=data,
                        home_team=home,
                        away_team=away,
                        minute=log["inference_minute"],
                        q_key=q_key,
                        q_home=ft_q_home,
                        q_away=ft_q_away,
                        league=league,
                        scheduled_ts=scheduled_ts,
                        confidence=log["confidence"]
                    )
                    
        update_schedule_status(match_id, status="done")
        log_info("DESCARGA", f"[FT] Final y liquidación de apuestas persistidos | {home} vs {away} (gp={gp_total})")
    except Exception as e:
        log_error("DESCARGA", f"[FT] Error procesando final {match_id}: {e}")
        update_schedule_status(match_id, status="done", skip_reason="final_fetch_failed")

async def _watch_match(match_id: str, match_row: dict, stop_event: asyncio.Event, ft_only: bool = False) -> None:
    """
    Máquina de estados independiente para el seguimiento en vivo de un partido.
    """
    home = match_row["home_team"]
    away = match_row["away_team"]
    league = match_row["league"]
    scheduled_ts = int(match_row["scheduled_utc_ts"])
    sched_label = datetime.fromtimestamp(scheduled_ts, tz=timezone(timedelta(hours=UTC_OFFSET_HOURS))).strftime("%H:%M")
    
    match_display = format_match_log(sched_label, match_id, home, away)
    log_info("MONITOREO", f"[WATCHER] Start: {match_display} | {COLOR_GREEN}SOLO FT{COLOR_RESET}" if ft_only else f"[WATCHER] Start: {match_display}")
    
    watcher_state = GameWatcherState(match_id)
    secs_per_gmin = float(SECS_PER_GAME_MIN)
    
    # 1. Validación de obsolescencia
    if time.time() - scheduled_ts > 12600:
        log_warning("MONITOREO", f"[WATCHER] Partido obsoleto | {match_display}")
        update_schedule_status(match_id, status="done", skip_reason="obsoleto_tiempo_superado")
        await _final_fetch_and_save(match_id, home, away)
        return
        
    # 2. Espera pasiva inicial si el partido es futuro
    now = time.time()
    if now < scheduled_ts - 60:
        sleep_secs = (scheduled_ts - 60) - now
        log_info("MONITOREO", f"[PROBE] {match_display} | {COLOR_GREEN}ETA {format_human_time(sleep_secs)}{COLOR_RESET}")
        
        remaining = sleep_secs
        while remaining > 0 and not stop_event.is_set():
            chunk = min(300.0, remaining)
            await asyncio.sleep(chunk)
            remaining = (scheduled_ts - 60) - time.time()
            
        if stop_event.is_set():
            return

    # Si es FT-Only, dormir hasta finalización estimada
    if ft_only:
        sleep_secs = max(60, (scheduled_ts + 7200) - time.time())
        log_info("MONITOREO", f"[FT-ONLY] Espera pasiva: {format_human_time(sleep_secs)} | {match_display}")
        await asyncio.sleep(sleep_secs)
        update_schedule_status(match_id, status="in_progress")
        await _final_fetch_and_save(match_id, home, away)
        return

    # 3. Modo Sonda Prestart
    in_probe_mode = True
    probe_delay = float(PRESTART_PROBE_MIN_SECS)
    
    while in_probe_mode and not stop_event.is_set():
        if time.time() - scheduled_ts > 12600:
            log_warning("MONITOREO", f"[PROBE] Partido obsoleto | {match_display}")
            await _final_fetch_and_save(match_id, home, away)
            return

        drift = time.time() - scheduled_ts
        if drift >= (secs_per_gmin * Q4_ONLY_EARLY_WAKE_MINUTE):
            log_info("MONITOREO", f"[PROBE] Drift Exit: {match_display}")
            break

        try:
            snapshot = await fetch_event_snapshot(match_id)
            status_type = snapshot.get("status_type", "").lower()
            
            if status_type in ("inprogress", "live"):
                log_info("MONITOREO", f"[PROBE] En Vivo: {match_display}")
                break
            elif status_type == "finished":
                log_info("MONITOREO", f"[PROBE] Finalizado detectado: {match_display}")
                await _final_fetch_and_save(match_id, home, away)
                return
        except Exception as e:
            log_error("MONITOREO", f"[PROBE] Error sondeo {match_display}: {e}")

        probe_sleep = calculate_jitter_sleep_secs(probe_delay, phase="error_retry")
        await asyncio.sleep(probe_sleep)
        probe_delay = min(float(PRESTART_PROBE_MAX_SECS), probe_delay * PRESTART_PROBE_BACKOFF)

    # 4. Espera hasta acercarse a la ventana de evaluación (min 27) usando el RELOJ
    #    FIABLE del snapshot (event.time.played), NO estimaciones de reloj de pared.
    #    Así, aunque el daemon se reinicie, el watcher sabe en qué minuto va el partido.
    EVAL_WAKE_MINUTE = max(0, Q4_ONLY_EARLY_WAKE_MINUTE - 2)
    while not stop_event.is_set():
        try:
            snap = await fetch_event_snapshot(match_id)
            st = snap.get("status_type", "").lower()
            if st == "finished":
                log_info("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Finalizado antes de la ventana → FT | {match_display}")
                await _final_fetch_and_save(match_id, home, away)
                return
            played = snap.get("game_seconds_played")
            if played is not None and (int(played) // 60) >= EVAL_WAKE_MINUTE:
                break
            # Respaldo si el API no expone el reloj: entrar al ver Q3/Q4.
            desc = (snap.get("status_description") or "").lower()
            if played is None and ("3rd quarter" in desc or "4th quarter" in desc):
                break
        except Exception as e:
            log_error("MONITOREO", f"[PROBE] Error sondeo {match_display}: {e}")
        await asyncio.sleep(POLL_NEAR_SECS)

    # 5. Monitoreo Activo en Vivo (Ventana Q3/Q4)
    try:
        with get_db_connection() as conn:
            existing_logs = conn.execute(
                "SELECT DISTINCT model_version FROM bet_monitor_log_v3 WHERE match_id = ? AND target_quarter = 4",
                (match_id,)
            ).fetchall()
            existing_models = {row["model_version"] for row in existing_logs}
    except Exception:
        existing_models = set()

    q4_done = {model: (model in existing_models) for model in ACTIVE_MODELS}
    last_gmin = None
    last_gmin_wall = 0.0

    log_info("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Iniciando monitoreo móvil en vivo | {match_display}")

    while not stop_event.is_set():
        if time.time() - scheduled_ts > 12600:
            log_warning("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Partido obsoleto | {match_display}")
            await _final_fetch_and_save(match_id, home, away)
            break

        try:
            full_data = await fetch_match_by_id(match_id, is_ft=False)
            match_meta = full_data.get("match", {})
            status_type = match_meta.get("status_type", "").lower()
            status_desc = match_meta.get("status_description", "") or ""
            score_data = full_data.get("score", {})
            home_score = score_data.get("home", 0)
            away_score = score_data.get("away", 0)

            if status_type == "finished":
                log_info("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Fin detectado → Derivando a FT | {match_display}")
                await _final_fetch_and_save(match_id, home, away)
                break

            # Minuto fiable: event.time.played = segundos acumulados del reloj de juego
            # (SofaScore lo expone aunque no muestre el minuto). Fallback a PBP/gráfica.
            played = match_meta.get("game_seconds_played")
            if played is not None:
                minute = int(played // 60)
            else:
                minute = _infer_minute_from_pbp(full_data)
                gp_count = len(full_data.get("graph_points", []) or [])
                if gp_count > 0 and (minute is None or gp_count > minute):
                    minute = gp_count

            # NOTA: el reloj `played` puede llegar a 40:00 (fin de regulación) mientras
            # el partido SIGUE en vivo (última posesión / tiros libres / revisiones) y el
            # marcador continúa cambiando. Por eso NO se usa `played >= regulación` para
            # cerrar el Q4: el ÚNICO indicador fiable es `status_type == "finished"`.

            period_lower = status_desc.lower()
            q_key = None
            if "1st quarter" in period_lower: q_key = "Q1"
            elif "2nd quarter" in period_lower: q_key = "Q2"
            elif "3rd quarter" in period_lower: q_key = "Q3"
            elif "4th quarter" in period_lower: q_key = "Q4"

            is_q4_active = (q_key == "Q4")
            home_col, away_col = colorize_scores(home_score, away_score)
            min_str = f"MIN~{minute}" if minute else "MIN~?"
            bracket_label = f"[{q_key or status_desc} | {min_str}]"
            
            log_info(
                "MONITOREO",
                f"{format_match_log(sched_label, match_id, home, away)} {bracket_label} Score: {home_col} - {away_col}",
                q4_orange=is_q4_active
            )

            if minute is not None:
                now_wall = time.monotonic()
                if last_gmin is not None and minute > last_gmin:
                    secs_per_gmin = calculate_ema_secs_per_gmin(secs_per_gmin, now_wall - last_gmin_wall, minute - last_gmin)
                last_gmin = minute
                last_gmin_wall = now_wall

                # Evaluación en Q4: AMBOS modelos activos (m27_v3 y v6_2) usan snapshot
                # minuto 27. (v6_2 tiene snapshot 36 por un fallo de diseño con leakage,
                # pero se evalúa igualmente al min 27; con el retraso no ha ido mal.)
                # Se corre en cuanto el reloj FIABLE alcanza el min 27; el clasificador
                # distingue ordinaria (<=33), tardía (34-35) y anulada (>=36).
                if minute >= Q4_ONLY_EARLY_WAKE_MINUTE and minute < 36:
                    eval_res = await evaluate_match_q4(match_id, full_data, watcher_state, forced_minute=minute)
                    if eval_res.get("ok"):
                        preds = eval_res.get("predictions", {})
                        operable_to_send = {}

                        for model, pred in preds.items():
                            if q4_done[model]:
                                continue

                            sig = pred.get("signal", "UNAVAILABLE")
                            if "BET" in sig:
                                operable_to_send[model] = pred
                            elif sig == "NO_BET":
                                inf_json = pred.get("inference_json") or {}
                                h2h = inf_json.get("q4", {}).get("h2h_available")
                                save_bet_log(
                                    match_id=match_id,
                                    model_version=model,
                                    inference_minute=pred.get("inference_minute"),
                                    graph_points_count=pred.get("graph_points_count"),
                                    raw_json=pred.get("raw_payload"),
                                    signal_type="NO_BET",
                                    picked_side="NONE",
                                    confidence=0.0,
                                    actual_home_score=pred.get("actual_home_score"),
                                    actual_away_score=pred.get("actual_away_score"),
                                    result="push",
                                    inference_json=pred.get("inference_json"),
                                    h2h_available=h2h
                                )
                                q4_done[model] = True
                                log_info("EVALUACION", f"[EVAL] NO_BET definitivo | Model: {model} | {match_display}", q4_orange=True)

                        if operable_to_send:
                            for model, pred in operable_to_send.items():
                                inf_json = pred.get("inference_json") or {}
                                h2h = inf_json.get("q4", {}).get("h2h_available")
                                save_bet_log(
                                    match_id=match_id,
                                    model_version=model,
                                    inference_minute=pred.get("inference_minute"),
                                    graph_points_count=pred.get("graph_points_count"),
                                    raw_json=pred.get("raw_payload"),
                                    signal_type=pred.get("signal"),
                                    picked_side=pred.get("pick"),
                                    confidence=pred.get("confidence"),
                                    actual_home_score=pred.get("actual_home_score"),
                                    actual_away_score=pred.get("actual_away_score"),
                                    inference_json=pred.get("inference_json"),
                                    h2h_available=h2h
                                )
                                q4_done[model] = True

                            # Enviar Telegram Alert
                            quarters_data = score_data.get("quarters", {})
                            # Mostrar el marcador del cuarto ACTUAL (no el de Q4, que aún va 0-0).
                            current_q_scores = quarters_data.get(q_key, {}) if q_key else {}
                            if len(operable_to_send) > 1:
                                await send_combined_bet_alert(
                                    predictions=operable_to_send,
                                    match_id=match_id,
                                    match_data=full_data,
                                    home_team=home,
                                    away_team=away,
                                    minute=minute,
                                    q_key=q_key,
                                    q_home=current_q_scores.get("home"),
                                    q_away=current_q_scores.get("away"),
                                    league=league,
                                    scheduled_ts=scheduled_ts
                                )
                            elif len(operable_to_send) == 1:
                                m_name, pred0 = next(iter(operable_to_send.items()))
                                picked_side = pred0.get("pick")
                                await send_bet_alert(
                                    model=m_name,
                                    picked_team=home if picked_side == "HOME" else away,
                                    is_home=(picked_side == "HOME"),
                                    signal=pred0.get("signal"),
                                    match_id=match_id,
                                    match_data=full_data,
                                    home_team=home,
                                    away_team=away,
                                    minute=minute,
                                    q_key=q_key,
                                    q_home=current_q_scores.get("home"),
                                    q_away=current_q_scores.get("away"),
                                    league=league,
                                    scheduled_ts=scheduled_ts,
                                    confidence=pred0.get("confidence")
                                )

                            update_schedule_q4(match_id, signal="BET_SENT", model=",".join(operable_to_send.keys()))

                        # Si todos los modelos evaluaron, podemos dormir hasta el final del juego
                        if all(q4_done.values()):
                            log_info("MONITOREO", f"{COLOR_GREEN}[EVAL COMPLETADA]{COLOR_RESET} Todos los modelos asentados → esperando FT | {match_display}", q4_orange=True)
                            await asyncio.sleep(POLL_INTERVAL_IDLE_SECS)
                            continue

        except Exception as e:
            log_error("MONITOREO", f"[LIVE] Error en ciclo {match_display}: {e}")

        # Polling espaciado
        poll_sleep = POLL_INTERVAL_LIVE_SECS if is_q4_active else POLL_NEAR_SECS
        await asyncio.sleep(poll_sleep)

async def _schedule_refresh_task(stop_event: asyncio.Event) -> None:
    """Refresca el itinerario para hoy y mañana cada 4 horas."""
    while not stop_event.is_set():
        try:
            today_str = get_utc6_now().strftime("%Y-%m-%d")
            tomorrow_str = (get_utc6_now() + timedelta(days=1)).strftime("%Y-%m-%d")
            log_info("SYSTEM", f"Actualizando itinerario: {today_str} y {tomorrow_str}")
            
            for d in [today_str, tomorrow_str]:
                events = await fetch_schedule_matches_for_date(d)
                if events:
                    save_schedule_matches(events)
                    log_info("SYSTEM", f"Itinerario {d} guardado: {len(events)} partidos")
        except Exception as e:
            log_error("SYSTEM", f"Error en tarea de itinerario: {e}")

        for _ in range(SCHEDULE_REFRESH_HOURS * 3600 // 10):
            if stop_event.is_set():
                break
            await asyncio.sleep(10)

async def _reconcile_loop_task(stop_event: asyncio.Event) -> None:
    """Audita periódicamente resultados de apuestas pendientes."""
    while not stop_event.is_set():
        try:
            reconcile_pending_results()
        except Exception as e:
            log_error("DATABASE", f"Error reconciliando resultados: {e}")
            
        for _ in range(PENDING_RECHECK_SECS // 10):
            if stop_event.is_set():
                break
            await asyncio.sleep(10)

async def _live_discovery_task(stop_event: asyncio.Event) -> None:
    """
    Sondea continuamente partidos en vivo globalmente (/sport/basketball/events/live)
    para descubrir partidos activos no contemplados en el itinerario inicial.
    """
    while not stop_event.is_set():
        try:
            live_events = await fetch_live_events()
            leagues_cfg = get_leagues_config()
            today_str = get_utc6_now().strftime("%Y-%m-%d")
            
            for ev in live_events:
                mid = str(ev.get("id"))
                if mid in _watcher_tasks and not _watcher_tasks[mid].done():
                    continue

                # No relanzar watchers de partidos ya finalizados/notificados
                # (aunque SofaScore los siga listando como "live" con datos congelados).
                try:
                    with get_db_connection() as conn:
                        srow = conn.execute(
                            "SELECT status, final_fetched FROM bet_monitor_schedule_v3 WHERE match_id = ?",
                            (mid,)
                        ).fetchone()
                    if srow and (srow["final_fetched"] or srow["status"] == "done"):
                        continue
                except Exception:
                    pass

                league_name = ev.get("tournament", {}).get("name", "")
                mode = get_league_mode(league_name, leagues_cfg)
                if mode == "EXCLUDE":
                    continue
                    
                # Guardar en base de datos e iniciar watcher
                ts = ev.get("startTimestamp", int(time.time()))
                match_dict = {
                    "match_id": mid,
                    "home_team": ev.get("homeTeam", {}).get("name", "Home"),
                    "away_team": ev.get("awayTeam", {}).get("name", "Away"),
                    "league": league_name,
                    "event_date": today_str,
                    "scheduled_utc_ts": ts,
                    "scheduled_utc": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "status_type": "inprogress",
                }
                save_schedule_matches([match_dict])
                
                is_ft_only = (mode == "FT_ONLY")
                task = asyncio.create_task(_watch_match(mid, match_dict, stop_event, ft_only=is_ft_only))
                _watcher_tasks[mid] = task
                
        except Exception as e:
            log_error("MONITOREO", f"Error en descubrimiento de partidos en vivo: {e}")

        await asyncio.sleep(POLL_INTERVAL_LIVE_SECS)

async def main_async() -> None:
    """Punto de entrada principal del daemon Monitor V3."""
    log_info("SYSTEM", "==================================================")
    log_info("SYSTEM", "🏀 INICIANDO MONITOR V3 (NATIVE MOBILE API + MULTI-JWT)")
    log_info("SYSTEM", f"Modelos Activos: {', '.join(ACTIVE_MODELS)}")
    log_info("SYSTEM", "==================================================")

    # 1. Verificar proxy local
    if not is_proxy_accessible():
        log_warning("SYSTEM", "El proxy local HTTP Toolkit (127.0.0.1:8000) no parece responder. Intentando conexión...")

    # 2. Inicializar tablas de BD y configuración
    init_tables()
    sync_leagues_config()
    load_models_to_cache()
    reconcile_pending_results()

    # 3. Inicializar el pool multi-JWT
    token_pool = get_token_pool()
    await token_pool.initialize()

    # Configurar señales de apagado limpio
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: stop_event.set())
        except NotImplementedError:
            pass

    # 4. Lanzar tareas de fondo
    bg_tasks = [
        asyncio.create_task(_schedule_refresh_task(stop_event)),
        asyncio.create_task(_reconcile_loop_task(stop_event)),
        asyncio.create_task(_live_discovery_task(stop_event)),
    ]

    log_info("SYSTEM", "Daemon V3 inicializado y operando. Monitoreando partidos en tiempo real...")

    try:
        while not stop_event.is_set():
            # Limpiar tareas terminadas
            done_keys = [mid for mid, task in _watcher_tasks.items() if task.done()]
            for mid in done_keys:
                del _watcher_tasks[mid]

            # Lanzar watchers para partidos programados pendientes
            all_pending = get_pending_schedule_matches()
            leagues_cfg = get_leagues_config()
            now_ts = time.time()

            for m in all_pending:
                mid = m["match_id"]
                if mid in _watcher_tasks and not _watcher_tasks[mid].done():
                    continue

                sched_ts = int(m["scheduled_utc_ts"])
                # Partidos que inician en las próximas 2 horas o iniciaron hace menos de 3.5 horas
                if (sched_ts - now_ts <= 7200) and (now_ts - sched_ts <= 12600):
                    mode = get_league_mode(m["league"], leagues_cfg)
                    if mode == "EXCLUDE":
                        continue

                    is_ft_only = (mode == "FT_ONLY")
                    task = asyncio.create_task(_watch_match(mid, m, stop_event, ft_only=is_ft_only))
                    _watcher_tasks[mid] = task

            await asyncio.sleep(15)

    except (KeyboardInterrupt, asyncio.CancelledError):
        log_info("SYSTEM", "Señal de interrupción recibida. Deteniendo daemon...")
    finally:
        stop_event.set()
        for t in bg_tasks:
            t.cancel()
        for t in _watcher_tasks.values():
            t.cancel()
        await get_mobile_client().close()
        log_info("SYSTEM", "Monitor V3 detenido de forma limpia y segura.")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

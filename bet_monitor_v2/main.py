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

import sys
import asyncio
import time
import yaml
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Cargar el path del workspace para importar los módulos de match
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.config.constants import (
    DB_FILE_PATH,
    LEAGUES_CONFIG_PATH,
    UTC_OFFSET_HOURS,
    SECS_PER_GAME_MIN,
    Q4_ONLY_EARLY_WAKE_MINUTE,
    Q4_ONLY_WAKE_LEAD_MINUTES,
    POLL_INTERVAL_FF,
    POLL_JITTER_FF,
    FINAL_FETCH_MIN_GP,
    IDLE_POLL_SECS,
    POLL_NEAR_SECS,
    SCHEDULE_REFRESH_HOURS,
    ACTIVE_MODELS,
    PRESTART_PROBE_MIN_SECS,
    PRESTART_PROBE_MAX_SECS,
    PRESTART_PROBE_BACKOFF
)
from bet_monitor_v2.database.connection import get_db_connection
from bet_monitor_v2.database.repository import (
    init_tables,
    sync_leagues_config,
    reconcile_pending_results,
    save_schedule_matches,
    get_pending_schedule_matches,
    update_schedule_status,
    save_bet_log,
    save_quarter_scores,
    save_eval_match_results
)
from bet_monitor_v2.scrapers.live_scraper import fetch_event_snapshot
from bet_monitor_v2.scrapers.browser_client import fetch_match_by_id
from bet_monitor_v2.scrapers.base_scraper import set_monitoring_lock, is_monitoring_locked
from bet_monitor_v2.models.evaluator import GameWatcherState, evaluate_match_q4
from bet_monitor_v2.notifications.telegram_bot import (
    send_bet_alert,
    send_final_confirmation,
    send_combined_bet_alert,
    send_combined_final_confirmation,
    send_stats_message
)
from bet_monitor_v2.utils.logger import log_info, log_warning, log_error, format_match_log, get_utc6_now, COLOR_BRIGHT_RED, COLOR_RESET, COLOR_GREEN, COLOR_WARNING
from bet_monitor_v2.utils.helpers import format_human_time, calculate_ema_secs_per_gmin, calculate_jitter_sleep_secs, ensure_obscura_running

# Reutilizar el schedule fetcher original para garantizar robustez
from match.bet_monitor import _fetch_all_events_for_date_sync

# Función de inferencia de minuto desde PBP (infer_match módulo de entrenamiento)
from match.training.infer_match import _infer_minute_from_pbp

# Variables de estado del loop
_watcher_tasks = {}
_probe_completed_count = 0
_total_watchers_spawned = 0

def get_leagues_config() -> dict:
    """Carga declarativa del archivo leagues.yaml."""
    yaml_path = ROOT / LEAGUES_CONFIG_PATH
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_league_mode(league: str, config: dict) -> str:
    """Retorna: 'EXCLUDE', 'FT_ONLY', o 'FULL_MONITOR'."""
    league_lower = league.lower()
    
    # 1. Comprobar exclusiones
    for pattern in config.get("excluded_leagues", {}).get("patterns", []):
        if pattern.lower() in league_lower:
            return "EXCLUDE"
            
    # 2. Comprobar FT-Only
    for pattern in config.get("ft_only_leagues", {}).get("patterns", []):
        if pattern.lower() in league_lower:
            return "FT_ONLY"
            
    return "FULL_MONITOR"


async def _final_fetch_and_save(match_id: str, home: str, away: str) -> None:
    """
    Descarga el payload final Playwright (Chrome Headless) y lo persiste de forma atómica.
    """
    try:
        # Reclamar slot y ejecutar fetch final (FT)
        data = await fetch_match_by_id(match_id, is_ft=True)
        gp_total = len(data.get("graph_points") or [])
        
        if gp_total < FINAL_FETCH_MIN_GP:
            log_warning("DESCARGA", f"[FT] Cierre con gráfica corta (gp={gp_total}) | {home} vs {away}")
            
        # 1. Persistir scores por cuarto en quarter_scores_v2
        score_data = data.get("score", {})
        quarters = score_data.get("quarters", {})
        save_quarter_scores(match_id, quarters)
        
        # 2. Guardar también en la base histórica legacy si db.py está disponible
        try:
            db_mod = __import__("db")
            legacy_conn = db_mod.get_conn(str(ROOT / "match" / "matches.db"))
            db_mod.init_db(legacy_conn)
            db_mod.save_match(legacy_conn, match_id, data)
            legacy_conn.close()
        except Exception:
            pass
            
        # 3. Liquidar / Reconciliar resultados de apuestas en logs
        reconcile_pending_results()
        
        # 4. Enviar notificaciones de confirmación de Telegram
        # Recuperar logs guardados para este partido y enviar ✅ o ❌
        
        # Extraer el marcador final del cuarto Q4 desde los datos FT descargados
        # (quarters ya está en memoria desde el paso 1; fallback a None si no disponible)
        q4_ft_data = quarters.get("Q4") or {}
        q4_ft_home = q4_ft_data.get("home")
        q4_ft_away = q4_ft_data.get("away")
        
        with get_db_connection() as conn:
            sched_row = conn.execute(
                "SELECT league, scheduled_utc_ts FROM bet_monitor_schedule_v2 WHERE match_id = ?",
                (match_id,)
            ).fetchone()
            league = sched_row["league"] if sched_row else None
            scheduled_ts = sched_row["scheduled_utc_ts"] if sched_row else None

            cursor = conn.execute("""
                SELECT id, model_version, picked_side, signal_type, result,
                       inference_minute, confidence, actual_home_score, actual_away_score, target_quarter
                FROM bet_monitor_log_v2
                WHERE match_id = ?
            """, (match_id,))
            logs = [dict(r) for r in cursor.fetchall()]
            
            # Filtrar para ver si hay al menos una apuesta operable real
            has_operable_bet = any("BET" in (log["signal_type"] or "") for log in logs)
            
            if has_operable_bet:
                if len(logs) > 1:
                    log0 = logs[0]
                    target_quarter = log0["target_quarter"] or 4
                    q_key = f"Q{target_quarter}"
                    # Usar scores reales FT del cuarto correspondiente; fallback al snapshot de apuesta
                    ft_q_data = quarters.get(q_key) or {}
                    ft_q_home = ft_q_data.get("home") if ft_q_data else q4_ft_home
                    ft_q_away = ft_q_data.get("away") if ft_q_data else q4_ft_away
                    
                    await send_combined_final_confirmation(
                        logs=logs,
                        match_id=match_id,
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
                    outcome = log["result"]
                    model = log["model_version"]
                    picked_side = log["picked_side"]
                    sig = log["signal_type"]
                    inference_minute = log["inference_minute"]
                    confidence = log["confidence"]
                    target_quarter = log["target_quarter"] or 4
                    q_key = f"Q{target_quarter}"
                    # Usar scores reales FT del cuarto correspondiente; fallback al snapshot de apuesta
                    ft_q_data = quarters.get(q_key) or {}
                    ft_q_home = ft_q_data.get("home") if ft_q_data else log["actual_home_score"]
                    ft_q_away = ft_q_data.get("away") if ft_q_data else log["actual_away_score"]
                    
                    await send_final_confirmation(
                        model=model,
                        picked_team=home if picked_side == "HOME" else away,
                        is_home=(picked_side == "HOME"),
                        signal=sig,
                        outcome=outcome,
                        match_id=match_id,
                        home_team=home,
                        away_team=away,
                        minute=inference_minute,
                        q_key=q_key,
                        q_home=ft_q_home,
                        q_away=ft_q_away,
                        league=league,
                        scheduled_ts=scheduled_ts,
                        confidence=confidence
                    )
                
                # Enviar el mensaje consolidado lado-a-lado de stats de los modelos
                try:
                    await send_stats_message()
                except Exception as stats_err:
                    log_error("DESCARGA", f"[FT] Error enviando stats unificadas de Telegram: {stats_err}")

                
        update_schedule_status(match_id, status="done")
        log_info("DESCARGA", f"[FT] Final y liquidación de apuestas persistidos | {home} vs {away} (gp={gp_total})")
    except Exception as e:
        err_msg = str(e)
        if "Incidents API" in err_msg:
            import re
            # Extraer el código de respuesta HTTP del mensaje del error
            code_match = re.search(r"(?:HTTP\s+)?(\d+)", err_msg)
            http_code = code_match.group(1) if code_match else "404"
            
            # Obtener horario del partido desde la base de datos
            try:
                with get_db_connection() as conn:
                    row = conn.execute("SELECT scheduled_utc_ts FROM bet_monitor_schedule_v2 WHERE match_id = ?", (match_id,)).fetchone()
                    scheduled_ts = row[0] if row else int(time.time())
            except Exception:
                scheduled_ts = int(time.time())
                
            sched_label = datetime.fromtimestamp(scheduled_ts, tz=timezone(timedelta(hours=UTC_OFFSET_HOURS))).strftime("%H:%M")
            match_display = format_match_log(sched_label, match_id, home, away)
            
            red_err = f"{COLOR_BRIGHT_RED}Incidents API HTTP {http_code}{COLOR_RESET}"
            log_error("DESCARGA", f"[FT] {match_display}: Incidents | {red_err}")
        else:
            err_str = str(e)
            if "locked" in err_str.lower():
                red_locked = f"{COLOR_BRIGHT_RED}DB LOCKED{COLOR_RESET}"
                log_error("DESCARGA", f"[FT] {home} vs {away} | {red_locked}")
            else:
                log_error("DESCARGA", f"[FT] Error en fetch final | {home} vs {away}: {e}")
            
        update_schedule_status(match_id, status="pending", skip_reason="final_fetch_failed")


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


async def _watch_match(match_id: str, match_row: dict, stop_event: asyncio.Event, ft_only: bool = False) -> None:
    """
    Máquina de Estados de Corrutina Watcher independiente para el ciclo de vida del partido.
    """
    global _probe_completed_count
    home = match_row["home_team"]
    away = match_row["away_team"]
    league = match_row["league"]
    scheduled_ts = int(match_row["scheduled_utc_ts"])
    sched_label = datetime.fromtimestamp(scheduled_ts, tz=timezone(timedelta(hours=UTC_OFFSET_HOURS))).strftime("%H:%M")
    
    match_display = format_match_log(sched_label, match_id, home, away)
    log_info("MONITOREO", f"[WATCHER] Start: {match_display} | ft_only={ft_only}")
    
    watcher_state = GameWatcherState(match_id)
    secs_per_gmin = float(SECS_PER_GAME_MIN)
    
    # Validar si el partido ya es obsoleto temporalmente (>3.5 horas transcurridas desde su hora teórica de inicio)
    # antes de realizar cualquier interacción con la API o esperar.
    if time.time() - scheduled_ts > 12600:
        log_warning("MONITOREO", f"[WATCHER] Partido obsoleto | {match_display}")
        update_schedule_status(match_id, status="done", skip_reason="obsoleto_tiempo_superado")
        await _final_fetch_and_save(match_id, home, away)
        return
        
    # 1. Flujo Degradado (FT-Only)
    if ft_only:
        # Espera pasiva hasta la hora teórica de inicio + 2 horas para cierres
        sleep_secs = max(60, (scheduled_ts + 7200) - time.time())
        log_info("MONITOREO", f"[FT-ONLY] Espera pasiva: {format_human_time(sleep_secs)} | {match_display}")
        await asyncio.sleep(sleep_secs)
        
        # Derivar directamente a descarga final
        update_schedule_status(match_id, status="in_progress")
        await _final_fetch_and_save(match_id, home, away)
        return

    # 2. Modo Sonda Prestart (Probe Mode)
    in_probe_mode = True
    probe_delay = float(PRESTART_PROBE_MIN_SECS)
    probe_start_time = time.time()
    is_first_probe = True
    
    while in_probe_mode and not stop_event.is_set():
        # Validar si el partido ya es obsoleto temporalmente durante la sonda
        if time.time() - scheduled_ts > 12600:
            log_warning("MONITOREO", f"[PROBE] Partido obsoleto | {match_display}")
            update_schedule_status(match_id, status="done", skip_reason="obsoleto_tiempo_superado")
            await _final_fetch_and_save(match_id, home, away)
            return

        # Validar timeout global de 6 horas desde la hora programada de inicio
        if time.time() - scheduled_ts > 21600:
            log_warning("MONITOREO", f"[PROBE] Timeout 6h desde inicio programado: {match_display}")
            update_schedule_status(match_id, status="discarded", skip_reason="probe_timeout")
            return
            
        # Drift de seguridad por drift temporal (evaluado por reloj antes de consultar la API)
        drift = time.time() - scheduled_ts
        if drift >= (secs_per_gmin * Q4_ONLY_EARLY_WAKE_MINUTE):
            if is_first_probe:
                _probe_completed_count += 1
                is_first_probe = False
            progress_label = f" [{_probe_completed_count}/{_total_watchers_spawned}]"
            log_info("MONITOREO", f"[PROBE]{progress_label} Drift Exit: {match_display}")
            in_probe_mode = False
            break
            
        try:
            snapshot = await fetch_event_snapshot(match_id)
            status_type = snapshot.get("status_type", "").lower()
            
            if is_first_probe:
                _probe_completed_count += 1
                is_first_probe = False
                
            progress_label = f" [{_probe_completed_count}/{_total_watchers_spawned}]"
            
            # Condiciones de salida del modo sonda
            if status_type in ("inprogress", "live"):
                log_info("MONITOREO", f"[PROBE]{progress_label} En Vivo: {match_display}")
                in_probe_mode = False
                break
            elif status_type == "finished":
                log_info("MONITOREO", f"[PROBE]{progress_label} Finalizado prematuro: {match_display}")
                update_schedule_status(match_id, status="done", skip_reason="prematuro_detectado")
                # Derivar directamente a descarga final y terminar la corrutina sin pasar al bucle de Q4 en vivo
                await _final_fetch_and_save(match_id, home, away)
                return
                
        except Exception as e:
            if is_first_probe:
                _probe_completed_count += 1
                is_first_probe = False
            progress_label = f" [{_probe_completed_count}/{_total_watchers_spawned}]"
            log_warning("MONITOREO", f"[PROBE]{progress_label} Error ligero | {match_display}: {e}")
            
        # Dormir con backoff exponencial
        probe_sleep = calculate_jitter_sleep_secs(probe_delay, phase="error_retry")
        await asyncio.sleep(probe_sleep)
        probe_delay = min(float(PRESTART_PROBE_MAX_SECS), probe_delay * PRESTART_PROBE_BACKOFF)

    # 3. Espera Adaptativa Q1/Q2 (un probe ligero + sleep hasta Q3 estimado)
    # Evita fetches pesados de Playwright mientras el partido está en Q1/Q2.
    # Calcula el tiempo estimado en que comenzará Q3 (~minuto de juego 20).
    Q3_START_GAME_MIN = 20  # minuto de juego aproximado donde empieza Q3
    estimated_q3_wall = scheduled_ts + secs_per_gmin * Q3_START_GAME_MIN
    now_ts = time.time()

    if now_ts < estimated_q3_wall - 60:
        # Un probe ligero para confirmar que el partido sigue en curso
        try:
            snap_early = await fetch_event_snapshot(match_id)
            early_status = snap_early.get("status_type", "").lower()
            if early_status == "finished":
                log_info("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Finalizado durante Q1/Q2 → FT | {match_display}")
                await _final_fetch_and_save(match_id, home, away)
                return
            early_period = snap_early.get("status_description", "") or early_status or "En Vivo"
            period_lower = early_period.lower()
            if "1st quarter" in period_lower or "q1" in period_lower:
                early_period = "Q1"
            elif "2nd quarter" in period_lower or "q2" in period_lower:
                early_period = "Q2"
            elif "3rd quarter" in period_lower or "q3" in period_lower:
                early_period = "Q3"
            elif "4th quarter" in period_lower or "q4" in period_lower:
                early_period = "Q4"
            wait_secs = estimated_q3_wall - 60 - now_ts
            log_info("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} {early_period} en Curso, esperando Q3 en ~{format_human_time(wait_secs)} | {match_display}")
        except Exception:
            wait_secs = estimated_q3_wall - 60 - now_ts

        # Dormir en chunks de 5 min con probes ligeros para detectar finalizaciones
        remaining = estimated_q3_wall - 60 - time.time()
        while remaining > 0 and not stop_event.is_set():
            chunk = min(300.0, remaining)
            await asyncio.sleep(chunk)
            remaining = estimated_q3_wall - 60 - time.time()
            if remaining > 60:
                try:
                    snap_chk = await fetch_event_snapshot(match_id)
                    if snap_chk.get("status_type", "").lower() == "finished":
                        log_info("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Finalizado durante espera Q1/Q2 → FT | {match_display}")
                        await _final_fetch_and_save(match_id, home, away)
                        return
                except Exception:
                    pass

    # 4. Monitoreo Pesado Activo (Q3/Q4 window)
    q4_done = {model: False for model in ACTIVE_MODELS}
    last_gmin = None
    last_gmin_wall = 0.0
    consecutive_timeouts = 0  # contador de timeouts consecutivos en fetch pesado
    
    # Activar candado Live de 5 minutos al haber partidos en ventana
    set_monitoring_lock(300)
    
    from bet_monitor_v2.config.constants import SOFASCORE_SCRAPER_BACKEND
    log_info("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Iniciando monitoreo pesado (backend={SOFASCORE_SCRAPER_BACKEND}) | {match_display}")
    
    while not stop_event.is_set():
        # Validar si el partido ya es obsoleto temporalmente (más de 3.5 horas transcurridas desde su hora teórica de inicio)
        if time.time() - scheduled_ts > 12600:
            log_warning("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Partido obsoleto | {match_display}")
            update_schedule_status(match_id, status="done", skip_reason="obsoleto_tiempo_superado")
            await _final_fetch_and_save(match_id, home, away)
            break
            
        try:
            snapshot = await fetch_event_snapshot(match_id)
            status_type = snapshot.get("status_type", "").lower()
            status_desc = snapshot.get("status_description", "") or ""
            home_score = snapshot.get("home_score", 0)
            away_score = snapshot.get("away_score", 0)
            
            if status_type == "finished":
                log_info("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Fin detectado → Derivando a FT | {match_display}")
                await _final_fetch_and_save(match_id, home, away)
                break
            
            # Descarga pesada completa para verificar el minuto exacto del PBP y las gráficas
            # Si el fetch pesado falla con timeout, usamos los datos del snapshot como fallback mínimo.
            full_data = None
            try:
                full_data = await fetch_match_by_id(match_id, is_ft=False)
                consecutive_timeouts = 0  # reset al obtener datos exitosos
            except Exception as fetch_err:
                fetch_err_msg = str(fetch_err)
                is_timeout_err = "timeout" in fetch_err_msg.lower() or "50s" in fetch_err_msg
                if is_timeout_err:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= 3:
                        # Fallback: construir full_data mínimo desde snapshot para permitir evaluación
                        q_snap = snapshot.get("status_description", "") or ""
                        full_data = {
                            "score": {
                                "home": home_score,
                                "away": away_score,
                                "quarters": {}
                            },
                            "play_by_play": {},
                            "graph_points": [],
                            "incidents": {},
                            "_snapshot_fallback": True,
                        }
                        log_warning("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} [FALLBACK] {consecutive_timeouts} timeouts consecutivos → usando snapshot para evaluación | {match_display}")
                    else:
                        log_error("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} {match_display} | {COLOR_WARNING}TIMEOUT{COLOR_RESET} 50s Playwright/Obscura")
                        await asyncio.sleep(POLL_NEAR_SECS)
                        continue
                else:
                    # Error no-timeout: re-lanzar para que lo capture el except exterior
                    raise
            minute = _infer_minute_from_pbp(full_data)
            # En modo fallback de snapshot, el PBP está vacío. Si el snapshot indica Q4, fijamos
            # el minuto en 35 para permitir la evaluación dentro de la ventana operable.
            if full_data.get("_snapshot_fallback") and minute is None:
                if "4th quarter" in status_desc.lower() or "q4" in status_desc.lower():
                    minute = 35

            
            # Construir etiqueta del período con minuto si disponible
            period_label = status_desc if status_desc else "En Vivo"
            period_lower = period_label.lower()
            q_key = None
            q_home = None
            q_away = None
            if "1st quarter" in period_lower:
                period_label = "Q1"
                q_key = "Q1"
            elif "2nd quarter" in period_lower:
                period_label = "Q2"
                q_key = "Q2"
            elif "3rd quarter" in period_lower:
                period_label = "Q3"
                q_key = "Q3"
            elif "4th quarter" in period_lower:
                period_label = "Q4"
                q_key = "Q4"
            elif "overtime" in period_lower:
                period_label = "OT"
                q_key = "OT"
            elif "pause" in period_lower or "halftime" in period_lower or "intermission" in period_lower:
                period_label = "Pause"
                
            # Si estamos en el cuarto Q4 pero el minuto está congelado en el final del Q3 (35) o no está disponible,
            # y el Q4 ya registra anotaciones parciales menores a 30 puntos totales (recién empezado en vivo)
            if q_key == "Q4":
                quarters_data = full_data.get("score", {}).get("quarters", {}) or {}
                q4_score = quarters_data.get("Q4", {}) or {}
                q4_home = q4_score.get("home")
                q4_away = q4_score.get("away")
                if q4_home is not None and q4_away is not None:
                    q4_total = q4_home + q4_away
                    if q4_total < 30 and (minute is None or minute < 36):
                        minute = 35
                
            # Coloreado dinámico de period_label para la consola
            period_colored = period_label
            if "Q3" in period_label:
                period_colored = f"{COLOR_WARNING}Q3{COLOR_RESET}"
            elif "Q4" in period_label:
                period_colored = f"{COLOR_GREEN}Q4{COLOR_RESET}"
            elif "Pause" in period_label:
                period_colored = f"{COLOR_BRIGHT_RED}Pause{COLOR_RESET}"
            
            # Construir la sección de brackets colorida y estructurada [period | MIN~minute | score_home - score_away ]
            bracket_parts = [period_colored]
            if minute is not None:
                bracket_parts.append(f"MIN~{minute}")
            
            # Intentar extraer el marcador del cuarto actual
            if q_key:
                quarters_data = full_data.get("score", {}).get("quarters", {}) or {}
                q_data = quarters_data.get(q_key)
                if q_data:
                    q_home = q_data.get("home")
                    q_away = q_data.get("away")
                    if q_home is not None and q_away is not None:
                        q_home_col, q_away_col = colorize_scores(q_home, q_away)
                        bracket_parts.append(f"{q_home_col} - {q_away_col}")
            
            bracket_content = " | ".join(bracket_parts)
            bracket_label = f"[{bracket_content} ]" if bracket_content else ""
            
            # Marcador global coloreado
            home_score_colored, away_score_colored = colorize_scores(home_score, away_score)
            
            log_info("MONITOREO", f"{format_match_log(sched_label, match_id, home, away)} {bracket_label} Score: {home_score_colored} - {away_score_colored}", q4_orange=True)
            
            if minute is not None:
                # Calibrar EMA
                now_wall = time.monotonic()
                if last_gmin is not None and minute > last_gmin:
                    secs_per_gmin = calculate_ema_secs_per_gmin(secs_per_gmin, now_wall - last_gmin_wall, minute - last_gmin)
                last_gmin = minute
                last_gmin_wall = now_wall
                
                # Si ya pasamos del minuto 38, la ventana Q4 está cerrada.
                # Aun así llamamos _final_fetch_and_save para reconciliar cualquier apuesta ya lanzada.
                if minute >= 38:
                    log_warning("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Ventana Q4 superada (min {minute}) | {match_display}")
                    await _final_fetch_and_save(match_id, home, away)
                    break
                    
                # Si estamos dentro de la ventana operable Q4 (Minuto 27 a 36)
                if minute >= Q4_ONLY_EARLY_WAKE_MINUTE and minute < 36:
                    # Evaluar predicciones
                    eval_res = await evaluate_match_q4(match_id, full_data, watcher_state, forced_minute=minute)
                    
                    if eval_res.get("ok"):
                        preds = eval_res.get("predictions", {})
                        
                        all_resolved = True
                        operable_to_send = {}
                        
                        for model, pred in preds.items():
                            if q4_done[model]:
                                continue
                                
                            sig = pred.get("signal", "UNAVAILABLE")
                            if sig == "ERROR":
                                reason = pred.get("reason", "?")
                                log_error("EVALUACION", f"[EVAL] Error modelo [{model}]: {reason} | {match_display}", q4_orange=True)
                                all_resolved = False
                                continue
                            if sig == "UNAVAILABLE":
                                all_resolved = False
                                continue
                                
                            if "BET" in sig:
                                operable_to_send[model] = pred
                            elif sig == "NO_BET":
                                # Asentamos NO_BET final tras el buffer
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
                                    inference_json=pred.get("inference_json")
                                )
                                q4_done[model] = True
                                log_info("EVALUACION", f"[EVAL] NO_BET definitivo | Model: {model} | {match_display}", q4_orange=True)
                                
                        if operable_to_send:
                            # 1. Guardar log transaccional para cada señal operable
                            for model, pred in operable_to_send.items():
                                picked_side = pred.get("pick")
                                confidence = pred.get("confidence")
                                save_bet_log(
                                    match_id=match_id,
                                    model_version=model,
                                    inference_minute=pred.get("inference_minute"),
                                    graph_points_count=pred.get("graph_points_count"),
                                    raw_json=pred.get("raw_payload"),
                                    signal_type=pred.get("signal"),
                                    picked_side=picked_side,
                                    confidence=confidence,
                                    actual_home_score=pred.get("actual_home_score"),
                                    actual_away_score=pred.get("actual_away_score"),
                                    inference_json=pred.get("inference_json")
                                )
                            
                            # 2. Despachar Telegram Alert (combinada o individual)
                            if len(operable_to_send) > 1:
                                tg_res = await send_combined_bet_alert(
                                    predictions=operable_to_send,
                                    match_id=match_id,
                                    match_data=full_data,
                                    home_team=home,
                                    away_team=away,
                                    minute=minute,
                                    q_key=q_key,
                                    q_home=q_home,
                                    q_away=q_away,
                                    league=league,
                                    scheduled_ts=scheduled_ts
                                )
                                if tg_res.get("ok"):
                                    match_url = tg_res.get("match_url") or ""
                                    url_line = f"\n   {match_url}" if match_url else ""
                                    models_str = ", ".join(operable_to_send.keys())
                                    log_info("TELEGRAM", f"{COLOR_GREEN}[SENT]{COLOR_RESET} COMBINED [{models_str}] | {match_display}{url_line}", q4_orange=True)
                                else:
                                    log_error("TELEGRAM", f"[FAIL] No se pudo enviar alerta combinada: {tg_res.get('description','?')} | {match_display}")
                            else:
                                # Solo hay 1 operable, enviamos de forma estándar
                                model, pred = next(iter(operable_to_send.items()))
                                picked_side = pred.get("pick")
                                confidence = pred.get("confidence")
                                sig = pred.get("signal")
                                tg_res = await send_bet_alert(
                                    model=model,
                                    picked_team=home if picked_side == "HOME" else away,
                                    is_home=(picked_side == "HOME"),
                                    signal=sig,
                                    match_id=match_id,
                                    match_data=full_data,
                                    home_team=home,
                                    away_team=away,
                                    minute=minute,
                                    q_key=q_key,
                                    q_home=q_home,
                                    q_away=q_away,
                                    league=league,
                                    scheduled_ts=scheduled_ts,
                                    confidence=confidence
                                )
                                if tg_res.get("ok"):
                                    match_url = tg_res.get("match_url") or ""
                                    url_line = f"\n   {match_url}" if match_url else ""
                                    log_info("TELEGRAM", f"{COLOR_GREEN}[SENT]{COLOR_RESET} {sig} [{model}] | {match_display}{url_line}", q4_orange=True)
                                else:
                                    log_error("TELEGRAM", f"[FAIL] No se pudo enviar alerta [{model}]: {tg_res.get('description','?')} | {match_display}")
                            
                            # Marcar todas las señales enviadas como resueltas
                            for model, pred in operable_to_send.items():
                                q4_done[model] = True
                                log_info("EVALUACION", f"[EVAL] Señal operable [{pred.get('signal')}] | Model: {model} | {match_display}", q4_orange=True)
                                
                        if all_resolved and all(q4_done.values()):
                            log_info("MONITOREO", f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET} Q4 Concluido. Esperando FT | {match_display}")
                            # Pasamos a espera pasiva hasta finalización
                            sleep_secs = max(60, (scheduled_ts + 7200) - time.time())
                            await asyncio.sleep(sleep_secs)
                            await _final_fetch_and_save(match_id, home, away)
                            break
                            
            # Tiered Sleep
            if minute is None:
                await asyncio.sleep(POLL_NEAR_SECS)
            else:
                mins_to_wake = Q4_ONLY_EARLY_WAKE_MINUTE - minute
                if mins_to_wake <= 1:
                    sleep_secs = 45.0
                    phase = "q4_window"
                elif mins_to_wake <= 2:
                    sleep_secs = 60.0
                    phase = "q4_window"
                elif mins_to_wake <= 5:
                    sleep_secs = 90.0
                    phase = "q4_window"
                elif mins_to_wake > 10:
                    sleep_secs = 180.0
                    phase = "q4_far"
                else:
                    sleep_secs = 120.0
                    phase = "q4_far"
                    
                jitter_sleep = calculate_jitter_sleep_secs(sleep_secs, phase=phase)
                await asyncio.sleep(jitter_sleep)
                
        except Exception as e:
            err_msg = str(e)
            live_label = f"{COLOR_BRIGHT_RED}[LIVE]{COLOR_RESET}"
            if "Incidents API" in err_msg:
                import re
                code_match = re.search(r"(?:HTTP\s+)?(\d+)", err_msg)
                http_code = code_match.group(1) if code_match else "404"
                
                # Si el error es específicamente un HTTP 404 (Sofascore no tiene incidentes de este partido),
                # descartamos y cerramos el partido inmediatamente en un solo log de error unificado.
                if http_code == "404":
                    red_err = f"Incidents API HTTP 404 → {COLOR_BRIGHT_RED}DESCARTADO{COLOR_RESET}"
                    log_error("MONITOREO", f"{live_label} {match_display}: {red_err}")
                    update_schedule_status(match_id, status="done", skip_reason="incidents_api_404_unsupported")
                    break
                else:
                    red_err = f"{COLOR_BRIGHT_RED}Incidents API HTTP {http_code}{COLOR_RESET}"
                    log_error("MONITOREO", f"{live_label} {match_display}: {red_err}")
            else:
                log_error("MONITOREO", f"{live_label} Error bucle | {match_display}: {e}")
            await asyncio.sleep(POLL_NEAR_SECS)


async def _recheck_pending_finished_schedule_once() -> dict:
    """
    Tarea asíncrona periódica (cada 2 horas) que audita y descarga resultados FT
    para schedule_v2 colgados (final_fetched=0 pero finalizados).
    """
    now = time.time()
    summary = {"found": 0, "checked": 0, "scraped_ok": 0, "scraped_fail": 0, "finished_saved": 0}
    
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT match_id, home_team, away_team, scheduled_utc_ts, status
            FROM bet_monitor_schedule_v2
            WHERE final_fetched = 0 AND scheduled_utc_ts < ?
        """, (int(now - 10800),)) # partidos que iniciaron hace más de 3 horas
        pending = [dict(r) for r in cursor.fetchall()]
        
    summary["found"] = len(pending)
    for m in pending:
        mid = m["match_id"]
        home = m["home_team"]
        away = m["away_team"]
        
        summary["checked"] += 1
        try:
            # Forzar Playwright download final (FT)
            await _final_fetch_and_save(mid, home, away)
            summary["scraped_ok"] += 1
            summary["finished_saved"] += 1
        except Exception:
            summary["scraped_fail"] += 1
            
    return summary


async def main_loop(stop_event: asyncio.Event) -> None:
    """
    Orquestador principal del event loop infinito controlable.
    """
    log_info("SYSTEM", "Iniciando bet_monitor_v2...")
    
    # 1. Inicialización de infraestructura y reconciliaciones
    init_tables()
    reconcile_pending_results()
    sync_leagues_config()
    
    # Auto-encendido inteligente de Obscura (CDP) si es requerido por la configuración
    if ensure_obscura_running():
        log_info("SYSTEM", "Servicio Obscura (CDP) verificado y activo en puerto 9222.")
    else:
        log_warning("SYSTEM", "Obscura (CDP) inactivo y no se pudo iniciar automáticamente.")
    
    last_refresh_wall = time.monotonic() - 3600 * SCHEDULE_REFRESH_HOURS
    last_recheck_schedule_wall = time.monotonic()
    
    log_info("SYSTEM", "Inicialización completada.")
    
    while not stop_event.is_set():
        now_wall = time.monotonic()
        
        # A) Cargar / refrescar itinerario cada 8 horas o al inicio
        if now_wall - last_refresh_wall >= SCHEDULE_REFRESH_HOURS * 3600:
            today_str = get_utc6_now().strftime("%Y-%m-%d")
            tomorrow_str = (get_utc6_now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            log_info("SYSTEM", f"Descargando itinerario: {today_str} | {tomorrow_str}")
            
            for date_str in [today_str, tomorrow_str]:
                try:
                    # Descargar vía _fetch_all_events_for_date_sync en hilo
                    events = await asyncio.to_thread(_fetch_all_events_for_date_sync, date_str)
                    save_schedule_matches(events)
                    log_info("SYSTEM", f"Itinerario {date_str} guardado: {len(events)} partidos")
                except Exception as e:
                    log_error("SYSTEM", f"Error itinerario {date_str}: {e}")
                    
            last_refresh_wall = now_wall
            
        # B) Lanzar Watchers para partidos pendientes
        all_pending = get_pending_schedule_matches()
        leagues_cfg = get_leagues_config()
        
        # Filtrar para procesar de forma activa únicamente los partidos que inician pronto (próximas 2 horas)
        # o que ya iniciaron recientemente (menos de 3.5 horas transcurridas para permitir recuperación en reinicios)
        now_ts = time.time()
        pending_matches = []
        for m in all_pending:
            sched_ts = int(m["scheduled_utc_ts"])
            if (sched_ts - now_ts <= 7200) and (now_ts - sched_ts <= 12600):
                pending_matches.append(m)
                
        # Calcular cuántos se van a procesar de forma activa
        active_pendings = [m for m in pending_matches if get_league_mode(m["league"], leagues_cfg) != "EXCLUDE"]
        
        global _total_watchers_spawned, _probe_completed_count
        # Si hay tareas activas nuevas que no estaban en _watcher_tasks, actualizamos el total y reseteamos el progreso
        new_spawns = [m for m in active_pendings if m["match_id"] not in _watcher_tasks]
        if new_spawns:
            _total_watchers_spawned = len(active_pendings)
            _probe_completed_count = 0
            
        for m in pending_matches:
            mid = m["match_id"]
            if mid in _watcher_tasks and not _watcher_tasks[mid].done():
                continue
                
            # Doble Filtro de Asignación Basado en Ligas
            league = m["league"]
            mode = get_league_mode(league, leagues_cfg)
            
            if mode == "EXCLUDE":
                # Omitir
                update_schedule_status(mid, status="discarded", skip_reason="excluida_por_config")
                log_info("SYSTEM", f"Excluido: {m['home_team']} vs {m['away_team']} (liga: {league})")
                continue
                
            ft_only = (mode == "FT_ONLY")
            
            # Lanzar watcher task
            task = asyncio.create_task(
                _watch_match(mid, m, stop_event, ft_only=ft_only)
            )
            _watcher_tasks[mid] = task
            
        # Limpiar tareas completadas
        finished_ids = [mid for mid, t in _watcher_tasks.items() if t.done()]
        for mid in finished_ids:
            del _watcher_tasks[mid]
            
        # C) Ejecutar re-chequeo periódico de resultados finalizados (cada 2 horas)
        if now_wall - last_recheck_schedule_wall >= 7200:
            # Mientras no haya locks por Live activo
            if not is_monitoring_locked():
                log_info("SYSTEM", "Iniciando re-chequeo FT periódico")
                summary = await _recheck_pending_finished_schedule_once()
                log_info("SYSTEM", f"Re-chequeo FT finalizado: {summary}")
            last_recheck_schedule_wall = now_wall
            
        # Dormir 60 segundos
        await asyncio.sleep(60)


if __name__ == "__main__":
    import signal
    
    stop_event = asyncio.Event()
    
    def handle_exit_signal(sig, frame):
        log_info("PROGRAMACION", "Señal de apagado recibida. Deteniendo watchers...")
        stop_event.set()
        
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)
    
    try:
        asyncio.run(main_loop(stop_event))
    except KeyboardInterrupt:
        pass

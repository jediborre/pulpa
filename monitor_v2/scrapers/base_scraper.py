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

import asyncio
import random
import time
import os
from monitor_v2.config.constants import (
    MAX_CONCURRENT_FETCHES,
    GLOBAL_FETCH_MIN_SPACING_SECS,
    GLOBAL_403_STREAK_TRIGGER,
    GLOBAL_403_COOLDOWN_SECS,
    SESSION_ROTATE_EVERY,
    SESSION_ROTATE_PAUSE_SECS,
    FETCH_TIMEOUT_SECS,
    FREE_PROXY_LIST,
)

# Semáforo de concurrencia compartido globalmente
_fetch_sem = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

# Variables de control de estado del Anti-Ban
_403_streak = 0
_cooldown_until = 0.0
_success_calls_count = 0
_monitoring_lock_until = 0.0

# Cola serializada para descargas de Playwright (evita abrir múltiples Chrome Headless simultáneos)
_ft_scrape_sem = asyncio.Semaphore(1)
_active_ft_scrapes = 0

# Proxy con prioridad: local → free list → premium
_proxy_tier = 0
_free_idx = 0
_start_time = time.time()
_PROXY_GRACE_SECS = 300  # 5 min con IP propia

# Guardar Smartproxy URL original (premium) y limpiar — deshabilitado
_premium_proxy_url = os.environ.get("SOFASCORE_PROXY_URL", "")
os.environ.pop("SOFASCORE_PROXY_URL_SMARTPROXY", None)
os.environ["SOFASCORE_PROXY_URL"] = ""

def _set_proxy_env(url: str) -> None:
    """Actualiza SOFASCORE_PROXY_URL para que el scraper lo lea."""
    os.environ["SOFASCORE_PROXY_URL"] = url or ""

def _current_proxy_label() -> str:
    """Retorna label legible del proxy actual."""
    global _proxy_tier, _free_idx
    if _proxy_tier == 0:
        return "local"
    if _proxy_tier == "premium":
        return "premium"
    return f"free#{_free_idx + 1}"

def _rotate_proxy(force_tier=None) -> None:
    """Avanza al siguiente proxy disponible según prioridad."""
    global _proxy_tier, _free_idx

    if force_tier is not None:
        _proxy_tier = force_tier
        _free_idx = 0
    elif _proxy_tier == 0:
        # local falló → probar free list
        _proxy_tier = 1
        _free_idx = 0
    elif isinstance(_proxy_tier, int) and _free_idx < len(FREE_PROXY_LIST) - 1:
        # siguiente free proxy
        _free_idx += 1
    elif isinstance(_proxy_tier, int):
        # se acabaron los free → premium
        _proxy_tier = "premium"
        _free_idx = 0
    else:
        # premium falló → volver a local tras cooldown
        _proxy_tier = 0
        _free_idx = 0

    # Actualizar env var
    premium_url = os.environ.get("SOFASCORE_PROXY_URL_SMARTPROXY", "")
    if _proxy_tier == 0:
        _set_proxy_env(None)
    elif _proxy_tier == "premium":
        _set_proxy_env(premium_url)
    elif isinstance(_proxy_tier, int) and _free_idx < len(FREE_PROXY_LIST):
        _set_proxy_env(FREE_PROXY_LIST[_free_idx])

    _log_proxy_change()

def _log_proxy_change() -> None:
    from monitor_v2.utils.logger import log_warning, COLOR_WARNING, COLOR_RESET
    label = _current_proxy_label()
    url_display = os.environ.get("SOFASCORE_PROXY_URL", "directo")[:80]
    log_warning("SYSTEM", f"{COLOR_WARNING}[PROXY]{COLOR_RESET} Cambio a: {label} ({url_display})")

def set_monitoring_lock(duration_secs: int) -> None:
    """Activa el candado de monitoreo Live."""
    global _monitoring_lock_until
    _monitoring_lock_until = time.time() + duration_secs

def is_monitoring_locked() -> bool:
    """Verifica si el candado Live está activo."""
    global _monitoring_lock_until
    return time.time() < _monitoring_lock_until

async def wait_if_cooldown() -> None:
    """Suspende las peticiones si el sistema se encuentra en estado de hibernación por 403."""
    global _cooldown_until
    now = time.time()
    if now < _cooldown_until:
        sleep_time = _cooldown_until - now
        await asyncio.sleep(sleep_time)

async def _proxy_timer_check() -> None:
    """Activa tier 1 si hay proxies configurados. No hace nada si solo hay local."""
    global _proxy_tier
    if _proxy_tier != 0:
        return
    if not FREE_PROXY_LIST and not os.environ.get("SOFASCORE_PROXY_URL_SMARTPROXY", ""):
        return  # sin proxies que activar
    if (time.time() - _start_time) >= _PROXY_GRACE_SECS:
        _rotate_proxy(force_tier=1)

async def check_403_streak(status_code: int) -> None:
    """Incrementa la racha de 403 y gestiona el fallback dinámico o cooldown de seguridad."""
    global _403_streak, _cooldown_until
    if status_code == 403:
        _403_streak += 1
        
        # Si 5 errores 403 consecutivos, mostrar banner y esperar Enter
        if _403_streak >= 5:
            from monitor_v2.utils.logger import log_error, COLOR_BRIGHT_RED, COLOR_RESET
            log_error(
                "SYSTEM",
                f"{COLOR_BRIGHT_RED}[BANEADO]{COLOR_RESET} 5 errores 403 consecutivos — "
                f"IP bloqueada por SofaScore. Presiona Enter para reintentar..."
            )
            await asyncio.to_thread(input)
            _403_streak = 0
        
        # Rotar proxy solo tras 3 403 consecutivos (evita quemar proxies por glitches)
        if _403_streak >= 3:
            if FREE_PROXY_LIST or os.environ.get("SOFASCORE_PROXY_URL_SMARTPROXY", ""):
                _rotate_proxy()
            _403_streak = 0
        
        # Hotswap dinámico: Si llevamos 3 errores 403 consecutivos en Obscura o Traditional,
        # rotamos en caliente a 'chrome' (el backend blindado) para auto-sanación instantánea.
        if _403_streak == 3:
            import monitor_v2.config.constants as constants
            current_live = getattr(constants, "SOFASCORE_SCRAPER_BACKEND_LIVE", "chrome")
            current_probe = getattr(constants, "SOFASCORE_SCRAPER_BACKEND_PROBE", "chrome")
            
            if current_live in {"obscura", "traditional"} or current_probe in {"obscura", "traditional"}:
                from monitor_v2.utils.logger import log_warning, COLOR_WARNING, COLOR_RESET
                log_warning(
                    "SYSTEM",
                    f"{COLOR_WARNING}[ANTI-BAN]{COLOR_RESET} 3 errores 403 consecutivos detectados → "
                    f"Auto-sanando red: Rotando backends en caliente a 'CHROME' nativo."
                )
                constants.SOFASCORE_SCRAPER_BACKEND_LIVE = "chrome"
                constants.SOFASCORE_SCRAPER_BACKEND_PROBE = "chrome"
                constants.SOFASCORE_SCRAPER_BACKEND_FT = "chrome"
                
        if _403_streak >= GLOBAL_403_STREAK_TRIGGER:
            _cooldown_until = time.time() + GLOBAL_403_COOLDOWN_SECS
            _403_streak = 0  # resetear racha tras activar cooldown
    else:
        _403_streak = 0

async def record_successful_call() -> None:
    """Registra una llamada exitosa y gestiona la pausa de rotación de cookies."""
    global _success_calls_count
    _success_calls_count += 1
    if _success_calls_count >= SESSION_ROTATE_EVERY:
        _success_calls_count = 0
        await asyncio.sleep(SESSION_ROTATE_PAUSE_SECS)

async def claim_ft_scrape_slot() -> None:
    """Reclama secuencialmente el semáforo para descarga pesada (FT), protegiendo la memoria RAM."""
    global _active_ft_scrapes
    await _ft_scrape_sem.acquire()
    _active_ft_scrapes += 1
        
async def release_ft_scrape_slot() -> None:
    """Libera el slot ocupado para descargas FT en el semáforo global."""
    global _active_ft_scrapes
    _active_ft_scrapes = max(0, _active_ft_scrapes - 1)
    _ft_scrape_sem.release()

async def execute_safe_fetch(fetch_coro):
    """
    Encapsula una corrutina de scraping aplicando semáforos, espaciado con jitter,
    cooldown por racha de 403, y rotación de sesión.
    """
    await wait_if_cooldown()
    await _proxy_timer_check()  # activa free list si pasaron 5min
    
    async with _fetch_sem:
        try:
            # Envolver con timeout estricto de 50s para evitar bloqueos indefinidos de CDP/Obscura
            result = await asyncio.wait_for(fetch_coro(), timeout=FETCH_TIMEOUT_SECS)
            await record_successful_call()
            await check_403_streak(200) # reset racha
            return result
        except asyncio.TimeoutError:
            from monitor_v2.config.constants import SOFASCORE_SCRAPER_BACKEND_LIVE
            backend_label = SOFASCORE_SCRAPER_BACKEND_LIVE or "chrome"
            raise RuntimeError(f"Timeout {FETCH_TIMEOUT_SECS}s [back={backend_label}]")
        except Exception as e:
            err_str = str(e)
            if "403" in err_str:
                await check_403_streak(403)
            raise e
        finally:
            # Garantía de espaciado y jitter microscópico tras liberar semáforo
            spacing = GLOBAL_FETCH_MIN_SPACING_SECS + random.uniform(0.0, 0.5)
            await asyncio.sleep(spacing)


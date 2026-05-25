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

import asyncio
import random
import time
from bet_monitor_v2.config.constants import (
    MAX_CONCURRENT_FETCHES,
    GLOBAL_FETCH_MIN_SPACING_SECS,
    GLOBAL_403_STREAK_TRIGGER,
    GLOBAL_403_COOLDOWN_SECS,
    SESSION_ROTATE_EVERY,
    SESSION_ROTATE_PAUSE_SECS
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

async def check_403_streak(status_code: int) -> None:
    """Incrementa la racha de 403 y dispara el cooldown de 15 minutos si se alcanza el trigger."""
    global _403_streak, _cooldown_until
    if status_code == 403:
        _403_streak += 1
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
    
    async with _fetch_sem:
        try:
            # Envolver con timeout estricto de 50s para evitar bloqueos indefinidos de CDP/Obscura
            result = await asyncio.wait_for(fetch_coro(), timeout=50.0)
            await record_successful_call()
            await check_403_streak(200) # reset racha
            return result
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout de conexión en Playwright/Obscura (50s superados)")
        except Exception as e:
            err_str = str(e)
            if "403" in err_str:
                await check_403_streak(403)
            raise e
        finally:
            # Garantía de espaciado y jitter microscópico tras liberar semáforo
            spacing = GLOBAL_FETCH_MIN_SPACING_SECS + random.uniform(0.0, 0.5)
            await asyncio.sleep(spacing)


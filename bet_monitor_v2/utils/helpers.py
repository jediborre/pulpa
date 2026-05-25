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

import random
import socket
import subprocess
import time
from pathlib import Path
from bet_monitor_v2.config.constants import SOFASCORE_SCRAPER_BACKEND

def is_obscura_port_open() -> bool:
    """Verifica mediante socket TCP si el puerto 9222 está activo en localhost."""
    try:
        with socket.create_connection(("127.0.0.1", 9222), timeout=1.0) as s:
            return True
    except Exception:
        return False

def ensure_obscura_running() -> bool:
    """
    Determina si el backend requiere Obscura y, si está apagado en el 9222,
    lo enciende automáticamente en segundo plano llamando a start_obscura.bat.
    """
    backend = str(SOFASCORE_SCRAPER_BACKEND).strip().lower()
    if backend not in {"obscura", "cdp"}:
        return True

    if is_obscura_port_open():
        return True

    root_dir = Path(__file__).resolve().parents[2]
    bat_path = root_dir / "start_obscura.bat"
    if not bat_path.exists():
        return False

    try:
        # Ejecutar start_obscura.bat de forma totalmente desacoplada en Windows
        subprocess.Popen(
            [str(bat_path)],
            shell=True,
            cwd=str(root_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Espera adaptativa de hasta 5 segundos a que se abra el puerto 9222
        for _ in range(10):
            time.sleep(0.5)
            if is_obscura_port_open():
                return True
        return False
    except Exception:
        return False

def format_human_time(seconds: float) -> str:
    """
    Convierte segundos en una cadena legible en formato humano.
    Ej: 1 dia 2h 15min / 45s
    """
    if seconds < 0:
        return "0s"
        
    s = int(seconds)
    if s < 60:
        return f"{s}s"
        
    m, s_rem = divmod(s, 60)
    if m < 60:
        return f"{m}min {s_rem}s" if s_rem else f"{m}min"
        
    h, m_rem = divmod(m, 60)
    if h < 24:
        return f"{h}h {m_rem}min"
        
    d, h_rem = divmod(h, 24)
    return f"{d} dia {h_rem}h {m_rem}min"


def calculate_ema_secs_per_gmin(current_val: float, wall_time_elapsed: float, minute_delta: float) -> float:
    """
    Calcula el ritmo de juego aplicando Media Móvil Exponencial (EMA)
    sobre el avance del minutero. Trunca el resultado final entre [60.0, 360.0] segundos.
    """
    if minute_delta <= 0:
        return current_val
        
    rate = wall_time_elapsed / minute_delta
    new_val = (0.7 * current_val) + (0.3 * rate)
    
    return max(60.0, min(new_val, 360.0))


def calculate_jitter_sleep_secs(base_sleep: float, phase: str, urgent: bool = False) -> float:
    """
    Aplica Jitter Estocástico al tiempo de suspensión base
    según la criticidad y fase del partido.
    
    Fases válidas: 'q4_far', 'q4_window', 'error_retry'
    """
    if base_sleep <= 0:
        return 0.0
        
    # Parámetros por defecto para q4_far
    jitter_pct = 0.20
    cap = 45.0
    floor = 20.0
    
    if phase == 'q4_window':
        jitter_pct = 0.12
        cap = 18.0
        floor = 12.0
    elif phase == 'error_retry':
        jitter_pct = 0.22
        cap = 55.0
        floor = 15.0
        
    # Si la fase es urgente (evaluación en Q4 en vivo), limitar jitter a max 5% o 6s
    if urgent:
        jitter_pct = min(jitter_pct, 0.05)
        cap = min(cap, 6.0)
        
    # Calcular rango del jitter aleatorio
    jitter_range = base_sleep * jitter_pct
    actual_jitter = random.uniform(-jitter_range, jitter_range)
    
    final_sleep = base_sleep + actual_jitter
    
    # Aplicar techos y suelos (restringidos por jitter)
    return max(floor, min(final_sleep, cap if base_sleep < cap else final_sleep))

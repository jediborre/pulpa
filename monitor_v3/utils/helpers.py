# =====================================================================
# REGLAS DE ARQUITECTURA, FORMATEO Y LOGGING DE PRODUCCIÓN (MANTENER):
# 1. ESTRUCTURA: Estrictamente modularizado (config, core, database, scrapers, 
#    models, notifications, utils) coordinados asíncronamente por main.py.
# 2. INFRAESTRUCTURA DB: El archivo base SQLite se localiza exclusivamente en 
#    matches.db (en la raíz del proyecto) y todas las tablas de V3 finalizan
#    con el sufijo '_v3'.
# 3. TABLA DE LOGS: 'bet_monitor_log_v3' se particiona por modelo y contiene 
#    obligatoriamente los campos 'raw_json' (TEXT), 'inference_minute' (INT), 
#    y 'graph_points_count' (INT) junto con marcadores reales del juego.
# 4. CONFIGURACIÓN DE LIGAS: Consumida declarativamente desde config/leagues.yaml.
# 5. Formato Log: {Fecha Hora} [INFO/WARNING/ERROR] [COMPONENTE]
#    - Colores ANSI: INFO=Azul, WARNING=Amarillo, ERROR=Rojo.
# 6. Formato Matches en Log: {horario_match} {match_id} {home} vs {away}
# 7. Errores de red críticos: Imprimir explícitamente "HTTP 403/404" en ROJO.
# 8. Monitoreo avanzado en progreso de cuarto final: Usar obligatoriamente "Q4 🟠".
# 9. Telegram Prefijos de Apuestas: 🟢 (Bettable), 🟡 (No Bettable), ⚪ (Tardía).
# 10. Telegram Resultados FT: Prefijar con ✅ (Ganada) o ❌ (Perdida).
# 11. Conversión de tiempos siempre legibles en formato humano.
# =====================================================================

import hashlib
import random
import socket
import time
from urllib.parse import urlparse
from monitor_v3.config.constants import (
    SOFASCORE_PROXY_URL,
    SOFASCORE_PKG,
    SOFASCORE_UA_VERSION,
    SOFASCORE_UA_SALT,
)

def build_signed_ua() -> str:
    """
    Reconstruye el User-Agent firmado de la app de SofaScore.

    La app lo calcula como:
        md5( str(unix_segundos // 100) + "sofa2012" )[:6]
    y lo antepone a "com.sofascore.results/260921/". La firma cambia cada 100s.
    """
    bucket = int(time.time() // 100)
    digest = hashlib.md5(f"{bucket}{SOFASCORE_UA_SALT}".encode()).hexdigest()
    return f"{SOFASCORE_PKG}/{SOFASCORE_UA_VERSION}/{digest[:6]}"

def build_mobile_headers(token: str | None = None) -> dict:
    """Cabeceras exactas que la app envía a la API móvil de SofaScore."""
    headers = {
        "User-Agent": build_signed_ua(),
        "X-Timestamp": str(int(time.time() * 1000)),
        "app-version": SOFASCORE_UA_VERSION,
        "Cache-Control": "max-age=0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "Connection": "Keep-Alive",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def is_proxy_accessible() -> bool:
    """Verifica si el proxy HTTP local o remoto está aceptando conexiones TCP."""
    if not SOFASCORE_PROXY_URL:
        return True
    try:
        parsed = urlparse(SOFASCORE_PROXY_URL)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=1.5):
            return True
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
        
    if urgent:
        jitter_pct = min(jitter_pct, 0.05)
        cap = min(cap, 6.0)
        
    jitter_range = base_sleep * jitter_pct
    actual_jitter = random.uniform(-jitter_range, jitter_range)
    final_sleep = base_sleep + actual_jitter
    
    return max(floor, min(final_sleep, cap if base_sleep < cap else final_sleep))

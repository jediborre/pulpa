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

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Códigos ANSI para colores de consola
COLOR_DEBUG = "\033[90m"      # Gris
COLOR_INFO = "\033[94m"       # Azul
COLOR_WARNING = "\033[93m"    # Amarillo
COLOR_ERROR = "\033[91m"      # Rojo
COLOR_RESET = "\033[0m"

COLOR_LATE_APUESTA = "\033[37m" # Blanco/Gris claro
COLOR_BRIGHT_RED = "\033[91;1m" # Rojo brillante
COLOR_GREEN = "\033[92m"      # Verde

def get_utc6_now() -> datetime:
    """Retorna la fecha y hora actual en la zona horaria UTC-6."""
    tz = timezone(timedelta(hours=-6))
    return datetime.now(tz)

def format_match_log(horario_match: str, match_id: str, home: str, away: str) -> str:
    """
    Formatea la visualización estándar de un match:
    {horario_match} en Amarillo, {match_id} en Azul.
    """
    fmt_horario = f"{COLOR_WARNING}{horario_match}{COLOR_RESET}"
    fmt_id = f"{COLOR_INFO}{match_id}{COLOR_RESET}"
    return f"{fmt_horario} {fmt_id} {home} vs {away}"

def format_critical_error(err_msg: str) -> str:
    """
    Formatea de forma explícita los errores críticos de red:
    HTTP 403/404 en Rojo brillante sin corromper IDs de partidos.
    """
    import re
    # 1. Reemplazar "HTTP 403" o "HTTP 404" explícitos primero
    err_msg = re.sub(r"\bHTTP\s+403\b", f"{COLOR_BRIGHT_RED}HTTP 403{COLOR_RESET}", err_msg)
    err_msg = re.sub(r"\bHTTP\s+404\b", f"{COLOR_BRIGHT_RED}HTTP 404{COLOR_RESET}", err_msg)
    
    # 2. Reemplazar "403" o "404" aislados (que no formen parte de un número de ID largo)
    # Evitamos formatear si forma parte de un ID de partido (ej. precedido por otros dígitos)
    err_msg = re.sub(r"(?<![\d\033\[;])\b403\b(?![\d\033])", f"{COLOR_BRIGHT_RED}HTTP 403{COLOR_RESET}", err_msg)
    err_msg = re.sub(r"(?<![\d\033\[;])\b404\b(?![\d\033])", f"{COLOR_BRIGHT_RED}HTTP 404{COLOR_RESET}", err_msg)
    
    return err_msg

def _write_file_log(level: str, component: str, msg: str) -> None:
    """Escribe los logs en archivos rotativos diarios de forma segura."""
    now = get_utc6_now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Resolver ruta de logs en el directorio raíz del proyecto /logs/
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"monitor_{date_str}.log"
        
        # Limpiar secuencias de escape ANSI para el archivo de texto plano
        clean_msg = msg
        for code in [COLOR_DEBUG, COLOR_INFO, COLOR_WARNING, COLOR_ERROR, COLOR_RESET, COLOR_LATE_APUESTA, COLOR_BRIGHT_RED, COLOR_GREEN]:
            clean_msg = clean_msg.replace(code, "")
            
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{time_str} [{level}] [{component}] {clean_msg}\n")
    except Exception:
        # Si no se puede escribir el log (ej. permisos), se ignora silenciosamente
        pass

def _log_base(level: str, color: str, component: str, msg: str, q4_orange: bool = False) -> None:
    """Escribe el log en consola colorizada y en el archivo rotativo correspondiente."""
    now = get_utc6_now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Aplicar indicador Q4 si corresponde
    prefix = "Q4 🟠 " if q4_orange else ""
    
    # Formatear errores de red críticos
    formatted_msg = format_critical_error(msg)
    
    # Escribir en consola
    sys.stdout.write(f"{time_str} {color}[{level}]{COLOR_RESET} [{component}] {prefix}{formatted_msg}\n")
    sys.stdout.flush()
    
    # Guardar en archivo
    _write_file_log(level, component, f"{prefix}{formatted_msg}")


def log_debug(component: str, msg: str, q4_orange: bool = False) -> None:
    _log_base("DEBUG", COLOR_DEBUG, component, msg, q4_orange)

def log_info(component: str, msg: str, q4_orange: bool = False) -> None:
    _log_base("INFO", COLOR_INFO, component, msg, q4_orange)

def log_warning(component: str, msg: str, q4_orange: bool = False) -> None:
    _log_base("WARNING", COLOR_WARNING, component, msg, q4_orange)

def log_error(component: str, msg: str, q4_orange: bool = False) -> None:
    _log_base("ERROR", COLOR_ERROR, component, msg, q4_orange)

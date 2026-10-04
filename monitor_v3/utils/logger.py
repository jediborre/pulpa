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

import sys
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Forzar salida UTF-8 en consola Windows para evitar UnicodeEncodeError con emojis.
for _stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Códigos ANSI para colores de consola
COLOR_DEBUG = "\033[90m"        # Gris
COLOR_INFO = "\033[94m"         # Azul
COLOR_WARNING = "\033[93m"      # Amarillo
COLOR_ERROR = "\033[91m"        # Rojo
COLOR_RESET = "\033[0m"

COLOR_LATE_APUESTA = "\033[37m" # Blanco/Gris claro
COLOR_BRIGHT_RED = "\033[91;1m" # Rojo brillante
COLOR_GREEN = "\033[92m"        # Verde
COLOR_ORANGE = "\033[38;5;208m" # Naranja
COLOR_CYAN = "\033[96m"         # Cyan
COLOR_MAGENTA = "\033[95m"      # Magenta

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
    err_msg = re.sub(r"\bHTTP\s+403\b", f"{COLOR_BRIGHT_RED}HTTP 403{COLOR_RESET}", err_msg)
    err_msg = re.sub(r"\bHTTP\s+404\b", f"{COLOR_BRIGHT_RED}HTTP 404{COLOR_RESET}", err_msg)
    err_msg = re.sub(r"(?<![\d\033\[;])\b403\b(?![\d\033])", f"{COLOR_BRIGHT_RED}HTTP 403{COLOR_RESET}", err_msg)
    err_msg = re.sub(r"(?<![\d\033\[;])\b404\b(?![\d\033])", f"{COLOR_BRIGHT_RED}HTTP 404{COLOR_RESET}", err_msg)
    return err_msg

def _write_file_log(level: str, component: str, msg: str) -> None:
    """Escribe los logs en archivos rotativos diarios de forma segura."""
    now = get_utc6_now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"monitor_v3_{date_str}.log"
        
        # Limpiar secuencias de escape ANSI
        clean_msg = re.sub(r"\033\[[0-9;]*m", "", msg)
            
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{time_str} [{level}] [{component}] {clean_msg}\n")
    except Exception:
        pass

def _log_base(level: str, color: str, component: str, msg: str, q4_orange: bool = False) -> None:
    """Escribe el log en consola colorizada y en el archivo rotativo correspondiente."""
    now = get_utc6_now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    prefix = "Q4 🟠 " if q4_orange else ""
    formatted_msg = format_critical_error(msg)
    
    # Colorizar componentes específicos en la consola
    fmt_component = f"[{component}]"
    if component in ("DESCARGA", "EVALUACION"):
        fmt_component = f"{COLOR_GREEN}[{component}]{COLOR_RESET}"
    elif component in ("MONITOREO", "TELEGRAM"):
        fmt_component = f"{COLOR_CYAN}[{component}]{COLOR_RESET}"
    elif component in ("SYSTEM", "TOKEN_POOL"):
        fmt_component = f"{COLOR_MAGENTA}[{component}]{COLOR_RESET}"
    
    # Colorizar pipes (|) y tags
    formatted_msg = formatted_msg.replace(" | ", f" {COLOR_GREEN}|{COLOR_RESET} ")
    formatted_msg = formatted_msg.replace("[PROBE]", f"{COLOR_ORANGE}[PROBE]{COLOR_RESET}")
    formatted_msg = formatted_msg.replace("[WATCHER]", f"{COLOR_ORANGE}[WATCHER]{COLOR_RESET}")
    
    sys.stdout.write(f"{time_str} {color}[{level}]{COLOR_RESET} {fmt_component} {prefix}{formatted_msg}\n")
    sys.stdout.flush()
    
    _write_file_log(level, component, f"{prefix}{formatted_msg}")

def log_debug(component: str, msg: str, q4_orange: bool = False) -> None:
    _log_base("DEBUG", COLOR_DEBUG, component, msg, q4_orange)

def log_info(component: str, msg: str, q4_orange: bool = False) -> None:
    _log_base("INFO", COLOR_INFO, component, msg, q4_orange)

def log_warning(component: str, msg: str, q4_orange: bool = False) -> None:
    _log_base("WARNING", COLOR_WARNING, component, msg, q4_orange)

def log_error(component: str, msg: str, q4_orange: bool = False) -> None:
    _log_base("ERROR", COLOR_ERROR, component, msg, q4_orange)

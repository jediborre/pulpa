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
from pathlib import Path

# Cargar variables de entorno desde el .env raíz del proyecto
# (funciona independientemente del directorio de ejecución)
try:
    from dotenv import load_dotenv as _load_dotenv
    _ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
    if _ROOT_ENV.exists():
        _load_dotenv(_ROOT_ENV, override=False)  # override=False: no sobreescribe vars ya definidas
except ImportError:
    pass  # dotenv opcional: si no está instalado se ignora

# --- Variables de Entorno Requeridas ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
SOFASCORE_SCRAPER_BACKEND = os.getenv("SOFASCORE_SCRAPER_BACKEND", "obscura").strip().lower()

# --- Rutas de Archivos de Configuración e infraestructura ---
DB_FILE_PATH = "/match/matches.db"          # Ubicación absoluta y estricta de la base de datos
LEAGUES_CONFIG_PATH = "config/leagues.yaml"  # Ruta del archivo dinámico de ligas

# --- Configuración de Red y Anti-Ban ---
MAX_CONCURRENT_FETCHES = 2         # Semáforo de control de concurrencia de red
GLOBAL_FETCH_MIN_SPACING_SECS = 1.5 # Espaciado mínimo absoluto entre peticiones consecutivas
GLOBAL_403_STREAK_TRIGGER = 6      # Número de errores 403 seguidos que disparan el cooldown
GLOBAL_403_COOLDOWN_SECS = 900     # Tiempo de enfriamiento (15 min) ante bloqueos
SESSION_ROTATE_EVERY = 40          # Rotación de sesión/cookies cada N llamadas exitosas
SESSION_ROTATE_PAUSE_SECS = 45     # Pausa obligatoria al rotar sesión
MONITORING_LOCK_DURATION_SECS = 300 # Lock de 5 min para pausar tareas secundarias al haber partidos Live

# --- Ritmo de Juego y Ventana de Monitoreo Q4 ---
SECS_PER_GAME_MIN = 170            # Estimación inicial de segundos reales por minuto de juego
Q4_ONLY_EARLY_WAKE_MINUTE = 27     # Minuto del partido para despertar el watcher completo
Q4_ONLY_WAKE_LEAD_MINUTES = 2      # Margen de seguridad previo al minuto de despertar
Q4_TOO_LATE_BET_MINUTE = 33        # Límite para alertas ordinarias (minutos superiores marcan TARDÍA)
Q4_TOO_LATE_HARD_MINUTE = 36       # Bloqueo total: No se procesan apuestas pasada esta marca de tiempo
Q4_LATE_BET_MAX_DISADVANTAGE = 5   # Máxima desventaja de puntos permitida para el pick en apuesta tardía
Q4_WAITING_MAX_TICKS = 8           # Ticks máximos de espera por sincronización de score de periodos previos antes de abortar
Q4_STALE_MAX_TICKS = 5             # Ticks máximos sin crecimiento en la gráfica antes de abortar

# --- Sondeos Prestart (Modo Sonda) ---
PRESTART_PROBE_MIN_SECS = 60       # Intervalo mínimo de sondeo para partidos no iniciados
PRESTART_PROBE_MAX_SECS = 300      # Intervalo máximo de sondeo (Cap del Backoff)
PRESTART_PROBE_BACKOFF = 1.25      # Factor multiplicador del temporizador exponencial
PROBE_GLOBAL_TIMEOUT_SECS = 21600  # Timeout global (6h) para evitar tareas colgadas en probe mode

# --- Polling y Descarga Final (FT) ---
POLL_INTERVAL_FF = 150             # Intervalo base para verificar partido finalizado
POLL_JITTER_FF = 45                # Jitter máximo para desincronizar polling final
FINAL_FETCH_EXTRA_SECS = 300       # Margen de seguridad tras tiempo estimado de fin
FINAL_FETCH_MIN_GP = 20            # Mínimo de puntos de gráfica requeridos para persistencia válida
FT_SCRAPE_SLOT_SPACING_BASE = 35.0 # Identificador base en segundos para encolar descargas FT

# --- Intervalos de Tareas Secundarias ---
SCHEDULE_REFRESH_HOURS = 8         # Frecuencia de actualización del itinerario
PENDING_RECHECK_SECS = 3600        # Recheck de resultados cada hora
POLL_NEAR_SECS = 45                # Polling corto dentro de la ventana de juego activa
IDLE_POLL_SECS = 180               # Polling largo en periodos de inactividad
NO_BET_CONFIRM_TICKS = 1           # Confirmaciones consecutivas necesarias para asentar NO_BET (1 = evaluación instantánea)
UTC_OFFSET_HOURS = -6              # Desfase horario local

# --- Catálogo de Modelos Activos ---
# Reemplazado v12 por m27_v3 por solicitud del usuario
ACTIVE_MODELS = ["v6_2", "m27_v3"]

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

import os
from pathlib import Path

# Cargar .env de la raíz
try:
    from dotenv import load_dotenv as _load_dotenv
    _ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
    if _ROOT_ENV.exists():
        _load_dotenv(_ROOT_ENV, override=False)
except ImportError:
    pass

# --- Variables de Entorno y Notificaciones ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# --- Rutas de Archivos e Infraestructura ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DB_FILE_PATH = "matches.db"
LEAGUES_CONFIG_PATH = str(Path(__file__).resolve().parent / "leagues.yaml")
TOKENS_CONFIG_PATH = str(Path(__file__).resolve().parent / "tokens.json")

# --- Configuración del Motor Móvil y Red ---
# El WAF de Fastly ya no se evade con proxy sino con huella TLS OkHttp Android +
# User-Agent firmado (ver docs/REVERSE_ENGINEERING_SOFASCORE.md). Por defecto se
# conecta DIRECTO (sin proxy). Solo se usa un proxy si el .env define uno explícito
# y distinto del smartproxy (que está muerto/407).
_env_proxy = os.getenv("SOFASCORE_PROXY_URL", "").strip()
if _env_proxy and "smartproxy" not in _env_proxy:
    SOFASCORE_PROXY_URL = _env_proxy
else:
    SOFASCORE_PROXY_URL = ""

SOFASCORE_CERT_PATH = os.getenv("SOFASCORE_CERT_PATH", r"C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem").strip()
SOFASCORE_MOBILE_UA = os.getenv("SOFASCORE_MOBILE_UA", "com.sofascore.results/260921/022538").strip()
SOFASCORE_API_BASE = "https://api.sofascore.com/api/v1"

# --- Contrato móvil firmado (extraído por reverse engineering del APK) ---
# UA = "{SOFASCORE_PKG}/{SOFASCORE_UA_VERSION}/{md5(str(unix//100)+SOFASCORE_UA_SALT)[:6]}"
SOFASCORE_PKG = "com.sofascore.results"
SOFASCORE_APP_VERSION = "260921003"
SOFASCORE_UA_VERSION = "260921"
SOFASCORE_UA_SALT = "sofa2012"
# Huella TLS/HTTP2 OkHttp Android (tls_client). NO usar httpx/requests/curl_cffi.
SOFASCORE_TLS_CLIENT_ID = os.getenv("SOFASCORE_TLS_CLIENT_ID", "okhttp4_android_13").strip()

# Concurrencia y Timeouts Móviles (ultra ligeros)
MAX_CONCURRENT_FETCHES = 8          # API móvil asíncrona pura soporta mayor concurrencia
FETCH_TIMEOUT_SECS = 15.0           # 15s de timeout para operaciones HTTP
FETCH_MIN_SPACING_SECS = 0.05       # 50ms de espaciado mínimo

# --- Pool de Tokens JWT Rotativo ---
TOKEN_ROTATION_POOL_SIZE = 5        # Cantidad objetivo de tokens concurrentes
TOKEN_MAX_FAILURES = 3              # Fallos consecutivos antes de revocar y regenerar token
TOKEN_REFRESH_MINUTES = 60 * 24 * 7 # Chequeo preventivo semanal (tokens duran ~6 meses)

# Modelos Activos para Inferencia
ACTIVE_MODELS = ["v6_2", "m27_v3"]

# --- Ritmo de Juego y Ventana de Monitoreo Q4 ---
SECS_PER_GAME_MIN = 170             # Estimación inicial de segundos reales por minuto de juego
Q4_ONLY_EARLY_WAKE_MINUTE = 27      # Minuto del partido para despertar el watcher completo
Q4_ONLY_WAKE_LEAD_MINUTES = 2       # Margen de seguridad previo al minuto de despertar
Q4_TOO_LATE_BET_MINUTE = 33         # Límite para alertas ordinarias (minutos superiores marcan TARDÍA)
Q4_TOO_LATE_HARD_MINUTE = 36        # Bloqueo total: No se procesan apuestas pasada esta marca de tiempo
Q4_LATE_BET_MAX_DISADVANTAGE = 5    # Máxima desventaja de puntos permitida para el pick en apuesta tardía
Q4_WAITING_MAX_TICKS = 8            # Ticks máximos esperando sync de cuartos previos antes de abortar
Q4_STALE_MAX_TICKS = 15             # Ticks máximos sin crecimiento en gráfica ni marcador
NO_BET_CONFIRM_TICKS = 1            # Confirmaciones consecutivas para asentar NO_BET

# --- Sondeos Prestart (Modo Sonda) ---
PRESTART_PROBE_MIN_SECS = 60        # Intervalo mínimo de sondeo para partidos no iniciados
PRESTART_PROBE_MAX_SECS = 300       # Intervalo máximo de sondeo (Cap del Backoff)
PRESTART_PROBE_BACKOFF = 1.25       # Factor multiplicador del temporizador exponencial
PROBE_GLOBAL_TIMEOUT_SECS = 21600   # Timeout global (6h) para evitar tareas colgadas

# --- Polling y Descarga Final (FT) ---
POLL_INTERVAL_LIVE_SECS = 15        # Polling frecuente en vivo para Q4
POLL_NEAR_SECS = 45                 # Polling en ventana activa previa
POLL_INTERVAL_IDLE_SECS = 120       # Polling cuando no hay partidos activos en Q4
FINAL_FETCH_EXTRA_SECS = 180        # Margen de seguridad tras tiempo estimado de fin
FINAL_FETCH_MIN_GP = 20             # Mínimo de puntos de gráfica requeridos para persistencia válida
FT_SCRAPE_SLOT_SPACING_BASE = 1.0   # Segundos base entre descargas FT

# --- Intervalos de Tareas Secundarias ---
SCHEDULE_REFRESH_HOURS = 4          # Frecuencia de actualización del itinerario
PENDING_RECHECK_SECS = 1800         # Recheck de resultados cada 30 minutos
UTC_OFFSET_HOURS = -6               # Horario local Ciudad de México / CST

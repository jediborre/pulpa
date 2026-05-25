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
import sqlite3
from pathlib import Path
from bet_monitor_v2.config.constants import DB_FILE_PATH

def get_real_db_path() -> str:
    """
    Normaliza y resuelve la ruta de la base de datos para asegurar compatibilidad
    entre Windows y Unix. Si la ruta absoluta /match/matches.db no es escribible,
    o está vacía/incompleta, hace fallback al directorio match/ del workspace.
    """
    p = Path(DB_FILE_PATH)
    workspace_db = Path(__file__).resolve().parents[2] / "match" / "matches.db"
    
    # Comprobar si el archivo /match/matches.db ya existe y tiene tamaño real (los 600MB)
    if p.exists() and p.stat().st_size > 1000000:
        return str(p.resolve())
        
    # Si existe en el workspace local con el tamaño de 600MB, usar la de local workspace
    if workspace_db.exists() and workspace_db.stat().st_size > 1000000:
        return str(workspace_db.resolve())
        
    # Fallback genérico resolviendo rutas
    if os.name == 'nt':
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return str(p.resolve())
        except Exception:
            return str(workspace_db.resolve())
    else:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return str(p.resolve())
        except Exception:
            return str(workspace_db.resolve())

def get_db_connection() -> sqlite3.Connection:
    """
    Obtiene una conexión SQLite a la base de datos configurada, aplicando
    las directivas de journal_mode=WAL y busy_timeout de forma atómica.
    """
    db_path = get_real_db_path()
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    
    # Activar el modo WAL y busy_timeout
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    
    return conn

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

import sys
import asyncio
from pathlib import Path

# Agregar el directorio raíz del proyecto al sys.path para poder importar match
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match.scraper import fetch_event_snapshot as ss_fetch_snapshot
from monitor_v2.scrapers.base_scraper import execute_safe_fetch

async def fetch_event_snapshot(match_id: str, backend: str | None = None) -> dict:
    """
    Sondea de forma optimizada el endpoint ligero /event/{id} vía el backend configurado
    encapsulado dentro del gestor de reintentos y anti-ban seguro.
    """
    from monitor_v2.config.constants import SOFASCORE_SCRAPER_BACKEND_PROBE
    
    backend_to_use = backend or SOFASCORE_SCRAPER_BACKEND_PROBE
    
    async def _fetch():
        # Ejecutar en hilo separado ya que el cliente de match.scraper usa Playwright síncrono
        return await asyncio.to_thread(ss_fetch_snapshot, match_id, backend=backend_to_use)
        
    return await execute_safe_fetch(_fetch)

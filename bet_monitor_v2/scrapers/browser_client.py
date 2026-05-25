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

import sys
import asyncio
from pathlib import Path

# Agregar el directorio raíz del proyecto al sys.path para poder importar match
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match.scraper import fetch_match_by_id as ss_fetch_match
from bet_monitor_v2.scrapers.base_scraper import execute_safe_fetch, claim_ft_scrape_slot, release_ft_scrape_slot
from bet_monitor_v2.config.constants import SOFASCORE_SCRAPER_BACKEND

async def fetch_match_by_id(match_id: str, is_ft: bool = False) -> dict:
    """
    Realiza una ráfaga secuencial controlada de peticiones para compilar
    la totalidad de las estructuras analíticas del partido, implementando
    la cola serializada si es un cierre de partido (FT).
    """
    if is_ft:
        # Reclamar slot de cola serializada FT
        await claim_ft_scrape_slot()
        
    try:
        async def _fetch():
            # Estrategia híbrida: las descargas finales de cierre (FT) fuerzan 'traditional',
            # mientras que el monitoreo activo en vivo utiliza el backend Obscura de constants.py.
            backend_to_use = "traditional" if is_ft else SOFASCORE_SCRAPER_BACKEND
            return await asyncio.to_thread(ss_fetch_match, match_id, backend=backend_to_use)
            
        return await execute_safe_fetch(_fetch)
    finally:
        if is_ft:
            await release_ft_scrape_slot()

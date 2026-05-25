# =====================================================================
# SCRIPT DE UTILIDAD: FUERZA LA DESCARGA DE PARTIDOS PENDIENTES DEL PASADO
# =====================================================================

import sys
import asyncio
import time
from pathlib import Path

# Cargar path del workspace
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.database.connection import get_db_connection
from bet_monitor_v2.main import _final_fetch_and_save
from bet_monitor_v2.utils.logger import log_info, log_warning, log_error
from bet_monitor_v2.utils.helpers import ensure_obscura_running

async def force_download_all():
    log_info("CLI", "Iniciando descarga forzada de partidos pendientes finalizados en el pasado...")
    
    # Auto-encendido inteligente de Obscura si es requerido
    if ensure_obscura_running():
        log_info("CLI", "Servicio Obscura (CDP) verificado y activo en puerto 9222.")
    else:
        log_warning("CLI", "Obscura (CDP) inactivo. Asegúrate de tener Obscura corriendo.")
        
    # Buscar partidos cuya hora de inicio programada ya pasó y siguen pendientes
    now_ts = int(time.time())
    
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT match_id, home_team, away_team, scheduled_utc_ts, status
            FROM bet_monitor_schedule_v2
            WHERE status = 'pending' 
              AND final_fetched = 0 
              AND scheduled_utc_ts < ?
              AND (skip_reason IS NULL OR skip_reason != 'final_fetch_failed')
            ORDER BY scheduled_utc_ts DESC
        """, (now_ts,))
        pending = [dict(r) for r in cursor.fetchall()]
        
    total_pending = len(pending)
    log_info("CLI", f"Se encontraron {total_pending} partidos pendientes del pasado para descargar.")
    
    if total_pending == 0:
        log_info("CLI", "¡No hay partidos pendientes del pasado por descargar!")
        return

    success_count = 0
    fail_count = 0
    
    for idx, m in enumerate(pending, 1):
        mid = m["match_id"]
        home = m["home_team"]
        away = m["away_team"]
        
        log_info("CLI", f"[{idx}/{total_pending}] Procesando descarga FT para: {home} vs {away} (ID: {mid})...")
        
        try:
            # Reutiliza la infraestructura robusta con semáforos, Playwright y timeouts
            await _final_fetch_and_save(mid, home, away)
            success_count += 1
            log_info("CLI", f"  [OK] Descargado y guardado con éxito: {home} vs {away}")
        except Exception as e:
            fail_count += 1
            log_error("CLI", f"  [ERROR] Falló descarga de {home} vs {away}: {e}")
            
    log_info("CLI", f"=== PROCESO TERMINADO ===")
    log_info("CLI", f"Descargados correctamente: {success_count}")
    log_info("CLI", f"Fallidos o con error: {fail_count}")

if __name__ == "__main__":
    # Soporte para emojis en Windows CMD
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    asyncio.run(force_download_all())

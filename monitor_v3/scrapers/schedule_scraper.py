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

import asyncio
from datetime import datetime, timezone
from monitor_v3.core.mobile_client import get_mobile_client
from monitor_v3.utils.logger import log_info, log_error

async def fetch_schedule_matches_for_date(date_str: str) -> list[dict]:
    """
    Descarga el itinerario completo de partidos de baloncesto para una fecha
    vía la API móvil y los mapea al esquema de bet_monitor_schedule_v3.
    """
    mc = get_mobile_client()
    try:
        raw_events = await mc.get_all_scheduled_events_for_date(date_str)
        mapped = []
        for ev in raw_events:
            ts = ev.get("startTimestamp", 0)
            utc_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
            
            mapped.append({
                "match_id": str(ev.get("id")),
                "home_team": ev.get("homeTeam", {}).get("name", "Home"),
                "away_team": ev.get("awayTeam", {}).get("name", "Away"),
                "league": ev.get("tournament", {}).get("name", ""),
                "event_date": date_str,
                "scheduled_utc_ts": ts,
                "scheduled_utc": utc_iso,
                "status_type": ev.get("status", {}).get("type", "notstarted"),
            })
        return mapped
    except Exception as e:
        log_error("DESCARGA", f"Error descargando itinerario para {date_str}: {e}")
        return []

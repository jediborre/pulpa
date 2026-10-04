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
import os
import sys
import time
from pathlib import Path
import httpx

from monitor_v3.config.constants import (
    SOFASCORE_API_BASE,
    SOFASCORE_MOBILE_UA,
    SOFASCORE_PROXY_URL,
    SOFASCORE_CERT_PATH,
    FETCH_TIMEOUT_SECS,
    FETCH_MIN_SPACING_SECS,
    MAX_CONCURRENT_FETCHES,
    UTC_OFFSET_HOURS,
)
from monitor_v3.core.token_manager import TokenPool, get_token_pool
from monitor_v3.utils.logger import log_error, log_warning, log_info

# Asegurar import de los parseadores canónicos de match
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from match.scraper import (
    _parse,
    _parse_h2h,
    _parse_team_statistics,
    _parse_period_stats,
    _parse_lineups,
    _parse_odds,
)

class MobileClient:
    """
    Cliente HTTP asíncrono puro optimizado para la API móvil de SofaScore.
    Utiliza el pool rotativo de JWTs (TokenPool), enrutamiento por túnel local
    con certificado SSL propio, control estricto de concurrencia y reintentos adaptativos.
    """
    def __init__(self, token_pool: TokenPool | None = None):
        self.token_pool = token_pool or get_token_pool()
        self._verify_cert = SOFASCORE_CERT_PATH if os.path.exists(SOFASCORE_CERT_PATH) else True
        self._proxy = SOFASCORE_PROXY_URL if SOFASCORE_PROXY_URL else None
        self._client: httpx.AsyncClient | None = None
        self._client_loop = None

    @property
    def sem(self) -> asyncio.Semaphore:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if not hasattr(self, "_sem_loop") or self._sem_loop != current_loop:
            self._sem_loop = current_loop
            self._sem_obj = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
        return self._sem_obj

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self) -> None:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if (
            self._client is None
            or self._client.is_closed
            or self._client_loop != current_loop
        ):
            self._client_loop = current_loop
            self._client = httpx.AsyncClient(
                proxy=self._proxy,
                verify=self._verify_cert,
                timeout=FETCH_TIMEOUT_SECS,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
            )

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            self._client_loop = None

    async def request(self, method: str, path: str, retry_count: int = 1, **kwargs) -> httpx.Response:
        """
        Ejecuta una petición HTTP autenticada con rotación de tokens y auto-sanación.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._client is None or self._client.is_closed or self._client_loop != current_loop:
            await self.start()

        url = path if path.startswith("http") else f"{SOFASCORE_API_BASE}/{path.lstrip('/')}"

        for attempt in range(retry_count + 1):
            token = await self.token_pool.get_token()
            headers = kwargs.pop("headers", {})
            headers.update({
                "User-Agent": SOFASCORE_MOBILE_UA,
                "x-timestamp": str(int(time.time() * 1000)),
                "Authorization": f"Bearer {token}",
                "Accept-Encoding": "gzip",
                "Connection": "Keep-Alive",
            })

            async with self.sem:
                try:
                    res = await self._client.request(method, url, headers=headers, **kwargs)
                    
                    # Si el token caduca o es rechazado, marcar fallo y reintentar con el siguiente del pool
                    if res.status_code in (401, 403):
                        await self.token_pool.mark_failure(token, status_code=res.status_code)
                        if attempt < retry_count:
                            await asyncio.sleep(0.1)
                            continue
                    
                    await asyncio.sleep(FETCH_MIN_SPACING_SECS)
                    return res

                except Exception as e:
                    if attempt < retry_count:
                        await asyncio.sleep(0.2)
                        continue
                    raise e

        return res

    # -------------------------------------------------------------
    # Métodos de Alto Nivel de la API Móvil
    # -------------------------------------------------------------

    async def get_live_events(self) -> list[dict]:
        """Obtiene en una sola llamada ultrarrápida todos los partidos de baloncesto en vivo globalmente."""
        try:
            res = await self.request("GET", "sport/basketball/events/live")
            if res.status_code == 200:
                return res.json().get("events", [])
            elif res.status_code == 404:
                return []
            else:
                log_warning("DESCARGA", f"get_live_events retorno HTTP {res.status_code}")
                return []
        except Exception as e:
            log_error("DESCARGA", f"Error en get_live_events: {e}")
            return []

    async def get_categories_for_date(self, date_str: str, tz_offset_seconds: int = UTC_OFFSET_HOURS * 3600) -> list[dict]:
        """Obtiene las categorías (países/torneos) con partidos de baloncesto para una fecha determinada."""
        try:
            path = f"sport/basketball/{date_str}/{tz_offset_seconds}/categories"
            res = await self.request("GET", path)
            if res.status_code == 200:
                cats = res.json().get("categories", [])
                # Filtrar categorías con partidos programados
                return [c for c in cats if c.get("totalEvents", 0) > 0]
            return []
        except Exception as e:
            log_error("DESCARGA", f"Error consultando categorías para {date_str}: {e}")
            return []

    async def get_category_scheduled_events(self, category_id: int, date_str: str) -> list[dict]:
        """Obtiene los partidos programados de una categoría completa para una fecha."""
        try:
            path = f"category/{category_id}/scheduled-events/{date_str}"
            res = await self.request("GET", path)
            if res.status_code == 200:
                return res.json().get("events", [])
            return []
        except Exception as e:
            log_error("DESCARGA", f"Error en eventos de categoría {category_id} ({date_str}): {e}")
            return []

    async def get_all_scheduled_events_for_date(self, date_str: str) -> list[dict]:
        """
        Descarga el calendario integral de baloncesto para una fecha consultando en paralelo
        todas las categorías activas.
        """
        active_cats = await self.get_categories_for_date(date_str)
        if not active_cats:
            return []

        tasks = [
            self.get_category_scheduled_events(c["category"]["id"], date_str)
            for c in active_cats
            if "category" in c and "id" in c["category"]
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_events: list[dict] = []
        seen_ids = set()

        for res_batch in results:
            if isinstance(res_batch, list):
                for ev in res_batch:
                    eid = ev.get("id")
                    if eid and eid not in seen_ids:
                        seen_ids.add(eid)
                        all_events.append(ev)

        return all_events

    async def get_event_snapshot(self, match_id: str) -> dict:
        """
        Obtiene el estado ligero en vivo (/event/{match_id}) para actualización de marcador
        y sincronización de periodos.
        """
        res = await self.request("GET", f"event/{match_id}")
        if res.status_code != 200:
            raise RuntimeError(f"HTTP {res.status_code}")

        body = res.json()
        ev = body.get("event", body)
        status = ev.get("status") or {}
        hs = ev.get("homeScore") or {}
        as_ = ev.get("awayScore") or {}

        return {
            "status_type": status.get("type", ""),
            "status_description": status.get("description", ""),
            "status_code": status.get("code", None),
            "home_score": hs.get("current", hs.get("normaltime", 0)),
            "away_score": as_.get("current", as_.get("normaltime", 0)),
            "raw_event": ev,
        }

    async def fetch_full_match(self, match_id: str) -> dict:
        """
        Realiza una ráfaga paralela y atómica de todos los endpoints analíticos del partido
        y genera la estructura canónica lista para inferencia de modelos e inserción en matches.db.
        """
        endpoints = {
            "event": f"event/{match_id}",
            "incidents": f"event/{match_id}/incidents",
            "graph": f"event/{match_id}/graph",
            "h2h": f"event/{match_id}/h2h",
            "statistics": f"event/{match_id}/statistics",
            "lineups": f"event/{match_id}/lineups",
            "odds": f"event/{match_id}/odds/1/all",
        }

        async def _fetch_ep(name: str, path: str):
            try:
                r = await self.request("GET", path)
                if r.status_code == 200:
                    return name, r.json()
                return name, {}
            except Exception:
                return name, {}

        tasks = [_fetch_ep(k, v) for k, v in endpoints.items()]
        results = dict(await asyncio.gather(*tasks))

        event_json = results.get("event", {})
        if not event_json or "event" not in event_json:
            raise RuntimeError(f"No se pudo obtener información del partido {match_id}")

        incidents_json = results.get("incidents", {})
        graph_json = results.get("graph", {})
        h2h_json = results.get("h2h", {})
        statistics_json = results.get("statistics", {})
        lineups_json = results.get("lineups", {})
        odds_json = results.get("odds", {})

        incidents = incidents_json.get("incidents", []) if isinstance(incidents_json, dict) else []
        graph_points = graph_json.get("graphPoints", []) if isinstance(graph_json, dict) else []

        # Parsear con los módulos canónicos de match
        parsed = _parse(event_json, incidents, graph_points)

        # H2H
        parsed["h2h"] = []
        if h2h_json:
            parsed["h2h"] = _parse_h2h(match_id, h2h_json)

        # Estadísticas de equipo
        if statistics_json:
            parsed["team_statistics"] = _parse_team_statistics(match_id, statistics_json)
        else:
            parsed["team_statistics"] = []

        # Estadísticas por periodo computadas desde incidents
        parsed["period_stats"] = []
        if incidents:
            ev = event_json.get("event", {})
            period_len = (ev.get("time") or {}).get("periodLength", 600)
            try:
                parsed["period_stats"] = _parse_period_stats(match_id, incidents, period_seconds=period_len)
            except Exception:
                pass

        # Alineaciones y stats de jugadores
        if lineups_json:
            lineup_rows, player_stat_rows = _parse_lineups(match_id, lineups_json)
            parsed["lineups"] = lineup_rows
            parsed["player_stats"] = player_stat_rows
        else:
            parsed["lineups"] = []
            parsed["player_stats"] = []

        # Cuotas pre-partido
        if odds_json:
            parsed["odds"] = _parse_odds(match_id, odds_json)
        else:
            parsed["odds"] = []

        return parsed

# Singleton para reutilización
_GLOBAL_MOBILE_CLIENT = MobileClient()

def get_mobile_client() -> MobileClient:
    return _GLOBAL_MOBILE_CLIENT

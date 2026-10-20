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
import threading
from pathlib import Path

from monitor_v3.config.constants import (
    SOFASCORE_API_BASE,
    SOFASCORE_PROXY_URL,
    SOFASCORE_TLS_CLIENT_ID,
    FETCH_TIMEOUT_SECS,
    FETCH_MIN_SPACING_SECS,
    MAX_CONCURRENT_FETCHES,
    UTC_OFFSET_HOURS,
)
from monitor_v3.core.token_manager import TokenPool, get_token_pool
from monitor_v3.utils.helpers import build_mobile_headers
from monitor_v3.utils.logger import log_error, log_warning

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

# Sesión TLS/HTTP2 thread-local con huella OkHttp Android (tls_client).
_TLS_LOCAL = threading.local()


def _frac_dec(frac) -> float | None:
    """Convierte cuota fraccionaria '53/100' a decimal."""
    import re

    if not frac or not isinstance(frac, str):
        return None
    m = re.match(r"\s*(\d+)/(\d+)", frac)
    if not m or int(m.group(2)) <= 0:
        return None
    return round(int(m.group(1)) / int(m.group(2)) + 1.0, 3)


# MarketPeriod de SofaScore -> clave interna para los mercados por periodo.
# "4th quarter" es el mercado del GANADOR DE Q4, que es el que apostamos.
_PERIOD_KEYS = {
    "1st quarter": "q1", "2nd quarter": "q2", "3rd quarter": "q3", "4th quarter": "q4",
    "1st period": "p1", "2nd period": "p2", "3rd period": "p3", "4th period": "p4",
    "1st half": "h1", "2nd half": "h2",
}


def summarize_live_odds(odds_json: dict | None) -> dict:
    """
    Resume los mercados IN-PLAY (isLive=True) de event/{id}/odds/1/all:
    ML en vivo, linea de handicap y total implicitas (donde ambos lados cotizan mas parejo).
    No alimenta m27_v5_3; se persiste en live_snapshots_v3 para entrenar con cuotas en vivo a futuro.
    """
    import re

    out: dict = {}
    spreads: list[tuple[float, float, float]] = []
    totals: list[tuple[float, float, float]] = []
    # Mercado Home/Away POR PERIODO (Q4, Q3...): es el mercado que realmente
    # apostamos (ganador del cuarto), asi que sin el no hay forma de calcular
    # el EV real de la apuesta. Se guardan como q4_ml_home / q4_ml_away.
    period_ml: dict[str, tuple[float, float]] = {}
    for m in (odds_json or {}).get("markets") or []:
        if not m.get("isLive") or m.get("suspended"):
            continue
        grp, per = m.get("marketGroup"), m.get("marketPeriod")
        ch = m.get("choices") or []
        if len(ch) < 2:
            continue
        p0, p1 = _frac_dec(ch[0].get("fractionalValue")), _frac_dec(ch[1].get("fractionalValue"))
        if not p0 or not p1:
            continue
        if grp == "Home/Away" and per == "Match":
            out["ml_home"], out["ml_away"] = p0, p1
        elif grp == "Home/Away" and per:
            key = _PERIOD_KEYS.get(str(per).strip().lower())
            if key:
                period_ml[key] = (p0, p1)
        elif grp == "Point spread":
            # El nombre de la opcion es tipo "Home (-3.5)": hay que BUSCAR el
            # parentesis, no anclar al inicio (con re.match nunca casaba y la
            # linea de handicap nunca se capturaba).
            sm = re.search(r"\(([+-]?\d+(?:\.\d+)?)\)", str(ch[0].get("name", "")))
            if sm:
                spreads.append((float(sm.group(1)), p0, p1))
        elif grp == "Over/Under":
            try:
                totals.append((float(m.get("choiceGroup")), p0, p1))
            except (TypeError, ValueError):
                continue
    for key, (a, b) in period_ml.items():
        out[f"{key}_ml_home"], out[f"{key}_ml_away"] = a, b
    for key, ladder in (("spread", spreads), ("total", totals)):
        if ladder:
            line, a, b = min(ladder, key=lambda r: abs(1.0 / r[1] - 1.0 / r[2]))
            out[f"{key}_line"] = line
            out[f"{key}_n"] = len(ladder)
    return out


def _get_tls_session():
    session = getattr(_TLS_LOCAL, "session", None)
    if session is None:
        import tls_client

        session = tls_client.Session(
            client_identifier=SOFASCORE_TLS_CLIENT_ID,
            random_tls_extension_order=False,
        )
        if SOFASCORE_PROXY_URL:
            session.proxies = {
                "http": SOFASCORE_PROXY_URL,
                "https": SOFASCORE_PROXY_URL,
            }
        _TLS_LOCAL.session = session
    return session


def _sync_request(method: str, url: str, headers: dict, kwargs: dict):
    session = _get_tls_session()
    return session.execute_request(method, url, headers=headers, **kwargs)


class MobileClient:
    """
    Cliente HTTP asíncrono para la API móvil de SofaScore.

    Usa la huella TLS/HTTP2 de OkHttp Android (`tls_client`) + el User-Agent
    firmado de la app (MD5 con ventana de 100s) para pasar el WAF de Fastly.
    Conserva el pool rotativo de JWTs, control de concurrencia y reintentos.
    Ver docs/REVERSE_ENGINEERING_SOFASCORE.md.
    """
    def __init__(self, token_pool: TokenPool | None = None):
        self.token_pool = token_pool or get_token_pool()
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
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def start(self) -> None:
        # tls_client gestiona sus propias conexiones (thread-local); nada que iniciar.
        return None

    async def close(self) -> None:
        return None

    async def request(self, method: str, path: str, retry_count: int = 2, **kwargs):
        """
        Ejecuta una petición HTTP autenticada con rotación de tokens y auto-sanación.

        Si el WAF responde 401/403 (token retado/baneado), el token se elimina del
        pool y se emite automáticamente uno nuevo vía /token/init (sin teléfono),
        reintentando la petición con el token fresco.
        """
        url = path if path.startswith("http") else f"{SOFASCORE_API_BASE}/{path.lstrip('/')}"
        kwargs.setdefault("timeout_seconds", int(FETCH_TIMEOUT_SECS))

        res = None
        last_exc: Exception | None = None
        for attempt in range(retry_count + 1):
            try:
                token = await self.token_pool.get_token()
            except RuntimeError:
                # Pool vacío: intentar regenerar tokens artificialmente.
                if not await self.token_pool.ensure_min_tokens():
                    raise
                token = await self.token_pool.get_token()

            headers = build_mobile_headers(token)

            async with self.sem:
                try:
                    res = await asyncio.to_thread(_sync_request, method, url, headers, kwargs)

                    # Token retado/caducado: purgar, regenerar y reintentar.
                    if res.status_code in (401, 403):
                        await self.token_pool.mark_failure(token, status_code=res.status_code)
                        await self.token_pool.ensure_min_tokens()
                        if attempt < retry_count:
                            await asyncio.sleep(0.5)
                            continue

                    await asyncio.sleep(FETCH_MIN_SPACING_SECS)
                    return res

                except Exception as e:
                    last_exc = e
                    if attempt < retry_count:
                        await asyncio.sleep(0.3)
                        continue
                    raise

        if res is not None:
            return res
        raise last_exc

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
            "game_seconds_played": (ev.get("time") or {}).get("played"),
            "period_length": (ev.get("time") or {}).get("periodLength"),
            "period_count": (ev.get("time") or {}).get("totalPeriodCount"),
            "raw_event": ev,
        }

    async def fetch_team_strength(
        self,
        home_team_id,
        away_team_id,
        home_team_name: str = "",
        away_team_name: str = "",
    ) -> list[dict]:
        """
        Fetches pregameForm (/team/{id}) and performance points (/team/{id}/performance)
        para ambos equipos vía API móvil (sin navegador).
        """
        import json as _json

        rows: list[dict] = []
        for team_id, team_name in ((home_team_id, home_team_name), (away_team_id, away_team_name)):
            if not team_id:
                continue
            row: dict = {"team_id": int(team_id), "team_name": team_name}
            try:
                r = await self.request("GET", f"team/{team_id}")
                if r.status_code == 200:
                    pf = (r.json() or {}).get("pregameForm") or {}
                    row["position"] = pf.get("position")
                    val = pf.get("value", "") or ""
                    if "-" in val:
                        parts = val.split("-")
                        row["wins"] = int(parts[0]) if parts[0].isdigit() else None
                        row["losses"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                    else:
                        row["wins"] = row["losses"] = None
                    row["form"] = _json.dumps(pf.get("form", []))
                r2 = await self.request("GET", f"team/{team_id}/performance")
                if r2.status_code == 200:
                    row["perf_points"] = _json.dumps((r2.json() or {}).get("points") or {})
            except Exception:
                pass
            rows.append(row)
        return rows

    async def fetch_full_match(self, match_id: str, fetch_team_strength: bool = False) -> dict:
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
                    return name, r.json(), 200
                return name, {}, r.status_code
            except Exception:
                return name, {}, -1

        tasks = [_fetch_ep(k, v) for k, v in endpoints.items()]
        raw = await asyncio.gather(*tasks)
        results = {name: data for name, data, _ in raw}
        status_map = {name: status for name, _, status in raw}

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
        parsed["match_id"] = str(match_id)
        if isinstance(parsed.get("match"), dict):
            parsed["match"]["match_id"] = str(match_id)
            ev_obj = event_json.get("event") or {}
            cat_name = ((ev_obj.get("tournament") or {}).get("category") or {}).get("name") or ""
            if cat_name:
                parsed["match"]["country"] = cat_name

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

        # Cuotas pre-partido (isLive=False). Las in-play se resumen aparte para no contaminar el prior de mercado.
        if odds_json:
            parsed["odds"] = _parse_odds(match_id, odds_json)
            parsed["odds_live"] = summarize_live_odds(odds_json)
        else:
            parsed["odds"] = []
            parsed["odds_live"] = {}

        # Team strength (opcional; 2 endpoints por equipo: /team/{id} y /performance)
        parsed["team_strength"] = []
        if fetch_team_strength:
            ev = event_json.get("event", {})
            home_tid = (ev.get("homeTeam") or {}).get("id")
            away_tid = (ev.get("awayTeam") or {}).get("id")
            home_name = (ev.get("homeTeam") or {}).get("name", "")
            away_name = (ev.get("awayTeam") or {}).get("name", "")
            if home_tid and away_tid:
                try:
                    parsed["team_strength"] = await self.fetch_team_strength(
                        home_tid, away_tid, home_name, away_name
                    )
                except Exception:
                    parsed["team_strength"] = []

        # Estado HTTP por endpoint (200/404/-1) para distinguir "dato inexistente"
        # (404) de fallos transitorios al hacer backfill.
        parsed["_endpoint_status"] = status_map

        return parsed


# Singleton para reutilización
_GLOBAL_MOBILE_CLIENT = MobileClient()


def get_mobile_client() -> MobileClient:
    return _GLOBAL_MOBILE_CLIENT

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
import asyncio
from pathlib import Path

from monitor_v3.config.constants import (
    ACTIVE_MODELS,
    Q4_STALE_MAX_TICKS,
    Q4_LATE_BET_MAX_DISADVANTAGE,
    Q4_TOO_LATE_BET_MINUTE,
    Q4_TOO_LATE_HARD_MINUTE,
    NO_BET_CONFIRM_TICKS,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models
from models.common.pbp_utils import infer_minute_from_pbp, infer_regulation_quarter_minutes

_ENGINE_CACHE = {}


_PRIORS_REFRESH_TS = 0.0


def _refresh_priors_if_stale(max_age_secs: float = 1800.0) -> None:
    """Actualiza (en hilo, cada ~30 min) los priors de m27_v5_3 con los partidos terminados recientes."""
    import time as _time

    global _PRIORS_REFRESH_TS
    if _time.time() - _PRIORS_REFRESH_TS < max_age_secs:
        return
    _PRIORS_REFRESH_TS = _time.time()
    try:
        from models.m27_v5_3.features import refresh_state
        refresh_state(str(ROOT / "matches.db"))
    except Exception:
        pass


def load_models_to_cache() -> None:
    """Precarga en memoria O(1) los modelos de m27_v5_3, m27_v5_2, m27_v5_1 y m27_v4 para latencia cero en vivo."""
    try:
        from models.m27_v5_3.predict import _load_production_models as load_v5_3
        load_v5_3()
    except Exception:
        pass
    try:
        from models.m27_v5_2.predict import _load_production_models as load_v5_2
        load_v5_2()
    except Exception:
        pass
    try:
        from models.m27_v5_1.predict import _load_production_models as load_v5_1
        load_v5_1()
    except Exception:
        pass
    try:
        from models.m27_v5.predict import _load_production_models as load_v5
        load_v5()
    except Exception:
        pass
    try:
        from models.m27_v4.features import get_priors_state
        from models.m27_v4.predict import _get_bundle, _get_toxic_map
        get_priors_state()
        _get_toxic_map()
        for arch in ("fiba_men_10m", "fiba_women_10m", "nba_12m"):
            _get_bundle(arch)
    except Exception:
        pass


class GameWatcherState:
    """Mantiene contadores y variables de estado para el ciclo de vida del watcher de un partido."""
    def __init__(self, match_id: str):
        self.match_id = match_id
        self.stale_ticks = 0
        self.last_graph_points_count = 0
        self.last_score_sum = 0
        self.no_bet_confirmations = {}
        self.unavailable_ticks = 0
        self.team_strength_cache: list[dict] | None = None
        self.team_strength_fetched: bool = False
        self.pregame_odds_rows: list[dict] | None = None
        self.last_snapshot_minute: int = -1

    def check_stale_graph(self, current_count: int, score_sum: int) -> bool:
        if current_count > 0 and current_count == self.last_graph_points_count:
            if score_sum != self.last_score_sum:
                self.stale_ticks = 0
                self.last_score_sum = score_sum
            else:
                self.stale_ticks += 1
        else:
            self.stale_ticks = 0
            self.last_graph_points_count = current_count
            self.last_score_sum = score_sum

        return self.stale_ticks >= Q4_STALE_MAX_TICKS


def _merge_pregame_odds(match_id: str, payload: dict, watcher_state: "GameWatcherState") -> None:
    """Agrega al payload las lineas pre-partido (spread/total) guardadas en matches.db, ya que en vivo la API solo
    expone mercados in-play. Se lee una sola vez por partido."""
    if watcher_state.pregame_odds_rows is None:
        rows: list[dict] = []
        try:
            import sqlite3
            conn = sqlite3.connect(str(ROOT / "matches.db"), timeout=5)
            try:
                for r in conn.execute(
                    "SELECT odds_type, market_name, market_period, home_value, away_value, line "
                    "FROM match_odds WHERE match_id = ?", (match_id,)
                ):
                    rows.append({
                        "odds_type": r[0], "market_name": r[1], "market_period": r[2],
                        "home_value": r[3], "away_value": r[4], "line": r[5],
                    })
            finally:
                conn.close()
        except Exception:
            rows = []
        watcher_state.pregame_odds_rows = rows
    have = {(o.get("odds_type"), o.get("market_period")) for o in (payload.get("odds") or []) if o.get("line") is not None}
    extra = [r for r in watcher_state.pregame_odds_rows if r.get("line") is not None and (r["odds_type"], r["market_period"]) not in have]
    if extra:
        payload["odds"] = list(payload.get("odds") or []) + extra


def _record_live_snapshot(match_id: str, minute: float, payload: dict) -> None:
    """Guarda telemetria compacta del minuto (jugadores/faltas/+- y cuotas in-play) en live_snapshots_v3."""
    from monitor_v3.database.repository import save_live_snapshot

    starters = {
        (l.get("team"), str(l.get("player_id"))): int(l.get("is_starter") or 0)
        for l in (payload.get("lineups") or []) if isinstance(l, dict)
    }
    players = [
        [p.get("team"), str(p.get("player_id")), starters.get((p.get("team"), str(p.get("player_id"))), 0),
         p.get("minutes_played"), p.get("points"), p.get("fouls"), p.get("plus_minus"), p.get("sofascore_rating")]
        for p in (payload.get("player_stats") or []) if isinstance(p, dict)
    ]
    team_stats = [
        [s.get("stat_key"), s.get("home_value"), s.get("away_value")]
        for s in (payload.get("team_statistics") or []) if isinstance(s, dict) and s.get("period") == "ALL"
    ]
    score = payload.get("score") or {}
    m_info = payload.get("match") or {}
    save_live_snapshot(
        match_id, minute, int(score.get("home") or 0), int(score.get("away") or 0),
        str(m_info.get("status_description") or ""), payload.get("odds_live"), players, team_stats,
    )


async def evaluate_match_q4(
    match_id: str,
    match_payload: dict,
    watcher_state: GameWatcherState,
    forced_minute: int | None = None,
) -> dict:
    """
    Ejecuta inferencias en paralelo para los modelos activos (m27_v4, v6_2 y m27_v3)
    aplicando validaciones de marcador, minutos límites y señales de apuesta.
    """
    # 0. Enriquecer una sola vez por partido con team_strength en vivo si no está en caché
    if not watcher_state.team_strength_fetched:
        watcher_state.team_strength_fetched = True
        try:
            m_info = match_payload.get("match") or {}
            h_tid = m_info.get("home_team_id")
            a_tid = m_info.get("away_team_id")
            h_name = m_info.get("home_team", "")
            a_name = m_info.get("away_team", "")
            if h_tid and a_tid:
                from monitor_v3.core.mobile_client import get_mobile_client
                watcher_state.team_strength_cache = await get_mobile_client().fetch_team_strength(
                    h_tid, a_tid, h_name, a_name
                )
        except Exception:
            watcher_state.team_strength_cache = []

    if watcher_state.team_strength_cache and not match_payload.get("team_strength"):
        match_payload["team_strength"] = watcher_state.team_strength_cache

    _merge_pregame_odds(match_id, match_payload, watcher_state)

    # Guardar en SQLite previo a la inferencia
    try:
        import match.db as db_mod
        db_path = str(ROOT / "matches.db")
        db_conn = db_mod.get_conn(db_path)
        try:
            db_mod.init_db(db_conn)
            db_mod.save_match(db_conn, match_id, match_payload)
        finally:
            db_conn.close()
    except Exception:
        pass

    # 1. Validar calidad de datos y duración de cuartos (10m FIBA vs 12m NBA)
    q_dur = int(round(infer_regulation_quarter_minutes(match_payload)))
    is_12m = (q_dur == 12)
    q3_end_min = 36 if is_12m else 30
    too_late_bet_min = 37 if is_12m else Q4_TOO_LATE_BET_MINUTE
    too_late_hard_min = 40 if is_12m else Q4_TOO_LATE_HARD_MINUTE

    graph_points = match_payload.get("graph_points", [])
    gp_count = len([p for p in graph_points if int(p.get("minute", 0)) <= too_late_hard_min])

    score = match_payload.get("score", {})
    home_score = int(score.get("home", 0))
    away_score = int(score.get("away", 0))
    score_sum = home_score + away_score

    if watcher_state.check_stale_graph(gp_count, score_sum):
        return {"ok": False, "reason": "graph_stale_timeout"}

    if forced_minute is not None:
        minute_est = forced_minute
    else:
        minute_est = infer_minute_from_pbp(match_payload) or q3_end_min

    # En ligas de 12 minutos (NBA/G-League/PBA), la ventana de evaluación pre-Q4 inicia en el minuto 32 (4:00 rest Q3)
    if is_12m and minute_est < 32:
        return {"ok": False, "reason": "waiting_nba_12m_q3_window"}

    if int(minute_est) != watcher_state.last_snapshot_minute:
        watcher_state.last_snapshot_minute = int(minute_est)
        asyncio.create_task(asyncio.to_thread(_record_live_snapshot, match_id, float(minute_est), match_payload))
    asyncio.create_task(asyncio.to_thread(_refresh_priors_if_stale))

    predictions = {}

    async def run_model_inference(model_version: str):
        try:
            pred = await asyncio.to_thread(
                models.predict,
                match_id=match_id,
                model_version=model_version,
                target="q4",
                match_data=match_payload,
                snapshot_minute=float(minute_est),
            )
            return model_version, pred
        except Exception as e:
            return model_version, models.PredictionResult.unavailable(
                model_version=model_version,
                target="q4",
                reason=str(e),
            )

    tasks = [run_model_inference(m) for m in ACTIVE_MODELS]
    results = await asyncio.gather(*tasks)

    # Verificar si algún modelo emitió señal de apuesta operable en este tick
    any_bet_in_tick = any(
        pred.available and (
            pred.signal in ("BET", "BET_STRONG", "BET_SNIPER", "LEAN")
            or str(pred.signal).startswith("BET")
        )
        for _, pred in results
    )

    m_info = match_payload.get("match") or {}
    status_desc = str(m_info.get("status_description") or "").lower()
    in_q4_now = ("4th quarter" in status_desc) or (minute_est >= q3_end_min)

    for model_version, pred in results:
        if not pred.available:
            predictions[model_version] = {
                "signal": "UNAVAILABLE",
                "reason": pred.reason or "no_data",
            }
            continue

        p_home = pred.p_home_win
        p_away = pred.p_away_win
        predicted_winner = pred.pick.lower()
        confidence = pred.confidence
        bet_signal = str(pred.signal or "NO_BET")

        final_signal = "NO_BET"
        picked_side = predicted_winner.upper()
        if picked_side in ("NONE", ""):
            picked_side = "HOME" if (p_home or 0.5) >= 0.5 else "AWAY"

        is_operable_sig = (
            bet_signal in ("BET", "BET_STRONG", "BET_SNIPER", "LEAN")
            or (bet_signal.startswith("BET") and "NO_BET" not in bet_signal)
        )

        if is_operable_sig:
            # Minuto <= límite ordinario: Apuesta ordinaria
            if minute_est <= too_late_bet_min:
                final_signal = f"BET_{picked_side}"
            # Ventana tardía con filtro de marcador
            elif minute_est < too_late_hard_min:
                picked_deficit = (away_score - home_score) if picked_side == "HOME" else (home_score - away_score)
                if picked_deficit > Q4_LATE_BET_MAX_DISADVANTAGE:
                    final_signal = "NO_BET"
                else:
                    final_signal = f"BET_{picked_side}_LATE"
            else:
                final_signal = "NO_BET"
        else:
            final_signal = "NO_BET"

        if final_signal == "NO_BET":
            # Si aún estamos en minutos previos al cierre del Q3 (ej. m27..m29) y m27_v5/m27_v4 (cuyo óptimo
            # es al cierre de Q3, m30/m36) aún no han disparado apuesta, seguimos monitoreando hasta el
            # cierre del Q3 por si el partido entra en zona verde.
            if (model_version in ("m27_v5_3", "m27_v5", "m27_v4") or not any_bet_in_tick) and not in_q4_now:
                predictions[model_version] = {
                    "signal": "UNAVAILABLE",
                    "reason": "monitoring_q3_window_for_green",
                }
                continue

            watcher_state.no_bet_confirmations[model_version] = watcher_state.no_bet_confirmations.get(model_version, 0) + 1
            if not any_bet_in_tick and watcher_state.no_bet_confirmations[model_version] < NO_BET_CONFIRM_TICKS:
                predictions[model_version] = {
                    "signal": "UNAVAILABLE",
                    "reason": "awaiting_no_bet_confirmations",
                }
                continue
        else:
            watcher_state.no_bet_confirmations[model_version] = 0

        predictions[model_version] = {
            "signal": final_signal,
            "raw_model_signal": bet_signal,
            "pick": picked_side,
            "confidence": confidence,
            "p_home_win": p_home,
            "p_away_win": p_away,
            "inference_minute": minute_est,
            "graph_points_count": gp_count,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "raw_payload": match_payload,
            "inference_json": pred.to_dict(),
        }

    return {"ok": True, "predictions": predictions}


# Cargar en el arranque
load_models_to_cache()

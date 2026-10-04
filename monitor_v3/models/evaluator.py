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
import joblib
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

_ENGINE_CACHE = {}

def load_models_to_cache() -> None:
    """Los modelos de models/ gestionan su propio caché interno de inferencia."""
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

async def evaluate_match_q4(match_id: str, match_payload: dict, watcher_state: GameWatcherState, forced_minute: int | None = None) -> dict:
    """
    Ejecuta inferencias en paralelo para los modelos activos (v6_2 y m27_v3)
    aplicando validaciones de marcador, minutos límites y señales de apuesta.
    """
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

    # 1. Validar calidad de datos
    graph_points = match_payload.get("graph_points", [])
    gp_count = len([p for p in graph_points if int(p.get("minute", 0)) <= 36])
    
    score = match_payload.get("score", {})
    home_score = int(score.get("home", 0))
    away_score = int(score.get("away", 0))
    score_sum = home_score + away_score
    
    if watcher_state.check_stale_graph(gp_count, score_sum):
        return {"ok": False, "reason": "graph_stale_timeout"}
        
    if forced_minute is not None:
        minute_est = forced_minute
    else:
        from models.common.pbp_utils import infer_minute_from_pbp
        minute_est = infer_minute_from_pbp(match_payload) or 36
    
    predictions = {}
    
    async def run_model_inference(model_version: str):
        try:
            pred = await asyncio.to_thread(
                models.predict,
                match_id=match_id,
                model_version=model_version,
                target="q4",
                match_data=match_payload,
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
        bet_signal = pred.signal
        
        final_signal = "NO_BET"
        picked_side = predicted_winner.upper()
        
        if bet_signal in ("BET", "LEAN"):
            # Minuto <= 33: Apuesta ordinaria
            if minute_est <= Q4_TOO_LATE_BET_MINUTE:
                final_signal = f"BET_{picked_side}"
            # Minuto 34..36: Apuesta tardía con filtro de marcador
            elif minute_est < Q4_TOO_LATE_HARD_MINUTE:
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
            watcher_state.no_bet_confirmations[model_version] = watcher_state.no_bet_confirmations.get(model_version, 0) + 1
            if watcher_state.no_bet_confirmations[model_version] < NO_BET_CONFIRM_TICKS:
                predictions[model_version] = {
                    "signal": "UNAVAILABLE",
                    "reason": "awaiting_no_bet_confirmations"
                }
                continue
        else:
            watcher_state.no_bet_confirmations[model_version] = 0
            
        predictions[model_version] = {
            "signal": final_signal,
            "pick": picked_side,
            "confidence": confidence,
            "p_home_win": p_home,
            "p_away_win": p_away,
            "inference_minute": minute_est,
            "graph_points_count": gp_count,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "raw_payload": match_payload,
            "inference_json": pred.to_dict()
        }
        
    return {"ok": True, "predictions": predictions}

# Cargar en el arranque
load_models_to_cache()

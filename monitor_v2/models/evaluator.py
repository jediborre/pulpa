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
import joblib
from pathlib import Path
from monitor_v2.config.constants import (
    ACTIVE_MODELS,
    Q4_STALE_MAX_TICKS,
    Q4_LATE_BET_MAX_DISADVANTAGE,
    Q4_TOO_LATE_BET_MINUTE,
    Q4_TOO_LATE_HARD_MINUTE,
    NO_BET_CONFIRM_TICKS
)

# Cargar el path del workspace para importar los módulos de match
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importar el paquete unificado de modelos
import models

# Singleton caché de motores analíticos
_ENGINE_CACHE = {}

def get_engine_cache():
    """Retorna el caché singleton de modelos."""
    return _ENGINE_CACHE

def load_models_to_cache() -> None:
    """
    Carga en caché los artefactos de v6_2 y m27_v3 para cumplir con el patrón Singleton.
    """
    global _ENGINE_CACHE
    try:
        # Cargar metadatos del campeón v6_2
        champ_path = ROOT / "models" / "v6_2" / "model_outputs" / "q4_champion.joblib"
        if champ_path.exists():
            _ENGINE_CACHE["v6_2"] = joblib.load(champ_path)
            
        # Cargar artefactos de m27_v3
        m27_dir = ROOT / "models" / "m27_v3" / "model_outputs"
        if m27_dir.exists():
            _ENGINE_CACHE["m27_v3"] = {
                "xgb": joblib.load(m27_dir / "m27_v3_xgb.joblib"),
                "hist": joblib.load(m27_dir / "m27_v3_histgb.joblib")
            }
    except Exception:
        pass


class GameWatcherState:
    """
    Mantiene contadores y variables de estado para el ciclo de vida del watcher de un partido.
    """
    def __init__(self, match_id: str):
        self.match_id = match_id
        self.stale_ticks = 0
        self.last_graph_points_count = 0
        self.last_score_sum = 0
        self.no_bet_confirmations = {} # model -> count
        self.unavailable_ticks = 0
        
    def check_stale_graph(self, current_count: int, score_sum: int) -> bool:
        """
        Retorna True si la gráfica y el marcador se mantienen estancados por Q4_STALE_MAX_TICKS.
        Si el score cambia, se asume que el partido sigue activo en vivo.
        """
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
            
        # Tolerancia alta de 15 ticks en lugar de Q4_STALE_MAX_TICKS para evitar falsos desiertos
        # por sequías anotadoras en ligas con PBP diferido/en vivo limitado
        return self.stale_ticks >= 15


async def evaluate_match_q4(match_id: str, match_payload: dict, watcher_state: GameWatcherState, forced_minute: int | None = None) -> dict:
    """
    Orquesta la ejecución paralela en hilos de los modelos v6_2 y m27_v3
    dentro de la ventana de Q4, aplicando validación de calidad de datos,
    filtros de marcador en apuestas tardías y buffers de confirmación.
    """
    # Guardar en la base de datos SQLite antes de la inferencia para que run_inference pueda
    # cargar los datos en vivo más recientes y actualizados.
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
    
    # Aborto por estancamiento de gráfica (solo si el score tampoco se mueve)
    if watcher_state.check_stale_graph(gp_count, score_sum):
        return {"ok": False, "reason": "graph_stale_timeout"}
        
    # Estimar minuto a partir de PBP o Snapshot
    if forced_minute is not None:
        minute_est = forced_minute
    else:
        from models.common.pbp_utils import infer_minute_from_pbp
        minute_est = infer_minute_from_pbp(match_payload) or 36
    
    # Ejecutar inferencias en paralelo utilizando asyncio.to_thread
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

    # Lanzar tareas concurrentes para cada modelo activo
    tasks = [run_model_inference(m) for m in ACTIVE_MODELS]
    results = await asyncio.gather(*tasks)
    
    for model_version, pred in results:
        if not pred.available:
            predictions[model_version] = {
                "signal": "UNAVAILABLE",
                "reason": pred.reason or "no_data",
            }
            continue
            
        # Extraer probabilidades y pick arrojado por el modelo
        p_home = pred.p_home_win
        p_away = pred.p_away_win
        predicted_winner = pred.pick.lower() # 'home' o 'away'
        confidence = pred.confidence
        bet_signal = pred.signal
        
        # Aplicar Reglas de Guardia Temporal y Filtros de Marcador en Q4
        final_signal = "NO_BET"
        picked_side = predicted_winner.upper() # 'HOME' o 'AWAY'
        
        if bet_signal in ("BET", "LEAN"):
            # A) Si el minuto es <= 33: APUESTA ORDINARIA Q4
            if minute_est <= Q4_TOO_LATE_BET_MINUTE:
                final_signal = f"BET_{picked_side}"
            # B) Si el minuto está en el rango [34..36]: Apuesta Tardía con filtro de marcador
            elif minute_est < Q4_TOO_LATE_HARD_MINUTE:
                # Calcular la diferencia de puntos del pick seleccionado
                picked_deficit = 0
                if picked_side == "HOME":
                    picked_deficit = away_score - home_score
                else:
                    picked_deficit = home_score - away_score
                    
                # Si el equipo va perdiendo por más de Q4_LATE_BET_MAX_DISADVANTAGE, se descarta
                if picked_deficit > Q4_LATE_BET_MAX_DISADVANTAGE:
                    final_signal = "NO_BET"
                else:
                    final_signal = f"BET_{picked_side}_LATE"
            # C) Bloqueo total si el minuto es >= 36
            else:
                final_signal = "NO_BET"
        else:
            final_signal = "NO_BET"
            
        # Buffer de confirmaciones consecutivas para asentar NO_BET
        if final_signal == "NO_BET":
            watcher_state.no_bet_confirmations[model_version] = watcher_state.no_bet_confirmations.get(model_version, 0) + 1
            if watcher_state.no_bet_confirmations[model_version] < NO_BET_CONFIRM_TICKS:
                # Mantener estado operable transitoriamente o reportar neutral
                predictions[model_version] = {
                    "signal": "UNAVAILABLE",
                    "reason": "awaiting_no_bet_confirmations"
                }
                continue
        else:
            # Resetear confirmador ante señal operable activa
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

# Inicializar cache en el arranque
load_models_to_cache()

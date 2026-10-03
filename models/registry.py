"""Registro y despachador dinámico universal de modelos.

Provee un punto de entrada desacoplado para ejecutar inferencias con cualquier versión
de modelo soportada (m27_v3, v6_2, v6_3, etc.) y persistir resultados automáticamente.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from models.common.data_loader import get_db_connection, load_match
from models.common.persistence import save_eval_results
from models.common.schema import PredictionResult

# Catálogo oficial de modelos soportados con metadatos
REGISTERED_MODELS: dict[str, dict[str, Any]] = {
    "m27_v3": {
        "name": "M27 V3 Champion Q4",
        "targets": ["q4"],
        "status": "production_champion",
        "description": "Snapshot min 27 + H2H + Ensamble XGBoost/HistGB + Calibrador Isotónico",
    },
    "v6_2": {
        "name": "V6.2 Champion Q3/Q4",
        "targets": ["q3", "q4"],
        "status": "production_champion",
        "description": "Ensamble XGBoost/HistGB con poda algorítmica de ligas (league pruning)",
    },
    "v6_3": {
        "name": "V6.3 Early Snapshot Q4",
        "targets": ["q4"],
        "status": "candidate",
        "description": "Snapshots tempranos min 27/30 con lista negra manual de ligas",
    },
    "v6": {
        "name": "V6 Baseline Q3/Q4",
        "targets": ["q3", "q4"],
        "status": "historical",
        "description": "Modelo base de presión y momentum",
    },
    "v2": {
        "name": "V2 Baseline Q3/Q4",
        "targets": ["q3", "q4"],
        "status": "historical",
        "description": "Modelo clásico con compuertas logísticas y random forest",
    },
}


def get_available_models() -> list[str]:
    """Retorna la lista de identificadores de modelos registrados."""
    return list(REGISTERED_MODELS.keys())


def predict(
    match_id: str,
    model_version: str,
    target: str = "q4",
    match_data: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> PredictionResult:
    """Ejecuta inferencia con el modelo especificado retornando un PredictionResult canónico."""
    v = model_version.lower().strip()
    close_conn = False

    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        if match_data is None:
            match_data = load_match(conn, match_id)

        if not match_data:
            return PredictionResult.unavailable(
                model_version=v,
                target=target,
                reason=f"Partido {match_id} no encontrado en matches.db",
            )

        if v == "m27_v3":
            from models.m27_v3.predict import predict as m27_predict
            return m27_predict(match_id=match_id, target=target, match_data=match_data, conn=conn)

        elif v == "v6_2":
            from models.v6_2.predict import predict as v62_predict
            return v62_predict(match_id=match_id, target=target, match_data=match_data, conn=conn)

        elif v == "v6_3":
            from models.v6_3.predict import predict as v63_predict
            return v63_predict(match_id=match_id, target=target, match_data=match_data, conn=conn)

        else:
            # Delegar a adaptador histórico si está disponible
            try:
                import match.training.infer_match as infer_legacy
                raw = infer_legacy.run_inference(
                    match_id=match_id,
                    metric="f1",
                    fetch_missing=False,
                    force_version=v,
                    refresh=False,
                    target_only=target,
                )
                pred_dict = raw.get("predictions", {}).get(target, {})
                if not pred_dict.get("available"):
                    return PredictionResult.unavailable(
                        model_version=v,
                        target=target,
                        reason=pred_dict.get("reason", "unavailable_in_legacy"),
                    )

                p_home = float(pred_dict.get("p_home_win", 0.5))
                p_away = float(pred_dict.get("p_away_win", 0.5))
                winner = str(pred_dict.get("predicted_winner", "home")).upper()
                conf = float(pred_dict.get("confidence", 0.0))
                sig = str(pred_dict.get("bet_signal", "NO_BET"))

                return PredictionResult(
                    available=True,
                    model_version=v,
                    target=target,
                    signal=sig,
                    pick=winner,
                    confidence=conf,
                    p_home_win=p_home,
                    p_away_win=p_away,
                    reason=f"Legacy {v} prediction",
                    raw_json=pred_dict,
                )
            except Exception as e:
                return PredictionResult.unavailable(
                    model_version=v,
                    target=target,
                    reason=f"Error en inferencia de modelo {v}: {e}",
                )
    finally:
        if close_conn:
            conn.close()


def predict_all(
    match_id: str,
    models: list[str] | None = None,
    target: str = "q4",
    match_data: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
    save_to_db: bool = False,
) -> dict[str, PredictionResult]:
    """Ejecuta inferencia multi-modelo y opcionalmente persiste en eval_match_results_v2."""
    if models is None:
        models = ["m27_v3", "v6_2"]

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        if match_data is None:
            match_data = load_match(conn, match_id)

        results: dict[str, PredictionResult] = {}
        for m in models:
            results[m] = predict(
                match_id=match_id,
                model_version=m,
                target=target,
                match_data=match_data,
                conn=conn,
            )

        if save_to_db and results:
            save_eval_results(conn, match_id, results)

        return results
    finally:
        if close_conn:
            conn.close()

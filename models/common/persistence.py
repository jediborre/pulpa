"""Capa de persistencia estandarizada para resultados de inferencia y auditoría.

Gestiona las escrituras y lecturas en:
- eval_match_results_v2: tabla matricial con columnas dinámicas por modelo.
- bet_monitor_log_v2: log transaccional de auditoría por modelo y cuarto.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from models.common.schema import PredictionResult


def ensure_eval_columns(conn: sqlite3.Connection, model_versions: list[str]) -> None:
    """Asegura mediante introspección que existan las columnas en eval_match_results_v2."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(eval_match_results_v2);")
    existing_cols = {row["name"] for row in cursor.fetchall()}

    for version in model_versions:
        clean_v = version.replace("-", "_").lower()
        for suffix in ["_signal", "_pick", "_confidence", "_outcome"]:
            col_name = f"q4{suffix}__{clean_v}"
            if col_name not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE eval_match_results_v2 ADD COLUMN {col_name} TEXT;")
                    existing_cols.add(col_name)
                except sqlite3.OperationalError:
                    pass


def save_eval_results(
    conn: sqlite3.Connection,
    match_id: str,
    predictions: dict[str, PredictionResult],
    available: int = 1,
) -> None:
    """Guarda o actualiza las predicciones de uno o varios modelos en eval_match_results_v2."""
    if not predictions:
        return

    model_versions = list(predictions.keys())
    ensure_eval_columns(conn, model_versions)

    # Cargar fila previa para no pisar predicciones existentes de otros modelos
    old_row = conn.execute(
        "SELECT * FROM eval_match_results_v2 WHERE match_id = ?", (str(match_id),)
    ).fetchone()

    data_map: dict[str, Any] = dict(old_row) if old_row else {"match_id": str(match_id), "available": available}
    data_map["match_id"] = str(match_id)
    data_map["available"] = available

    for version, pred in predictions.items():
        clean_v = version.replace("-", "_").lower()
        prefix = f"{pred.target.lower()}"
        data_map[f"{prefix}_signal__{clean_v}"] = pred.signal
        data_map[f"{prefix}_pick__{clean_v}"] = pred.pick
        data_map[f"{prefix}_confidence__{clean_v}"] = float(pred.confidence)
        data_map[f"{prefix}_outcome__{clean_v}"] = "pending"

    cols = list(data_map.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    values = [data_map[c] for c in cols]

    conn.execute(
        f"INSERT OR REPLACE INTO eval_match_results_v2 ({col_str}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def log_bet_decision(
    conn: sqlite3.Connection,
    match_id: str,
    pred: PredictionResult,
    inference_minute: int | None = None,
    graph_points_count: int | None = None,
    actual_home_score: int | None = None,
    actual_away_score: int | None = None,
    result: str = "pending",
) -> int:
    """Inserta un registro de auditoría en bet_monitor_log_v2."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_q = 4 if pred.target.lower() == "q4" else 3
    inf_json = json.dumps(pred.to_dict())

    cursor = conn.execute(
        """
        INSERT INTO bet_monitor_log_v2 (
            match_id, model_version, target_quarter, inference_minute,
            graph_points_count, raw_json, signal_type, picked_side,
            confidence, actual_home_score, actual_away_score, result,
            created_at, inference_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(match_id),
            pred.model_version,
            target_q,
            inference_minute,
            graph_points_count,
            inf_json,
            pred.signal,
            pred.pick,
            pred.confidence,
            actual_home_score,
            actual_away_score,
            result,
            now_str,
            inf_json,
        ),
    )
    conn.commit()
    return cursor.lastrowid or 0

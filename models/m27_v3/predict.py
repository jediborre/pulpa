"""Motor de inferencia y predicción para el modelo campeón Q4 m27_v3.

Carga los artefactos de model_outputs/ (XGBoost, HistGradientBoosting, Vectorizer,
Isotonic Calibrator) y genera un PredictionResult canónico.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from models.common.data_loader import get_db_connection, load_match
from models.common.schema import PredictionResult
from models.m27_v3.features import extract_features

MODEL_DIR = Path(__file__).resolve().parent / "model_outputs"

_CACHE: dict[str, Any] | None = None


def load_artifacts() -> dict[str, Any]:
    """Carga y almacena en caché singleton los artefactos entrenados de m27_v3."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    vec_path = MODEL_DIR / "m27_v3_vectorizer.joblib"
    xgb_path = MODEL_DIR / "m27_v3_xgb.joblib"
    hist_path = MODEL_DIR / "m27_v3_histgb.joblib"
    cal_path = MODEL_DIR / "m27_v3_calibrator.joblib"

    if not (vec_path.exists() and xgb_path.exists() and hist_path.exists() and cal_path.exists()):
        raise FileNotFoundError(
            f"Artefactos de modelo m27_v3 incompletos en {MODEL_DIR}. Verifique la carpeta model_outputs."
        )

    vec = joblib.load(vec_path)
    xgb_m = joblib.load(xgb_path)
    hist_m = joblib.load(hist_path)
    cal = joblib.load(cal_path)

    # Pesos balanceados del ensamble validados en validación histórica (~50/50)
    w_xgb = 0.50004
    w_hist = 0.49996

    _CACHE = {
        "vec": vec,
        "xgb": xgb_m,
        "hist": hist_m,
        "cal": cal,
        "w_xgb": w_xgb,
        "w_hist": w_hist,
    }
    return _CACHE


def predict(
    match_id: str,
    target: str = "q4",
    match_data: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
    threshold_bet: float = 0.50,
) -> PredictionResult:
    """Ejecuta inferencia con m27_v3 retornando un PredictionResult canónico."""
    if target.lower() != "q4":
        return PredictionResult.unavailable(
            model_version="m27_v3",
            target=target,
            reason="m27_v3 solo está diseñado y calibrado para target='q4'",
        )

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        if match_data is None:
            match_data = load_match(conn, match_id)

        if not match_data:
            return PredictionResult.unavailable(
                model_version="m27_v3",
                target=target,
                reason=f"No se encontraron datos para el match {match_id}",
            )

        match_meta = match_data.get("match", {}) or {}
        ht = match_meta.get("home_team", "")
        at = match_meta.get("away_team", "")
        if not ht or not at:
            return PredictionResult.unavailable(
                model_version="m27_v3",
                target=target,
                reason="Equipos no definidos en metadatos del partido",
            )

        features = extract_features(match_data, conn)
        artifacts = load_artifacts()

        x_mat = artifacts["vec"].transform([features])
        xgb_p = artifacts["xgb"].predict_proba(x_mat)[0, 1]
        hist_p = artifacts["hist"].predict_proba(
            x_mat.toarray() if hasattr(x_mat, "toarray") else x_mat
        )[0, 1]

        ens_p = float(artifacts["w_xgb"] * xgb_p + artifacts["w_hist"] * hist_p)
        cal_p = float(np.clip(artifacts["cal"].transform(np.clip([[ens_p]], 0.0, 1.0))[0], 0.0, 1.0))

        p_home = round(cal_p, 6)
        p_away = round(1.0 - cal_p, 6)
        winner = "HOME" if cal_p >= 0.5 else "AWAY"
        confidence = round(max(cal_p, 1.0 - cal_p), 6)

        # Determinar señal
        signal = "BET" if confidence >= threshold_bet else "NO_BET"

        return PredictionResult(
            available=True,
            model_version="m27_v3",
            target="q4",
            signal=signal,
            pick=winner,
            confidence=confidence,
            p_home_win=p_home,
            p_away_win=p_away,
            snapshot_minute=27,
            reason=f"P(Home)={p_home:.4f}, Conf={confidence:.4f}",
            raw_json={
                "xgb_p": round(float(xgb_p), 6),
                "hist_p": round(float(hist_p), 6),
                "ensemble_p": round(ens_p, 6),
                "calibrated_p": p_home,
                "features_used": len(features),
            },
        )
    finally:
        if close_conn:
            conn.close()

"""
Inferencia en tiempo real e histórica para el modelo M27_V5_2 (Lean 27 Features + Market Odds & Standings).
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any
import numpy as np

from models.common.data_loader import get_db_connection, load_match
from models.common.schema import PredictionResult
from models.m27_v5_2.features import (
    FEATURE_NAMES_27, MODEL_DIR, extract_features_v5_2, resolve_archetype_v5_2,
)

_MODELS_CACHE: dict[str, Any] | None = None
_CALIBRATION_CACHE: dict[str, Any] | None = None


def _get_league_calibration() -> dict[str, Any]:
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is None:
        calib_path = MODEL_DIR / "league_calibration.json"
        if calib_path.exists():
            try:
                _CALIBRATION_CACHE = json.loads(calib_path.read_text(encoding="utf-8"))
            except Exception:
                _CALIBRATION_CACHE = {}
        else:
            _CALIBRATION_CACHE = {}
    return _CALIBRATION_CACHE


def _load_production_models() -> dict[str, Any]:
    global _MODELS_CACHE
    if _MODELS_CACHE is None:
        try:
            with open(MODEL_DIR / "xgb_model.pkl", "rb") as f:
                xgb_model = pickle.load(f)
            with open(MODEL_DIR / "hgb_model.pkl", "rb") as f:
                hgb_model = pickle.load(f)
            with open(MODEL_DIR / "lr_model.pkl", "rb") as f:
                lr_model = pickle.load(f)
            with open(MODEL_DIR / "reg_model.pkl", "rb") as f:
                reg_model = pickle.load(f)
            with open(MODEL_DIR / "calibrator.pkl", "rb") as f:
                calibrator = pickle.load(f)

            _MODELS_CACHE = {
                "xgb": xgb_model,
                "hgb": hgb_model,
                "lr": lr_model,
                "reg": reg_model,
                "calibrator": calibrator,
            }
        except Exception:
            _MODELS_CACHE = {}
    return _MODELS_CACHE


def predict(
    match_id: str,
    target: str = "q4",
    match_data: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
    snapshot_minute: float | None = None,
) -> PredictionResult:
    """Ejecuta inferencia con el modelo M27_V5_2 retornando un PredictionResult canónico."""
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        if match_data is None:
            match_data = load_match(conn, match_id)
        if not match_data:
            return PredictionResult.unavailable(
                model_version="m27_v5_2",
                target=target,
                reason=f"Partido {match_id} no encontrado en matches.db",
            )

        models = _load_production_models()
        if not models:
            return PredictionResult.unavailable(
                model_version="m27_v5_2",
                target=target,
                reason="Artefactos de modelos no encontrados en models/m27_v5_2/model_outputs/",
            )

        snap = snapshot_minute if snapshot_minute is not None else 27.0
        # Ventana de entrenamiento: v5_2 se entreno con el marcador de Q3
        # cerrado (min 30 en ligas de 10 min, 36 en las de 12). `score_upto`
        # acumula los puntos hasta el minuto pedido, asi que mas alla del
        # cierre de Q3 el vector MEZCLA puntos de Q4 con los de Q1-Q3 y deja
        # de ser el vector entrenado. Se rechaza en vez de mezclar.
        _, q_dur_v52, _, _ = resolve_archetype_v5_2(match_data)
        max_valid_minute = 36.0 if int(q_dur_v52) == 12 else 30.0
        if float(snap) > max_valid_minute + 1e-6:
            return PredictionResult.unavailable(
                model_version="m27_v5_2",
                target=target,
                reason=f"Fuera de la ventana de entrenamiento (min {float(snap):.1f} > {max_valid_minute}); "
                       f"v5_2 solo infiere con el marcador de Q3 cerrado",
            )
        x_vec = extract_features_v5_2(match_data, target_minute=snap)
        X = x_vec.reshape(1, -1)

        xgb_m = models["xgb"]
        hgb_m = models["hgb"]
        lr_m = models["lr"]
        reg_m = models["reg"]
        calibrator = models["calibrator"]

        p_xgb = float(xgb_m.predict_proba(X)[0, 1])
        p_hgb = float(hgb_m.predict_proba(X)[0, 1])
        p_lr = float(lr_m.predict_proba(X)[0, 1])
        p_raw = 0.40 * p_xgb + 0.35 * p_hgb + 0.25 * p_lr

        p_cal = float(calibrator.predict(np.array([p_raw]))[0])
        p_cal = float(np.clip(p_cal, 0.01, 0.99))
        exp_margin = float(reg_m.predict(X)[0])

        m_info = match_data.get("match") or {}
        league = str(m_info.get("league") or match_data.get("league") or "").strip()

        # Umbral calibrado de señal verde de alta seguridad (P >= 0.75 -> Win Rate 78.13%, ROI @ 1.40: +9.39%)
        min_conf = 0.75
        min_margin = 1.8

        is_green = False
        signal = "HOLD"
        confidence = 0.50

        if p_cal >= min_conf and p_xgb > 0.5 and exp_margin >= min_margin:
            is_green = True
            signal = "HOME"
            confidence = p_cal
        elif p_cal <= (1.0 - min_conf) and p_xgb < 0.5 and exp_margin <= -min_margin:
            is_green = True
            signal = "AWAY"
            confidence = 1.0 - p_cal
        else:
            confidence = max(p_cal, 1.0 - p_cal)

        pred_winner = "HOME" if p_cal >= 0.5 else "AWAY"

        metadata = {
            "p_cal": round(p_cal, 4),
            "p_raw": round(p_raw, 4),
            "p_xgb": round(p_xgb, 4),
            "p_hgb": round(p_hgb, 4),
            "p_lr": round(p_lr, 4),
            "expected_q4_margin": round(exp_margin, 2),
            "is_green": is_green,
            "min_conf_threshold": min_conf,
            "min_margin_threshold": min_margin,
            "snapshot_minute": snap,
            "league": league,
            "features_count": len(x_vec),
        }

        return PredictionResult(
            available=True,
            model_version="m27_v5_2",
            target=target,
            signal="BET" if is_green else "PASS",
            pick=pred_winner,
            confidence=round(confidence, 4),
            p_home_win=round(p_cal, 4),
            p_away_win=round(1.0 - p_cal, 4),
            snapshot_minute=int(snap),
            raw_json=metadata,
        )

    except Exception as exc:
        return PredictionResult.unavailable(
            model_version="m27_v5_2",
            target=target,
            reason=f"Error en inferencia: {exc}",
        )
    finally:
        if close_conn and conn:
            conn.close()


def predict_batch(match_ids: list[str], target: str = "q4") -> dict[str, PredictionResult]:
    """Inferencia por lotes optimizada reutilizando conexión a base de datos."""
    conn = get_db_connection()
    results = {}
    try:
        for mid in match_ids:
            results[mid] = predict(mid, target=target, conn=conn)
    finally:
        conn.close()
    return results


def predict_with_metadata(match_data: dict[str, Any], snapshot_minute: float = 27.0) -> dict[str, Any]:
    """Inferencia directa desde diccionario match_data (diseñada para monitor_v3 live loop)."""
    res = predict("live", match_data=match_data, snapshot_minute=snapshot_minute)
    return {
        "model_version": res.model_version,
        "available": res.available,
        "pick": res.pick,
        "confidence": res.confidence,
        "p_home_win": res.p_home_win,
        "p_away_win": res.p_away_win,
        "signal": res.signal,
        "raw_json": res.raw_json,
    }

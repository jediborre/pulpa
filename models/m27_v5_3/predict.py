"""
Inferencia de m27_v5_3 (live y offline). Usa el extractor unico `features.extract_features_v5_3`.

Pipeline: vector -> (vista original + vista espejo) -> ensamble XGB/HGB/LR -> promedio simetrico -> calibrador isotonico
fuera de muestra -> reglas de apuesta de `model_config.json` (umbral P, concordancia con el margen esperado, liga no toxica).
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import warnings
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", message="Skipping features without any observed values")

from models.common.data_loader import get_db_connection, load_match
from models.common.schema import PredictionResult
from models.m27_v5_3.features import (
    FEATURE_NAMES, IDX, MODEL_DIR, SNAPSHOT_MAX, SNAPSHOT_MIN, ctx_from_payload, get_state,
    is_blocked_league, mirror, snapshot_features,
)

MODEL_VERSION = "m27_v5_3"
MIN_VALID_MINUTE = SNAPSHOT_MIN
MAX_VALID_MINUTE = SNAPSHOT_MAX
_MODELS: dict[str, Any] | None = None
_DEFAULT_CFG = {
    "min_conf": 0.70, "min_margin": 0.0, "weights": {"xgb": 0.40, "hgb": 0.35, "lr": 0.25},
    "p_clip": [0.03, 0.97], "block_toxic_leagues": True,
}


def _load_production_models() -> dict[str, Any]:
    """Carga (una vez) pesos, calibrador, configuracion y estado de priors."""
    global _MODELS
    if _MODELS is None:
        try:
            out: dict[str, Any] = {}
            for key, fname in (("xgb", "xgb_model.pkl"), ("hgb", "hgb_model.pkl"), ("lr", "lr_model.pkl"),
                               ("reg", "reg_model.pkl"), ("calibrator", "calibrator.pkl")):
                with open(MODEL_DIR / fname, "rb") as f:
                    out[key] = pickle.load(f)
            cfg_path = MODEL_DIR / "model_config.json"
            try:
                out["xgb"].set_params(n_jobs=1)  # un vector: los hilos de XGB solo agregan latencia
            except Exception:
                pass
            out["cfg"] = {**_DEFAULT_CFG, **(json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {})}
            get_state()
            _MODELS = out
        except Exception:
            _MODELS = {}
    return _MODELS


def _ensemble(models: dict[str, Any], X: np.ndarray) -> np.ndarray:
    w = models["cfg"]["weights"]
    return (w["xgb"] * models["xgb"].predict_proba(X)[:, 1]
            + w["hgb"] * models["hgb"].predict_proba(X)[:, 1]
            + w["lr"] * models["lr"].predict_proba(X)[:, 1])


def predict_vector(models: dict[str, Any], x: np.ndarray) -> dict[str, float]:
    """Probabilidad calibrada y margen simetricos para un vector (N,). Ambas vistas en una sola llamada por modelo."""
    X2 = np.vstack([x.reshape(1, -1), mirror(x.reshape(1, -1))])
    pe = _ensemble(models, X2)
    p_raw = 0.5 * (pe[0] + 1.0 - pe[1])
    lo, hi = models["cfg"]["p_clip"]
    p_cal = float(np.clip(models["calibrator"].predict(np.array([p_raw]))[0], lo, hi))
    mg = models["reg"].predict(X2)
    return {"p_raw": float(p_raw), "p_cal": p_cal, "margin": 0.5 * float(mg[0] - mg[1])}


def predict(
    match_id: str,
    target: str = "q4",
    match_data: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
    snapshot_minute: float | None = None,
) -> PredictionResult:
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True
    try:
        if match_data is None:
            match_data = load_match(conn, match_id)
        if not match_data:
            return PredictionResult.unavailable(MODEL_VERSION, target, f"Partido {match_id} no encontrado en matches.db")

        models = _load_production_models()
        if not models:
            return PredictionResult.unavailable(MODEL_VERSION, target, "Artefactos de m27_v5_3 no encontrados (ejecutar el entrenamiento)")

        state = get_state()
        ctx = ctx_from_payload(match_data, snapshot_minute, state)
        if ctx is None:
            return PredictionResult.unavailable(MODEL_VERSION, target, "Datos insuficientes (cuartos Q1/Q2 o marcador)")
        if int(ctx["info"].get("q_dur", 10)) == 12:
            return PredictionResult.unavailable(MODEL_VERSION, target, "Ligas de cuartos de 12 min no soportadas por m27_v5_3")
        # Ventana de entrenamiento: el modelo solo se entreno con snapshots en
        # [25.5, 30.0]. `ctx["minute"]` viene CLAMPEADO a 30 cuando el partido
        # ya esta en Q4, asi que el minuto real sin clampear es `minute_raw`:
        # a partir del 31 el modelo prediria el ganador de un Q4 ya empezado
        # usando solo el cierre de Q3 (y el monitor si permite apostar hasta
        # el min 33). Se rechaza en vez de extrapolar.
        minute_raw = float(ctx.get("minute_raw", ctx["minute"]))
        if minute_raw > MAX_VALID_MINUTE + 1e-6:
            return PredictionResult.unavailable(
                MODEL_VERSION, target,
                f"Fuera de la ventana de entrenamiento (min {minute_raw:.1f} > {MAX_VALID_MINUTE}); "
                f"m27_v5_3 solo infiere en [{MIN_VALID_MINUTE}, {MAX_VALID_MINUTE}]"
            )
        if float(ctx["minute"]) < MIN_VALID_MINUTE:
            return PredictionResult.unavailable(MODEL_VERSION, target, f"Muy temprano (min {ctx['minute']:.1f} < {MIN_VALID_MINUTE}); entrenado en min 25.5-30")

        x = snapshot_features(ctx, state)
        res = predict_vector(models, x)
        cfg = models["cfg"]
        p_cal, margin = res["p_cal"], res["margin"]
        league = ctx["league"]
        toxic = bool(cfg.get("block_toxic_leagues", True)) and is_blocked_league(league)

        thr, mm = float(cfg["min_conf"]), float(cfg["min_margin"])
        default_pick = "HOME" if p_cal >= 0.5 else "AWAY"
        signal, pick, confidence = "PASS", default_pick, max(p_cal, 1.0 - p_cal)
        if not toxic:
            if p_cal >= thr and margin >= mm:
                signal, pick, confidence = "BET", "HOME", p_cal
            elif p_cal <= 1.0 - thr and margin <= -mm:
                signal, pick, confidence = "BET", "AWAY", 1.0 - p_cal

        feats = {n: (None if np.isnan(v) else round(float(v), 4)) for n, v in zip(FEATURE_NAMES, x)}
        raw = {
            **{k: round(v, 4) for k, v in res.items()},
            "expected_q4_margin": round(margin, 2),
            "is_green": signal == "BET",
            "toxic_league_blocked": toxic,
            "min_conf_threshold": thr,
            "snapshot_minute_used": round(float(ctx["minute"]), 2),
            "features_missing": int(np.isnan(x).sum()),
            "has_ml": bool(x[IDX["has_ml"]]),
            "has_spread": not np.isnan(x[IDX["mkt_margin_spread"]]),
            "league": league,
            "features": feats,
        }
        return PredictionResult(
            available=True,
            model_version=MODEL_VERSION,
            target=target,
            signal=signal,
            pick=pick,
            confidence=round(float(confidence), 4),
            p_home_win=round(p_cal, 4),
            p_away_win=round(1.0 - p_cal, 4),
            snapshot_minute=int(ctx["minute"]),
            raw_json=raw,
        )
    except Exception as exc:
        return PredictionResult.unavailable(MODEL_VERSION, target, f"Error en inferencia: {exc}")
    finally:
        if close_conn and conn:
            conn.close()


def predict_batch(match_ids: list[str], target: str = "q4") -> dict[str, PredictionResult]:
    conn = get_db_connection()
    try:
        return {mid: predict(mid, target=target, conn=conn) for mid in match_ids}
    finally:
        conn.close()


def predict_with_metadata(match_data: dict[str, Any], snapshot_minute: float = 27.0) -> dict[str, Any]:
    res = predict("live", match_data=match_data, snapshot_minute=snapshot_minute)
    return {
        "model_version": res.model_version, "available": res.available, "pick": res.pick,
        "confidence": res.confidence, "p_home_win": res.p_home_win, "p_away_win": res.p_away_win,
        "signal": res.signal, "raw_json": res.raw_json,
    }

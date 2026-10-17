"""Inferencia en tiempo real e histórica para el modelo M27 V4 (Tríada Canónica Multi-Head)."""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Any
import numpy as np

from models.common.data_loader import get_db_connection, load_match
from models.common.schema import PredictionResult
from models.m27_v4.features import MODEL_DIR, extract_m27_v4_features

_BUNDLES_CACHE: dict[str, dict[str, Any]] = {}


def _get_bundle(archetype: str) -> dict[str, Any] | None:
    if archetype not in _BUNDLES_CACHE:
        path = MODEL_DIR / f"model_v4_{archetype}.pkl"
        if not path.exists():
            path = MODEL_DIR / "model_v4_fiba_men_10m.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            _BUNDLES_CACHE[archetype] = pickle.load(f)
    return _BUNDLES_CACHE[archetype]


def predict(
    match_id: str,
    target: str = "q4",
    match_data: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
    snapshot_minute: float | None = None,
) -> PredictionResult:
    """
    Ejecuta la inferencia Multi-Head M27 V4 sobre un partido.
    Soporta target='q4' (por defecto), target='q4_total' y target='ft'.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        if match_data is None:
            match_data = load_match(conn, match_id)
        if not match_data:
            return PredictionResult.unavailable(
                model_version="m27_v4",
                target=target,
                reason=f"Partido {match_id} no encontrado en matches.db",
            )

        X, meta = extract_m27_v4_features(match_data, conn=conn, snapshot_minute=snapshot_minute)
        arch = meta["archetype"]
        bundle = _get_bundle(arch)
        if bundle is None:
            return PredictionResult.unavailable(
                model_version="m27_v4",
                target=target,
                reason=f"Bundle entrenado para arquetipo {arch} no encontrado",
            )

        w_xgb, w_hgb, w_lr = bundle.get("weights", (0.45, 0.40, 0.15))
        p_xgb = float(bundle["xgb_q4_win"].predict_proba(X)[0, 1])
        p_hgb = float(bundle["hgb_q4_win"].predict_proba(X)[0, 1])
        p_lr = float(bundle["lr_q4_win"].predict_proba(X)[0, 1])
        raw_prob = w_xgb * p_xgb + w_hgb * p_hgb + w_lr * p_lr
        p_home_q4 = float(bundle["iso_q4_win"].predict(np.array([raw_prob]))[0])
        p_away_q4 = 1.0 - p_home_q4

        exp_q4_margin = float(bundle["reg_q4_margin"].predict(X)[0])
        exp_q4_total = float(bundle["reg_q4_total"].predict(X)[0])
        p_home_ft = float(bundle["clf_ft_win"].predict_proba(X)[0, 1])

        lg_avg_q4 = float(meta["lg_avg_q4"])
        ou_diff = exp_q4_total - lg_avg_q4
        ou_signal = "OVER_Q4" if ou_diff >= 2.5 else ("UNDER_Q4" if ou_diff <= -2.5 else "NEUTRAL")

        conf_q4 = max(p_home_q4, p_away_q4)
        pick_q4 = "HOME" if p_home_q4 >= 0.5 else "AWAY"

        # Gate Dual-Head: probabilidad calibrada >= 0.68 y margen esperado alineado >= 1.8 pts
        margin_aligned = (p_home_q4 >= 0.5 and exp_q4_margin >= 1.8) or (p_home_q4 < 0.5 and exp_q4_margin <= -1.8)
        if conf_q4 >= 0.70 and margin_aligned:
            sig = "BET_STRONG"
        elif conf_q4 >= 0.68 and margin_aligned:
            sig = "BET"
        else:
            sig = "NO_BET"

        reason_str = (
            f"[{arch} | m{int(round(meta['snapshot_minute']))}] "
            f"Q4 Pick={pick_q4} ({conf_q4*100:.1f}%, E[dQ4]={exp_q4_margin:+.1f} pts) | "
            f"E[TotQ4]={exp_q4_total:.1f} vs Prior={lg_avg_q4:.1f} ({ou_signal}) | "
            f"P(FT Home)={p_home_ft*100:.1f}%"
        )

        raw_payload = {
            "archetype": arch,
            "snapshot_minute": meta["snapshot_minute"],
            "p_home_win": round(p_home_q4, 4),
            "p_away_win": round(p_away_q4, 4),
            "confidence": round(conf_q4, 4),
            "predicted_winner": pick_q4.lower(),
            "bet_signal": sig,
            "expected_q4_margin": round(exp_q4_margin, 2),
            "expected_q4_total": round(exp_q4_total, 2),
            "league_q4_prior": round(lg_avg_q4, 2),
            "q4_over_under_signal": ou_signal,
            "p_ft_home_win": round(p_home_ft, 4),
            "p_ft_away_win": round(1.0 - p_home_ft, 4),
            "components": {
                "p_xgb": round(p_xgb, 4),
                "p_hgb": round(p_hgb, 4),
                "p_lr": round(p_lr, 4),
            },
            "meta": meta,
        }

        return PredictionResult(
            available=True,
            model_version="m27_v4",
            target=target,
            signal=sig,
            pick=pick_q4,
            confidence=round(conf_q4, 4),
            p_home_win=round(p_home_q4, 4),
            p_away_win=round(p_away_q4, 4),
            reason=reason_str,
            raw_json=raw_payload,
        )
    finally:
        if close_conn:
            conn.close()

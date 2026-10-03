"""Motor de inferencia y predicción para el modelo campeón Q3/Q4 v6_2.

Carga q3_champion.joblib o q4_champion.joblib (XGBoost + HistGB con poda de ligas)
y genera un PredictionResult canónico.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import joblib

from models.common.data_loader import get_db_connection, load_match
from models.common.schema import PredictionResult
from models.v6_2.features import extract_features

MODEL_DIR = Path(__file__).resolve().parent / "model_outputs"

_CHAMPION_CACHE: dict[str, Any] = {}


def load_champion_metadata(target: str) -> dict[str, Any]:
    """Carga y almacena en caché singleton los artefactos campeón de v6_2 para Q3 o Q4."""
    t = target.lower()
    if t in _CHAMPION_CACHE:
        return _CHAMPION_CACHE[t]

    champ_file = MODEL_DIR / f"{t}_champion.joblib"
    if not champ_file.exists():
        raise FileNotFoundError(f"Artefacto campeón {champ_file} no encontrado.")

    data = joblib.load(champ_file)
    _CHAMPION_CACHE[t] = data
    return data


def predict(
    match_id: str,
    target: str = "q4",
    match_data: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> PredictionResult:
    """Ejecuta inferencia con v6_2 para target='q3' o 'q4'."""
    t = target.lower()
    if t not in ("q3", "q4"):
        return PredictionResult.unavailable(
            model_version="v6_2",
            target=target,
            reason=f"Target {target} no soportado por v6_2 (solo 'q3' y 'q4')",
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
                model_version="v6_2",
                target=t,
                reason=f"No se encontraron datos para el match {match_id}",
            )

        features = extract_features(match_data, conn, target=t)
        meta = load_champion_metadata(t)

        vec = meta["vectorizer"]
        models = meta["models"]

        # Aplicar poda de ligas
        league = str(features.get("league", ""))
        keep_leagues = set(meta.get("league_filter", {}).get("kept_leagues", []))
        other_token = meta.get("league_filter", {}).get("other_token", "LEAGUE_OTHER_SIGNAL_WEAK")

        features_copy = dict(features)
        if keep_leagues and league not in keep_leagues:
            features_copy["league"] = other_token
            features_copy["league_bucket"] = other_token

        x_mat = vec.transform([features_copy])

        if t == "q3":
            prob_home = float(models["xgb"].predict_proba(x_mat)[0, 1])
            bet_thr = 0.18
            lean_thr = 0.10
            snap_min = 24
        else:
            p_xgb = float(models["xgb"].predict_proba(x_mat)[0, 1])
            p_hgb = float(models["hist_gb"].predict_proba(x_mat)[0, 1])
            prob_home = float(0.6 * p_xgb + 0.4 * p_hgb)
            bet_thr = 0.14
            lean_thr = 0.08
            snap_min = 36

        prob_home = round(prob_home, 6)
        prob_away = round(1.0 - prob_home, 6)
        confidence = round(abs(prob_home - 0.5) * 2.0, 6)
        winner = "HOME" if prob_home >= 0.5 else "AWAY"

        if confidence >= bet_thr:
            signal = "BET"
        elif confidence >= lean_thr:
            signal = "LEAN"
        else:
            signal = "NO_BET"

        return PredictionResult(
            available=True,
            model_version="v6_2",
            target=t,
            signal=signal,
            pick=winner,
            confidence=confidence,
            p_home_win=prob_home,
            p_away_win=prob_away,
            snapshot_minute=snap_min,
            reason=f"P(Home)={prob_home:.4f}, Conf={confidence:.4f}",
            raw_json={
                "league_kept": league in keep_leagues,
                "confidence": confidence,
                "threshold_bet": bet_thr,
                "threshold_lean": lean_thr,
            },
        )
    finally:
        if close_conn:
            conn.close()

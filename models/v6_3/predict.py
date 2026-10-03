"""Motor de inferencia para el modelo v6_3 con lista negra de ligas y snapshots dinámicos.

Carga q4_m27_champion.joblib o q4_m30_champion.joblib y aplica el filtro de ligas negras.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import joblib

from models.common.data_loader import get_db_connection, load_match
from models.common.schema import PredictionResult
from models.v6_3.features import extract_features
from models.v6_3.league_blacklist import V63LeagueBlacklist

MODEL_DIR = Path(__file__).resolve().parent / "model_outputs"

_BLACKLIST = V63LeagueBlacklist()
_CHAMPION_CACHE: dict[str, Any] = {}


def load_champion_metadata(target: str = "q4", snapshot_minute: int = 27) -> dict[str, Any]:
    """Carga y almacena en caché el artefacto campeón de v6_3."""
    snap = 27 if snapshot_minute <= 28 else 30
    cache_key = f"{target}_m{snap}"
    if cache_key in _CHAMPION_CACHE:
        return _CHAMPION_CACHE[cache_key]

    champ_path = MODEL_DIR / f"{target}_m{snap}_champion.joblib"
    if not champ_path.exists():
        champ_path = MODEL_DIR / f"{target}_champion.joblib"

    if not champ_path.exists():
        raise FileNotFoundError(f"Artefacto campeón v6_3 {champ_path} no encontrado.")

    data = joblib.load(champ_path)
    _CHAMPION_CACHE[cache_key] = data
    return data


def predict(
    match_id: str,
    target: str = "q4",
    match_data: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
    snapshot_minute: int = 27,
) -> PredictionResult:
    """Ejecuta inferencia con v6_3 aplicando filtro de blacklist y ensamble 50/50."""
    t = target.lower()
    if t != "q4":
        return PredictionResult.unavailable(
            model_version="v6_3",
            target=target,
            reason="v6_3 solo opera sobre target='q4'",
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
                model_version="v6_3",
                target=t,
                reason=f"No se encontraron datos para el match {match_id}",
            )

        league = str((match_data.get("match") or {}).get("league", ""))
        blocked, bl_reason = _BLACKLIST.is_blocked(league)
        if blocked:
            return PredictionResult.unavailable(
                model_version="v6_3",
                target=t,
                reason=f"v6_3_blacklist: {bl_reason}",
            )

        features = extract_features(match_data, conn, target=t)
        meta = load_champion_metadata(t, snapshot_minute)

        vec = meta["vectorizer"]
        models = meta["models"]

        keep_leagues = set(meta.get("league_filter", {}).get("kept_leagues", []))
        other_token = meta.get("league_filter", {}).get("other_token", "LEAGUE_OTHER_SIGNAL_WEAK")

        features_copy = dict(features)
        if keep_leagues and league not in keep_leagues:
            features_copy["league"] = other_token
            features_copy["league_bucket"] = other_token

        x_mat = vec.transform([features_copy])
        p_xgb = float(models["xgb"].predict_proba(x_mat)[0, 1])
        p_hgb = float(models["hist_gb"].predict_proba(x_mat)[0, 1])

        prob_home = round(0.5 * p_xgb + 0.5 * p_hgb, 6)
        prob_away = round(1.0 - prob_home, 6)
        confidence = round(abs(prob_home - 0.5) * 2.0, 6)
        winner = "HOME" if prob_home >= 0.5 else "AWAY"

        snap = 27 if snapshot_minute <= 28 else 30
        bet_thr = 0.18 if snap <= 30 else 0.14
        lean_thr = 0.11 if snap <= 30 else 0.08

        if confidence >= bet_thr:
            signal = "BET"
        elif confidence >= lean_thr:
            signal = "LEAN"
        else:
            signal = "NO_BET"

        return PredictionResult(
            available=True,
            model_version="v6_3",
            target=t,
            signal=signal,
            pick=winner,
            confidence=confidence,
            p_home_win=prob_home,
            p_away_win=prob_away,
            snapshot_minute=snap,
            reason=f"P(Home)={prob_home:.4f}, Conf={confidence:.4f}",
            raw_json={
                "snapshot_minute": snap,
                "confidence": confidence,
                "threshold_bet": bet_thr,
            },
        )
    finally:
        if close_conn:
            conn.close()

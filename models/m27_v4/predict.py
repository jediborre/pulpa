"""Inferencia en tiempo real e histórica para el modelo M27 V4 (Tríada Canónica Multi-Head)."""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any
import numpy as np

from models.common.data_loader import get_db_connection, load_match
from models.common.schema import PredictionResult
from models.m27_v4.features import MODEL_DIR, extract_m27_v4_features

_BUNDLES_CACHE: dict[str, dict[str, Any]] = {}
_TOXIC_CACHE: dict[str, list[str]] | None = None


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


def _get_toxic_map() -> dict[str, list[str]]:
    global _TOXIC_CACHE
    if _TOXIC_CACHE is None:
        toxic_path = MODEL_DIR / "toxic_leagues.json"
        if toxic_path.exists():
            try:
                _TOXIC_CACHE = json.loads(toxic_path.read_text(encoding="utf-8"))
            except Exception:
                _TOXIC_CACHE = {}
        else:
            _TOXIC_CACHE = {}
    return _TOXIC_CACHE


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

        X, meta = extract_m27_v4_features(
            match_data,
            conn=conn,
            snapshot_minute=snapshot_minute,
            match_id=match_id,
        )
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
        pick_sign = 1.0 if p_home_q4 >= 0.5 else -1.0

        # 1. Unanimidad de los 3 submodelos (XGBoost, HistGB, Linear L2)
        c_xgb = p_xgb if pick_sign > 0 else (1.0 - p_xgb)
        c_hgb = p_hgb if pick_sign > 0 else (1.0 - p_hgb)
        c_lr = p_lr if pick_sign > 0 else (1.0 - p_lr)
        min_submodel_conf = min(c_xgb, c_hgb, c_lr)

        # 2. Alineación de Margen Esperado Q4, ELO Q4 y Calidad de Tiros (Pintura + FT)
        exp_margin_fav = pick_sign * exp_q4_margin
        cur_d_fav = pick_sign * float(X[0, 0])
        elo_q4_fav = pick_sign * float(X[0, 14])
        q4_bayes_fav = pick_sign * float(X[0, 17])

        # 3. Filtro de Ligas Tóxicas (toxic_leagues.json)
        toxic_map = _get_toxic_map()
        toxic_list = toxic_map.get(arch, [])
        lg_name = str(meta.get("league", "")).strip()
        country = str(meta.get("country", "")).strip()
        full_lg_label = f"{country} - {lg_name}".strip(" -")
        is_toxic_league = any(
            t.lower() in lg_name.lower() or t.lower() in full_lg_label.lower()
            for t in toxic_list
        )

        # 4. Protección contra Trampa de Descanso de Estrellas (Star-Rest / Garbage-Time Trap):
        # No apostar a favor del equipo que YA va ganando por paliza al cierre del Q3,
        # porque sienta a sus titulares >5 min en Q4 (60.3% pierde Q4 en NBA, 44.3% en FIBA Men).
        max_lead_for_pick = 14.0 if arch == "nba_12m" else (24.0 if arch == "fiba_women_10m" else 18.0)
        is_star_rest_trap = cur_d_fav >= max_lead_for_pick

        # 5. Verificación de Información Pre-Partido y Bloqueo de Ligas Infantiles (U13-U17)
        elo_rel = float(X[0, 16])
        has_odds = float(X[0, 19])
        d_rol_pm = float(X[0, 29])
        h2h_avail = bool(meta.get("h2h_available", False))
        has_prior_info = (elo_rel >= 0.08) or (has_odds > 0.5) or (abs(d_rol_pm) > 0.05) or h2h_avail
        is_youth_league = any(
            u in lg_name.lower() for u in ("u13", "u14", "u15", "u16", "u17", "u18", "youth")
        )

        # 6. Condiciones Óptimas por Arquetipo para Maximizar Verdes (77.7% - 85.7% WR)
        min_elo_q4_req = 0.0 if arch == "nba_12m" else -15.0
        elo_q4_aligned = (elo_q4_fav >= min_elo_q4_req) or (q4_bayes_fav >= 0.0)

        sniper_aligned = (
            not is_toxic_league
            and not is_star_rest_trap
            and not is_youth_league
            and has_prior_info
            and conf_q4 >= 0.72
            and exp_margin_fav >= 3.0
            and min_submodel_conf >= 0.66
            and elo_q4_fav >= 0.0
        )
        strong_aligned = (
            not is_toxic_league
            and not is_star_rest_trap
            and not is_youth_league
            and has_prior_info
            and conf_q4 >= 0.70
            and exp_margin_fav >= 2.0
            and min_submodel_conf >= 0.64
            and elo_q4_aligned
        )
        bet_aligned = (
            not is_toxic_league
            and not is_star_rest_trap
            and not is_youth_league
            and has_prior_info
            and conf_q4 >= 0.68
            and exp_margin_fav >= 1.8
            and min_submodel_conf >= 0.62
            and elo_q4_aligned
        )

        if is_toxic_league or is_star_rest_trap or is_youth_league or not has_prior_info:
            sig = "NO_BET"
        elif sniper_aligned:
            sig = "BET_SNIPER"
        elif strong_aligned:
            sig = "BET_STRONG"
        elif bet_aligned:
            sig = "BET"
        else:
            sig = "NO_BET"

        block_tag = ""
        if is_toxic_league:
            block_tag = " [TOXIC_LG_BLOCKED]"
        elif is_youth_league:
            block_tag = " [YOUTH_LG_BLOCKED]"
        elif not has_prior_info:
            block_tag = " [NO_PRIOR_INFO]"
        elif is_star_rest_trap:
            block_tag = " [STAR_REST_TRAP_BLOCKED]"

        reason_str = (
            f"[{arch} | m{int(round(meta['snapshot_minute']))}] "
            f"Q4 Pick={pick_q4} ({conf_q4*100:.1f}%, minSub={min_submodel_conf*100:.1f}%, E[dQ4]={exp_q4_margin:+.1f} pts){block_tag} | "
            f"E[TotQ4]={exp_q4_total:.1f} vs Prior={lg_avg_q4:.1f} ({ou_signal}) | "
            f"P(FT Home)={p_home_ft*100:.1f}%"
        )

        raw_payload = {
            "archetype": arch,
            "snapshot_minute": meta["snapshot_minute"],
            "p_home_win": round(p_home_q4, 4),
            "p_away_win": round(p_away_q4, 4),
            "confidence": round(conf_q4, 4),
            "min_submodel_confidence": round(min_submodel_conf, 4),
            "is_toxic_league": is_toxic_league,
            "is_star_rest_trap": is_star_rest_trap,
            "predicted_winner": pick_q4.lower(),
            "bet_signal": sig,
            "expected_q4_margin": round(exp_q4_margin, 2),
            "expected_q4_total": round(exp_q4_total, 2),
            "league_q4_prior": round(lg_avg_q4, 2),
            "q4_over_under_signal": ou_signal,
            "p_ft_home_win": round(p_home_ft, 4),
            "p_ft_away_win": round(1.0 - p_home_ft, 4),
            "q4": {
                "h2h_available": bool(meta.get("h2h_available", False)),
                "expected_q4_margin": round(exp_q4_margin, 2),
                "min_submodel_confidence": round(min_submodel_conf, 4),
            },
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

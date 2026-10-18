"""Inferencia en tiempo real e histórica para el modelo M27 V5 (Roster ELO Dinámico + Perfil Táctico)."""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any
import numpy as np

from models.common.data_loader import get_db_connection, load_match
from models.common.schema import PredictionResult
from models.m27_v5.features import FEATURE_NAMES_V5, MODEL_DIR, extract_m27_v5_features

_MODELS_CACHE: dict[str, Any] | None = None
_TOXIC_CACHE: dict[str, list[str]] | None = None
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
        except Exception as e:
            _MODELS_CACHE = {}
    return _MODELS_CACHE


def _get_toxic_map() -> dict[str, list[str]]:
    global _TOXIC_CACHE
    if _TOXIC_CACHE is None:
        toxic_path = MODEL_DIR.parent.parent / "m27_v4" / "model_outputs" / "toxic_leagues.json"
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
    """Ejecuta inferencia con el modelo M27 V5 retornando un PredictionResult canónico."""
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        if match_data is None:
            match_data = load_match(conn, match_id)
        if not match_data:
            return PredictionResult.unavailable(
                model_version="m27_v5",
                target=target,
                reason=f"Partido {match_id} no encontrado en matches.db",
            )

        models = _load_production_models()
        if not models:
            return PredictionResult.unavailable(
                model_version="m27_v5",
                target=target,
                reason="Modelos de producción m27_v5 no encontrados en disco",
            )

        X, meta = extract_m27_v5_features(match_data, snapshot_minute=snapshot_minute)
        X_2d = X.reshape(1, -1)

        xgb_m = models["xgb"]
        hgb_m = models["hgb"]
        lr_m = models["lr"]
        reg_m = models["reg"]
        calib = models["calibrator"]

        p_xgb = float(xgb_m.predict_proba(X_2d)[0, 1])
        p_hgb = float(hgb_m.predict_proba(X_2d)[0, 1])
        p_lr = float(lr_m.predict_proba(X_2d)[0, 1])

        p_raw = 0.40 * p_xgb + 0.35 * p_hgb + 0.25 * p_lr
        p_home = float(np.clip(calib.predict([p_raw])[0], 0.02, 0.98))
        p_away = float(1.0 - p_home)

        exp_q4_margin = float(reg_m.predict(X_2d)[0])

        if p_home >= 0.50:
            pick = "HOME"
            confidence = p_home
            min_submodel_conf = min(p_xgb, p_hgb, p_lr)
            fav_expected_q4_margin = exp_q4_margin
        else:
            pick = "AWAY"
            confidence = p_away
            min_submodel_conf = min(1.0 - p_xgb, 1.0 - p_hgb, 1.0 - p_lr)
            fav_expected_q4_margin = -exp_q4_margin

        # Filtros de Operabilidad, Calibración de Ligas y Verdes
        league_str = meta.get("league", "").strip()
        country_str = meta.get("country", "").strip()
        full_league_key = f"{country_str} - {league_str}" if country_str else league_str
        toxic_map = _get_toxic_map()
        toxic_list = toxic_map.get("toxic_leagues", [])

        is_toxic_league = any(
            t.lower() in league_str.lower() or t.lower() in full_league_key.lower()
            for t in toxic_list
            if len(t) >= 4
        )

        calib_cfg = _get_league_calibration()
        blacklisted_patterns = calib_cfg.get("blacklisted_patterns", [
            "friendly", "preseason", "relegation", "playout", "rel/prom", "qualifying round"
        ])
        is_blacklisted_league = any(
            p.lower() in league_str.lower() or p.lower() in full_league_key.lower()
            for p in blacklisted_patterns
        )

        # Calibración adaptativa de umbral P y margen mínimo según arquetipo y liga
        league_up = league_str.upper()
        is_women_league = any(tag in league_up for tag in ["WOMEN", "FEMENIL", "FEMENINA", "FEMININA", "WNBA", "DAMEN", "KOBIET", "NAISTE"])

        if is_women_league:
            req_confidence = 0.60
            req_margin = 1.4
            league_tier_note = "Women Tier (WR 78-89%)"
        elif any(k in league_up for k in ["NBA", "G LEAGUE", "CBA"]):
            req_confidence = 0.68
            req_margin = 2.2
            league_tier_note = "NBA/CBA Tier (Filtro Sniper P>=0.68)"
        elif any(k in league_up for k in ["ACB", "EUROLEAGUE"]):
            req_confidence = 0.62
            req_margin = 1.6
            league_tier_note = "Euro Elite Tier (WR 74-76%)"
        else:
            req_confidence = 0.63
            req_margin = 1.6
            league_tier_note = "Standard FIBA Tier (WR 71-73%)"

        # Filtro de ligas juveniles U13..U18
        is_youth = any(tag in league_str.upper() for tag in ["U13", "U14", "U15", "U16", "U17", "U18", "JUNIOR", "CADETE"])
        # Filtro de cyber / esports
        is_cyber = any(tag in (meta.get("home_team", "") + meta.get("away_team", "")).lower() for tag in ["cyber", "esports", "2k"])

        # Filtro de basura / descanso de estrellas en palizas
        cur_lead = meta.get("cur_diff", 0.0)
        q_dur = meta.get("quarter_duration", 10)
        blowout_limit = 18.0 if q_dur == 10 else 14.0
        is_star_rest_trap = (
            (pick == "HOME" and cur_lead >= blowout_limit) or
            (pick == "AWAY" and cur_lead <= -blowout_limit)
        )

        signal = "NO_BET"
        if not is_toxic_league and not is_blacklisted_league and not is_youth and not is_cyber and not is_star_rest_trap:
            if (
                confidence >= 0.72 and
                fav_expected_q4_margin >= (req_margin + 0.8) and
                min_submodel_conf >= 0.65
            ):
                signal = "BET_SNIPER"
            elif (
                confidence >= max(0.66, req_confidence) and
                fav_expected_q4_margin >= (req_margin + 0.3) and
                min_submodel_conf >= 0.60
            ):
                signal = "BET_STRONG"
            elif (
                confidence >= req_confidence and
                fav_expected_q4_margin >= req_margin and
                min_submodel_conf >= (req_confidence - 0.05)
            ):
                signal = "BET"

        reason = (
            f"V5 Consensus: Conf={confidence*100:.1f}% ({pick}) [{league_tier_note}] | "
            f"Submodels: XGB={p_xgb:.2f}, HGB={p_hgb:.2f}, LR={p_lr:.2f} | "
            f"Exp Margin Q4={exp_q4_margin:+.1f} (Req>={req_margin:.1f}) | "
            f"Roster ELO Q4 Diff={meta.get('d_lineup_starters_elo_q4', 0.0):+.1f}"
        )
        if is_blacklisted_league:
            reason += " [BLOCKED: Liga Sobreestimada/Pretemporada/Descenso]"
        elif is_toxic_league:
            reason += " [BLOCKED: Liga Tóxica]"
        elif is_youth:
            reason += " [BLOCKED: Liga Juvenil]"
        elif is_cyber:
            reason += " [BLOCKED: Partido Cyber/Esports]"
        elif is_star_rest_trap:
            reason += " [BLOCKED: Trampa de Descanso de Estrellas en Paliza]"
        elif signal == "NO_BET" and confidence >= 0.60 and confidence < req_confidence:
            reason += f" [NO_BET: Confianza {confidence*100:.1f}% no alcanza umbral de seguridad para {league_tier_note} ({req_confidence*100:.0f}%)]"

        raw_payload = {
            "features": {FEATURE_NAMES_V5[i]: float(X[i]) for i in range(len(FEATURE_NAMES_V5))},
            "submodel_predictions": {
                "xgb_p_home": round(p_xgb, 4),
                "hgb_p_home": round(p_hgb, 4),
                "lr_p_home": round(p_lr, 4),
                "expected_q4_margin": round(exp_q4_margin, 2),
            },
            "thresholds": {
                "lean": 0.58,
                "bet": 0.63,
                "strong": 0.66,
                "sniper": 0.70,
            },
            "meta": meta,
        }

        return PredictionResult(
            available=True,
            model_version="m27_v5",
            target=target,
            signal=signal,
            pick=pick,
            confidence=round(confidence, 4),
            p_home_win=round(p_home, 4),
            p_away_win=round(p_away, 4),
            reason=reason,
            snapshot_minute=int(snapshot_minute) if snapshot_minute is not None else None,
            raw_json=raw_payload,
        )
    finally:
        if close_conn and conn:
            conn.close()

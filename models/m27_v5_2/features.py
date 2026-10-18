"""
Extractor de 27 features nucleares para el modelo M27_V5_2 (Lean Pruned + Market Odds & Standings).
Resiliente tanto para partidos históricos como para inferencia en tiempo real desde SofaScore Mobile API.
"""

from __future__ import annotations

import math
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

from models.common.pbp_utils import (
    graph_stats_upto,
    infer_gender,
    infer_minute_from_pbp,
    infer_regulation_quarter_minutes,
    pbp_events_upto,
    quarter_points,
    recent_window_features,
    score_upto,
)

MODEL_DIR = Path(__file__).resolve().parent / "model_outputs"
_STATE_CACHE: dict[str, Any] | None = None

FEATURE_NAMES_27 = [
    "cur_score_diff",
    "abs_cur_score_diff",
    "ht_diff",
    "q1_diff",
    "q2_diff",
    "q3_eff_diff",
    "q3_vs_ht_momentum",
    "recent_2m_run_diff",
    "pace_pts_per_min",
    "pace_vs_league_prior",
    "gp_val",
    "d_lineup_starters_elo_ft",
    "d_lineup_starters_elo_q4",
    "d_lineup_bench_elo_q4",
    "d_lineup_top3_elo_q4",
    "d_roster_drift_adv",
    "d_strat_q4_net",
    "d_q4_bayes",
    "d_wr_bayes",
    "odds_log_ratio",          # Restaurada de V5
    "expected_ft_margin",
    "h2h_bayes_ft_margin",
    "ts_delta_wp_bayes",       # Restaurada de V5
    "pbp_3pt_diff_q1_q3",
    "rubber_band_gap",
    "white_flag_risk_zone",
    "lg_avg_q4_prior",
]


def get_v5_2_priors_state() -> dict[str, Any]:
    """Carga en caché el estado de producción de m27_v5_2 (priors de equipo, jugador y ligas)."""
    global _STATE_CACHE
    if _STATE_CACHE is None:
        state_path = MODEL_DIR / "v5_2_production_state.pkl"
        if not state_path.exists():
            state_path = MODEL_DIR.parent.parent / "m27_v5" / "model_outputs" / "v5_production_state.pkl"
        if state_path.exists():
            with open(state_path, "rb") as f:
                _STATE_CACHE = pickle.load(f)
        else:
            _STATE_CACHE = {"team_priors": {}, "player_priors": {}, "lc_map": {}}
    return _STATE_CACHE


def resolve_archetype_v5_2(match_data: dict[str, Any]) -> tuple[str, int, float, float]:
    """Determina si el partido es 10m (FIBA) o 12m (NBA) y priors de anotación."""
    state = get_v5_2_priors_state()
    lc_map = state.get("lc_map", {})
    m_info = match_data.get("match") or {}
    league = str(m_info.get("league") or match_data.get("league") or "").strip()
    ht = str(m_info.get("home_team") or match_data.get("home_team") or "")
    at = str(m_info.get("away_team") or match_data.get("away_team") or "")

    p_len = m_info.get("period_length")
    if p_len == 720:
        return "nba_12m", 12, 218.0, 53.5

    if league in lc_map:
        info = lc_map[league]
        return info.get("arch", "fiba_men_10m"), info.get("q_dur", 10), info.get("lg_avg_ft", 158.0), info.get("lg_avg_q4", 39.5)

    q_dur = int(round(infer_regulation_quarter_minutes(match_data)))
    is_women = infer_gender(league, ht, at) == "women"
    if q_dur == 12:
        return "nba_12m", 12, 218.0, 53.5
    if is_women:
        return "fiba_women_10m", 10, 140.0, 34.5
    return "fiba_men_10m", 10, 158.0, 39.5


def _extract_odds_from_payload(raw_odds: list[dict[str, Any]]) -> tuple[float, float]:
    """Extrae momios de victoria final y calcula el log-ratio."""
    if not raw_odds or not isinstance(raw_odds, list):
        return 0.0, 0.0
    best_pair = None
    for entry in raw_odds:
        if not isinstance(entry, dict):
            continue
        items = entry.get("choices") or entry.get("markets") or [entry]
        for item in items:
            if not isinstance(item, dict):
                continue
            ho = item.get("home_value") or item.get("home_odd")
            ao = item.get("away_value") or item.get("away_odd")
            if not ho or not ao:
                continue
            try:
                fh, fa = float(ho), float(ao)
            except (ValueError, TypeError):
                continue
            if fh <= 1.0 or fa <= 1.0:
                continue
            otype = str(item.get("odds_type") or "").lower()
            mper = str(item.get("market_period") or "").lower()
            mname = str(item.get("market_name") or "").lower()
            if ("home/away" in otype or "full" in mname) and ("match" in mper or "full" in mper or not mper):
                return 1.0, float(np.clip(math.log(fa / fh), -2.5, 2.5))
            if best_pair is None and "spread" not in otype and "total" not in otype:
                best_pair = (fh, fa)
        if best_pair is not None:
            return 1.0, float(np.clip(math.log(best_pair[1] / best_pair[0]), -2.5, 2.5))
    return 0.0, 0.0


def extract_features_v5_2(match_data: dict[str, Any], target_minute: float = 27.0) -> np.ndarray:
    """Extrae las 27 features nucleares ortogonales en el snapshot especificado."""
    state = get_v5_2_priors_state()
    team_priors = state.get("team_priors", {})
    player_priors = state.get("player_priors", {})

    arch, q_dur, lg_avg_ft, lg_avg_q4 = resolve_archetype_v5_2(match_data)
    is_12m = (q_dur == 12)
    eff_min = float(target_minute) if target_minute is not None else (34.5 if is_12m else 27.0)
    q3_end_min = 36.0 if is_12m else 30.0

    # 1. Marcador y momentum
    ch, ca = score_upto(match_data, eff_min)
    cur_score_diff = float(ch - ca)
    abs_cur_score_diff = abs(cur_score_diff)

    q1h, q1a = quarter_points(match_data, "Q1")
    q2h, q2a = quarter_points(match_data, "Q2")
    q3h, q3a = quarter_points(match_data, "Q3")
    q1h, q1a = int(q1h or 0), int(q1a or 0)
    q2h, q2a = int(q2h or 0), int(q2a or 0)
    q3h, q3a = int(q3h or 0), int(q3a or 0)

    q1_diff = float(q1h - q1a)
    q2_diff = float(q2h - q2a)
    q3_diff = float(q3h - q3a)
    ht_diff = q1_diff + q2_diff

    q3_tot = float(q3h + q3a)
    q3_eff_diff = (q3_diff / (q3_tot + 1.0)) * 10.0
    q3_vs_ht_momentum = q3_diff - (ht_diff * 0.5)

    recent_2m_run_diff = recent_window_features(match_data, eff_min, window_minutes=2.0).get("run_diff", 0.0)

    pace_pts_per_min = float(ch + ca) / max(1.0, eff_min)
    lg_expected_ppm = (lg_avg_ft * 0.75) / q3_end_min
    pace_vs_league_prior = pace_pts_per_min - lg_expected_ppm

    gp_dict = graph_stats_upto(match_data.get("graph_points") or [], int(eff_min))
    gp_val = float(gp_dict.get("gp_last", 0.0))

    # 2. Priors de Equipo y ELO
    m_info = match_data.get("match") or {}
    ht = str(m_info.get("home_team") or match_data.get("home_team") or "")
    at = str(m_info.get("away_team") or match_data.get("away_team") or "")

    p_h = team_priors.get(ht, {})
    p_a = team_priors.get(at, {})

    d_elo_ft = float(p_h.get("elo_ft", 1500.0) - p_a.get("elo_ft", 1500.0))
    d_elo_q4 = float(p_h.get("elo_q4", 1500.0) - p_a.get("elo_q4", 1500.0))
    d_q4_bayes = d_elo_q4 / 35.0
    d_wr_bayes = d_elo_ft / 180.0

    # 3. Roster ELO
    lineups = match_data.get("lineups") or match_data.get("players") or []
    d_lineup_starters_elo_ft = d_elo_ft
    d_lineup_starters_elo_q4 = d_elo_q4
    d_lineup_bench_elo_q4 = 0.0
    d_lineup_top3_elo_q4 = d_elo_q4
    d_roster_drift_adv = 0.0

    if lineups and player_priors:
        h_st, a_st = [], []
        h_bn, a_bn = [], []
        for p in lineups:
            if not isinstance(p, dict):
                continue
            pname = p.get("name") or p.get("player_name") or ""
            pid = str(p.get("id") or p.get("player_id") or "")
            is_st = bool(p.get("is_starter") or p.get("starter"))
            is_home = p.get("is_home") if "is_home" in p else ("home" in str(p.get("team", "")).lower() or p.get("team") == ht)

            p_data = player_priors.get(pid) or player_priors.get(pname)
            p_elo_q4 = p_data.get("elo_q4", 1500.0) if p_data else 1500.0

            if is_home:
                if is_st:
                    h_st.append(p_elo_q4)
                else:
                    h_bn.append(p_elo_q4)
            else:
                if is_st:
                    a_st.append(p_elo_q4)
                else:
                    a_bn.append(p_elo_q4)

        if h_st and a_st:
            d_lineup_starters_elo_q4 = float(np.mean(h_st) - np.mean(a_st))
            d_lineup_starters_elo_ft = d_lineup_starters_elo_q4 * 1.15
        if h_bn and a_bn:
            d_lineup_bench_elo_q4 = float(np.mean(h_bn) - np.mean(a_bn))
        if h_st and a_st:
            top_h = sorted(h_st, reverse=True)[:3]
            top_a = sorted(a_st, reverse=True)[:3]
            d_lineup_top3_elo_q4 = float(np.mean(top_h) - np.mean(top_a))
        d_roster_drift_adv = d_lineup_starters_elo_q4 - d_elo_q4

    # 4. Estrategia y H2H
    h_q_prof = p_h.get("q_profile", {})
    a_q_prof = p_a.get("q_profile", {})
    h_q4_net = float(h_q_prof.get("q4_net_avg", 0.0))
    a_q4_net = float(a_q_prof.get("q4_net_avg", 0.0))
    d_strat_q4_net = h_q4_net - a_q4_net

    h2h_data = match_data.get("h2h") or {}
    h2h_bayes_ft_margin = float(h2h_data.get("ft_margin_avg", 0.0)) if isinstance(h2h_data, dict) else 0.0

    # 5. Mercado, Consenso y Standing Oficial (Features Restauradas)
    odds = match_data.get("odds") or {}
    odds_log_ratio = 0.0
    expected_ft_margin = (d_elo_ft / 25.0)

    if odds and isinstance(odds, dict):
        ho = float(odds.get("home_win") or odds.get("home") or 0.0)
        ao = float(odds.get("away_win") or odds.get("away") or 0.0)
        if ho > 1.0 and ao > 1.0:
            odds_log_ratio = float(np.clip(math.log(ao / ho), -2.5, 2.5))
            expected_ft_margin = float(np.clip(odds_log_ratio * 6.5, -28.0, 28.0))
    elif isinstance(odds, list) and odds:
        has_o, olr = _extract_odds_from_payload(odds)
        if has_o:
            odds_log_ratio = olr
            expected_ft_margin = float(np.clip(odds_log_ratio * 6.5, -28.0, 28.0))

    # Standing oficial
    ts_delta_wp_bayes = 0.0
    ts_data = match_data.get("team_strength")
    if isinstance(ts_data, dict):
        h_ts = ts_data.get("home") or {}
        a_ts = ts_data.get("away") or {}
        hw, hl = int(h_ts.get("wins", 0)), int(h_ts.get("losses", 0))
        aw, al = int(a_ts.get("wins", 0)), int(a_ts.get("losses", 0))
        if (hw + hl) > 0 and (aw + al) > 0:
            h_wp = (hw + 3.0) / (hw + hl + 6.0)
            a_wp = (aw + 3.0) / (aw + al + 6.0)
            ts_delta_wp_bayes = h_wp - a_wp

    # 6. PBP 3-Pointers y Situacional
    pbp_events = pbp_events_upto(match_data.get("pbp") or match_data.get("play_by_play") or [], eff_min)
    h_3pt, a_3pt = 0, 0
    for ev in pbp_events:
        pts = ev.get("points") or ev.get("score_value") or 0
        if pts == 3:
            tm = ev.get("team") or ev.get("side")
            if tm == "home" or tm == ht:
                h_3pt += 1
            else:
                a_3pt += 1
    pbp_3pt_diff_q1_q3 = float(h_3pt - a_3pt)

    rubber_band_gap = 0.0
    if abs_cur_score_diff >= 14.0:
        rubber_band_gap = -cur_score_diff * 0.15

    white_flag_risk_zone = 0.0
    if cur_score_diff >= 22.0:
        white_flag_risk_zone = -1.0
    elif cur_score_diff <= -22.0:
        white_flag_risk_zone = 1.0

    lg_avg_q4_prior = float(lg_avg_q4)

    # Vector exacto de 27 features en orden canónico
    features = [
        cur_score_diff,
        abs_cur_score_diff,
        ht_diff,
        q1_diff,
        q2_diff,
        q3_eff_diff,
        q3_vs_ht_momentum,
        recent_2m_run_diff,
        pace_pts_per_min,
        pace_vs_league_prior,
        gp_val,
        d_lineup_starters_elo_ft,
        d_lineup_starters_elo_q4,
        d_lineup_bench_elo_q4,
        d_lineup_top3_elo_q4,
        d_roster_drift_adv,
        d_strat_q4_net,
        d_q4_bayes,
        d_wr_bayes,
        odds_log_ratio,
        expected_ft_margin,
        h2h_bayes_ft_margin,
        ts_delta_wp_bayes,
        pbp_3pt_diff_q1_q3,
        rubber_band_gap,
        white_flag_risk_zone,
        lg_avg_q4_prior,
    ]

    return np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_m27_v5_2_features(
    match_data: dict[str, Any],
    snapshot_minute: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibilidad con API de tupla (vector, meta)."""
    x = extract_features_v5_2(match_data, target_minute=snapshot_minute or 27.0)
    m_info = match_data.get("match") or {}
    meta = {
        "match_id": m_info.get("id") or match_data.get("id"),
        "league": str(m_info.get("league") or match_data.get("league") or ""),
        "home_team": str(m_info.get("home_team") or match_data.get("home_team") or ""),
        "away_team": str(m_info.get("away_team") or match_data.get("away_team") or ""),
    }
    return x, meta

"""
Extractor de 37 features para el modelo M27 V5 (Roster ELO Dinámico + Perfil Táctico de Cuartos).
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

FEATURE_NAMES_V5 = [
    # Grupo A: Marcador y Momentum In-Game (0..11)
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
    "gp_slope_3m",
    # Grupo B: Roster ELO y Estrategia Táctica (12..23)
    "fraction_q3_completed",
    "d_lineup_starters_elo_ft",
    "d_lineup_starters_elo_q4",
    "d_lineup_bench_elo_q4",
    "d_lineup_top3_elo_q4",
    "d_roster_drift_adv",
    "d_strat_q4_net",
    "q3_tactical_surprise",
    "d_elo_q1",
    "elo_reliability",
    "d_q4_bayes",
    "d_wr_bayes",
    # Grupo C: Mercado, Margen Esperado y H2H (24..29)
    "has_odds",
    "odds_log_ratio",
    "expected_ft_margin",
    "h2h_bayes_ft_margin",
    "h2h_bayes_q4_margin",
    "ts_delta_wp_bayes",
    # Grupo D: Tiros, Fatiga, Situacional y Priors (30..36)
    "pbp_2pt_diff_q1_q3",
    "pbp_3pt_diff_q1_q3",
    "pbp_ft_diff_q1_q3",
    "rubber_band_gap",
    "white_flag_risk_zone",
    "blowout_garbage_time",
    "lg_avg_q4_prior",
]


def get_v5_priors_state() -> dict[str, Any]:
    """Carga en caché el estado de producción de m27_v5 (priors de equipo, jugador y ligas)."""
    global _STATE_CACHE
    if _STATE_CACHE is None:
        state_path = MODEL_DIR / "v5_production_state.pkl"
        if state_path.exists():
            with open(state_path, "rb") as f:
                _STATE_CACHE = pickle.load(f)
        else:
            _STATE_CACHE = {"team_priors": {}, "player_priors": {}, "lc_map": {}}
    return _STATE_CACHE


def resolve_archetype_v5(match_data: dict[str, Any]) -> tuple[str, int, float, float]:
    """Determina si el partido es 10m (FIBA) o 12m (NBA) y priors de anotación."""
    state = get_v5_priors_state()
    lc_map = state.get("lc_map", {})
    m_info = match_data.get("match") or {}
    league = str(m_info.get("league") or match_data.get("league") or "").strip()
    ht = str(m_info.get("home_team") or "")
    at = str(m_info.get("away_team") or "")

    # Chequeo explícito de period_length en payload de SofaScore (720 -> 12m, 600 -> 10m)
    p_len = m_info.get("period_length")
    if p_len == 720:
        return "nba_12m", 12, 218.0, 53.5

    if league in lc_map:
        info = lc_map[league]
        return info["arch"], int(info["q_dur"]), float(info["lg_avg_ft"]), float(info["lg_avg_q4"])

    q_dur = int(round(infer_regulation_quarter_minutes(match_data)))
    is_w = infer_gender(league, ht, at) == "women"
    if q_dur == 12:
        return "nba_12m", 12, 218.0, 53.5
    if is_w:
        return "fiba_women_10m", 10, 138.0, 34.5
    return "fiba_men_10m", 10, 159.0, 39.8


def _extract_odds_from_payload(raw_odds: Any) -> tuple[float, float]:
    """Extrae (has_odds, odds_log_ratio) soportando list[dict] y dict."""
    if isinstance(raw_odds, dict):
        ho = raw_odds.get("home_odd") or raw_odds.get("home_value")
        ao = raw_odds.get("away_odd") or raw_odds.get("away_value")
        if ho and ao:
            try:
                fh, fa = float(ho), float(ao)
                if fh > 1.0 and fa > 1.0:
                    return 1.0, float(np.clip(math.log(fa / fh), -2.5, 2.5))
            except (ValueError, TypeError):
                pass
    elif isinstance(raw_odds, list):
        best_pair = None
        for item in raw_odds:
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


def extract_m27_v5_features(
    match_data: dict[str, Any],
    snapshot_minute: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Extrae el vector de 37 features para inferencia con M27 V5.
    Retorna (X_vector, meta_dict).
    """
    state = get_v5_priors_state()
    team_priors = state.get("team_priors", {})
    player_priors = state.get("player_priors", {})

    arch, q_dur, lg_avg_ft, lg_avg_q4 = resolve_archetype_v5(match_data)
    is_12m = (q_dur == 12)
    q3_end_min = 36.0 if is_12m else 30.0

    if snapshot_minute is None:
        pbp_min = infer_minute_from_pbp(match_data)
        eff_min = pbp_min if pbp_min is not None else (27.0 if not is_12m else 32.0)
    else:
        eff_min = float(snapshot_minute)

    # 1. Marcadores y Momentum
    ch, ca = score_upto(match_data, int(eff_min))

    # Fallback con marcador live si PBP viene retrasado
    live_sc = match_data.get("score") or {}
    if live_sc and isinstance(live_sc, dict):
        lh = live_sc.get("home") or live_sc.get("home_score")
        la = live_sc.get("away") or live_sc.get("away_score")
        if lh is not None and la is not None and (int(lh) + int(la)) > (ch + ca):
            ch, ca = int(lh), int(la)

    cur_diff = float(ch - ca)
    abs_cur_score_diff = abs(cur_diff)

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
    gp_slope_3m = float(gp_dict.get("gp_slope_3m", 0.0))

    q2_end_min = 24.0 if is_12m else 20.0
    fraction_q3_completed = float(np.clip((eff_min - q2_end_min) / float(q_dur), 0.0, 1.0))

    # 2. Priors de Equipo y ELO
    m_info = match_data.get("match") or {}
    ht = str(m_info.get("home_team") or match_data.get("home_team") or "")
    at = str(m_info.get("away_team") or match_data.get("away_team") or "")
    league = str(m_info.get("league") or match_data.get("league") or "")
    country = str(m_info.get("country") or match_data.get("country") or "")
    match_id = str(m_info.get("match_id") or match_data.get("match_id") or "")

    p_h = team_priors.get(ht, {})
    p_a = team_priors.get(at, {})

    d_elo_ft = float(p_h.get("elo_ft", 1500.0) - p_a.get("elo_ft", 1500.0))
    d_elo_q4 = float(p_h.get("elo_q4", 1500.0) - p_a.get("elo_q4", 1500.0))
    d_elo_q1 = float(p_h.get("elo_q1", 1500.0) - p_a.get("elo_q1", 1500.0))
    elo_reliability = 1.0 if (ht in team_priors and at in team_priors) else 0.40
    d_q4_bayes = d_elo_q4 / 35.0
    d_wr_bayes = d_elo_ft / 180.0

    # 3. ROSTER ELO (Titulares, Banca, Top-3 Estrellas)
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
            is_h = bool(p.get("is_home", True) or str(p.get("team") or "").lower() in ("home", "1"))
            is_starter = int(p.get("is_starter", p.get("starter", 0)) or 0)
            pid = str(p.get("player_id") or p.get("id") or p.get("name") or "")
            p_prior = player_priors.get(pid, {})
            p_elo_q4 = p_prior.get("elo_q4", 1500.0)
            p_elo_ft = p_prior.get("elo_ft", 1500.0)

            target_list = (h_st if is_starter else h_bn) if is_h else (a_st if is_starter else a_bn)
            target_list.append((p_elo_ft, p_elo_q4))

        if len(h_st) >= 3 and len(a_st) >= 3:
            h_st_ft = float(np.mean([x[0] for x in h_st]))
            a_st_ft = float(np.mean([x[0] for x in a_st]))
            h_st_q4 = float(np.mean([x[1] for x in h_st]))
            a_st_q4 = float(np.mean([x[1] for x in a_st]))

            d_lineup_starters_elo_ft = h_st_ft - a_st_ft
            d_lineup_starters_elo_q4 = h_st_q4 - a_st_q4
            d_roster_drift_adv = d_lineup_starters_elo_q4 - d_elo_q4

            if h_bn and a_bn:
                d_lineup_bench_elo_q4 = float(np.mean([x[1] for x in h_bn]) - np.mean([x[1] for x in a_bn]))

            # Top-3 estrellas de los titulares
            h_top3 = sorted(h_st, key=lambda x: x[1], reverse=True)[:3]
            a_top3 = sorted(a_st, key=lambda x: x[1], reverse=True)[:3]
            d_lineup_top3_elo_q4 = float(np.mean([x[1] for x in h_top3]) - np.mean([x[1] for x in a_top3]))

    # 4. Perfil Táctico de Cuartos
    h_q4_net = float(p_h.get("q4_net_hist", 0.0))
    a_q4_net = float(p_a.get("q4_net_hist", 0.0))
    d_strat_q4_net = h_q4_net - a_q4_net

    h_q3_net = float(p_h.get("q3_net_hist", 0.0))
    a_q3_net = float(p_a.get("q3_net_hist", 0.0))
    q3_tactical_surprise = q3_diff - (h_q3_net - a_q3_net)

    # 5. Mercado y Momios
    raw_odds = match_data.get("odds") or []
    has_odds, odds_log_ratio = _extract_odds_from_payload(raw_odds)
    if has_odds:
        expected_ft_margin = 6.5 * odds_log_ratio + 0.50 * (d_elo_ft / 28.0)
    else:
        expected_ft_margin = d_elo_ft / 28.0

    # H2H priors
    h2h_list = match_data.get("h2h") or []
    if h2h_list:
        ft_m, q4_m = [], []
        for g in h2h_list[:8]:
            if not isinstance(g, dict):
                continue
            is_h = str(g.get("home_team") or "").strip().lower() == ht.strip().lower()
            m_ft = float(g.get("ft_diff") or (float(g.get("home_score", 0)) - float(g.get("away_score", 0))))
            m_q4 = float(g.get("q4_diff", 0.0))
            ft_m.append(m_ft if is_h else -m_ft)
            q4_m.append(m_q4 if is_h else -m_q4)
        if ft_m:
            shrink = len(ft_m) / (len(ft_m) + 4.0)
            h2h_bayes_ft_margin = float(np.mean(ft_m)) * shrink
            h2h_bayes_q4_margin = float(np.mean(q4_m)) * shrink
        else:
            h2h_bayes_ft_margin, h2h_bayes_q4_margin = 0.0, 0.0
    else:
        h2h_bayes_ft_margin, h2h_bayes_q4_margin = 0.0, 0.0

    # Team Strength
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

    # 6. Tiros PBP
    pbp_events = pbp_events_upto(match_data, q3_end_min)
    h_2pt, a_2pt, h_3pt, a_3pt, h_ft, a_ft = 0, 0, 0, 0, 0, 0
    for ev in pbp_events:
        pts = ev.get("points", 0)
        tm = str(ev.get("team", "")).lower()
        is_h = tm in ("home", "1") or tm == ht.lower()
        if pts == 2:
            if is_h: h_2pt += 1
            else: a_2pt += 1
        elif pts == 3:
            if is_h: h_3pt += 1
            else: a_3pt += 1
        elif pts == 1:
            if is_h: h_ft += 1
            else: a_ft += 1

    pbp_2pt_diff_q1_q3 = float(h_2pt - a_2pt)
    pbp_3pt_diff_q1_q3 = float(h_3pt - a_3pt)
    pbp_ft_diff_q1_q3 = float(h_ft - a_ft)

    # 7. Garbage Time & Situacional
    rubber_band_gap = cur_diff - (expected_ft_margin * 0.75)
    white_flag_threshold = 24.0 if arch == "fiba_women_10m" else (18.0 if not is_12m else 14.0)
    white_flag_risk_zone = 1.0 if abs_cur_score_diff >= white_flag_threshold else 0.0
    blowout_garbage_time = 1.0 if abs_cur_score_diff >= (white_flag_threshold + 6.0) else 0.0
    lg_avg_q4_prior = float(lg_avg_q4)

    raw_features = [
        cur_diff,
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
        gp_slope_3m,
        fraction_q3_completed,
        d_lineup_starters_elo_ft,
        d_lineup_starters_elo_q4,
        d_lineup_bench_elo_q4,
        d_lineup_top3_elo_q4,
        d_roster_drift_adv,
        d_strat_q4_net,
        q3_tactical_surprise,
        d_elo_q1,
        elo_reliability,
        d_q4_bayes,
        d_wr_bayes,
        has_odds,
        odds_log_ratio,
        expected_ft_margin,
        h2h_bayes_ft_margin,
        h2h_bayes_q4_margin,
        ts_delta_wp_bayes,
        pbp_2pt_diff_q1_q3,
        pbp_3pt_diff_q1_q3,
        pbp_ft_diff_q1_q3,
        rubber_band_gap,
        white_flag_risk_zone,
        blowout_garbage_time,
        lg_avg_q4_prior,
    ]

    X = np.nan_to_num(np.array(raw_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    meta = {
        "match_id": match_id,
        "league": league,
        "country": country,
        "home_team": ht,
        "away_team": at,
        "archetype": arch,
        "quarter_duration": q_dur,
        "cur_diff": cur_diff,
        "has_odds": bool(has_odds > 0),
        "d_elo_ft": d_elo_ft,
        "d_lineup_starters_elo_q4": d_lineup_starters_elo_q4,
    }

    return X, meta

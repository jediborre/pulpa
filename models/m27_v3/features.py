"""Extracción de features para el modelo campeón Q4 m27_v3.

Combina features estructurales de mitad de tiempo, métricas del snapshot minuto 27,
momentum de Q3, rachas anotadoras recientes y el historial Head-to-Head (H2H).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from models.common.h2h import compute_h2h_for_match
from models.common.pbp_utils import (
    current_scoring_run,
    graph_stats_upto,
    infer_gender,
    infer_regulation_quarter_minutes,
    margin_bin,
    max_scoring_run,
    pbp_events_upto,
    quarter_points,
    recent_window_features,
    score_upto,
    winner_tag,
)
from models.common.data_loader import compute_team_prior_wr

SNAPSHOT_MINUTE = 27


def _pbp_density_features(match_data: dict[str, Any], cutoff_minute: int) -> dict[str, Any]:
    events = pbp_events_upto(match_data, cutoff_minute)
    n_events = len(events)
    home_pts = sum(int(e.get("points", 0) or 0) for e in events if e.get("team") == "home")
    away_pts = sum(int(e.get("points", 0) or 0) for e in events if e.get("team") == "away")
    return {
        "pbp_events_upto_27": n_events,
        "pbp_scoring_density": round(n_events / float(cutoff_minute), 3) if cutoff_minute else 0.0,
        "pbp_points_per_event_home": round(home_pts / float(n_events), 3) if n_events else 0.0,
        "pbp_points_per_event_away": round(away_pts / float(n_events), 3) if n_events else 0.0,
    }


def extract_features(
    match_data: dict[str, Any],
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Extrae el vector de features completo para inferencia con m27_v3."""
    match_meta = match_data.get("match", {}) or {}
    ht = match_meta.get("home_team", "")
    at = match_meta.get("away_team", "")
    league = match_meta.get("league", "")
    sample_dt = match_meta.get("date", "")
    sample_time = match_meta.get("time", "23:59")

    # 1. Puntos de cuartos previos (Q1, Q2)
    q1h, q1a = quarter_points(match_data, "Q1")
    q2h, q2a = quarter_points(match_data, "Q2")
    q1h = int(q1h or 0)
    q1a = int(q1a or 0)
    q2h = int(q2h or 0)
    q2a = int(q2a or 0)

    ht_home = q1h + q2h
    ht_away = q1a + q2a
    ht_diff = ht_home - ht_away
    ht_total = ht_home + ht_away
    q1_winner = winner_tag(q1h, q1a)
    q2_winner = winner_tag(q2h, q2a)
    halftime_leader = winner_tag(ht_home, ht_away)

    # 2. Marcador acumulado hasta minuto 27
    est_home, est_away = score_upto(match_data, SNAPSHOT_MINUTE)
    est_home = int(est_home)
    est_away = int(est_away)
    score_diff = est_home - est_away
    q3_partial_home = max(0, est_home - ht_home)
    q3_partial_away = max(0, est_away - ht_away)
    q3_partial_diff = q3_partial_home - q3_partial_away
    q3_partial_total = q3_partial_home + q3_partial_away
    q3_partial_leader = winner_tag(q3_partial_home, q3_partial_away)

    current_trailing_side = "tied"
    if score_diff < 0:
        current_trailing_side = "home"
    elif score_diff > 0:
        current_trailing_side = "away"

    halftime_trailing_side = "tied"
    if ht_diff < 0:
        halftime_trailing_side = "home"
    elif ht_diff > 0:
        halftime_trailing_side = "away"

    # 3. Ventanas recientes y gráficas
    recent_3m = recent_window_features(match_data, SNAPSHOT_MINUTE, 3.0, "recent_3m")
    recent_2m = recent_window_features(match_data, SNAPSHOT_MINUTE, 2.0, "recent_2m")
    graph = graph_stats_upto(match_data.get("graph_points", []), SNAPSHOT_MINUTE)

    pbp_events = pbp_events_upto(match_data, SNAPSHOT_MINUTE)
    pbp_density = _pbp_density_features(match_data, SNAPSHOT_MINUTE)
    current_run_home = current_scoring_run(pbp_events, "home")
    current_run_away = current_scoring_run(pbp_events, "away")
    max_run_all_home = max_scoring_run(pbp_events, "home")
    max_run_all_away = max_scoring_run(pbp_events, "away")

    score_halftime_diff_ratio = round(ht_diff / max(ht_total, 1), 3)
    score_q1_share = round((q1h - q1a) / max(abs(ht_diff), 1), 3) if ht_diff != 0 else 0.0
    score_q3_vs_ht_momentum = q3_partial_diff - ht_diff

    trailing_now_recent_run_3m = 0
    trailing_now_recent_run_2m = 0
    if current_trailing_side == "home":
        trailing_now_recent_run_3m = int(recent_3m["recent_3m_points_diff"] > 0)
        trailing_now_recent_run_2m = int(recent_2m["recent_2m_points_diff"] > 0)
    elif current_trailing_side == "away":
        trailing_now_recent_run_3m = int(recent_3m["recent_3m_points_diff"] < 0)
        trailing_now_recent_run_2m = int(recent_2m["recent_2m_points_diff"] < 0)

    halftime_trailer_cutting_in_q3 = 0
    if halftime_trailing_side == "home":
        halftime_trailer_cutting_in_q3 = int(q3_partial_diff > 0)
    elif halftime_trailing_side == "away":
        halftime_trailer_cutting_in_q3 = int(q3_partial_diff < 0)

    qmin = infer_regulation_quarter_minutes(match_data)
    q3_start = qmin * 2.0
    q4_start = qmin * 3.0
    q3_elapsed = max(0.0, min(float(SNAPSHOT_MINUTE), q4_start) - q3_start)

    if q3_elapsed > 0:
        h_rate = q3_partial_home / q3_elapsed
        a_rate = q3_partial_away / q3_elapsed
        diff_rate = h_rate - a_rate
    else:
        h_rate = 0.0
        a_rate = 0.0
        diff_rate = 0.0

    # 4. Prior win rates históricos
    home_prior_wr = compute_team_prior_wr(conn, ht, sample_dt, sample_time, window=12)
    away_prior_wr = compute_team_prior_wr(conn, at, sample_dt, sample_time, window=12)

    features = {
        "gender_bucket": infer_gender(league, ht, at),
        "home_prior_wr": home_prior_wr,
        "away_prior_wr": away_prior_wr,
        "prior_wr_diff": home_prior_wr - away_prior_wr,
        "prior_wr_sum": home_prior_wr + away_prior_wr,
        "q1_diff": q1h - q1a,
        "q2_diff": q2h - q2a,
        "q1_winner": q1_winner,
        "q2_winner": q2_winner,
        "q1_q2_same_winner": int(q1_winner == q2_winner and q1_winner != "tied"),
        "home_wins_first2_count": int(q1_winner == "home") + int(q2_winner == "home"),
        "away_wins_first2_count": int(q1_winner == "away") + int(q2_winner == "away"),
        "halftime_home": ht_home,
        "halftime_away": ht_away,
        "halftime_diff": ht_diff,
        "halftime_total": ht_total,
        "halftime_leader": halftime_leader,
        "halftime_margin_bin": margin_bin(ht_diff),
        "halftime_trailing_side": halftime_trailing_side,
        "score_est_home": est_home,
        "score_est_away": est_away,
        "score_est_diff": score_diff,
        "current_margin_bin": margin_bin(score_diff),
        "current_trailing_side": current_trailing_side,
        "q3_partial_home": q3_partial_home,
        "q3_partial_away": q3_partial_away,
        "q3_partial_diff": q3_partial_diff,
        "q3_partial_total": q3_partial_total,
        "q3_partial_leader": q3_partial_leader,
        "q3_partial_home_share": float(q3_partial_home / q3_partial_total) if q3_partial_total else 0.5,
        "q3_partial_home_rate": round(h_rate, 3),
        "q3_partial_away_rate": round(a_rate, 3),
        "q3_partial_diff_rate": round(diff_rate, 3),
        "halftime_trailer_cutting_in_q3": halftime_trailer_cutting_in_q3,
        "trailing_now_recent_run_3m": trailing_now_recent_run_3m,
        "trailing_now_recent_run_2m": trailing_now_recent_run_2m,
        "trailing_now_is_home": int(current_trailing_side == "home"),
        "trailing_now_is_away": int(current_trailing_side == "away"),
        "trailing_now_deficit_abs": abs(score_diff),
        "halftime_deficit_abs": abs(ht_diff),
        "current_run_home": current_run_home,
        "current_run_away": current_run_away,
        "max_run_all_home": max_run_all_home,
        "max_run_all_away": max_run_all_away,
        "score_halftime_diff_ratio": score_halftime_diff_ratio,
        "score_q1_share": score_q1_share,
        "score_q3_vs_ht_momentum": score_q3_vs_ht_momentum,
    }
    features.update(pbp_density)
    features.update(graph)
    features.update(recent_3m)
    features.update(recent_2m)

    # Eliminar features muertas
    _DEAD: set[str] = {
        "halftime_leader",
        "halftime_trailing_side",
        "q3_partial_leader",
        "trailing_now_is_home",
        "trailing_now_is_away",
        "gp_count",
        "pbp_scoring_density",
    }
    features = {k: v for k, v in features.items() if k not in _DEAD}

    # 5. Features H2H históricas
    h2h_feats = compute_h2h_for_match(conn, ht, at, sample_dt)
    features.update(h2h_feats)

    return features

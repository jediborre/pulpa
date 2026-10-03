"""Simulador Monte Carlo y análisis de presión anotadora (V6.x).

Implementa la simulación de posesiones basada en ratios de tiros triples,
ritmo de juego y probabilidades condicionadas para la familia de modelos V6.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def stable_seed(value: str) -> int:
    """Genera una semilla entera determinista y estable a partir de un string."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def point_probs(points_per_play: float, three_point_share: float) -> np.ndarray:
    """Calcula las probabilidades de anotar 0, 2 o 3 puntos por posesión."""
    p3 = float(np.clip(three_point_share, 0.0, 0.7))
    remaining_ppp = max(0.0, points_per_play - 3.0 * p3)
    p2 = float(np.clip(remaining_ppp / 2.0, 0.0, 1.0 - p3))
    p0 = max(0.0, 1.0 - p2 - p3)
    total = p0 + p2 + p3
    if total <= 0:
        return np.array([1.0, 0.0, 0.0])
    return np.array([p0 / total, p2 / total, p3 / total])


def simulate_team_points(
    rng: np.random.Generator,
    possessions: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Simula los puntos anotados por un equipo a lo largo de un vector de posesiones."""
    p0, p2, p3 = probs
    threes = rng.binomial(possessions, p3)
    remaining = possessions - threes
    two_prob = 0.0 if p0 + p2 <= 0 else p2 / (p0 + p2)
    twos = rng.binomial(remaining, two_prob)
    return 2 * twos + 3 * threes


def monte_carlo_features(
    *,
    match_id: str,
    target: str,
    score_home: int,
    score_away: int,
    pbp_home_plays: int,
    pbp_away_plays: int,
    pbp_home_pts_per_play: float,
    pbp_home_3pt_play_share: float,
    pbp_away_pts_per_play: float,
    pbp_away_3pt_play_share: float,
    elapsed_minutes: float,
    minutes_left: float,
    num_sims: int = 2000,
) -> dict[str, float]:
    """Genera métricas sintéticas mediante simulación Monte Carlo de posesiones."""
    total_plays = pbp_home_plays + pbp_away_plays
    if total_plays <= 0 or elapsed_minutes <= 0 or minutes_left <= 0:
        win_prob = 0.5 if score_home == score_away else float(score_home > score_away)
        return {
            "mc_home_win_prob": win_prob,
            "mc_expected_diff": float(score_home - score_away),
            "mc_cover_rate": 0.0,
            "mc_std_diff": 0.0,
            "mc_comeback_rate": 0.0,
        }

    possessions_left = max(
        1, int(round(_safe_rate(total_plays, elapsed_minutes) * minutes_left))
    )
    home_play_share = float(
        np.clip(_safe_rate(pbp_home_plays, total_plays), 0.05, 0.95)
    )
    home_probs = point_probs(pbp_home_pts_per_play, pbp_home_3pt_play_share)
    away_probs = point_probs(pbp_away_pts_per_play, pbp_away_3pt_play_share)
    rng = np.random.default_rng(stable_seed(f"{match_id}:{target}"))

    home_possessions = rng.binomial(possessions_left, home_play_share, size=num_sims)
    away_possessions = possessions_left - home_possessions
    home_scores = score_home + simulate_team_points(rng, home_possessions, home_probs)
    away_scores = score_away + simulate_team_points(rng, away_possessions, away_probs)

    start_diff = score_home - score_away
    final_diff = home_scores - away_scores
    current_leader = 1 if start_diff > 0 else (-1 if start_diff < 0 else 0)
    final_leader = np.where(final_diff > 0, 1, np.where(final_diff < 0, -1, 0))
    ties = np.abs(final_diff) < 0.5

    if current_leader == 0:
        comeback_rate = 0.0
    else:
        comeback_rate = float(
            np.mean((final_leader != current_leader) & (final_leader != 0))
        )

    return {
        "mc_home_win_prob": float(np.mean(final_diff > 0) + 0.5 * np.mean(ties)),
        "mc_expected_diff": float(np.mean(final_diff)),
        "mc_cover_rate": float(np.mean(final_diff > start_diff)),
        "mc_std_diff": float(np.std(final_diff)),
        "mc_comeback_rate": comeback_rate,
    }


def score_pressure_features(
    *,
    score_home: int,
    score_away: int,
    pbp_home_plays: int,
    pbp_away_plays: int,
    pbp_home_3pt: int,
    pbp_away_3pt: int,
    elapsed_minutes: float,
    minutes_left: float,
) -> dict[str, Any]:
    """Calcula métricas de presión anotadora (urgencia, gap por minuto, puntos para empatar)."""
    diff = score_home - score_away
    abs_diff = abs(diff)

    if diff > 0:
        trailing_side = "away"
        trailing_score = score_away
        trailing_plays = pbp_away_plays
        trailing_3pt = pbp_away_3pt
        leading_score = score_home
    elif diff < 0:
        trailing_side = "home"
        trailing_score = score_home
        trailing_plays = pbp_home_plays
        trailing_3pt = pbp_home_3pt
        leading_score = score_away
    else:
        trailing_side = "tied"
        trailing_score = score_home
        trailing_plays = max(pbp_home_plays, pbp_away_plays)
        trailing_3pt = max(pbp_home_3pt, pbp_away_3pt)
        leading_score = trailing_score

    points_to_tie = abs_diff
    points_to_lead = abs_diff + (0 if trailing_side == "tied" else 1)

    total_plays = pbp_home_plays + pbp_away_plays
    pace_events_per_min = _safe_rate(total_plays, elapsed_minutes)
    trailing_play_share = _safe_rate(trailing_plays, total_plays)
    trailing_plays_per_min = _safe_rate(trailing_plays, elapsed_minutes)

    trailing_points_per_min = _safe_rate(trailing_score, elapsed_minutes)
    leading_points_per_min = _safe_rate(leading_score, elapsed_minutes)
    trailing_points_per_play = _safe_rate(trailing_score, trailing_plays)

    required_ppm_tie = _safe_rate(points_to_tie, minutes_left)
    required_ppm_lead = _safe_rate(points_to_lead, minutes_left)

    exp_total_events_left = pace_events_per_min * minutes_left
    exp_trailing_events_left = exp_total_events_left * trailing_play_share
    req_pts_per_trailing_event = _safe_rate(
        points_to_tie, exp_trailing_events_left
    )

    pressure_ratio_tie = _safe_rate(required_ppm_tie, trailing_points_per_min)
    pressure_ratio_lead = _safe_rate(required_ppm_lead, trailing_points_per_min)
    scoring_gap_per_min = trailing_points_per_min - leading_points_per_min
    urgency_index = _safe_rate(points_to_lead, minutes_left) * (
        1.0 + max(0.0, -scoring_gap_per_min)
    )

    return {
        "global_diff": diff,
        "global_abs_diff": abs_diff,
        "is_tied": int(diff == 0),
        "trailing_is_home": int(trailing_side == "home"),
        "trailing_is_away": int(trailing_side == "away"),
        "trailing_points_to_tie": points_to_tie,
        "trailing_points_to_lead": points_to_lead,
        "remaining_minutes_target": minutes_left,
        "required_ppm_tie": required_ppm_tie,
        "required_ppm_lead": required_ppm_lead,
        "trailing_points_per_min": trailing_points_per_min,
        "leading_points_per_min": leading_points_per_min,
        "trailing_points_per_play": trailing_points_per_play,
        "trailing_play_share": trailing_play_share,
        "trailing_plays_per_min": trailing_plays_per_min,
        "req_pts_per_trailing_event": req_pts_per_trailing_event,
        "pressure_ratio_tie": pressure_ratio_tie,
        "pressure_ratio_lead": pressure_ratio_lead,
        "scoring_gap_per_min": scoring_gap_per_min,
        "urgency_index": urgency_index,
        "trailing_3pt_rate": _safe_rate(trailing_3pt, trailing_plays),
    }

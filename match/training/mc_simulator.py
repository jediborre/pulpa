"""Monte Carlo simulator for Q4 winner prediction.

Simulates the remainder of a game from a given snapshot minute using
PBP-derived scoring rates, then returns P(home_win) as a standalone
probability.  Designed to be blended with ML model predictions.
"""

from __future__ import annotations

import numpy as np

import infer_match as infer_live


def _score_upto(data: dict, cutoff_minute: int) -> tuple[int, int]:
    home, away = infer_live._score_upto(data, cutoff_minute)
    return int(home), int(away)


def _pbp_totals_upto(data: dict, cutoff_minute: int) -> dict:
    """Return total PBP counts and points per team up to cutoff_minute."""
    events = infer_live._pbp_events_upto(data, cutoff_minute)
    home_plays = 0
    away_plays = 0
    home_pts = 0
    away_pts = 0
    for e in events:
        team = e.get("team")
        pts = int(e.get("points", 0) or 0)
        if team == "home":
            home_plays += 1
            home_pts += pts
        elif team == "away":
            away_plays += 1
            away_pts += pts
    return {
        "home_plays": home_plays,
        "away_plays": away_plays,
        "home_pts": home_pts,
        "away_pts": away_pts,
    }


def _infer_regulation_quarter_minutes(match_data: dict) -> float:
    pbp = match_data.get("play_by_play", {}) or {}
    best_clock_seconds = 0
    for quarter_label, plays in pbp.items():
        q_idx = infer_live._quarter_index(str(quarter_label))
        if q_idx is None or q_idx < 1 or q_idx > 4:
            continue
        for play in plays or []:
            rem_sec = infer_live._clock_to_seconds(str(play.get("time", "")))
            if rem_sec is None or rem_sec > (12 * 60):
                continue
            if rem_sec > best_clock_seconds:
                best_clock_seconds = rem_sec
    if best_clock_seconds >= (11 * 60):
        return 12.0
    if best_clock_seconds >= (9 * 60):
        return 10.0
    quarters = (match_data.get("score") or {}).get("quarters") or {}
    n_played_quarters = sum(
        1
        for q in ("Q1", "Q2", "Q3", "Q4")
        if isinstance(quarters.get(q), dict)
        and quarters[q].get("home") is not None
        and quarters[q].get("away") is not None
    )
    gp_minutes = [
        int(point.get("minute", 0))
        for point in (match_data.get("graph_points") or [])
        if point.get("minute") is not None
    ]
    if gp_minutes and n_played_quarters:
        approx_q_minutes = max(gp_minutes) / float(n_played_quarters)
        return min((10.0, 12.0), key=lambda v: abs(v - approx_q_minutes))
    league = str(match_data.get("league") or "")
    nba_leagues = {"NBA", "NBA G League", "WNBA"}
    if league in nba_leagues:
        return 12.0
    return 10.0


def simulate_win_prob(
    match_data: dict,
    snapshot_minute: int,
    num_sims: int = 5000,
    rng: np.random.Generator | None = None,
) -> float:
    """Simulate the remainder of a game from snapshot_minute and return P(home_win).

    Uses a Geometric-Brownian-Motion-inspired model:
        score(t+dt) = score(t) + ppm * dt + noise * sqrt(var * dt)

    where ppm and variance are estimated from PBP data up to snapshot_minute.
    Returns a float in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    home_score, away_score = _score_upto(match_data, snapshot_minute)
    pbp = _pbp_totals_upto(match_data, snapshot_minute)

    reg_q_min = _infer_regulation_quarter_minutes(match_data)
    total_game_min = reg_q_min * 4.0
    minutes_left = max(0.0, total_game_min - float(snapshot_minute))
    elapsed = float(snapshot_minute)

    home_ppm = pbp["home_pts"] / elapsed if elapsed > 0 else 0.0
    away_ppm = pbp["away_pts"] / elapsed if elapsed > 0 else 0.0

    if home_ppm <= 0 and away_ppm <= 0:
        if home_score == away_score:
            return 0.5
        return 1.0 if home_score > away_score else 0.0

    var_home = max(0.01, home_ppm * 1.3)
    var_away = max(0.01, away_ppm * 1.3)

    sim_home = rng.normal(
        loc=home_score + home_ppm * minutes_left,
        scale=np.sqrt(var_home * minutes_left),
        size=num_sims,
    )
    sim_away = rng.normal(
        loc=away_score + away_ppm * minutes_left,
        scale=np.sqrt(var_away * minutes_left),
        size=num_sims,
    )

    home_wins = int(np.sum(sim_home > sim_away))
    ties = int(np.sum(np.abs(sim_home - sim_away) < 0.5))
    return float(home_wins + 0.5 * ties) / float(num_sims)

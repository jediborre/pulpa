"""Extracción de features para el modelo campeón Q3/Q4 v6_2.

Implementa la extracción de features base (F0-F15), presión anotadora y ventanas
recientes con soporte para poda de ligas (league pruning).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from models.common.data_loader import compute_team_prior_wr
from models.common.monte_carlo import score_pressure_features
from models.common.pbp_utils import (
    graph_stats_upto,
    infer_gender,
    quarter_points,
    recent_window_features,
    safe_rate,
)


def _get_top_buckets(conn: sqlite3.Connection, top_leagues: int = 20, top_teams: int = 120):
    league_rows = conn.execute(
        "SELECT league, COUNT(*) AS n FROM matches GROUP BY league ORDER BY n DESC"
    ).fetchall()
    top_leagues_set = {str(row[0]) if row[0] else "" for row in league_rows[:top_leagues]}

    team_counter: dict[str, int] = {}
    rows = conn.execute("SELECT home_team, away_team FROM matches").fetchall()
    for home_team, away_team in rows:
        if home_team:
            team_counter[str(home_team)] = team_counter.get(str(home_team), 0) + 1
        if away_team:
            team_counter[str(away_team)] = team_counter.get(str(away_team), 0) + 1

    sorted_teams = sorted(team_counter.items(), key=lambda kv: kv[1], reverse=True)
    top_teams_set = {team for team, _ in sorted_teams[:top_teams]}
    return top_leagues_set, top_teams_set


def _pbp_stats_upto(pbp: dict[str, list[dict]], quarters: list[str]) -> dict[str, Any]:
    home_plays = 0
    away_plays = 0
    home_3pt = 0
    away_3pt = 0
    home_pts = 0
    away_pts = 0

    for qtr in quarters:
        for play in pbp.get(qtr, []):
            team = play.get("team")
            pts = int(play.get("points", 0) or 0)
            if team == "home":
                home_plays += 1
                home_pts += pts
                if pts == 3:
                    home_3pt += 1
            elif team == "away":
                away_plays += 1
                away_pts += pts
                if pts == 3:
                    away_3pt += 1

    total_plays = home_plays + away_plays
    total_3pt = home_3pt + away_3pt
    return {
        "pbp_home_plays": home_plays,
        "pbp_away_plays": away_plays,
        "pbp_plays_diff": home_plays - away_plays,
        "pbp_home_3pt": home_3pt,
        "pbp_away_3pt": away_3pt,
        "pbp_3pt_diff": home_3pt - away_3pt,
        "pbp_home_plays_share": safe_rate(home_plays, total_plays),
        "pbp_home_3pt_share": safe_rate(home_3pt, total_3pt),
        "pbp_away_3pt_share": safe_rate(away_3pt, total_3pt),
        "pbp_home_pts": home_pts,
        "pbp_away_pts": away_pts,
        "pbp_home_pts_per_play": safe_rate(home_pts, home_plays),
        "pbp_away_pts_per_play": safe_rate(away_pts, away_plays),
        "pbp_pts_per_play_diff": safe_rate(home_pts, home_plays) - safe_rate(away_pts, away_plays),
    }


def extract_features(
    match_data: dict[str, Any],
    conn: sqlite3.Connection,
    target: str = "q4",
) -> dict[str, Any]:
    """Extrae el diccionario de features requerido por v6_2 para Q3 o Q4."""
    m = match_data.get("match", {}) or {}
    pbp = match_data.get("play_by_play", {}) or {}
    gp = match_data.get("graph_points", []) or []

    q1h, q1a = quarter_points(match_data, "Q1")
    q2h, q2a = quarter_points(match_data, "Q2")
    q3h, q3a = quarter_points(match_data, "Q3")

    home_team = m.get("home_team", "")
    away_team = m.get("away_team", "")
    league = m.get("league", "")
    match_date = m.get("date", "")
    match_time = m.get("time", "23:59")

    home_prior_wr = compute_team_prior_wr(conn, home_team, match_date, match_time, window=12)
    away_prior_wr = compute_team_prior_wr(conn, away_team, match_date, match_time, window=12)

    top_leagues, top_teams = _get_top_buckets(conn)

    def bucket(value: str, top_set: set[str], prefix: str) -> str:
        return value if value in top_set else f"{prefix}_OTHER"

    base = {
        "league": league,
        "gender_bucket": infer_gender(league, home_team, away_team),
        "home_prior_wr": home_prior_wr,
        "away_prior_wr": away_prior_wr,
        "prior_wr_diff": home_prior_wr - away_prior_wr,
        "prior_wr_sum": home_prior_wr + away_prior_wr,
        "q1_diff": (q1h or 0) - (q1a or 0),
        "q2_diff": (q2h or 0) - (q2a or 0),
        "league_bucket": bucket(league, top_leagues, "LEAGUE"),
        "home_team_bucket": bucket(home_team, top_teams, "TEAM"),
        "away_team_bucket": bucket(away_team, top_teams, "TEAM"),
    }

    ht_home = (q1h or 0) + (q2h or 0)
    ht_away = (q1a or 0) + (q2a or 0)

    if target.lower() == "q3":
        feat = dict(base)
        feat.update({
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_diff": ht_home - ht_away,
            "ht_total": ht_home + ht_away,
        })
        feat.update(graph_stats_upto(gp, 24))
        q3_pbp = _pbp_stats_upto(pbp, ["Q1", "Q2"])
        feat.update(q3_pbp)
        feat.update(
            score_pressure_features(
                score_home=ht_home,
                score_away=ht_away,
                pbp_home_plays=q3_pbp["pbp_home_plays"],
                pbp_away_plays=q3_pbp["pbp_away_plays"],
                pbp_home_3pt=q3_pbp["pbp_home_3pt"],
                pbp_away_3pt=q3_pbp["pbp_away_3pt"],
                elapsed_minutes=24.0,
                minutes_left=12.0,
            )
        )
        recent_24 = recent_window_features(match_data, cutoff_minute=24.0, window_minutes=6.0)
        # Adaptar nombres de clutch a formato v6_2
        feat.update({
            "clutch_window_minutes": 6.0,
            "clutch_scoring_events": recent_24["recent_home_points"] + recent_24["recent_away_points"],
            "clutch_home_points": recent_24["recent_home_points"],
            "clutch_away_points": recent_24["recent_away_points"],
            "clutch_points_diff": recent_24["recent_points_diff"],
            "clutch_home_event_share": recent_24["recent_home_event_share"],
            "clutch_home_max_run_pts": recent_24["recent_home_max_run"],
            "clutch_away_max_run_pts": recent_24["recent_away_max_run"],
            "clutch_run_diff": recent_24["recent_run_diff"],
            "clutch_last_scoring_home": recent_24["recent_last_scoring_home"],
            "clutch_last_scoring_away": recent_24["recent_last_scoring_away"],
        })
        return feat

    # Para Q4:
    feat = dict(base)
    feat.update({
        "q3_diff": (q3h or 0) - (q3a or 0),
        "q3_total": (q3h or 0) + (q3a or 0),
        "score_3q_home": ht_home + (q3h or 0),
        "score_3q_away": ht_away + (q3a or 0),
        "score_3q_diff": (ht_home + (q3h or 0)) - (ht_away + (q3a or 0)),
    })
    feat.update(graph_stats_upto(gp, 36))
    q4_pbp = _pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"])
    feat.update(q4_pbp)
    feat.update(
        score_pressure_features(
            score_home=ht_home + (q3h or 0),
            score_away=ht_away + (q3a or 0),
            pbp_home_plays=q4_pbp["pbp_home_plays"],
            pbp_away_plays=q4_pbp["pbp_away_plays"],
            pbp_home_3pt=q4_pbp["pbp_home_3pt"],
            pbp_away_3pt=q4_pbp["pbp_away_3pt"],
            elapsed_minutes=36.0,
            minutes_left=12.0,
        )
    )
    recent_36 = recent_window_features(match_data, cutoff_minute=36.0, window_minutes=6.0)
    feat.update({
        "clutch_window_minutes": 6.0,
        "clutch_scoring_events": recent_36["recent_home_points"] + recent_36["recent_away_points"],
        "clutch_home_points": recent_36["recent_home_points"],
        "clutch_away_points": recent_36["recent_away_points"],
        "clutch_points_diff": recent_36["recent_points_diff"],
        "clutch_home_event_share": recent_36["recent_home_event_share"],
        "clutch_home_max_run_pts": recent_36["recent_home_max_run"],
        "clutch_away_max_run_pts": recent_36["recent_away_max_run"],
        "clutch_run_diff": recent_36["recent_run_diff"],
        "clutch_last_scoring_home": recent_36["recent_last_scoring_home"],
        "clutch_last_scoring_away": recent_36["recent_last_scoring_away"],
    })
    return feat

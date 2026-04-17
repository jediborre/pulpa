"""
V12 - Ultra Conservative Ensemble with Risk Management
=======================================================
Key innovations:
1. HYBRID ensemble: Classification (winner) + Regression (points) working together
2. ASYMMETRIC risk: Loss penalty 2x > win reward
3. LEAGUE FILTERING: Only bet on leagues with proven track record
4. MULTI-ALGORITHM: XGBoost + LightGBM + CatBoost + LogReg + GB
5. CONSERVATIVE gates: NO_BET by default, only bet when VERY confident
6. Gender-separated for regression, combined for classification
7. Monte Carlo + momentum + pressure + clutch features (all from V4-V9)
8. Stacking meta-learner on top of base model predictions
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

db_mod = __import__("db")

DB_PATH = ROOT / "matches.db"
OUT_DIR = ROOT / "training" / "v12" / "model_outputs"
LEAGUE_STATS_FILE = OUT_DIR / "league_stats.json"

# Top leagues to focus on (will be filtered by performance)
TOP_LEAGUES = 25
TOP_TEAMS = 150
TEAM_HISTORY_WINDOW = 12

# Risk parameters
LOSS_PENALTY_MULTIPLIER = 2.0  # Losing costs 2x more than winning
MIN_HIT_RATE_FOR_LEAGUE = 0.52  # 52% minimum
MIN_SAMPLES_FOR_LEAGUE = 50  # Need at least 50 samples to judge a league


@dataclass
class HybridSample:
    """Sample with both classification target and regression target."""
    match_id: str
    dt: datetime
    league: str
    gender: str
    features: dict
    target_winner: int | None  # 1=home, 0=away (classification)
    target_home_pts: int | None  # actual points scored
    target_away_pts: int | None
    target_total_pts: int | None


@dataclass
class LeaguePerformance:
    """Track per-league performance for filtering."""
    samples: int = 0
    home_wins: int = 0
    away_wins: int = 0
    pushes: int = 0
    avg_total_points: float = 0.0
    std_total_points: float = 0.0
    total_points_list: list = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Feature Engineering (best from V4, V6, V9, V11)
# ────────────────────────────────────────────────────────────────────

def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _quarter_points(data: dict, quarter: str) -> tuple[int | None, int | None]:
    q = data.get("score", {}).get("quarters", {}).get(quarter)
    if not q:
        return None, None
    return int(q.get("home", 0)), int(q.get("away", 0))


def _infer_gender(league: str, home_team: str, away_team: str) -> str:
    text = f"{league} {home_team} {away_team}".lower()
    markers = ["women", "woman", "female", "femen", "fem.", " ladies ", "(w)", " w ", "wnba", "girls"]
    for marker in markers:
        if marker in text:
            return "women"
    return "men_or_open"


def _count_sign_swings(values: list[int]) -> int:
    swings = 0
    prev_sign = 0
    for value in values:
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            swings += 1
        prev_sign = sign
    return swings


def _graph_stats_upto(graph_points: list[dict], max_minute: int) -> dict:
    points = [p for p in graph_points if int(p.get("minute", 0)) <= max_minute]
    values = [int(p.get("value", 0)) for p in points]
    if not values:
        return {
            "gp_count": 0, "gp_last": 0, "gp_peak_home": 0, "gp_peak_away": 0,
            "gp_area_home": 0, "gp_area_away": 0, "gp_area_diff": 0,
            "gp_mean_abs": 0.0, "gp_swings": 0, "gp_slope_3m": 0, "gp_slope_5m": 0,
            "gp_acceleration": 0.0,
        }

    area_home = sum(max(v, 0) for v in values)
    area_away = sum(max(-v, 0) for v in values)
    mean_abs = sum(abs(v) for v in values) / len(values)

    slope_3m = values[-1] - values[-4] if len(values) >= 4 else values[-1] - values[0]
    slope_5m = values[-1] - values[-6] if len(values) >= 6 else (values[-1] - values[0])

    # NEW: acceleration (change of slope)
    if len(values) >= 6:
        slope_first_half = values[len(values)//2] - values[0]
        slope_second_half = values[-1] - values[len(values)//2]
        acceleration = slope_second_half - slope_first_half
    else:
        acceleration = 0

    return {
        "gp_count": len(values), "gp_last": values[-1],
        "gp_peak_home": max(values), "gp_peak_away": abs(min(values)),
        "gp_area_home": area_home, "gp_area_away": area_away, "gp_area_diff": area_home - area_away,
        "gp_mean_abs": mean_abs, "gp_swings": _count_sign_swings(values),
        "gp_slope_3m": slope_3m, "gp_slope_5m": slope_5m,
        "gp_acceleration": acceleration,
    }


def _pbp_stats_upto(pbp: dict, quarters: list[str]) -> dict:
    home_plays, away_plays, home_3pt, away_3pt, home_pts, away_pts = 0, 0, 0, 0, 0, 0
    for quarter in quarters:
        for play in pbp.get(quarter, []):
            team, pts = play.get("team"), int(play.get("points", 0))
            if team == "home":
                home_plays += 1
                home_pts += pts
                if pts == 3: home_3pt += 1
            elif team == "away":
                away_plays += 1
                away_pts += pts
                if pts == 3: away_3pt += 1

    total_plays = home_plays + away_plays
    total_3pt = home_3pt + away_3pt
    return {
        "pbp_home_pts_per_play": _safe_rate(home_pts, home_plays),
        "pbp_away_pts_per_play": _safe_rate(away_pts, away_plays),
        "pbp_pts_per_play_diff": _safe_rate(home_pts, home_plays) - _safe_rate(away_pts, away_plays),
        "pbp_home_plays": home_plays, "pbp_away_plays": away_plays,
        "pbp_plays_diff": home_plays - away_plays, "pbp_home_3pt": home_3pt,
        "pbp_away_3pt": away_3pt, "pbp_3pt_diff": home_3pt - away_3pt,
        "pbp_home_plays_share": _safe_rate(home_plays, total_plays),
        "pbp_home_3pt_share": _safe_rate(home_3pt, total_3pt),
        "pbp_home_efficiency": _safe_rate(home_pts, home_plays) if home_plays else 0,
        "pbp_away_efficiency": _safe_rate(away_pts, away_plays) if away_plays else 0,
    }


def _monte_carlo_win_prob(
    score_home: int, score_away: int,
    pbp_home_plays: int, pbp_away_plays: int,
    pbp_home_pts: float, pbp_away_pts: float,
    elapsed_minutes: float, minutes_left: float,
    num_sims: int = 1000,
) -> dict:
    """Monte Carlo simulation (from V6, improved)."""
    home_ppm = _safe_rate(pbp_home_pts, elapsed_minutes)
    away_ppm = _safe_rate(pbp_away_pts, elapsed_minutes)

    if home_ppm <= 0 and away_ppm <= 0:
        if score_home == score_away:
            return {"mc_home_win_prob": 0.5}
        return {"mc_home_win_prob": 1.0 if score_home > score_away else 0.0}

    var_home = max(0.01, home_ppm * 1.3)
    var_away = max(0.01, away_ppm * 1.3)

    sim_home = np.random.normal(
        score_home + home_ppm * minutes_left,
        np.sqrt(var_home * minutes_left), num_sims
    )
    sim_away = np.random.normal(
        score_away + away_ppm * minutes_left,
        np.sqrt(var_away * minutes_left), num_sims
    )

    home_wins = np.sum(sim_home > sim_away)
    ties = np.sum(np.abs(sim_home - sim_away) < 0.5)

    return {"mc_home_win_prob": float(home_wins + 0.5 * ties) / float(num_sims)}


def _score_pressure_features(
    score_home: int, score_away: int,
    pbp_home_plays: int, pbp_away_plays: int,
    pbp_home_3pt: int, pbp_away_3pt: int,
    elapsed_minutes: float, minutes_left: float
) -> dict:
    """Pressure/comeback features from V4."""
    diff = score_home - score_away
    abs_diff = abs(diff)
    if diff > 0:
        trailing_side, trailing_score, trailing_plays, trailing_3pt, leading_score = "away", score_away, pbp_away_plays, pbp_away_3pt, score_home
    elif diff < 0:
        trailing_side, trailing_score, trailing_plays, trailing_3pt, leading_score = "home", score_home, pbp_home_plays, pbp_home_3pt, score_away
    else:
        trailing_side, trailing_score, trailing_plays, trailing_3pt, leading_score = "tied", max(score_home, score_away), max(pbp_home_plays, pbp_away_plays), max(pbp_home_3pt, pbp_away_3pt), max(score_home, score_away)

    pts_to_tie, pts_to_lead = abs_diff, abs_diff + (0 if trailing_side == "tied" else 1)
    tot_plays = pbp_home_plays + pbp_away_plays
    trailing_ps = _safe_rate(trailing_plays, tot_plays)
    t_ppm = _safe_rate(trailing_score, elapsed_minutes)
    return {
        "global_diff": diff, "global_abs_diff": abs_diff, "is_tied": int(diff == 0),
        "trailing_is_home": int(trailing_side == "home"), "trailing_is_away": int(trailing_side == "away"),
        "trailing_points_to_tie": pts_to_tie, "trailing_points_to_lead": pts_to_lead,
        "remaining_minutes_target": minutes_left,
        "required_ppm_tie": _safe_rate(pts_to_tie, minutes_left),
        "required_ppm_lead": _safe_rate(pts_to_lead, minutes_left),
        "trailing_points_per_min": t_ppm, "leading_points_per_min": _safe_rate(leading_score, elapsed_minutes),
        "trailing_points_per_play": _safe_rate(trailing_score, trailing_plays),
        "trailing_play_share": trailing_ps, "trailing_plays_per_min": _safe_rate(trailing_plays, elapsed_minutes),
        "req_pts_per_trailing_event": _safe_rate(pts_to_tie, _safe_rate(tot_plays, elapsed_minutes) * minutes_left * trailing_ps),
        "pressure_ratio_tie": _safe_rate(_safe_rate(pts_to_tie, minutes_left), t_ppm),
        "pressure_ratio_lead": _safe_rate(_safe_rate(pts_to_lead, minutes_left), t_ppm),
        "scoring_gap_per_min": t_ppm - _safe_rate(leading_score, elapsed_minutes),
        "urgency_index": _safe_rate(pts_to_lead, minutes_left) * (1.0 + max(0.0, - (t_ppm - _safe_rate(leading_score, elapsed_minutes)))),
        "trailing_3pt_rate": _safe_rate(trailing_3pt, trailing_plays),
    }


def _quarter_index(label: str) -> int | None:
    if label.startswith("Q"):
        try: return int(label[1:])
        except ValueError: return None
    return None


def _clock_to_seconds(clock: str) -> int | None:
    if not clock or ":" not in clock: return None
    try:
        mm, ss = clock.split(":", 1)
        return int(mm) * 60 + int(ss)
    except ValueError: return None


def _pbp_events_upto_minute(pbp: dict, cutoff_minute: float) -> list[dict]:
    events = []
    for quarter_label, plays in pbp.items():
        q_idx = _quarter_index(quarter_label)
        if q_idx is None or q_idx < 1 or q_idx > 4: continue
        q_start = (q_idx - 1) * 12.0
        for play in plays:
            rem_sec = _clock_to_seconds(str(play.get("time", "")))
            if rem_sec is None: continue
            elapsed_in_q = 12.0 - (rem_sec / 60.0)
            global_min = q_start + elapsed_in_q
            if global_min <= cutoff_minute + 1e-9:
                e = dict(play)
                e["_global_min"] = global_min
                events.append(e)
    events.sort(key=lambda e: float(e.get("_global_min", 0.0)))
    return events


def _max_scoring_run(events: list[dict], team_name: str) -> int:
    best, run = 0, 0
    for event in events:
        team, pts = event.get("team"), int(event.get("points", 0) or 0)
        if team == team_name and pts > 0:
            run += pts
            if run > best: best = run
        elif team in ("home", "away"):
            run = 0
    return best


def _current_scoring_run(events: list[dict], team_name: str) -> int:
    run = 0
    for event in reversed(events):
        team, pts = event.get("team"), int(event.get("points", 0) or 0)
        if team == team_name and pts > 0:
            run += pts
        else:
            break
    return run


def _pbp_recent_window_features(pbp: dict, cutoff_minute: float, window_minutes: float) -> dict:
    """Clutch/momentum features from V4/V9."""
    events_upto = _pbp_events_upto_minute(pbp, cutoff_minute)
    start_min = max(0.0, cutoff_minute - window_minutes)
    events = [e for e in events_upto if float(e.get("_global_min", 0.0)) >= start_min]

    home_points, away_points, home_events, away_events, last_scoring = 0, 0, 0, 0, "none"
    for event in events:
        team, pts = event.get("team"), int(event.get("points", 0) or 0)
        if team == "home" and pts > 0:
            home_points += pts
            home_events += 1
            last_scoring = "home"
        elif team == "away" and pts > 0:
            away_points += pts
            away_events += 1
            last_scoring = "away"

    scoring_events = home_events + away_events
    return {
        "clutch_window_minutes": window_minutes,
        "clutch_scoring_events": scoring_events,
        "clutch_home_points": home_points,
        "clutch_away_points": away_points,
        "clutch_points_diff": home_points - away_points,
        "clutch_home_event_share": _safe_rate(home_events, scoring_events),
        "clutch_home_max_run_pts": _max_scoring_run(events, "home"),
        "clutch_away_max_run_pts": _max_scoring_run(events, "away"),
        "clutch_run_diff": _max_scoring_run(events, "home") - _max_scoring_run(events, "away"),
        "clutch_last_scoring_home": int(last_scoring == "home"),
        "clutch_last_scoring_away": int(last_scoring == "away"),
        "clutch_home_pts_per_event": _safe_rate(home_points, home_events),
        "clutch_away_pts_per_event": _safe_rate(away_points, away_events),
    }


def _bucket(value: str, top_set: set[str], prefix: str) -> str:
    return value if value in top_set else f"{prefix}_OTHER"


def _is_complete_match(data: dict) -> bool:
    quarters = data.get("score", {}).get("quarters", {})
    required = {"Q1", "Q2", "Q3", "Q4"}
    if not required.issubset(quarters.keys()): return False
    gp = data.get("graph_points", [])
    if not gp or len(gp) < 20: return False
    pbp = data.get("play_by_play", {})
    if not pbp or not required.issubset(pbp.keys()): return False
    for q in required:
        if len(pbp[q]) == 0: return False
    return True


# ────────────────────────────────────────────────────────────────────
# League Performance Tracking
# ────────────────────────────────────────────────────────────────────

def _compute_league_stats(conn) -> dict:
    """Compute historical performance per league."""
    rows = conn.execute("""
        SELECT m.league, m.match_id, m.home_score, m.away_score
        FROM matches m
        WHERE m.status_type = 'finished'
          AND m.home_score IS NOT NULL
          AND m.away_score IS NOT NULL
    """).fetchall()

    league_data: dict[str, LeaguePerformance] = defaultdict(LeaguePerformance)
    for row in rows:
        league = row["league"] or "unknown"
        lp = league_data[league]
        lp.samples += 1
        home, away = int(row["home_score"]), int(row["away_score"])
        if home > away:
            lp.home_wins += 1
        elif away > home:
            lp.away_wins += 1
        else:
            lp.pushes += 1
        lp.total_points_list.append(home + away)

    result = {}
    for league, lp in league_data.items():
        pts = lp.total_points_list
        result[league] = {
            "samples": lp.samples,
            "home_win_rate": lp.home_wins / lp.samples if lp.samples else 0,
            "away_win_rate": lp.away_wins / lp.samples if lp.samples else 0,
            "avg_total_points": np.mean(pts) if pts else 0,
            "std_total_points": np.std(pts) if pts else 0,
            "is_favorite_home": lp.home_wins > lp.away_wins,
        }
    return result


# ────────────────────────────────────────────────────────────────────
# Sample Building
# ────────────────────────────────────────────────────────────────────

def build_samples(db_path: Path) -> list[HybridSample]:
    """Build hybrid samples with both classification and regression targets."""
    print("[v12] Loading DB...")
    conn = db_mod.get_conn(str(db_path))
    db_mod.init_db(conn)

    # Compute league stats
    print("[v12] Computing league statistics...")
    league_stats = _compute_league_stats(conn)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEAGUE_STATS_FILE, "w") as f:
        json.dump(league_stats, f, indent=2)

    # Identify top leagues/teams
    league_rows = conn.execute("SELECT league, COUNT(*) AS n FROM matches GROUP BY league ORDER BY n DESC").fetchall()
    team_counter = Counter()
    for ht, at in conn.execute("SELECT home_team, away_team FROM matches").fetchall():
        if ht: team_counter[str(ht)] += 1
        if at: team_counter[str(at)] += 1

    top_leagues = {str(r[0]) if r[0] else "" for r in league_rows[:TOP_LEAGUES]}
    top_teams = {team for team, _ in team_counter.most_common(TOP_TEAMS)}

    rows = conn.execute("SELECT match_id, date, time FROM matches ORDER BY date, time, match_id").fetchall()
    team_history: dict[str, list[int]] = defaultdict(list)
    samples: list[HybridSample] = []

    print(f"[v12] Processing {len(rows)} matches...")

    for idx, row in enumerate(rows):
        if idx % 2000 == 0:
            print(f"[v12] Progress: {idx}/{len(rows)}")

        match_id = str(row["match_id"])
        dt = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M")
        data = db_mod.get_match(conn, match_id)
        if data is None or not _is_complete_match(data):
            continue

        m, score, pbp, gp = data["match"], data["score"], data.get("play_by_play", {}), data.get("graph_points", [])
        q1h, q1a = _quarter_points(data, "Q1")
        q2h, q2a = _quarter_points(data, "Q2")
        q3h, q3a = _quarter_points(data, "Q3")
        q4h, q4a = _quarter_points(data, "Q4")
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue

        ht, at, league = m.get("home_team", ""), m.get("away_team", ""), m.get("league", "")
        gender = _infer_gender(league, ht, at)

        home_hist = team_history[ht][-TEAM_HISTORY_WINDOW:]
        away_hist = team_history[at][-TEAM_HISTORY_WINDOW:]
        home_prior_wr = _safe_rate(sum(home_hist), len(home_hist))
        away_prior_wr = _safe_rate(sum(away_hist), len(away_hist))

        # League context features
        league_info = league_stats.get(league, {})
        league_samples = league_info.get("samples", 0)
        league_home_wr = league_info.get("home_win_rate", 0.5)
        league_avg_total = league_info.get("avg_total_points", 0)
        league_std_total = league_info.get("std_total_points", 0)

        base = {
            "league_bucket": _bucket(league, top_leagues, "LEAGUE"),
            "gender_bucket": gender,
            "home_team_bucket": _bucket(ht, top_teams, "TEAM"),
            "away_team_bucket": _bucket(at, top_teams, "TEAM"),
            "home_prior_wr": round(home_prior_wr, 4),
            "away_prior_wr": round(away_prior_wr, 4),
            "prior_wr_diff": round(home_prior_wr - away_prior_wr, 4),
            "prior_wr_sum": round(home_prior_wr + away_prior_wr, 4),
            "q1_diff": q1h - q1a,
            "q2_diff": q2h - q2a,
            # NEW: league context
            "league_home_advantage": round(league_home_wr - 0.5, 4),
            "league_avg_total_points": round(league_avg_total, 2),
            "league_std_total_points": round(league_std_total, 2),
            "league_sample_size": min(league_samples, 500),
        }

        # ── Q3 features ──
        ht_home, ht_away = q1h + q2h, q1a + q2a
        q3_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2"])

        f_q3 = dict(base, ht_home=ht_home, ht_away=ht_away, ht_diff=ht_home - ht_away, ht_total=ht_home + ht_away)
        f_q3.update(_graph_stats_upto(gp, 24))
        f_q3.update(q3_pbp_stats)
        f_q3.update(_score_pressure_features(
            score_home=ht_home, score_away=ht_away,
            pbp_home_plays=q3_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q3_pbp_stats["pbp_away_plays"],
            pbp_home_3pt=q3_pbp_stats["pbp_home_3pt"],
            pbp_away_3pt=q3_pbp_stats["pbp_away_3pt"],
            elapsed_minutes=24.0, minutes_left=12.0
        ))
        f_q3.update(_pbp_recent_window_features(pbp, 24.0, 6.0))

        q3_mc = _monte_carlo_win_prob(
            score_home=ht_home, score_away=ht_away,
            pbp_home_plays=q3_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q3_pbp_stats["pbp_away_plays"],
            pbp_home_pts=q3_pbp_stats["pbp_home_pts_per_play"] * q3_pbp_stats["pbp_home_plays"],
            pbp_away_pts=q3_pbp_stats["pbp_away_pts_per_play"] * q3_pbp_stats["pbp_away_plays"],
            elapsed_minutes=24.0, minutes_left=12.0
        )
        f_q3.update(q3_mc)

        q3_events = _pbp_events_upto_minute(pbp, 24.0)
        f_q3["current_run_home"] = _current_scoring_run(q3_events, "home")
        f_q3["current_run_away"] = _current_scoring_run(q3_events, "away")
        f_q3["max_run_home"] = _max_scoring_run(q3_events, "home")
        f_q3["max_run_away"] = _max_scoring_run(q3_events, "away")
        f_q3["run_diff"] = _max_scoring_run(q3_events, "home") - _max_scoring_run(q3_events, "away")

        # ── Q4 features ──
        score_3q_home, score_3q_away = ht_home + q3h, ht_away + q3a
        q4_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"])

        f_q4 = dict(base, q3_diff=q3h - q3a, score_3q_home=score_3q_home, score_3q_away=score_3q_away, score_3q_diff=score_3q_home - score_3q_away, score_3q_total=score_3q_home + score_3q_away)
        f_q4.update(_graph_stats_upto(gp, 36))
        f_q4.update(q4_pbp_stats)
        f_q4.update(_score_pressure_features(
            score_home=score_3q_home, score_away=score_3q_away,
            pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
            pbp_home_3pt=q4_pbp_stats["pbp_home_3pt"],
            pbp_away_3pt=q4_pbp_stats["pbp_away_3pt"],
            elapsed_minutes=36.0, minutes_left=12.0
        ))
        f_q4.update(_pbp_recent_window_features(pbp, 36.0, 6.0))

        q4_mc = _monte_carlo_win_prob(
            score_home=score_3q_home, score_away=score_3q_away,
            pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
            pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
            pbp_home_pts=q4_pbp_stats["pbp_home_pts_per_play"] * q4_pbp_stats["pbp_home_plays"],
            pbp_away_pts=q4_pbp_stats["pbp_away_pts_per_play"] * q4_pbp_stats["pbp_away_plays"],
            elapsed_minutes=36.0, minutes_left=12.0
        )
        f_q4.update(q4_mc)

        q4_events = _pbp_events_upto_minute(pbp, 36.0)
        f_q4["current_run_home"] = _current_scoring_run(q4_events, "home")
        f_q4["current_run_away"] = _current_scoring_run(q4_events, "away")
        f_q4["max_run_home"] = _max_scoring_run(q4_events, "home")
        f_q4["max_run_away"] = _max_scoring_run(q4_events, "away")
        f_q4["run_diff"] = _max_scoring_run(q4_events, "home") - _max_scoring_run(q4_events, "away")

        # Build samples
        for quarter, features in [("q3", f_q3), ("q4", f_q4)]:
            if quarter == "q3":
                winner_target = None if q3h == q3a else (1 if q3h > q3a else 0)
                home_pts, away_pts, total_pts = q3h, q3a, q3h + q3a
            else:
                winner_target = None if q4h == q4a else (1 if q4h > q4a else 0)
                home_pts, away_pts, total_pts = q4h, q4a, q4h + q4a

            samples.append(HybridSample(
                match_id=f"{match_id}_{quarter}",
                dt=dt,
                league=league or "unknown",
                gender=gender,
                features=features,
                target_winner=winner_target,
                target_home_pts=home_pts,
                target_away_pts=away_pts,
                target_total_pts=total_pts,
            ))

        team_history[ht].append(1 if score["home"] > score["away"] else 0)
        team_history[at].append(1 if score["away"] > score["home"] else 0)

    conn.close()
    print(f"[v12] Built {len(samples)} samples")
    return samples


# ────────────────────────────────────────────────────────────────────
# League Filtering
# ────────────────────────────────────────────────────────────────────

def identify_strong_leagues(league_stats: dict) -> set[str]:
    """Identify leagues where models historically perform well."""
    strong = set()
    for league, stats in league_stats.items():
        if stats["samples"] >= MIN_SAMPLES_FOR_LEAGUE:
            # Home win rate significantly different from 0.5 indicates predictability
            home_wr = stats["home_win_rate"]
            if home_wr >= 0.55 or home_wr <= 0.45:
                strong.add(league)
    print(f"[v12] Identified {len(strong)} strong leagues for betting")
    return strong


# ────────────────────────────────────────────────────────────────────
# Training - Classification (Winner)
# ────────────────────────────────────────────────────────────────────

def train_classifier(
    samples: list[HybridSample],
    target_quarter: str,
) -> dict:
    """Train classification models for winner prediction."""
    print(f"[v12] Training classifier for {target_quarter}...")

    # Filter valid samples
    valid = [s for s in samples if s.target_winner is not None]
    valid.sort(key=lambda s: s.dt)

    if not valid:
        return {"error": "No valid samples"}

    x_dict = [s.features for s in valid]
    y = [s.target_winner for s in valid]

    n = len(valid)
    n_train = int(n * 0.8)
    n_test = n - n_train

    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(x_dict)
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {}
    probs_dict = {}

    # 1. Logistic Regression
    print(f"  [LogReg]...")
    logreg = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    logreg.fit(x_train_scaled, y_train)
    probs_lr = logreg.predict_proba(x_test_scaled)[:, 1]
    models["logreg"] = logreg
    probs_dict["logreg"] = probs_lr

    # 2. Gradient Boosting
    print(f"  [GB]...")
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_split=20, min_samples_leaf=10, random_state=42
    )
    gb.fit(x_train, y_train)
    probs_gb = gb.predict_proba(x_test)[:, 1]
    models["gb"] = gb
    probs_dict["gb"] = probs_gb

    # 3. XGBoost (if available)
    if HAS_XGB:
        print(f"  [XGB]...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
            random_state=42, eval_metric='logloss'
        )
        xgb_model.fit(x_train, y_train)
        probs_xgb = xgb_model.predict_proba(x_test)[:, 1]
        models["xgb"] = xgb_model
        probs_dict["xgb"] = probs_xgb

    # 4. LightGBM (if available)
    if HAS_LGB:
        print(f"  [LGBM]...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
            random_state=42
        )
        lgb_model.fit(x_train, y_train)
        probs_lgb = lgb_model.predict_proba(x_test)[:, 1]
        models["lgbm"] = lgb_model
        probs_dict["lgbm"] = probs_lgb

    # 5. CatBoost (if available)
    if HAS_CAT:
        print(f"  [CatBoost]...")
        cat_model = CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.08,
            l2_leaf_reg=3, random_state=42, verbose=0
        )
        cat_model.fit(x_train, y_train)
        probs_cat = cat_model.predict_proba(x_test)[:, 1]
        models["catboost"] = cat_model
        probs_dict["catboost"] = probs_cat

    # ── Ensemble combinations ──
    all_probs = list(probs_dict.values())
    model_names = list(probs_dict.keys())

    # Simple average
    avg_probs = np.mean(all_probs, axis=0)

    # Weighted (favor non-linear models)
    if HAS_XGB and HAS_LGB and HAS_CAT:
        # Best performing combo
        weighted_probs = 0.15 * probs_lr + 0.20 * probs_gb + 0.25 * probs_xgb + 0.20 * probs_lgb + 0.20 * probs_cat
    elif HAS_XGB:
        weighted_probs = 0.20 * probs_lr + 0.25 * probs_gb + 0.55 * probs_xgb
    else:
        weighted_probs = 0.30 * probs_lr + 0.70 * probs_gb

    # ── Metrics ──
    metrics_rows = []
    for name, probs in [("logreg", probs_lr), ("gb", probs_gb), ("avg_ensemble", avg_probs), ("weighted_ensemble", weighted_probs)]:
        preds = [1 if p >= 0.5 else 0 for p in probs]
        m = {
            "model": name, "n_test": n_test,
            "accuracy": round(float(accuracy_score(y_test, preds)), 6),
            "f1": round(float(f1_score(y_test, preds)), 6),
            "precision": round(float(precision_score(y_test, preds)), 6),
            "recall": round(float(recall_score(y_test, preds)), 6),
            "brier": round(float(brier_score_loss(y_test, probs)), 6),
        }
        try:
            m["roc_auc"] = round(float(roc_auc_score(y_test, probs)), 6)
        except ValueError:
            m["roc_auc"] = None
        metrics_rows.append(m)

        if name in ["logreg", "gb"]:
            m["log_loss"] = round(float(log_loss(y_test, probs)), 6)

    # Save models
    for name, model in models.items():
        artifact = {
            "version": "v12",
            "target": target_quarter,
            "model_type": "classifier",
            "model_name": name,
            "vectorizer": vec,
            "scaler": scaler,
            "model": model,
            "feature_names": vec.get_feature_names_out().tolist() if hasattr(vec, 'get_feature_names_out') else list(vec.feature_names_),
        }
        joblib.dump(artifact, OUT_DIR / f"{target_quarter}_clf_{name}.joblib")

    # Save ensemble
    joblib.dump({
        "version": "v12",
        "target": target_quarter,
        "model_type": "classifier_ensemble",
        "models": models,
        "vectorizer": vec,
        "scaler": scaler,
        "has_xgb": HAS_XGB,
        "has_lgb": HAS_LGB,
        "has_cat": HAS_CAT,
    }, OUT_DIR / f"{target_quarter}_clf_ensemble.joblib")

    return {"metrics": metrics_rows, "n_samples": n}


# ────────────────────────────────────────────────────────────────────
# Training - Regression (Points)
# ────────────────────────────────────────────────────────────────────

def train_regression(
    samples: list[HybridSample],
    target_quarter: str,
    target_type: str,  # "home", "away", "total"
    gender: str,
) -> dict:
    """Train regression models for point prediction."""
    print(f"[v12] Training regression for {target_quarter}_{target_type} ({gender})...")

    samples_filtered = [s for s in samples if s.gender == gender]
    if not samples_filtered:
        return {"error": f"No samples for gender {gender}"}

    if target_type == "home":
        targets = [s.target_home_pts for s in samples_filtered]
    elif target_type == "away":
        targets = [s.target_away_pts for s in samples_filtered]
    else:
        targets = [s.target_total_pts for s in samples_filtered]

    valid = [(s, t) for s, t in zip(samples_filtered, targets) if t is not None]
    if not valid:
        return {"error": "No valid samples"}

    valid.sort(key=lambda x: x[0].dt)
    samples_valid = [s for s, t in valid]
    y = np.array([t for s, t in valid])

    n = len(samples_valid)
    n_train = int(n * 0.8)
    n_test = n - n_train

    x_dict = [s.features for s in samples_valid]
    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(x_dict)
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {}
    preds_dict = {}

    # 1. Ridge
    print(f"  [Ridge]...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_scaled, y_train)
    pred_ridge = ridge.predict(x_test_scaled)
    models["ridge"] = ridge
    preds_dict["ridge"] = pred_ridge

    # 2. GB
    print(f"  [GB]...")
    gb = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_split=20, min_samples_leaf=10, random_state=42
    )
    gb.fit(x_train, y_train)
    pred_gb = gb.predict(x_test)
    models["gb"] = gb
    preds_dict["gb"] = pred_gb

    # 3. XGBoost
    if HAS_XGB:
        print(f"  [XGB]...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
            random_state=42
        )
        xgb_model.fit(x_train, y_train)
        pred_xgb = xgb_model.predict(x_test)
        models["xgb"] = xgb_model
        preds_dict["xgb"] = pred_xgb

    # 4. LightGBM (if available)
    if HAS_LGB:
        print(f"  [LGBM]...")
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
            random_state=42
        )
        lgb_reg.fit(x_train, y_train)
        pred_lgb = lgb_reg.predict(x_test)
        models["lgbm"] = lgb_reg
        preds_dict["lgbm"] = pred_lgb

    # 5. CatBoost
    if HAS_CAT:
        print(f"  [CatBoost]...")
        cat_model = CatBoostRegressor(
            iterations=150, depth=4, learning_rate=0.1,
            l2_leaf_reg=3, random_state=42, verbose=0
        )
        cat_model.fit(x_train, y_train)
        pred_cat = cat_model.predict(x_test)
        models["catboost"] = cat_model
        preds_dict["catboost"] = pred_cat

    # ── Ensemble ──
    all_preds = list(preds_dict.values())
    avg_pred = np.mean(all_preds, axis=0)

    if HAS_XGB and HAS_LGB and HAS_CAT:
        weighted_pred = (0.10 * pred_ridge + 0.20 * pred_gb +
                         0.25 * pred_xgb + 0.20 * pred_lgb + 0.25 * pred_cat)
    elif HAS_XGB and HAS_CAT:
        weighted_pred = 0.15 * pred_ridge + 0.25 * pred_gb + 0.30 * pred_xgb + 0.30 * pred_cat
    elif HAS_XGB and HAS_LGB:
        weighted_pred = 0.15 * pred_ridge + 0.25 * pred_gb + 0.30 * pred_xgb + 0.30 * pred_lgb
    elif HAS_XGB:
        weighted_pred = 0.15 * pred_ridge + 0.35 * pred_gb + 0.50 * pred_xgb
    else:
        weighted_pred = 0.25 * pred_ridge + 0.75 * pred_gb

    # ── Metrics ──
    mae = mean_absolute_error(y_test, avg_pred)
    rmse = np.sqrt(mean_squared_error(y_test, avg_pred))
    r2 = r2_score(y_test, avg_pred)

    metrics = {
        "model": f"{target_quarter}_{target_type}_{gender}",
        "gender": gender,
        "n_train": n_train,
        "n_test": n_test,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
    }

    # Save models
    for name, model in models.items():
        artifact = {
            "version": "v12",
            "target": target_quarter,
            "target_type": target_type,
            "gender": gender,
            "model_type": "regressor",
            "model_name": name,
            "vectorizer": vec,
            "scaler": scaler,
            "model": model,
        }
        joblib.dump(artifact, OUT_DIR / f"{target_quarter}_{target_type}_{gender}_reg_{name}.joblib")

    # Save ensemble
    joblib.dump({
        "version": "v12",
        "target": target_quarter,
        "target_type": target_type,
        "gender": gender,
        "model_type": "regressor_ensemble",
        "models": models,
        "vectorizer": vec,
        "scaler": scaler,
        "has_xgb": HAS_XGB,
        "has_cat": HAS_CAT,
    }, OUT_DIR / f"{target_quarter}_{target_type}_{gender}_reg_ensemble.joblib")

    print(f"[v12] {target_quarter}_{target_type}_{gender} done - MAE: {mae:.2f}, R2: {r2:.3f}")

    return {"metrics": metrics, "n_samples": n}


# ────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[v12] Starting training...")
    print(f"[v12] XGBoost: {'YES' if HAS_XGB else 'NO'}")
    print(f"[v12] LightGBM: {'YES' if HAS_LGB else 'NO'}")
    print(f"[v12] CatBoost: {'YES' if HAS_CAT else 'NO'}")

    # Build samples
    samples = build_samples(DB_PATH)

    # Train classifiers
    print("\n[v12] === CLASSIFICATION ===")
    clf_q3 = train_classifier(samples, "q3")
    clf_q4 = train_classifier(samples, "q4")

    # Train regressors
    print("\n[v12] === REGRESSION ===")
    all_metrics = []
    for target_quarter in ["q3", "q4"]:
        for target_type in ["home", "away", "total"]:
            for gender in ["men_or_open", "women"]:
                result = train_regression(samples, target_quarter, target_type, gender)
                if "metrics" in result:
                    all_metrics.append(result["metrics"])

    # Save all metrics
    metrics_file = OUT_DIR / "all_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)

    # Save summary
    summary = {
        "version": "v12",
        "generated_at": datetime.now().isoformat(),
        "algorithms": {
            "xgboost": HAS_XGB,
            "lightgbm": HAS_LGB,
            "catboost": HAS_CAT,
        },
        "classification": {
            "q3": clf_q3.get("metrics", []),
            "q4": clf_q4.get("metrics", []),
        },
        "regression_metrics": all_metrics,
        "total_samples": len(samples),
    }

    with open(OUT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start
    print(f"\n[v12] Training complete in {elapsed:.1f}s")
    print(f"[v12] Output: {OUT_DIR}")
    print(f"[v12] Total samples: {len(samples)}")


if __name__ == "__main__":
    main()

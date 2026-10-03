"""Train an independent M30_V1 Q4 model focused on minute-30 signal quality.

This model is intentionally independent from M27_V1. It follows the same
general philosophy of using cleaner live-state features, but it owns its
feature set, training artifacts, cache, and metadata.

The minute-30 version leans harder on information already visible deep into
Q3, especially:

- halftime-to-minute30 comeback transitions
- current leader / trailer state after more Q3 development
- selective Q3 partial state and momentum windows
- short graph slopes

For mixed datasets, quarter duration is inferred per match so minute 30 maps
correctly for both 10-minute and 12-minute regulations.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import xgboost as xgb

import infer_match as infer_live
import train_q3_q4_models_v6 as v6

ROOT = v6.ROOT
DB_PATH = v6.DB_PATH
OUT_DIR = ROOT / "training" / "model_outputs_m30_v1"
DYNAMIC_ROWS_CACHE = OUT_DIR / "dynamic_rows_cache.joblib"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
CV_N_SPLITS = 5
SNAPSHOT_MINUTE = 30
FEATURE_SCHEMA_VERSION = "m30_v1_momentum_v1"
BASE_FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION
TENM_SCHEMA_VERSION = "m30_v1_10m_momentum_v1"

FILTER_10M = False
"""Set True via --filter-10m. When active:
- Only 10-minute regulation games are included.
- Constant/redundant features for 10m are removed post-cache.
- Output goes to model_outputs_m30_v1/q4_10m_only/."""


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_auc(y_true: list[int], probs: list[float]) -> float | None:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return None


def _metric_row(
    model_name: str,
    split: str,
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    y_true: list[int],
    probs: list[float],
) -> dict:
    preds = [1 if prob >= 0.5 else 0 for prob in probs]
    row = {
        "target": "q4",
        "model": model_name,
        "split": split,
        "samples_total": n_total,
        "samples_train": n_train,
        "samples_val": n_val,
        "samples_test": n_test,
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 6),
        "precision": round(
            float(precision_score(y_true, preds, zero_division=0)),
            6,
        ),
        "recall": round(
            float(recall_score(y_true, preds, zero_division=0)),
            6,
        ),
        "log_loss": round(float(log_loss(y_true, probs, labels=[0, 1])), 6),
        "brier": round(float(brier_score_loss(y_true, probs)), 6),
    }
    auc = _safe_auc(y_true, probs)
    row["roc_auc"] = None if auc is None else round(auc, 6)
    return row


def _winner_tag(home: int, away: int) -> str:
    if home > away:
        return "home"
    if away > home:
        return "away"
    return "tied"


def _margin_bin(value: int) -> str:
    abs_value = abs(int(value))
    if abs_value <= 3:
        return "01_03"
    if abs_value <= 7:
        return "04_07"
    return "08_plus"


def _graph_stats_upto(graph_points: list[dict], max_minute: int) -> dict:
    points = [
        point
        for point in graph_points
        if int(point.get("minute", 0)) <= max_minute
    ]
    values = [int(point.get("value", 0)) for point in points]
    if not values:
        return {
            "gp_count": 0,
            "gp_last": 0,
            "gp_slope_3m": 0,
            "gp_slope_5m": 0,
        }
    slope_3m = (
        values[-1] - values[-4] if len(values) >= 4 else values[-1] - values[0]
    )
    slope_5m = (
        values[-1] - values[-6] if len(values) >= 6 else values[-1] - values[0]
    )
    return {
        "gp_count": len(values),
        "gp_last": values[-1],
        "gp_slope_3m": slope_3m,
        "gp_slope_5m": slope_5m,
    }


def _compute_q3_momentum_features(
    graph_points: list[dict],
    regulation_quarter_minutes: float,
    snapshot_minute: float = 30.0,
) -> dict:
    """Extract momentum/dynamics features from graph_points value time series.

    graph_points value = home_score - away_score (score differential).
    Captures Q3 trajectory: erosion, acceleration, volatility, recovery effort.
    """
    q3_start = regulation_quarter_minutes * 2.0
    q3_points = [
        p for p in graph_points
        if q3_start <= float(p.get("minute", 0)) <= snapshot_minute + 1e-9
    ]
    all_upto = [
        p for p in graph_points
        if float(p.get("minute", 0)) <= snapshot_minute + 1e-9
    ]

    if len(q3_points) < 2:
        return {
            "gp_q3_lead_erosion": 0,
            "gp_q3_late_momentum": 0,
            "gp_q3_volatility": 0.0,
            "gp_q3_turning_points": 0,
            "gp_q3_recovery_spent": 0,
            "gp_q3_peak_to_final_ratio": 1.0,
            "gp_q3_last_3min_momentum": 0,
            "gp_q3_momentum_accel": 0.0,
        }

    q3_vals = [int(p.get("value", 0)) for p in q3_points]
    q3_final = q3_vals[-1]
    q3_peak_abs = float(max(abs(v) for v in q3_vals))
    final_abs = float(abs(q3_final))

    # 1. Lead erosion: how much the lead shrunk from its Q3 peak
    lead_erosion = int(q3_peak_abs - final_abs) if q3_peak_abs > 0 else 0

    # 2. Late Q3 momentum (last 5 min of Q3)
    late_min = max(q3_start, snapshot_minute - 5.0)
    late_q3 = [p for p in q3_points if float(p.get("minute", 0)) >= late_min]
    late_momentum = int(late_q3[-1]["value"] - late_q3[0]["value"]) if len(late_q3) >= 2 else 0

    # 3. Q3 volatility (std of minute-to-minute changes)
    q3_diffs = np.diff(q3_vals)
    q3_volatility = round(float(np.std(q3_diffs)), 2) if len(q3_diffs) > 0 else 0.0

    # 4. Turning points in Q3 (momentum direction changes)
    q3_turns = int(sum(
        1 for i in range(1, len(q3_diffs))
        if q3_diffs[i - 1] * q3_diffs[i] < 0
    )) if len(q3_diffs) > 1 else 0

    # 5. Recovery spent: how many points the current leader climbed from their Q3 low
    q3_min = min(q3_vals)
    q3_max = max(q3_vals)
    if q3_final > 0:
        # Home leads: climb from Q3 low to current
        recovery_spent = int(q3_final - q3_min)
    elif q3_final < 0:
        # Away leads (negative): climb up (toward zero) from Q3 max
        recovery_spent = int(q3_max - q3_final)
    else:
        recovery_spent = 0

    # 6. Peak-to-final ratio: 1.0 = lead at peak, 0.0 = completely eroded
    peak_to_final_ratio = round(final_abs / q3_peak_abs, 3) if q3_peak_abs > 0 else 1.0

    # 7. Last 3 minutes momentum
    last3_min = max(q3_start, snapshot_minute - 3.0)
    last3_q3 = [p for p in q3_points if float(p.get("minute", 0)) >= last3_min]
    last_3m_momentum = int(last3_q3[-1]["value"] - last3_q3[0]["value"]) if len(last3_q3) >= 2 else 0

    # 8. Momentum acceleration (late vs early Q3)
    if len(q3_diffs) >= 6:
        split = len(q3_diffs) // 2
        early_mean = float(np.mean(q3_diffs[:split]))
        late_mean = float(np.mean(q3_diffs[split:]))
        accel = round(late_mean - early_mean, 3)
    else:
        accel = 0.0

    return {
        "gp_q3_lead_erosion": lead_erosion,
        "gp_q3_late_momentum": late_momentum,
        "gp_q3_volatility": q3_volatility,
        "gp_q3_turning_points": q3_turns,
        "gp_q3_recovery_spent": recovery_spent,
        "gp_q3_peak_to_final_ratio": peak_to_final_ratio,
        "gp_q3_last_3min_momentum": last_3m_momentum,
        "gp_q3_momentum_accel": accel,
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
        return min(
            (10.0, 12.0),
            key=lambda value: abs(value - approx_q_minutes),
        )

    league = str(match_data.get("league") or "")
    if "nba" in league.lower():
        return 12.0
    return 10.0


def _pbp_events_upto_m30(match_data: dict, cutoff_minute: float) -> list[dict]:
    quarter_minutes = _infer_regulation_quarter_minutes(match_data)
    pbp = match_data.get("play_by_play", {}) or {}
    events: list[dict] = []

    for quarter_label, plays in pbp.items():
        q_idx = infer_live._quarter_index(str(quarter_label))
        if q_idx is None or q_idx < 1 or q_idx > 4:
            continue

        q_start = (q_idx - 1) * quarter_minutes
        for play in plays or []:
            rem_sec = infer_live._clock_to_seconds(str(play.get("time", "")))
            if rem_sec is None or rem_sec > int(quarter_minutes * 60.0):
                continue
            elapsed_in_q = quarter_minutes - (rem_sec / 60.0)
            global_min = q_start + elapsed_in_q
            if global_min <= float(cutoff_minute) + 1e-9:
                event = dict(play)
                event["_global_min"] = global_min
                events.append(event)

    events.sort(key=lambda event: float(event.get("_global_min", 0.0)))
    return events


def _score_upto_m30(match_data: dict, cutoff_minute: float) -> tuple[int, int]:
    events = _pbp_events_upto_m30(match_data, cutoff_minute)
    if not events:
        return 0, 0

    home = 0
    away = 0
    for event in events:
        hs = event.get("home_score")
        as_ = event.get("away_score")
        if hs is not None and as_ is not None:
            home = int(hs)
            away = int(as_)
    return home, away


def _max_scoring_run(events: list[dict], team_name: str) -> int:
    best = 0
    run = 0
    for event in events:
        team = event.get("team")
        pts = int(event.get("points", 0) or 0)
        if team == team_name and pts > 0:
            run += pts
            if run > best:
                best = run
        elif team in ("home", "away"):
            run = 0
    return best


def _pbp_recent_window_features_m30(
    match_data: dict,
    cutoff_minute: float,
    window_minutes: float,
) -> dict:
    events_upto = _pbp_events_upto_m30(match_data, cutoff_minute)
    start_min = max(0.0, float(cutoff_minute) - window_minutes)
    events = [
        event
        for event in events_upto
        if float(event.get("_global_min", 0.0)) >= start_min
    ]

    home_points = 0
    away_points = 0
    home_events = 0
    away_events = 0
    last_scoring = "none"
    for event in events:
        team = event.get("team")
        pts = int(event.get("points", 0) or 0)
        if team == "home" and pts > 0:
            home_points += pts
            home_events += 1
            last_scoring = "home"
        elif team == "away" and pts > 0:
            away_points += pts
            away_events += 1
            last_scoring = "away"

    scoring_events = home_events + away_events
    home_run = _max_scoring_run(events, "home")
    away_run = _max_scoring_run(events, "away")

    return {
        "clutch_window_minutes": window_minutes,
        "clutch_scoring_events": scoring_events,
        "clutch_home_points": home_points,
        "clutch_away_points": away_points,
        "clutch_points_diff": home_points - away_points,
        "clutch_home_event_share": (
            float(home_events / scoring_events) if scoring_events else 0.0
        ),
        "clutch_home_max_run_pts": home_run,
        "clutch_away_max_run_pts": away_run,
        "clutch_run_diff": home_run - away_run,
        "clutch_last_scoring_home": int(last_scoring == "home"),
        "clutch_last_scoring_away": int(last_scoring == "away"),
    }


def _recent_window_features(
    match_data: dict,
    cutoff_minute: int,
    window_minutes: float,
    prefix: str,
) -> dict:
    recent = _pbp_recent_window_features_m30(
        match_data,
        cutoff_minute,
        window_minutes,
    )
    return {
        f"{prefix}_home_points": int(recent.get("clutch_home_points", 0)),
        f"{prefix}_away_points": int(recent.get("clutch_away_points", 0)),
        f"{prefix}_points_diff": int(recent.get("clutch_points_diff", 0)),
        f"{prefix}_home_event_share": float(
            recent.get("clutch_home_event_share", 0.0)
        ),
        f"{prefix}_home_max_run": int(
            recent.get("clutch_home_max_run_pts", 0)
        ),
        f"{prefix}_away_max_run": int(
            recent.get("clutch_away_max_run_pts", 0)
        ),
        f"{prefix}_run_diff": int(recent.get("clutch_run_diff", 0)),
        f"{prefix}_last_scoring_home": int(
            recent.get("clutch_last_scoring_home", 0)
        ),
        f"{prefix}_last_scoring_away": int(
            recent.get("clutch_last_scoring_away", 0)
        ),
    }


def _side_flag(side: str, target: str) -> int:
    return int(side == target)


def _build_m30_v1_features(sample: v6.MatchSample, match_data: dict) -> dict:
    q1h, q1a = v6._quarter_points(match_data, "Q1")
    q2h, q2a = v6._quarter_points(match_data, "Q2")
    q1h = int(q1h or 0)
    q1a = int(q1a or 0)
    q2h = int(q2h or 0)
    q2a = int(q2a or 0)

    ht_home = q1h + q2h
    ht_away = q1a + q2a
    ht_diff = ht_home - ht_away
    ht_total = ht_home + ht_away
    q1_winner = _winner_tag(q1h, q1a)
    q2_winner = _winner_tag(q2h, q2a)
    halftime_leader = _winner_tag(ht_home, ht_away)
    regulation_quarter_minutes = _infer_regulation_quarter_minutes(match_data)
    q3_start_minute = regulation_quarter_minutes * 2.0
    q4_start_minute = regulation_quarter_minutes * 3.0

    est_home, est_away = _score_upto_m30(match_data, SNAPSHOT_MINUTE)
    est_home = int(est_home)
    est_away = int(est_away)
    score_diff = est_home - est_away
    current_leader = _winner_tag(est_home, est_away)

    q3_partial_home = max(0, est_home - ht_home)
    q3_partial_away = max(0, est_away - ht_away)
    q3_partial_diff = q3_partial_home - q3_partial_away
    q3_partial_total = q3_partial_home + q3_partial_away
    q3_partial_leader = _winner_tag(q3_partial_home, q3_partial_away)
    q3_elapsed_minutes = max(
        0.0,
        min(float(SNAPSHOT_MINUTE), q4_start_minute) - q3_start_minute,
    )
    q3_remaining_minutes = max(0.0, q4_start_minute - float(SNAPSHOT_MINUTE))
    q3_partial_completion = min(
        1.0,
        max(0.0, q3_elapsed_minutes / regulation_quarter_minutes),
    )

    if q3_elapsed_minutes > 0:
        q3_partial_home_pace = q3_partial_home / q3_elapsed_minutes
        q3_partial_away_pace = q3_partial_away / q3_elapsed_minutes
        q3_partial_projected_home = (
            q3_partial_home_pace * regulation_quarter_minutes
        )
        q3_partial_projected_away = (
            q3_partial_away_pace * regulation_quarter_minutes
        )
    else:
        q3_partial_home_pace = 0.0
        q3_partial_away_pace = 0.0
        q3_partial_projected_home = 0.0
        q3_partial_projected_away = 0.0
    q3_partial_projected_diff = (
        q3_partial_projected_home - q3_partial_projected_away
    )

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

    graph = _graph_stats_upto(
        match_data.get("graph_points", []),
        SNAPSHOT_MINUTE,
    )

    # New: PBP window features (1m, 2m in addition to existing)
    pbp_windows_1m = _recent_window_features(
        match_data, SNAPSHOT_MINUTE, 1.0, "recent_1m"
    )
    pbp_windows_2m = _recent_window_features(
        match_data, SNAPSHOT_MINUTE, 2.0, "recent_2m"
    )

    # New: Q3 momentum features from graph_points
    q3_momentum = _compute_q3_momentum_features(
        match_data.get("graph_points", []),
        regulation_quarter_minutes,
        SNAPSHOT_MINUTE,
    )

    halftime_trailer_cutting_in_q3 = 0
    halftime_trailer_won_q3_partial = 0
    halftime_trailer_now_leads = 0
    halftime_trailer_now_tied = 0
    halftime_trailer_neutralized_deficit = 0
    halftime_leader_lost_lead = 0
    halftime_trailer_gain = 0
    halftime_to_current_margin_delta = score_diff - ht_diff
    abs_margin_delta_from_halftime = abs(score_diff) - abs(ht_diff)
    if halftime_trailing_side == "home":
        halftime_trailer_cutting_in_q3 = int(q3_partial_diff > 0)
        halftime_trailer_won_q3_partial = int(q3_partial_leader == "home")
        halftime_trailer_now_leads = int(score_diff > 0)
        halftime_trailer_now_tied = int(score_diff == 0)
        halftime_trailer_neutralized_deficit = int(score_diff >= 0)
        halftime_leader_lost_lead = int(current_leader != "away")
        halftime_trailer_gain = q3_partial_diff
    elif halftime_trailing_side == "away":
        halftime_trailer_cutting_in_q3 = int(q3_partial_diff < 0)
        halftime_trailer_won_q3_partial = int(q3_partial_leader == "away")
        halftime_trailer_now_leads = int(score_diff < 0)
        halftime_trailer_now_tied = int(score_diff == 0)
        halftime_trailer_neutralized_deficit = int(score_diff <= 0)
        halftime_leader_lost_lead = int(current_leader != "home")
        halftime_trailer_gain = -q3_partial_diff

    halftime_deficit_abs = abs(ht_diff)
    halftime_trailer_recovery_ratio = 0.0
    if halftime_deficit_abs > 0:
        halftime_trailer_recovery_ratio = (
            halftime_trailer_gain / halftime_deficit_abs
        )

    lead_flip_from_halftime = int(
        halftime_leader not in {"tied", current_leader}
        and current_leader != "tied"
    )
    is_q3_q4_boundary_10m = int(
        abs(regulation_quarter_minutes - 10.0) < 1e-9
        and abs(q4_start_minute - float(SNAPSHOT_MINUTE)) < 1e-9
    )
    q3_close_flip_or_tie_after_biglead_10m = int(
        is_q3_q4_boundary_10m
        and halftime_deficit_abs >= 8
        and (lead_flip_from_halftime or current_leader == "tied")
    )
    q3_close_biglead_now_04_07_10m = int(
        is_q3_q4_boundary_10m
        and halftime_deficit_abs >= 8
        and 4 <= abs(score_diff) <= 7
    )

    home_wins_first2_plus_q3_partial_count = (
        int(q1_winner == "home")
        + int(q2_winner == "home")
        + int(q3_partial_leader == "home")
    )
    away_wins_first2_plus_q3_partial_count = (
        int(q1_winner == "away")
        + int(q2_winner == "away")
        + int(q3_partial_leader == "away")
    )

    out = {
        "gender_bucket": sample.features_q4.get(
            "gender_bucket",
            "men_or_open",
        ),
        "home_prior_wr": sample.features_q4.get("home_prior_wr", 0.0),
        "away_prior_wr": sample.features_q4.get("away_prior_wr", 0.0),
        "prior_wr_diff": sample.features_q4.get("prior_wr_diff", 0.0),
        "q1_diff": q1h - q1a,
        "q2_diff": q2h - q2a,
        "q1_winner": q1_winner,
        "q2_winner": q2_winner,
        "home_wins_first2_count": int(q1_winner == "home")
        + int(q2_winner == "home"),
        "away_wins_first2_count": int(q1_winner == "away")
        + int(q2_winner == "away"),
        "home_wins_first2_plus_q3_partial_count": (
            home_wins_first2_plus_q3_partial_count
        ),
        "away_wins_first2_plus_q3_partial_count": (
            away_wins_first2_plus_q3_partial_count
        ),
        "halftime_home": ht_home,
        "halftime_away": ht_away,
        "halftime_diff": ht_diff,
        "halftime_total": ht_total,
        "halftime_leader": halftime_leader,
        "halftime_margin_bin": _margin_bin(ht_diff),
        "halftime_trailing_side": halftime_trailing_side,
        "score_est_home": est_home,
        "score_est_away": est_away,
        "score_est_diff": score_diff,
        "current_leader": current_leader,
        "current_margin_bin": _margin_bin(score_diff),
        "current_trailing_side": current_trailing_side,
        "q3_partial_home": q3_partial_home,
        "q3_partial_away": q3_partial_away,
        "q3_partial_diff": q3_partial_diff,
        "q3_partial_leader": q3_partial_leader,
        "q3_partial_margin_bin": _margin_bin(q3_partial_diff),
        "regulation_quarter_minutes": regulation_quarter_minutes,
        "q3_start_minute": q3_start_minute,
        "q4_start_minute": q4_start_minute,
        "q3_remaining_minutes": q3_remaining_minutes,
        "q3_partial_completion": q3_partial_completion,
        "q3_partial_home_pace_per_min": q3_partial_home_pace,
        "q3_partial_away_pace_per_min": q3_partial_away_pace,
        "q3_partial_projected_home": q3_partial_projected_home,
        "q3_partial_projected_away": q3_partial_projected_away,
        "q3_partial_projected_diff": q3_partial_projected_diff,
        "q3_partial_home_share": (
            float(q3_partial_home / q3_partial_total)
            if q3_partial_total
            else 0.5
        ),
        "halftime_trailer_cutting_in_q3": halftime_trailer_cutting_in_q3,
        "halftime_trailer_won_q3_partial": halftime_trailer_won_q3_partial,
        "halftime_trailer_gain": halftime_trailer_gain,
        "halftime_to_current_margin_delta": halftime_to_current_margin_delta,
        "abs_margin_delta_from_halftime": abs_margin_delta_from_halftime,
        "halftime_trailer_strong_recovery": int(
            halftime_trailing_side != "tied"
            and halftime_trailer_recovery_ratio >= 0.5
        ),
        "big_halftime_lead_now_close": int(
            halftime_deficit_abs >= 8 and abs(score_diff) <= 7
        ),
        "q3_close_biglead_now_04_07_10m": q3_close_biglead_now_04_07_10m,
        "q3_close_flip_or_tie_after_biglead_10m": (
            q3_close_flip_or_tie_after_biglead_10m
        ),
        "halftime_deficit_abs": halftime_deficit_abs,
        "current_leader_won_q3_partial": _side_flag(
            q3_partial_leader,
            current_leader,
        ),
    }
    out.update(graph)
    out.update(pbp_windows_1m)
    out.update(pbp_windows_2m)
    out.update(q3_momentum)
    return out


def _build_base_samples_and_data(
    db_path: Path,
) -> tuple[list[v6.MatchSample], dict[str, dict]]:
    captured: dict[str, dict] = {}
    original_get_match = v6.db_mod.get_match

    def _capturing(conn, match_id: str) -> dict | None:
        result = original_get_match(conn, match_id)
        if result is not None:
            captured[str(match_id)] = result
        return result

    v6.db_mod.get_match = _capturing
    try:
        samples = v6._build_samples(db_path)
    finally:
        v6.db_mod.get_match = original_get_match
    return samples, captured


def _window_is_eligible_m30(match_data: dict) -> bool:
    graph_points = [
        point
        for point in (match_data.get("graph_points") or [])
        if int(point.get("minute", 0)) <= SNAPSHOT_MINUTE
    ]
    pbp_count = len(_pbp_events_upto_m30(match_data, SNAPSHOT_MINUTE))
    thresholds = infer_live._sufficiency_thresholds("q4", SNAPSHOT_MINUTE)
    return (
        len(graph_points) >= int(thresholds["min_graph_points"])
        and pbp_count >= int(thresholds["min_pbp_events"])
    )


def _build_dynamic_samples(
    base_samples: list[v6.MatchSample],
    preloaded: dict[str, dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    conn = None if preloaded is not None else v6.db_mod.get_conn(str(DB_PATH))
    if conn is not None:
        v6.db_mod.init_db(conn)
    source = "cache" if preloaded is not None else "DB"
    try:
        for sample in tqdm(
            base_samples,
            desc=f"[m30_v1] Construyendo samples dinamicos (desde {source})",
            unit="partido",
        ):
            if sample.target_q4 is None:
                continue
            if preloaded is not None:
                match_data = preloaded.get(str(sample.match_id))
            else:
                match_data = v6.db_mod.get_match(conn, str(sample.match_id))
            if not match_data or not _window_is_eligible_m30(match_data):
                continue
            # When in 10m-only mode, skip non-10m games at build time
            if FILTER_10M:
                qmin = _infer_regulation_quarter_minutes(match_data)
                if abs(qmin - 10.0) > 1e-9:
                    continue
            rows.append(
                {
                    "features": _build_m30_v1_features(sample, match_data),
                    "target": int(sample.target_q4),
                    "dt": sample.dt,
                    "match_id": sample.match_id,
                    "snapshot_minute": SNAPSHOT_MINUTE,
                }
            )
    finally:
        if conn is not None:
            conn.close()
    rows.sort(key=lambda row: (row["dt"], row["match_id"]))
    return rows


def _load_or_build_dynamic_samples(
    samples: list[v6.MatchSample],
    preloaded: dict[str, dict] | None = None,
    force_rebuild: bool = False,
) -> list[dict]:
    if not force_rebuild and DYNAMIC_ROWS_CACHE.exists():
        print(
            f"[m30_v1] Cargando dynamic_rows desde cache: "
            f"{DYNAMIC_ROWS_CACHE}"
        )
        cached = joblib.load(DYNAMIC_ROWS_CACHE)
        if isinstance(cached, dict):
            cache_version = cached.get("schema_version")
            cache_rows = cached.get("rows")
            if (
                cache_version == FEATURE_SCHEMA_VERSION
                and isinstance(cache_rows, list)
            ):
                return cache_rows
            print(
                "[m30_v1] Cache invalido por cambio de schema; "
                "reconstruyendo dynamic_rows..."
            )
        elif isinstance(cached, list):
            print(
                "[m30_v1] Cache legacy sin version de schema; "
                "reconstruyendo dynamic_rows..."
            )
        else:
            print(
                "[m30_v1] Cache dynamic_rows no reconocido; "
                "reconstruyendo..."
            )
    rows = _build_dynamic_samples(samples, preloaded=preloaded)
    DYNAMIC_ROWS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "rows": rows,
        },
        DYNAMIC_ROWS_CACHE,
    )
    print(f"[m30_v1] dynamic_rows guardados en cache: {DYNAMIC_ROWS_CACHE}")
    return rows


def _split_rows_temporal_by_match(
    rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    if not rows:
        return [], [], [], {
            "matches_total": 0,
            "matches_train": 0,
            "matches_val": 0,
            "matches_test": 0,
        }
    match_first_dt: dict[str, object] = {}
    for row in rows:
        match_id = str(row["match_id"])
        current_dt = row["dt"]
        previous_dt = match_first_dt.get(match_id)
        if previous_dt is None or current_dt < previous_dt:
            match_first_dt[match_id] = current_dt
    ordered_ids = [
        match_id
        for match_id, _ in sorted(
            match_first_dt.items(),
            key=lambda item: (item[1], item[0]),
        )
    ]
    n_matches = len(ordered_ids)
    n_train_matches = int(n_matches * TRAIN_RATIO)
    n_val_matches = int(n_matches * VAL_RATIO)
    train_ids = set(ordered_ids[:n_train_matches])
    val_ids = set(ordered_ids[n_train_matches:n_train_matches + n_val_matches])
    test_ids = set(ordered_ids[n_train_matches + n_val_matches:])
    train_rows = [row for row in rows if str(row["match_id"]) in train_ids]
    val_rows = [row for row in rows if str(row["match_id"]) in val_ids]
    test_rows = [row for row in rows if str(row["match_id"]) in test_ids]
    for block in (train_rows, val_rows, test_rows):
        block.sort(key=lambda row: (row["dt"], row["match_id"]))
    return train_rows, val_rows, test_rows, {
        "matches_total": n_matches,
        "matches_train": len(train_ids),
        "matches_val": len(val_ids),
        "matches_test": len(test_ids),
    }


def _make_models() -> dict[str, object]:
    return {
        "xgb": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        ),
    }


def _isotonic_calibrate(
    model,
    x_val: np.ndarray,
    y_val: list[int],
    x_test: np.ndarray,
) -> list[float]:
    raw_val = model.predict_proba(x_val)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_val, y_val)
        return list(iso.predict(raw_test))
    except Exception:
        return list(raw_test)


def _timeseries_cv(
    x_all: np.ndarray,
    y: list[int],
    n_splits: int = CV_N_SPLITS,
) -> list[dict]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(
            splitter.split(x_all),
            total=n_splits,
            desc="[m30_v1] CV temporal",
            unit="fold",
        )
    ):
        x_train = x_all[train_idx]
        x_test = x_all[test_idx]
        y_train = [y[index] for index in train_idx]
        y_test = [y[index] for index in test_idx]
        if len(set(y_test)) < 2:
            continue
        fold_models = _make_models()
        proba_map: dict[str, list[float]] = {}
        for model_name, model in fold_models.items():
            model.fit(x_train, y_train)
            probs = list(model.predict_proba(x_test)[:, 1])
            proba_map[model_name] = probs
            row = _metric_row(
                model_name=f"m30_v1_{model_name}",
                split=f"cv_fold_{fold_idx}",
                n_total=len(y),
                n_train=len(train_idx),
                n_val=0,
                n_test=len(test_idx),
                y_true=y_test,
                probs=probs,
            )
            row["fold"] = fold_idx
            rows.append(row)
        ensemble = [
            (proba_map["xgb"][index] + proba_map["hist_gb"][index]) / 2.0
            for index in range(len(y_test))
        ]
        ensemble_row = _metric_row(
            model_name="m30_v1_champion_q4_ensemble_avg",
            split=f"cv_fold_{fold_idx}",
            n_total=len(y),
            n_train=len(train_idx),
            n_val=0,
            n_test=len(test_idx),
            y_true=y_test,
            probs=ensemble,
        )
        ensemble_row["fold"] = fold_idx
        rows.append(ensemble_row)
    return rows


TENM_CONSTANT_FEATURES = {
    "regulation_quarter_minutes",
    "q3_start_minute",
    "q4_start_minute",
    "q3_remaining_minutes",
    "q3_partial_completion",
}

TENM_REDUNDANT_FEATURES = {
    "q3_partial_home_pace_per_min",
    "q3_partial_away_pace_per_min",
    "q3_partial_projected_home",
    "q3_partial_projected_away",
    "q3_partial_projected_diff",
}

TENM_DROP_FEATURES = TENM_CONSTANT_FEATURES | TENM_REDUNDANT_FEATURES


def _train(
    samples: list[v6.MatchSample],
    preloaded: dict[str, dict] | None,
    force_rebuild: bool,
) -> dict:
    print("[m30_v1] Construyendo samples dinamicos...")
    start = time.perf_counter()
    dynamic_rows = _load_or_build_dynamic_samples(
        samples,
        preloaded=preloaded,
        force_rebuild=force_rebuild,
    )
    elapsed = time.perf_counter() - start
    print(f"[m30_v1] Total samples: {len(dynamic_rows)} ({elapsed:.1f}s)")
    if len(dynamic_rows) < 300:
        raise RuntimeError(f"[m30_v1] Muy pocas filas: {len(dynamic_rows)}")

    # Apply 10m filter in-memory (reuses cache regardless of setting)
    if FILTER_10M:
        pre_filter_count = len(dynamic_rows)
        dynamic_rows = [
            row for row in dynamic_rows
            if abs(row["features"].get("regulation_quarter_minutes", 10.0) - 10.0) < 1e-9
        ]
        print(f"[m30_v1] Filtrado a 10m: {pre_filter_count} -> {len(dynamic_rows)} filas")
        # Remove constant/redundant features for 10m
        for row in dynamic_rows:
            for feat in TENM_DROP_FEATURES:
                row["features"].pop(feat, None)
        print(f"[m30_v1] Features eliminadas (ctes/redundantes 10m): {len(TENM_DROP_FEATURES)}")

    train_rows, val_rows, test_rows, split_info = (
        _split_rows_temporal_by_match(dynamic_rows)
    )
    n_total = len(dynamic_rows)
    n_train = len(train_rows)
    n_val = len(val_rows)
    n_test = len(test_rows)
    if min(n_train, n_val, n_test) == 0:
        raise RuntimeError("[m30_v1] Split vacio")

    y_train = [row["target"] for row in train_rows]
    y_val = [row["target"] for row in val_rows]
    y_test = [row["target"] for row in test_rows]
    if len(set(y_train)) < 2 or len(set(y_val)) < 2 or len(set(y_test)) < 2:
        raise RuntimeError("[m30_v1] Clases insuficientes en algun split")

    x_train_dict = [row["features"] for row in train_rows]
    x_val_dict = [row["features"] for row in val_rows]
    x_test_dict = [row["features"] for row in test_rows]

    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(x_train_dict)
    x_val = vectorizer.transform(x_val_dict)
    x_test = vectorizer.transform(x_test_dict)

    print(
        "[m30_v1] split filas: "
        f"train={n_train} val={n_val} test={n_test} | "
        f"matches: {split_info['matches_train']}/"
        f"{split_info['matches_val']}/"
        f"{split_info['matches_test']}"
    )

    x_cv = np.vstack([x_train, x_val])
    y_cv = y_train + y_val
    print("[m30_v1] Ejecutando validacion temporal (CV)...")
    cv_rows = _timeseries_cv(x_cv, y_cv) if len(y_cv) > CV_N_SPLITS + 1 else []
    if cv_rows:
        print(f"[m30_v1] CV listo: {len(cv_rows)} filas de metricas")
    else:
        print("[m30_v1] CV omitido por tamano insuficiente")

    metrics_rows: list[dict] = []
    models = _make_models()
    proba_map: dict[str, list[float]] = {}
    proba_map_cal: dict[str, list[float]] = {}

    model_tag = "m30_v1_10m" if FILTER_10M else "m30_v1"
    metric_prefix = model_tag  # use same tag for metrics

    print("[m30_v1] Entrenando modelos holdout...")
    for model_name, model in tqdm(
        models.items(),
        desc="[m30_v1] Entrenando modelos",
        unit="modelo",
    ):
        model.fit(x_train, y_train)
        probs_test = list(model.predict_proba(x_test)[:, 1])
        proba_map[model_name] = probs_test
        metrics_rows.append(
            _metric_row(
                f"{metric_prefix}_{model_name}",
                "test",
                n_total,
                n_train,
                n_val,
                n_test,
                y_test,
                probs_test,
            )
        )

        probs_val = list(model.predict_proba(x_val)[:, 1])
        metrics_rows.append(
            _metric_row(
                f"{metric_prefix}_{model_name}",
                "val",
                n_total,
                n_train,
                n_val,
                n_test,
                y_val,
                probs_val,
            )
        )

        probs_cal = _isotonic_calibrate(model, x_val, y_val, x_test)
        proba_map_cal[model_name] = probs_cal
        metrics_rows.append(
            _metric_row(
                f"{metric_prefix}_{model_name}_cal",
                "test",
                n_total,
                n_train,
                n_val,
                n_test,
                y_test,
                probs_cal,
            )
        )

        artifact = {
            "version": model_tag,
            "target": "q4",
            "snapshot_minute": SNAPSHOT_MINUTE,
            "filter_10m": FILTER_10M,
            "model_name": f"{metric_prefix}_{model_name}",
            "vectorizer": vectorizer,
            "model": model,
            "trained_rows": n_total,
            "feature_count": len(vectorizer.feature_names_),
            "split": {"train": n_train, "val": n_val, "test": n_test},
            "split_matches": split_info,
        }
        joblib.dump(artifact, OUT_DIR / f"q4_{metric_prefix}_{model_name}.joblib")
        print(f"[m30_v1] artifact guardado: q4_{metric_prefix}_{model_name}.joblib")

    champion_probs = [
        (proba_map["xgb"][index] + proba_map["hist_gb"][index]) / 2.0
        for index in range(len(y_test))
    ]
    metrics_rows.append(
        _metric_row(
            f"{metric_prefix}_champion_q4_ensemble_avg",
            "test",
            n_total,
            n_train,
            n_val,
            n_test,
            y_test,
            champion_probs,
        )
    )

    champion_probs_cal = [
        (proba_map_cal["xgb"][index] + proba_map_cal["hist_gb"][index]) / 2.0
        for index in range(len(y_test))
    ]
    metrics_rows.append(
        _metric_row(
            f"{metric_prefix}_champion_q4_ensemble_avg_cal",
            "test",
            n_total,
            n_train,
            n_val,
            n_test,
            y_test,
            champion_probs_cal,
        )
    )

    champion_artifact = {
        "version": model_tag,
        "target": "q4",
        "snapshot_minute": SNAPSHOT_MINUTE,
        "filter_10m": FILTER_10M,
        "model_name": f"{metric_prefix}_champion_q4_ensemble_avg",
        "vectorizer": vectorizer,
        "models": models,
        "champion_strategy": "avg_prob_xgb_hist_gb",
        "trained_rows": n_total,
        "feature_count": len(vectorizer.feature_names_),
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "split_matches": split_info,
        "feature_spec": {
            "kept": [
                "gender_bucket",
                "prior_wr_*",
                "Q1/Q2 winner context",
                "halftime state and minute-30 global state",
                "per-match quarter duration inference (10m/12m)",
                "Q3 partial leader and margin state",
                "halftime-to-minute30 comeback transitions",
                "strong halftime deficit recovery markers",
                "Q3->Q4 boundary comeback state for 10m games",
                "graph last value and short slopes",
            ],
            "removed": [
                "raw league one-hot",
                "team bucket one-hot",
                "duplicated score families",
                "categorical noise not tied to comeback state",
                "weak Q1/Q2/Q3 partial winner combos",
                "weak Q3-partial to current leader combos",
                "low-signal halftime/current winner combos",
                "low-signal Q3 total and projected-total pace features",
                "redundant recovery-ratio and prior-strength sum features",
                "recent 4m and 2m run windows",
                "redundant halftime leader tied/ahead flags",
                "weak halftime trailer gain bins",
                "near-zero-signal current-state cross flags (won_q1, won_q3_partial cross, trailer_was_leader, leader_was_trailer, trailing_deficit_abs)",
                "near-zero-signal halftime transition flags (trailer_now_leads, trailer_now_tied, neutralized_deficit, leader_lost_lead, lead_flip)",
                "redundant trailing_now_is_home/away (r~0.77 with score_est_diff)",
            ],
        },
    }
    joblib.dump(champion_artifact, OUT_DIR / f"q4_{metric_prefix}_champion.joblib")
    print(f"[m30_v1] artifact guardado: q4_{metric_prefix}_champion.joblib")

    return {
        "metrics": metrics_rows,
        "cv_rows": cv_rows,
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "split_matches": split_info,
        "feature_names": list(vectorizer.feature_names_),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenar modelo independiente M30_V1 para Q4"
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignorar cache de dynamic_rows y reconstruir desde cero",
    )
    parser.add_argument(
        "--filter-10m",
        action="store_true",
        help="Solo partidos de 10m de reglamento (minuto 30 = fin de Q3)",
    )
    args = parser.parse_args()

    global FILTER_10M, OUT_DIR, DYNAMIC_ROWS_CACHE, FEATURE_SCHEMA_VERSION
    if args.filter_10m:
        FILTER_10M = True
        OUT_DIR = ROOT / "training" / "model_outputs_m30_v1" / "q4_10m_only"
        DYNAMIC_ROWS_CACHE = OUT_DIR / "dynamic_rows_cache.joblib"
        FEATURE_SCHEMA_VERSION = TENM_SCHEMA_VERSION
        print("[m30_v1] Modo 10m-only activado")
        print(f"[m30_v1] Output dir: {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_bar = tqdm(total=5, desc="[m30_v1] Pipeline", unit="fase")
    print("[m30_v1] Construyendo samples base con pipeline V6...")
    start = time.perf_counter()
    samples, match_data_cache = _build_base_samples_and_data(DB_PATH)
    print(
        f"[m30_v1] Pipeline base listo: {len(samples)} samples en "
        f"{time.perf_counter() - start:.1f}s"
    )
    pipeline_bar.update(1)

    result = _train(
        samples,
        preloaded=match_data_cache,
        force_rebuild=args.rebuild_cache,
    )
    pipeline_bar.update(1)

    print("[m30_v1] Guardando metricas CSV...")
    _write_csv(OUT_DIR / "q4_metrics.csv", result["metrics"])
    _write_csv(OUT_DIR / "q4_cv_metrics.csv", result["cv_rows"])
    pipeline_bar.update(1)

    summary_model_tag = "m30_v1_10m" if FILTER_10M else "m30_v1"
    summary = {
        "version": summary_model_tag,
        "target": "q4_only",
        "snapshot_minute": SNAPSHOT_MINUTE,
        "filter_10m": FILTER_10M,
        "split": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": round(1.0 - TRAIN_RATIO - VAL_RATIO, 2),
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "n_test": result["n_test"],
            "n_total": result["n_total"],
        },
        "cv_splits": CV_N_SPLITS,
        "split_matches": result["split_matches"],
        "feature_count": len(result["feature_names"]),
        "feature_names": result["feature_names"],
    }
    print("[m30_v1] Guardando run_summary.json...")
    with (OUT_DIR / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    pipeline_bar.update(1)

    print("[m30_v1] Finalizando...")
    pipeline_bar.update(1)
    pipeline_bar.close()
    print("[m30_v1] done")
    print(f"[m30_v1] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()

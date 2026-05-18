"""Train an independent M27_V2 Q4 model with improved features.

V2 adds normalized Q3 rate features (always on), optional Q3 pace projections
and regulation-quarter-length detection (--enable-pace-features), and optional
filtering to 10-minute regulation leagues (--filter-10m).
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
OUT_DIR = ROOT / "training" / "model_outputs_m27_v3"
DYNAMIC_ROWS_CACHE = OUT_DIR / "dynamic_rows_cache.joblib"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
CV_N_SPLITS = 5
SNAPSHOT_MINUTE = 27
FILTER_10M = False
ENABLE_PACE_FEATURES = False


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
    probs_in: list[float],
) -> dict:
    probs = [max(0.0, min(1.0, p)) for p in probs_in]
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
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 6),
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
    points = [point for point in graph_points if int(point.get("minute", 0)) <= max_minute]
    values = [int(point.get("value", 0)) for point in points]
    if not values:
        return {"gp_count": 0, "gp_last": 0, "gp_slope_3m": 0, "gp_slope_5m": 0}
    slope_3m = values[-1] - values[-4] if len(values) >= 4 else values[-1] - values[0]
    slope_5m = values[-1] - values[-6] if len(values) >= 6 else values[-1] - values[0]
    return {"gp_count": len(values), "gp_last": values[-1], "gp_slope_3m": slope_3m, "gp_slope_5m": slope_5m}


def _recent_window_features(match_data: dict, cutoff_minute: int, window_minutes: float, prefix: str) -> dict:
    recent = infer_live._pbp_recent_window_features(match_data, cutoff_minute, window_minutes)
    return {
        f"{prefix}_home_points": int(recent.get("clutch_home_points", 0)),
        f"{prefix}_away_points": int(recent.get("clutch_away_points", 0)),
        f"{prefix}_points_diff": int(recent.get("clutch_points_diff", 0)),
        f"{prefix}_home_event_share": float(recent.get("clutch_home_event_share", 0.0)),
        f"{prefix}_home_max_run": int(recent.get("clutch_home_max_run_pts", 0)),
        f"{prefix}_away_max_run": int(recent.get("clutch_away_max_run_pts", 0)),
        f"{prefix}_run_diff": int(recent.get("clutch_run_diff", 0)),
        f"{prefix}_last_scoring_home": int(recent.get("clutch_last_scoring_home", 0)),
        f"{prefix}_last_scoring_away": int(recent.get("clutch_last_scoring_away", 0)),
    }


def _max_scoring_run(events: list[dict], team_name: str) -> int:
    best, run = 0, 0
    for event in events:
        team, pts = event.get("team"), int(event.get("points", 0) or 0)
        if team == team_name and pts > 0:
            run += pts
            if run > best:
                best = run
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


def _pbp_density_features(match_data: dict, cutoff_minute: int) -> dict:
    events = infer_live._pbp_events_upto(match_data, cutoff_minute)
    n_events = len(events)
    home_pts = sum(int(e.get("points", 0) or 0) for e in events if e.get("team") == "home")
    away_pts = sum(int(e.get("points", 0) or 0) for e in events if e.get("team") == "away")
    home_3pt = sum(1 for e in events if e.get("team") == "home" and int(e.get("points", 0) or 0) == 3)
    away_3pt = sum(1 for e in events if e.get("team") == "away" and int(e.get("points", 0) or 0) == 3)
    total_pts = home_pts + away_pts
    return {
        "pbp_count": n_events,
        "pbp_home_pts": home_pts,
        "pbp_away_pts": away_pts,
        "pbp_pts_per_event": round(total_pts / n_events, 3) if n_events else 0.0,
        "pbp_home_3pt_rate": round(home_3pt / max(n_events, 1), 3),
        "pbp_away_3pt_rate": round(away_3pt / max(n_events, 1), 3),
        "pbp_scoring_density": round(n_events / float(cutoff_minute), 3) if cutoff_minute else 0.0,
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


def _build_m27_v3_features(
    sample: v6.MatchSample,
    match_data: dict,
) -> dict:
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

    est_home, est_away = infer_live._score_upto(match_data, SNAPSHOT_MINUTE)
    est_home = int(est_home)
    est_away = int(est_away)
    score_diff = est_home - est_away
    q3_partial_home = max(0, est_home - ht_home)
    q3_partial_away = max(0, est_away - ht_away)
    q3_partial_diff = q3_partial_home - q3_partial_away
    q3_partial_total = q3_partial_home + q3_partial_away
    q3_partial_leader = _winner_tag(q3_partial_home, q3_partial_away)

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

    recent_3m = _recent_window_features(match_data, SNAPSHOT_MINUTE, 3.0, "recent_3m")
    recent_2m = _recent_window_features(match_data, SNAPSHOT_MINUTE, 2.0, "recent_2m")
    graph = _graph_stats_upto(match_data.get("graph_points", []), SNAPSHOT_MINUTE)

    pbp_events = infer_live._pbp_events_upto(match_data, SNAPSHOT_MINUTE)
    pbp_density = _pbp_density_features(match_data, SNAPSHOT_MINUTE)
    current_run_home = _current_scoring_run(pbp_events, "home")
    current_run_away = _current_scoring_run(pbp_events, "away")
    max_run_all_home = _max_scoring_run(pbp_events, "home")
    max_run_all_away = _max_scoring_run(pbp_events, "away")

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

    qmin = _infer_regulation_quarter_minutes(match_data)
    q3_start = qmin * 2.0
    q4_start = qmin * 3.0
    q3_elapsed = max(0.0, min(float(SNAPSHOT_MINUTE), q4_start) - q3_start)
    q3_remaining = max(0.0, q4_start - float(SNAPSHOT_MINUTE))
    q3_pct = min(1.0, max(0.0, q3_elapsed / qmin)) if qmin > 0 else 0.0

    if q3_elapsed > 0:
        h_rate = q3_partial_home / q3_elapsed
        a_rate = q3_partial_away / q3_elapsed
        diff_rate = h_rate - a_rate
        h_pace = h_rate * qmin
        a_pace = a_rate * qmin
        proj_diff = h_pace - a_pace
    else:
        h_rate = 0.0
        a_rate = 0.0
        diff_rate = 0.0
        h_pace = 0.0
        a_pace = 0.0
        proj_diff = 0.0

    out = {
        "gender_bucket": sample.features_q4.get("gender_bucket", "men_or_open"),
        "home_prior_wr": sample.features_q4.get("home_prior_wr", 0.0),
        "away_prior_wr": sample.features_q4.get("away_prior_wr", 0.0),
        "prior_wr_diff": sample.features_q4.get("prior_wr_diff", 0.0),
        "prior_wr_sum": sample.features_q4.get("prior_wr_sum", 0.0),
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
        "halftime_margin_bin": _margin_bin(ht_diff),
        "halftime_trailing_side": halftime_trailing_side,
        "score_est_home": est_home,
        "score_est_away": est_away,
        "score_est_diff": score_diff,
        "current_margin_bin": _margin_bin(score_diff),
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
    out.update(pbp_density)
    if ENABLE_PACE_FEATURES:
        out.update({
            "regulation_quarter_minutes": qmin,
            "q3_remaining_minutes": q3_remaining,
            "q3_partial_completion": q3_pct,
            "q3_partial_home_pace_per_min": round(h_pace, 3),
            "q3_partial_away_pace_per_min": round(a_pace, 3),
            "q3_partial_projected_diff": round(proj_diff, 3),
        })
    out.update(graph)
    out.update(recent_3m)
    out.update(recent_2m)

    # Drop features with zero tree importance (XGB imp = 0.0)
    _DEAD: set[str] = {
        "halftime_leader",        # all 3 levels dead
        "halftime_trailing_side", # all 3 levels dead
        "q3_partial_leader",      # all 3 levels dead
        "trailing_now_is_home",   # redundant with current_trailing_side
        "trailing_now_is_away",   # redundant with current_trailing_side
        "gp_count",               # constant at snapshot 27 (NaN corr)
        "pbp_scoring_density",    # XGB imp = 0.0, corr = 0.0003
    }
    out = {k: v for k, v in out.items() if k not in _DEAD}
    return out


TENM_CONSTANT_FEATURES: set[str] = {
    "regulation_quarter_minutes",
    "q3_remaining_minutes",
    "q3_partial_completion",
}

TENM_REDUNDANT_FEATURES: set[str] = {
    "q3_partial_home_pace_per_min",
    "q3_partial_away_pace_per_min",
    "q3_partial_projected_diff",
}


def _build_base_samples_and_data(db_path: Path) -> tuple[list[v6.MatchSample], dict[str, dict]]:
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


def _window_is_eligible_m27(match_data: dict) -> bool:
    graph_points = [
        point
        for point in (match_data.get("graph_points") or [])
        if int(point.get("minute", 0)) <= SNAPSHOT_MINUTE
    ]
    pbp_count = len(infer_live._pbp_events_upto(match_data, SNAPSHOT_MINUTE))
    thresholds = infer_live._sufficiency_thresholds("q4", SNAPSHOT_MINUTE)
    return (
        len(graph_points) >= int(thresholds["min_graph_points"])
        and pbp_count >= int(thresholds["min_pbp_events"])
    )


def _build_h2h_lookup() -> dict[tuple[str, str], list[dict]]:
    conn = v6.db_mod.get_conn(str(DB_PATH))
    v6.db_mod.init_db(conn)

    all_matches = conn.execute(
        "SELECT match_id, home_team, away_team, date FROM matches"
    ).fetchall()
    match_info: dict[str, dict] = {}
    for mid, ht, at, dt in all_matches:
        match_info[mid] = {"home_team": ht, "away_team": at, "date": _parse_dt(dt)}

    qs_rows = conn.execute(
        "SELECT match_id, quarter, home, away FROM quarter_scores"
    ).fetchall()
    conn.close()

    match_quarters: dict[str, dict[str, tuple[int, int]]] = {}
    for mid, qtr, hs, aw in qs_rows:
        if hs is not None and aw is not None:
            match_quarters.setdefault(mid, {})[qtr] = (int(hs), int(aw))

    lookup: dict[tuple[str, str], list[dict]] = {}
    for mid, info in match_info.items():
        qs = match_quarters.get(mid)
        if not qs:
            continue
        pair = tuple(sorted([info["home_team"], info["away_team"]]))
        lookup.setdefault(pair, []).append({
            "mid": mid,
            "dt": info["date"],
            "home_team": info["home_team"],
            "away_team": info["away_team"],
            "quarters": qs,
        })

    for pair in lookup:
        lookup[pair].sort(key=lambda g: g["dt"])

    return lookup


def _parse_dt(s: str):
    from datetime import datetime
    return datetime.strptime(s.split(" ")[0], "%Y-%m-%d")


def _compute_h2h_features(
    ht: str, at: str, sample_dt, lookup: dict[tuple[str, str], list[dict]]
) -> dict:
    pair = tuple(sorted([ht, at]))
    past = [g for g in lookup.get(pair, []) if g["dt"] < sample_dt]
    if not past:
        return {}

    n = len(past)

    q1_diffs, q2_diffs, q3_diffs, q4_diffs = [], [], [], []
    home_won_q4_count = 0
    home_won_count = 0

    for g in past:
        is_home = g["home_team"] == ht
        sign = 1 if is_home else -1
        qs = g["quarters"]

        def _quarter_diff(qlabel: str):
            q = qs.get(qlabel)
            return (q[0] - q[1]) * sign if q else None

        d1 = _quarter_diff("Q1")
        d2 = _quarter_diff("Q2")
        d3 = _quarter_diff("Q3")
        d4 = _quarter_diff("Q4")

        if d1 is not None: q1_diffs.append(d1)
        if d2 is not None: q2_diffs.append(d2)
        if d3 is not None: q3_diffs.append(d3)
        if d4 is not None: q4_diffs.append(d4)

        total_h = sum(v[0] for v in g["quarters"].values())
        total_a = sum(v[1] for v in g["quarters"].values())
        home_won = (total_h > total_a) == is_home
        if home_won:
            home_won_count += 1

        q4 = qs.get("Q4")
        if q4:
            if (q4[0] > q4[1]) == is_home:
                home_won_q4_count += 1

    feats: dict = {}

    feats["h2h_avg_q1_diff"] = round(sum(q1_diffs) / len(q1_diffs), 3) if q1_diffs else 0.0

    recent_3 = past[-3:] if n >= 3 else past
    home_won_recent_3 = sum(
        1 for g in recent_3
        if ((sum(v[0] for v in g["quarters"].values()) > sum(v[1] for v in g["quarters"].values())) == (g["home_team"] == ht))
    )
    feats["h2h_recent3_home_won"] = home_won_recent_3 / len(recent_3) if recent_3 else 0.0

    last_g = past[-1]
    is_home_last = last_g["home_team"] == ht
    total_h_last = sum(v[0] for v in last_g["quarters"].values())
    total_a_last = sum(v[1] for v in last_g["quarters"].values())
    feats["h2h_last_home_won"] = int((total_h_last > total_a_last) == is_home_last)

    return feats


def _build_dynamic_samples(
    preloaded: dict[str, dict] | None = None,
) -> list[dict]:
    conn = None if preloaded is not None else v6.db_mod.get_conn(str(DB_PATH))
    if conn is not None:
        v6.db_mod.init_db(conn)

    base_samples, preloaded = _build_base_samples_and_data(DB_PATH)

    if conn is not None:
        conn.close()

    print("Building H2H lookup...")
    h2h_lookup = _build_h2h_lookup()

    rows: list[dict] = []
    for sample in tqdm(base_samples, desc="Building dynamic samples"):
        if sample.target_q4 is None:
            continue
        match_id = str(sample.match_id)
        match_data = preloaded.get(match_id)
        if not match_data or not _window_is_eligible_m27(match_data):
            continue

        feat = _build_m27_v3_features(sample, match_data)

        ht = match_data["match"]["home_team"]
        at = match_data["match"]["away_team"]
        h2h_feats = _compute_h2h_features(ht, at, sample.dt, h2h_lookup)
        feat.update(h2h_feats)

        rows.append({
            "features": feat,
            "target": int(sample.target_q4),
            "match_id": match_id,
            "dt": str(sample.dt),
            "snapshot_minute": SNAPSHOT_MINUTE,
        })

    rows.sort(key=lambda r: (r["dt"], r["match_id"]))
    return rows


def _split_rows_temporal_by_match(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
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
        match_id for match_id, _ in sorted(match_first_dt.items(), key=lambda item: (item[1], item[0]))
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


def _run_cv(
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
    vectorizer: DictVectorizer | None,
) -> dict:
    x_train_dict = [r["features"] for r in train_rows]
    x_val_dict = [r["features"] for r in val_rows]
    x_test_dict = [r["features"] for r in test_rows]
    y_train = [r["target"] for r in train_rows]
    y_val = [r["target"] for r in val_rows]
    y_test = [r["target"] for r in test_rows]

    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform(x_train_dict)
        x_val = vectorizer.transform(x_val_dict)
        x_test = vectorizer.transform(x_test_dict)
    else:
        x_train = vectorizer.transform(x_train_dict)
        x_val = vectorizer.transform(x_val_dict)
        x_test = vectorizer.transform(x_test_dict)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(x_train, y_train)

    x_train_dense = x_train.toarray() if hasattr(x_train, "toarray") else x_train
    x_val_dense = x_val.toarray() if hasattr(x_val, "toarray") else x_val
    x_test_dense = x_test.toarray() if hasattr(x_test, "toarray") else x_test

    hist_model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    hist_model.fit(x_train_dense, y_train)

    xgb_train_prob = xgb_model.predict_proba(x_train)[:, 1]
    xgb_val_prob = xgb_model.predict_proba(x_val)[:, 1]
    xgb_test_prob = xgb_model.predict_proba(x_test)[:, 1]

    hist_train_prob = hist_model.predict_proba(x_train_dense)[:, 1]
    hist_val_prob = hist_model.predict_proba(x_val_dense)[:, 1]
    hist_test_prob = hist_model.predict_proba(x_test_dense)[:, 1]

    xgb_val_auc = _safe_auc(y_val, xgb_val_prob.tolist()) or 0.5
    hist_val_auc = _safe_auc(y_val, hist_val_prob.tolist()) or 0.5
    total_w = xgb_val_auc + hist_val_auc
    w_xgb = xgb_val_auc / total_w if total_w > 0 else 0.5
    w_hist = hist_val_auc / total_w if total_w > 0 else 0.5
    ens_train_prob = w_xgb * xgb_train_prob + w_hist * hist_train_prob
    ens_val_prob = w_xgb * xgb_val_prob + w_hist * hist_val_prob
    ens_test_prob = w_xgb * xgb_test_prob + w_hist * hist_test_prob

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(ens_val_prob, y_val)
    cal_train_prob = np.clip(calibrator.transform(np.clip(ens_train_prob, 0.0, 1.0)), 0.0, 1.0)
    cal_val_prob = np.clip(calibrator.transform(np.clip(ens_val_prob, 0.0, 1.0)), 0.0, 1.0)
    cal_test_prob = np.clip(calibrator.transform(np.clip(ens_test_prob, 0.0, 1.0)), 0.0, 1.0)

    n_total = len(train_rows) + len(val_rows) + len(test_rows)
    results = {
        "vectorizer": vectorizer,
        "xgb_model": xgb_model,
        "hist_model": hist_model,
        "calibrator": calibrator,
        "metrics": [],
    }

    def _c(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, 0.0, 1.0)

    for split_label, y, probs_dict in [
        ("train", y_train, {
            "xgb": _c(xgb_train_prob),
            "hist": _c(hist_train_prob),
            "ensemble": _c(ens_train_prob),
            "cal_ensemble": _c(cal_train_prob),
        }),
        ("val", y_val, {
            "xgb": _c(xgb_val_prob),
            "hist": _c(hist_val_prob),
            "ensemble": _c(ens_val_prob),
            "cal_ensemble": _c(cal_val_prob),
        }),
        ("test", y_test, {
            "xgb": _c(xgb_test_prob),
            "hist": _c(hist_test_prob),
            "ensemble": _c(ens_test_prob),
            "cal_ensemble": _c(cal_test_prob),
        }),
    ]:
        for model_name, probs in probs_dict.items():
            results["metrics"].append(
                _metric_row(
                    model_name, split_label,
                    n_total, len(train_rows), len(val_rows), len(test_rows),
                    y, probs.tolist(),
                )
            )

    return results


def train() -> dict:
    global ENABLE_PACE_FEATURES
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    cache_path = DYNAMIC_ROWS_CACHE
    if cache_path.exists():
        all_rows = joblib.load(cache_path)
        print(f"Loaded {len(all_rows)} cached rows")
    else:
        print("Building dynamic samples ...")
        t0 = time.time()
        all_rows = _build_dynamic_samples()
        print(f"  Built {len(all_rows)} rows in {time.time() - t0:.1f}s")
        joblib.dump(all_rows, cache_path)

    n = len(all_rows)
    if n < 300:
        raise RuntimeError(f"Too few rows: {n}")

    if FILTER_10M:
        if not ENABLE_PACE_FEATURES:
            print("WARNING: --filter-10m requires --enable-pace-features to detect quarter length")
            print("Re-building rows with pace features enabled for filtering...")
            ENABLE_PACE_FEATURES = True
            # Rebuild all rows so regulation_quarter_minutes is in features
            all_rows = _build_dynamic_samples()
            joblib.dump(all_rows, cache_path)
        pre_filter_count = len(all_rows)
        all_rows = [
            row for row in all_rows
            if abs(row["features"].get("regulation_quarter_minutes", 10.0) - 10.0) < 1e-9
        ]
        print(f"Filtered to 10m: {pre_filter_count} -> {len(all_rows)} rows")
        drop_feats = TENM_CONSTANT_FEATURES | TENM_REDUNDANT_FEATURES
        for row in all_rows:
            for feat in drop_feats:
                row["features"].pop(feat, None)
        print(f"Dropped constant/redundant features: {len(drop_feats)}")

    train_rows, val_rows, test_rows, split_info = _split_rows_temporal_by_match(all_rows)
    n_train = len(train_rows)
    n_val = len(val_rows)
    n_test = len(test_rows)
    if min(n_train, n_val, n_test) == 0:
        raise RuntimeError("Empty split")

    y_train_vals = [r["target"] for r in train_rows]
    y_val_vals = [r["target"] for r in val_rows]
    y_test_vals = [r["target"] for r in test_rows]
    if len(set(y_train_vals)) < 2 or len(set(y_val_vals)) < 2 or len(set(y_test_vals)) < 2:
        raise RuntimeError("Insufficient classes in at least one split")

    print(f"Train: {n_train} ({split_info['matches_train']} matches), "
          f"Val: {n_val} ({split_info['matches_val']} matches), "
          f"Test: {n_test} ({split_info['matches_test']} matches)")

    outer = _run_cv(train_rows, val_rows, test_rows, vectorizer=None)
    all_cv_metrics = list(outer["metrics"])

    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)
    print(f"Running {CV_N_SPLITS}-fold TimeSeries CV ...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_rows)):
        fold_train = [train_rows[i] for i in train_idx]
        fold_val = [train_rows[i] for i in val_idx]
        fold_result = _run_cv(
            fold_train, fold_val, test_rows,
            vectorizer=outer["vectorizer"],
        )
        for m in fold_result["metrics"]:
            m["fold"] = fold
        all_cv_metrics.extend(fold_result["metrics"])
        print(f"  Fold {fold}: train={len(fold_train)} val={len(fold_val)}")

    results = {
        "config": {
            "version": "m27_v3",
            "snapshot": SNAPSHOT_MINUTE,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "cv_n_splits": CV_N_SPLITS,
        },
        "samples": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "total": len(all_rows),
        },
        "metrics": all_cv_metrics,
        "model_files": {},
        "trained_at": time.time(),
    }

    vectorizer_path = OUT_DIR / "m27_v3_vectorizer.joblib"
    joblib.dump(outer["vectorizer"], vectorizer_path)
    results["model_files"]["vectorizer"] = str(vectorizer_path)

    xgb_path = OUT_DIR / "m27_v3_xgb.joblib"
    joblib.dump(outer["xgb_model"], xgb_path)
    results["model_files"]["xgb"] = str(xgb_path)

    hist_path = OUT_DIR / "m27_v3_histgb.joblib"
    joblib.dump(outer["hist_model"], hist_path)
    results["model_files"]["histgb"] = str(hist_path)

    cal_path = OUT_DIR / "m27_v3_calibrator.joblib"
    joblib.dump(outer["calibrator"], cal_path)
    results["model_files"]["calibrator"] = str(cal_path)

    summary = {"m27_v3": results}
    summary_path = OUT_DIR / "training_summary_m27_v3.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")

    test_metrics = [m for m in all_cv_metrics if m["split"] == "test" and m.get("fold") is None]
    print("\nTest set metrics (fold 0):")
    for m in sorted(test_metrics, key=lambda x: x["model"]):
        auc_str = f"auc={m['roc_auc']:.4f}" if m["roc_auc"] is not None else "auc=N/A"
        print(f"  {m['model']:20s} acc={m['accuracy']:.4f}  {auc_str}  brier={m['brier']:.4f}  f1={m['f1']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train m27_v3 Q4 model")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild dynamic rows cache")
    parser.add_argument("--filter-10m", action="store_true", help="Only 10-minute regulation leagues")
    parser.add_argument("--enable-pace-features", action="store_true", help="Enable Q3 pace/regulation features")
    args = parser.parse_args()

    if args.rebuild_cache:
        cache = DYNAMIC_ROWS_CACHE
        if cache.exists():
            cache.unlink()
            print(f"Removed cache {cache}")

    global FILTER_10M, ENABLE_PACE_FEATURES
    if args.filter_10m:
        FILTER_10M = True
        print("10m filter enabled")
    if args.enable_pace_features:
        ENABLE_PACE_FEATURES = True
        print("Pace features enabled")

    train()


if __name__ == "__main__":
    main()

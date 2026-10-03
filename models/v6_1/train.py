"""Train V6.1 multi-model predictors for quarter winners (Q3, Q4).

V6.1 keeps the V6 feature engineering family and upgrades the training flow:
temporal train/validation/test split, isotonic calibration, optimized
thresholds, validation-AUC weighted ensembles, and possession-level Monte Carlo.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_q3_q4_models_v6 as v6

ROOT = v6.ROOT
DB_PATH = v6.DB_PATH
OUT_DIR = ROOT / "training" / "model_outputs_v6_1"
TOP_LEAGUES = 50
TOP_TEAMS = 300
TEAM_HISTORY_WINDOW = v6.TEAM_HISTORY_WINDOW
ENSEMBLE_AUC_WINDOW = 0.03
CALIBRATION_LOGLOSS_TOLERANCE = 0.002
CHAMPION_MIN_AUC_DELTA = 0.001


def _safe_rate(num: float, den: float) -> float:
    return v6._safe_rate(num, den)


def _write_csv(path: Path, rows: list[dict]) -> None:
    v6._write_csv(path, rows)


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _pbp_stats_upto(pbp: dict, quarters: list[str]) -> dict:
    home_plays = 0
    away_plays = 0
    home_3pt = 0
    away_3pt = 0
    home_pts = 0
    away_pts = 0
    for quarter in quarters:
        for play in pbp.get(quarter, []):
            team = play.get("team")
            pts = int(play.get("points", 0))
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
        "pbp_home_pts_per_play": _safe_rate(home_pts, home_plays),
        "pbp_away_pts_per_play": _safe_rate(away_pts, away_plays),
        "pbp_pts_per_play_diff": _safe_rate(home_pts, home_plays)
        - _safe_rate(away_pts, away_plays),
        "pbp_home_plays": home_plays,
        "pbp_away_plays": away_plays,
        "pbp_plays_diff": home_plays - away_plays,
        "pbp_home_3pt": home_3pt,
        "pbp_away_3pt": away_3pt,
        "pbp_3pt_diff": home_3pt - away_3pt,
        "pbp_home_plays_share": _safe_rate(home_plays, total_plays),
        "pbp_home_3pt_share": _safe_rate(home_3pt, total_3pt),
        "pbp_away_3pt_share": _safe_rate(away_3pt, total_3pt),
        "pbp_home_3pt_play_share": _safe_rate(home_3pt, home_plays),
        "pbp_away_3pt_play_share": _safe_rate(away_3pt, away_plays),
    }


def _point_probs(points_per_play: float, three_point_share: float) -> np.ndarray:
    p3 = float(np.clip(three_point_share, 0.0, 0.7))
    remaining_ppp = max(0.0, points_per_play - 3.0 * p3)
    p2 = float(np.clip(remaining_ppp / 2.0, 0.0, 1.0 - p3))
    p0 = max(0.0, 1.0 - p2 - p3)
    total = p0 + p2 + p3
    if total <= 0:
        return np.array([1.0, 0.0, 0.0])
    return np.array([p0 / total, p2 / total, p3 / total])


def _simulate_team_points(
    rng: np.random.Generator,
    possessions: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    p0, p2, p3 = probs
    threes = rng.binomial(possessions, p3)
    remaining = possessions - threes
    two_prob = 0.0 if p0 + p2 <= 0 else p2 / (p0 + p2)
    twos = rng.binomial(remaining, two_prob)
    return 2 * twos + 3 * threes


def _monte_carlo_rest_game_features(
    *,
    score_home: int,
    score_away: int,
    pbp_home_plays: int,
    pbp_away_plays: int,
    pbp_home_pts_per_play: float,
    pbp_home_3pt_share: float,
    pbp_away_pts_per_play: float,
    pbp_away_3pt_share: float,
    elapsed_minutes: float,
    minutes_left: float,
    seed_key: str,
    num_sims: int = 2000,
) -> dict:
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

    possessions_left = max(1, int(round(_safe_rate(total_plays, elapsed_minutes) * minutes_left)))
    home_play_share = float(np.clip(_safe_rate(pbp_home_plays, total_plays), 0.05, 0.95))
    home_probs = _point_probs(pbp_home_pts_per_play, pbp_home_3pt_share)
    away_probs = _point_probs(pbp_away_pts_per_play, pbp_away_3pt_share)
    rng = np.random.default_rng(_stable_seed(seed_key))

    home_possessions = rng.binomial(possessions_left, home_play_share, size=num_sims)
    away_possessions = possessions_left - home_possessions
    home_scores = score_home + _simulate_team_points(rng, home_possessions, home_probs)
    away_scores = score_away + _simulate_team_points(rng, away_possessions, away_probs)

    start_diff = score_home - score_away
    final_diff = home_scores - away_scores
    current_leader = 1 if start_diff > 0 else (-1 if start_diff < 0 else 0)
    final_leader = np.where(final_diff > 0, 1, np.where(final_diff < 0, -1, 0))
    ties = np.abs(final_diff) < 0.5

    if current_leader == 0:
        comeback_rate = 0.0
    else:
        comeback_rate = float(np.mean((final_leader != current_leader) & (final_leader != 0)))

    return {
        "mc_home_win_prob": float(np.mean(final_diff > 0) + 0.5 * np.mean(ties)),
        "mc_expected_diff": float(np.mean(final_diff)),
        "mc_cover_rate": float(np.mean(final_diff > start_diff)),
        "mc_std_diff": float(np.std(final_diff)),
        "mc_comeback_rate": comeback_rate,
    }


def _build_samples(db_path: Path) -> list[v6.MatchSample]:
    conn = v6.db_mod.get_conn(str(db_path))
    v6.db_mod.init_db(conn)

    top_leagues, top_teams = v6._collect_top_buckets(
        conn,
        top_leagues=TOP_LEAGUES,
        top_teams=TOP_TEAMS,
    )

    rows = conn.execute(
        "SELECT match_id, date, time FROM matches "
        "ORDER BY date, time, match_id"
    ).fetchall()

    team_history: dict[str, list[int]] = defaultdict(list)
    samples: list[v6.MatchSample] = []

    for row in rows:
        match_id = str(row["match_id"])
        dt = datetime.strptime(
            f"{row['date']} {row['time']}",
            "%Y-%m-%d %H:%M",
        )
        data = v6.db_mod.get_match(conn, match_id)
        if data is None or not v6._is_complete_match(data):
            continue

        match_info = data["match"]
        score = data["score"]
        pbp = data.get("play_by_play", {})
        graph_points = data.get("graph_points", [])

        q1h, q1a = v6._quarter_points(data, "Q1")
        q2h, q2a = v6._quarter_points(data, "Q2")
        q3h, q3a = v6._quarter_points(data, "Q3")
        q4h, q4a = v6._quarter_points(data, "Q4")
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue

        home_team = match_info.get("home_team", "")
        away_team = match_info.get("away_team", "")
        league = match_info.get("league", "")

        home_hist = team_history[home_team][-TEAM_HISTORY_WINDOW:]
        away_hist = team_history[away_team][-TEAM_HISTORY_WINDOW:]
        home_prior_wr = _safe_rate(sum(home_hist), len(home_hist))
        away_prior_wr = _safe_rate(sum(away_hist), len(away_hist))

        base = {
            "league": league,
            "league_bucket": v6._bucket(league, top_leagues, "LEAGUE"),
            "gender_bucket": v6._infer_gender(league, home_team, away_team),
            "home_team_bucket": v6._bucket(home_team, top_teams, "TEAM"),
            "away_team_bucket": v6._bucket(away_team, top_teams, "TEAM"),
            "home_prior_wr": home_prior_wr,
            "away_prior_wr": away_prior_wr,
            "prior_wr_diff": home_prior_wr - away_prior_wr,
            "prior_wr_sum": home_prior_wr + away_prior_wr,
            "q1_diff": q1h - q1a,
            "q2_diff": q2h - q2a,
        }

        ht_home = q1h + q2h
        ht_away = q1a + q2a

        features_q3 = dict(base)
        features_q3.update({
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_diff": ht_home - ht_away,
            "ht_total": ht_home + ht_away,
        })
        q3_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2"])
        features_q3.update(v6._graph_stats_upto(graph_points, 24))
        features_q3.update(q3_pbp_stats)
        features_q3.update(
            v6._score_pressure_features(
                score_home=ht_home,
                score_away=ht_away,
                pbp_home_plays=q3_pbp_stats["pbp_home_plays"],
                pbp_away_plays=q3_pbp_stats["pbp_away_plays"],
                pbp_home_3pt=q3_pbp_stats["pbp_home_3pt"],
                pbp_away_3pt=q3_pbp_stats["pbp_away_3pt"],
                elapsed_minutes=24.0,
                minutes_left=12.0,
            )
        )
        features_q3.update(
            v6._pbp_recent_window_features(
                pbp,
                cutoff_minute=24.0,
                window_minutes=6.0,
            )
        )
        features_q3.update(
            _monte_carlo_rest_game_features(
                score_home=ht_home,
                score_away=ht_away,
                pbp_home_plays=q3_pbp_stats["pbp_home_plays"],
                pbp_away_plays=q3_pbp_stats["pbp_away_plays"],
                pbp_home_pts_per_play=q3_pbp_stats["pbp_home_pts_per_play"],
                pbp_home_3pt_share=q3_pbp_stats["pbp_home_3pt_play_share"],
                pbp_away_pts_per_play=q3_pbp_stats["pbp_away_pts_per_play"],
                pbp_away_3pt_share=q3_pbp_stats["pbp_away_3pt_play_share"],
                elapsed_minutes=24.0,
                minutes_left=12.0,
                seed_key=f"{match_id}:q3",
            )
        )

        features_q4 = dict(base)
        score_3q_home = ht_home + q3h
        score_3q_away = ht_away + q3a
        features_q4.update({
            "q3_diff": q3h - q3a,
            "q3_total": q3h + q3a,
            "score_3q_home": score_3q_home,
            "score_3q_away": score_3q_away,
            "score_3q_diff": score_3q_home - score_3q_away,
        })
        q4_pbp_stats = _pbp_stats_upto(pbp, ["Q1", "Q2", "Q3"])
        features_q4.update(v6._graph_stats_upto(graph_points, 36))
        features_q4.update(q4_pbp_stats)
        features_q4.update(
            v6._score_pressure_features(
                score_home=score_3q_home,
                score_away=score_3q_away,
                pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
                pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
                pbp_home_3pt=q4_pbp_stats["pbp_home_3pt"],
                pbp_away_3pt=q4_pbp_stats["pbp_away_3pt"],
                elapsed_minutes=36.0,
                minutes_left=12.0,
            )
        )
        features_q4.update(
            v6._pbp_recent_window_features(
                pbp,
                cutoff_minute=36.0,
                window_minutes=6.0,
            )
        )
        features_q4.update(
            _monte_carlo_rest_game_features(
                score_home=score_3q_home,
                score_away=score_3q_away,
                pbp_home_plays=q4_pbp_stats["pbp_home_plays"],
                pbp_away_plays=q4_pbp_stats["pbp_away_plays"],
                pbp_home_pts_per_play=q4_pbp_stats["pbp_home_pts_per_play"],
                pbp_home_3pt_share=q4_pbp_stats["pbp_home_3pt_play_share"],
                pbp_away_pts_per_play=q4_pbp_stats["pbp_away_pts_per_play"],
                pbp_away_3pt_share=q4_pbp_stats["pbp_away_3pt_play_share"],
                elapsed_minutes=36.0,
                minutes_left=12.0,
                seed_key=f"{match_id}:q4",
            )
        )

        target_q3 = None if q3h == q3a else int(q3h > q3a)
        target_q4 = None if q4h == q4a else int(q4h > q4a)

        samples.append(
            v6.MatchSample(
                match_id=match_id,
                dt=dt,
                features_q3=features_q3,
                target_q3=target_q3,
                features_q4=features_q4,
                target_q4=target_q4,
            )
        )

        home_win = int(score["home"] > score["away"])
        away_win = 1 - home_win
        team_history[home_team].append(home_win)
        team_history[away_team].append(away_win)

    conn.close()
    return samples


def _best_threshold(y_true: list[int], probs: list[float]) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.50, 0.901, 0.01):
        preds = [1 if p >= threshold else 0 for p in probs]
        score = float(f1_score(y_true, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(round(threshold, 2))
    return best_t, best_f1


def _safe_auc(y_true: list[int], probs: list[float]) -> float | None:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return None


def _iter_weight_vectors(n_items: int, step: float):
    units = int(round(1.0 / step))

    def rec(remaining_items: int, remaining_units: int, prefix: list[int]):
        if remaining_items == 1:
            yield prefix + [remaining_units]
            return
        for units_value in range(remaining_units + 1):
            yield from rec(
                remaining_items - 1,
                remaining_units - units_value,
                prefix + [units_value],
            )

    for vector in rec(n_items, units, []):
        yield [value / units for value in vector]


def _weighted_probs(
    proba_map: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    return sum(proba_map[name] * weights.get(name, 0.0) for name in proba_map)


def _best_grid_ensemble_weights(
    y_true: list[int],
    proba_map: dict[str, np.ndarray],
    candidate_models: list[str],
) -> tuple[dict[str, float], np.ndarray, float | None, float | None, float]:
    if not candidate_models:
        candidate_models = list(proba_map)

    step = 0.01 if len(candidate_models) == 2 else 0.05
    best_weights = {name: 1.0 / len(candidate_models) for name in candidate_models}
    best_probs = _weighted_probs(proba_map, best_weights)
    best_auc = _safe_auc(y_true, list(best_probs))
    best_log_loss = float(log_loss(y_true, best_probs, labels=[0, 1]))

    for vector in _iter_weight_vectors(len(candidate_models), step):
        weights = {
            name: float(weight)
            for name, weight in zip(candidate_models, vector)
        }
        probs = _weighted_probs(proba_map, weights)
        auc = _safe_auc(y_true, list(probs))
        if auc is None:
            continue
        prob_log_loss = float(log_loss(y_true, probs, labels=[0, 1]))
        if (
            best_auc is None
            or auc > best_auc + 1e-12
            or (abs(auc - best_auc) <= 1e-12 and prob_log_loss < best_log_loss)
        ):
            best_auc = auc
            best_log_loss = prob_log_loss
            best_weights = weights
            best_probs = probs

    full_weights = {name: 0.0 for name in proba_map}
    full_weights.update(best_weights)
    return full_weights, best_probs, best_auc, best_log_loss, step


def _metric_row(
    target: str,
    model_name: str,
    split: str,
    n_total: int,
    n_train: int,
    n_validation: int,
    n_test: int,
    y_true: list[int],
    probs: list[float],
    threshold: float,
) -> dict:
    preds = [1 if p >= threshold else 0 for p in probs]
    metric = {
        "target": target,
        "model": model_name,
        "split": split,
        "samples_total": n_total,
        "samples_train": n_train,
        "samples_validation": n_validation,
        "samples_test": n_test,
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 6),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 6),
        "log_loss": round(float(log_loss(y_true, probs, labels=[0, 1])), 6),
        "brier": round(float(brier_score_loss(y_true, probs)), 6),
    }

    auc = _safe_auc(y_true, probs)
    metric["roc_auc"] = None if auc is None else round(auc, 6)
    return metric


def _probability_diagnostics(
    *,
    target: str,
    model_name: str,
    y_true: list[int],
    raw_probs: list[float],
    calibrated_probs: list[float],
) -> dict:
    raw_auc = _safe_auc(y_true, raw_probs)
    calibrated_auc = _safe_auc(y_true, calibrated_probs)
    raw_log_loss = float(log_loss(y_true, raw_probs, labels=[0, 1]))
    calibrated_log_loss = float(log_loss(y_true, calibrated_probs, labels=[0, 1]))
    raw_brier = float(brier_score_loss(y_true, raw_probs))
    calibrated_brier = float(brier_score_loss(y_true, calibrated_probs))
    use_calibrated = calibrated_log_loss <= raw_log_loss + CALIBRATION_LOGLOSS_TOLERANCE

    return {
        "target": target,
        "model": model_name,
        "raw_validation_auc": None if raw_auc is None else round(raw_auc, 6),
        "calibrated_validation_auc": None
        if calibrated_auc is None
        else round(calibrated_auc, 6),
        "raw_log_loss": round(raw_log_loss, 6),
        "calibrated_log_loss": round(calibrated_log_loss, 6),
        "raw_brier": round(raw_brier, 6),
        "calibrated_brier": round(calibrated_brier, 6),
        "ensemble_probability_source": "calibrated" if use_calibrated else "raw",
        "log_loss_delta_calibrated_minus_raw": round(
            calibrated_log_loss - raw_log_loss,
            6,
        ),
    }


def _make_calibrator(fitted_model):
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(fitted_model), method="isotonic")
    except ImportError:
        return CalibratedClassifierCV(fitted_model, method="isotonic", cv="prefit")


def _model_specs() -> dict:
    return {
        "xgb": {
            "model": xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                early_stopping_rounds=50,
            ),
            "matrix": "sparse",
        },
        "hist_gb": {
            "model": HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.03,
                max_depth=5,
                random_state=42,
            ),
            "matrix": "dense",
        },
        "mlp": {
            "model": make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=42,
                ),
            ),
            "matrix": "dense",
        },
    }


def _matrix_for(spec: dict, sparse_matrix, dense_matrix):
    return sparse_matrix if spec["matrix"] == "sparse" else dense_matrix


def _fit_model(
    model_name: str,
    model,
    x_train,
    y_train: list[int],
    x_validation,
    y_validation: list[int],
) -> None:
    if model_name == "xgb":
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        return
    model.fit(x_train, y_train)


def _split_label(index: int, n_train: int, validation_end: int) -> str:
    if index < n_train:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def _new_support_record(key: str) -> dict:
    record = {
        "key": key,
        "total_rows": 0,
        "train_rows": 0,
        "validation_rows": 0,
        "test_rows": 0,
        "positive_rows": 0,
        "first_date": None,
        "last_date": None,
        "train_first_date": None,
        "train_last_date": None,
        "validation_first_date": None,
        "validation_last_date": None,
        "test_first_date": None,
        "test_last_date": None,
    }
    return record


def _update_date_range(record: dict, prefix: str, date_value: str) -> None:
    first_key = f"{prefix}_first_date" if prefix else "first_date"
    last_key = f"{prefix}_last_date" if prefix else "last_date"
    if record[first_key] is None or date_value < record[first_key]:
        record[first_key] = date_value
    if record[last_key] is None or date_value > record[last_key]:
        record[last_key] = date_value


def _support_by_key(rows: list[dict], key_name: str, target_col: str) -> dict[str, dict]:
    support: dict[str, dict] = {}
    for row in rows:
        key = str(row.get(key_name, ""))
        record = support.setdefault(key, _new_support_record(key))
        split = row["support_split"]
        date_value = str(row["datetime"])[:10]

        record["total_rows"] += 1
        record[f"{split}_rows"] += 1
        record["positive_rows"] += int(row[target_col])
        _update_date_range(record, "", date_value)
        _update_date_range(record, split, date_value)

    for record in support.values():
        record["target_positive_rate"] = round(
            _safe_rate(record["positive_rows"], record["total_rows"]),
            6,
        )
        for key, value in list(record.items()):
            if value is None:
                record[key] = ""
    return support


def _support_rows(
    target_name: str,
    support: dict[str, dict],
    key_name: str,
) -> list[dict]:
    rows = []
    for key, record in sorted(
        support.items(),
        key=lambda item: (-int(item[1]["total_rows"]), item[0]),
    ):
        row = {"target": target_name, key_name: key}
        for field, value in record.items():
            if field != "key":
                row[field] = value
        rows.append(row)
    return rows


def _add_support_columns(
    rows: list[dict],
    support: dict[str, dict],
    row_key: str,
    prefix: str,
) -> None:
    for row in rows:
        record = support[str(row.get(row_key, ""))]
        row[f"{prefix}_total_rows"] = record["total_rows"]
        row[f"{prefix}_train_rows"] = record["train_rows"]
        row[f"{prefix}_validation_rows"] = record["validation_rows"]
        row[f"{prefix}_test_rows"] = record["test_rows"]
        row[f"{prefix}_target_positive_rate"] = record["target_positive_rate"]
        row[f"{prefix}_first_date"] = record["first_date"]
        row[f"{prefix}_last_date"] = record["last_date"]
        row[f"{prefix}_train_first_date"] = record["train_first_date"]
        row[f"{prefix}_train_last_date"] = record["train_last_date"]
        row[f"{prefix}_validation_first_date"] = record["validation_first_date"]
        row[f"{prefix}_validation_last_date"] = record["validation_last_date"]
        row[f"{prefix}_test_first_date"] = record["test_first_date"]
        row[f"{prefix}_test_last_date"] = record["test_last_date"]


def _dataset_rows_with_support(
    samples: list[v6.MatchSample],
    target_name: str,
) -> tuple[list[dict], dict[str, list[dict]]]:
    if target_name == "q3":
        target_rows = [s for s in samples if s.target_q3 is not None]
        feature_attr = "features_q3"
        target_attr = "target_q3"
        target_col = "target_q3_home_win"
    else:
        target_rows = [s for s in samples if s.target_q4 is not None]
        feature_attr = "features_q4"
        target_attr = "target_q4"
        target_col = "target_q4_home_win"

    target_rows = sorted(target_rows, key=lambda item: item.dt)
    n_total = len(target_rows)
    n_train = int(n_total * 0.70)
    validation_end = int(n_total * 0.85)

    rows = []
    for idx, sample in enumerate(target_rows):
        row = {
            "match_id": sample.match_id,
            "datetime": sample.dt.isoformat(),
            "support_split": _split_label(idx, n_train, validation_end),
        }
        row.update(getattr(sample, feature_attr))
        row[target_col] = getattr(sample, target_attr)
        rows.append(row)

    league_support = _support_by_key(rows, "league", target_col)
    league_bucket_support = _support_by_key(rows, "league_bucket", target_col)
    home_team_bucket_support = _support_by_key(rows, "home_team_bucket", target_col)
    away_team_bucket_support = _support_by_key(rows, "away_team_bucket", target_col)

    _add_support_columns(rows, league_support, "league", "support_league")
    _add_support_columns(
        rows,
        league_bucket_support,
        "league_bucket",
        "support_league_bucket",
    )
    for row in rows:
        home_support = home_team_bucket_support[str(row.get("home_team_bucket", ""))]
        away_support = away_team_bucket_support[str(row.get("away_team_bucket", ""))]
        row["support_home_team_bucket_total_rows"] = home_support["total_rows"]
        row["support_home_team_bucket_first_date"] = home_support["first_date"]
        row["support_home_team_bucket_last_date"] = home_support["last_date"]
        row["support_away_team_bucket_total_rows"] = away_support["total_rows"]
        row["support_away_team_bucket_first_date"] = away_support["first_date"]
        row["support_away_team_bucket_last_date"] = away_support["last_date"]

    support_exports = {
        "league": _support_rows(target_name, league_support, "league"),
        "league_bucket": _support_rows(
            target_name,
            league_bucket_support,
            "league_bucket",
        ),
        "home_team_bucket": _support_rows(
            target_name,
            home_team_bucket_support,
            "home_team_bucket",
        ),
        "away_team_bucket": _support_rows(
            target_name,
            away_team_bucket_support,
            "away_team_bucket",
        ),
    }
    return rows, support_exports


def _train_target(samples: list[v6.MatchSample], target_name: str) -> dict:
    if target_name == "q3":
        target_rows = [s for s in samples if s.target_q3 is not None]
        target_rows = sorted(target_rows, key=lambda item: item.dt)
        x_dict = [s.features_q3 for s in target_rows]
        y = [int(s.target_q3) for s in target_rows]
    else:
        target_rows = [s for s in samples if s.target_q4 is not None]
        target_rows = sorted(target_rows, key=lambda item: item.dt)
        x_dict = [s.features_q4 for s in target_rows]
        y = [int(s.target_q4) for s in target_rows]

    n_total = len(target_rows)
    if n_total < 200:
        raise RuntimeError(
            f"Not enough rows for {target_name}. Need >=200, got {n_total}."
        )

    n_train = int(n_total * 0.70)
    n_validation = int(n_total * 0.85) - n_train
    n_test = n_total - n_train - n_validation
    validation_end = n_train + n_validation

    vec = DictVectorizer(sparse=True)
    x_all_sparse = vec.fit_transform(x_dict)
    x_all_dense = x_all_sparse.toarray()

    y_train = y[:n_train]
    y_validation = y[n_train:validation_end]
    y_test = y[validation_end:]

    models = _model_specs()
    metrics_rows = []
    threshold_rows = []
    weights_rows = []
    calibration_rows = []
    champion_rows = []
    feature_importance_rows = []
    validation_proba_map = {}
    test_proba_map = {}
    ensemble_validation_proba_map = {}
    ensemble_test_proba_map = {}
    validation_auc_map = {}
    ensemble_validation_auc_map = {}
    ensemble_source_map = {}
    threshold_map = {}

    for model_name, spec in models.items():
        x_model = _matrix_for(spec, x_all_sparse, x_all_dense)
        x_train = x_model[:n_train]
        x_validation = x_model[n_train:validation_end]
        x_test = x_model[validation_end:]

        base_model = spec["model"]
        _fit_model(
            model_name,
            base_model,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        calibrated_model = _make_calibrator(base_model)
        calibrated_model.fit(x_validation, y_validation)

        raw_validation_probs = base_model.predict_proba(x_validation)[:, 1]
        raw_test_probs = base_model.predict_proba(x_test)[:, 1]
        validation_probs = calibrated_model.predict_proba(x_validation)[:, 1]
        test_probs = calibrated_model.predict_proba(x_test)[:, 1]
        threshold, validation_f1 = _best_threshold(y_validation, list(validation_probs))
        validation_auc = _safe_auc(y_validation, list(validation_probs))
        raw_validation_auc = _safe_auc(y_validation, list(raw_validation_probs))
        diagnostic = _probability_diagnostics(
            target=target_name,
            model_name=model_name,
            y_true=y_validation,
            raw_probs=list(raw_validation_probs),
            calibrated_probs=list(validation_probs),
        )
        calibration_rows.append(diagnostic)
        use_calibrated_for_ensemble = (
            diagnostic["ensemble_probability_source"] == "calibrated"
        )
        ensemble_validation_probs = (
            validation_probs if use_calibrated_for_ensemble else raw_validation_probs
        )
        ensemble_test_probs = test_probs if use_calibrated_for_ensemble else raw_test_probs
        ensemble_validation_auc = (
            validation_auc if use_calibrated_for_ensemble else raw_validation_auc
        )

        validation_proba_map[model_name] = np.asarray(validation_probs)
        test_proba_map[model_name] = np.asarray(test_probs)
        ensemble_validation_proba_map[model_name] = np.asarray(ensemble_validation_probs)
        ensemble_test_proba_map[model_name] = np.asarray(ensemble_test_probs)
        validation_auc_map[model_name] = validation_auc
        ensemble_validation_auc_map[model_name] = ensemble_validation_auc
        ensemble_source_map[model_name] = diagnostic["ensemble_probability_source"]
        threshold_map[model_name] = threshold

        threshold_rows.append(
            {
                "target": target_name,
                "model": model_name,
                "threshold": threshold,
                "validation_f1": round(validation_f1, 6),
                "validation_auc": None
                if validation_auc is None
                else round(validation_auc, 6),
            }
        )
        metrics_rows.append(
            _metric_row(
                target=target_name,
                model_name=model_name,
                split="validation",
                n_total=n_total,
                n_train=n_train,
                n_validation=n_validation,
                n_test=n_test,
                y_true=y_validation,
                probs=list(validation_probs),
                threshold=threshold,
            )
        )
        metrics_rows.append(
            _metric_row(
                target=target_name,
                model_name=model_name,
                split="test",
                n_total=n_total,
                n_train=n_train,
                n_validation=n_validation,
                n_test=n_test,
                y_true=y_test,
                probs=list(test_probs),
                threshold=threshold,
            )
        )

        if model_name == "xgb":
            importances = getattr(base_model, "feature_importances_", None)
            if importances is not None:
                for feature, importance in sorted(
                    zip(vec.feature_names_, importances),
                    key=lambda item: float(item[1]),
                    reverse=True,
                ):
                    feature_importance_rows.append(
                        {
                            "target": target_name,
                            "feature": feature,
                            "importance": float(importance),
                        }
                    )

        artifact = {
            "version": "v6.1",
            "target": target_name,
            "model_name": model_name,
            "vectorizer": vec,
            "model": calibrated_model,
            "base_model": base_model,
            "threshold": threshold,
            "calibration_info": {
                "method": "isotonic",
                "calibration_split": "validation",
                "calibrated_probabilities_used_for_metrics": True,
                "ensemble_probability_source": diagnostic[
                    "ensemble_probability_source"
                ],
                "raw_validation_auc": diagnostic["raw_validation_auc"],
                "calibrated_validation_auc": diagnostic["calibrated_validation_auc"],
                "raw_log_loss": diagnostic["raw_log_loss"],
                "calibrated_log_loss": diagnostic["calibrated_log_loss"],
            },
            "matrix": spec["matrix"],
            "trained_rows": n_total,
            "split": {
                "train": n_train,
                "validation": n_validation,
                "test": n_test,
            },
            "feature_count": len(vec.feature_names_),
        }
        joblib.dump(artifact, OUT_DIR / f"{target_name}_{model_name}.joblib")

    usable_aucs = {
        name: auc
        for name, auc in ensemble_validation_auc_map.items()
        if auc is not None
    }
    best_candidate_auc = max(usable_aucs.values()) if usable_aucs else 0.0
    candidate_models = [
        name
        for name, auc in usable_aucs.items()
        if auc >= best_candidate_auc - ENSEMBLE_AUC_WINDOW
    ]
    if not candidate_models:
        candidate_models = list(models)

    ensemble_weights, ensemble_validation_probs, ensemble_validation_auc, ensemble_validation_log_loss, grid_step = (
        _best_grid_ensemble_weights(
            y_validation,
            ensemble_validation_proba_map,
            candidate_models,
        )
    )
    ensemble_test_probs = _weighted_probs(
        ensemble_test_proba_map,
        ensemble_weights,
    )
    ensemble_threshold, ensemble_validation_f1 = _best_threshold(
        y_validation,
        list(ensemble_validation_probs),
    )

    for model_name in models:
        weights_rows.append(
            {
                "target": target_name,
                "model": model_name,
                "validation_auc": None
                if ensemble_validation_auc_map[model_name] is None
                else round(ensemble_validation_auc_map[model_name], 6),
                "probability_source": ensemble_source_map[model_name],
                "eligible": int(model_name in candidate_models),
                "grid_step": grid_step,
                "ensemble_validation_log_loss": None
                if ensemble_validation_log_loss is None
                else round(ensemble_validation_log_loss, 6),
                "weight": round(float(ensemble_weights[model_name]), 8),
            }
        )

    threshold_rows.append(
        {
            "target": target_name,
            "model": "ensemble_weighted_auc",
            "threshold": ensemble_threshold,
            "validation_f1": round(ensemble_validation_f1, 6),
            "validation_auc": None
            if ensemble_validation_auc is None
            else round(ensemble_validation_auc, 6),
        }
    )
    metrics_rows.append(
        _metric_row(
            target=target_name,
            model_name="ensemble_weighted_auc",
            split="validation",
            n_total=n_total,
            n_train=n_train,
            n_validation=n_validation,
            n_test=n_test,
            y_true=y_validation,
            probs=list(ensemble_validation_probs),
            threshold=ensemble_threshold,
        )
    )

    best_individual_name = max(
        validation_auc_map,
        key=lambda name: -1.0
        if validation_auc_map[name] is None
        else float(validation_auc_map[name]),
    )
    best_individual_validation_auc = validation_auc_map[best_individual_name]
    if (
        ensemble_validation_auc is not None
        and best_individual_validation_auc is not None
        and ensemble_validation_auc
        > best_individual_validation_auc + CHAMPION_MIN_AUC_DELTA
    ):
        champion_name = "ensemble_weighted_auc"
        champion_threshold = ensemble_threshold
        champion_validation_probs = ensemble_validation_probs
        champion_test_probs = ensemble_test_probs
        champion_validation_auc = ensemble_validation_auc
    else:
        champion_name = best_individual_name
        champion_threshold = threshold_map[best_individual_name]
        champion_validation_probs = validation_proba_map[best_individual_name]
        champion_test_probs = test_proba_map[best_individual_name]
        champion_validation_auc = best_individual_validation_auc

    champion_validation_metric = _metric_row(
        target=target_name,
        model_name="champion",
        split="validation",
        n_total=n_total,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        y_true=y_validation,
        probs=list(champion_validation_probs),
        threshold=champion_threshold,
    )
    champion_test_metric = _metric_row(
        target=target_name,
        model_name="champion",
        split="test",
        n_total=n_total,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        y_true=y_test,
        probs=list(champion_test_probs),
        threshold=champion_threshold,
    )
    metrics_rows.append(champion_validation_metric)
    metrics_rows.append(champion_test_metric)
    champion_rows.append(
        {
            "target": target_name,
            "candidate_models": "|".join(candidate_models),
            "best_individual_model": best_individual_name,
            "best_individual_validation_auc": None
            if best_individual_validation_auc is None
            else round(best_individual_validation_auc, 6),
            "ensemble_validation_auc": None
            if ensemble_validation_auc is None
            else round(ensemble_validation_auc, 6),
            "selected_model": champion_name,
            "threshold": round(float(champion_threshold), 2),
            "test_accuracy": champion_test_metric["accuracy"],
            "test_f1": champion_test_metric["f1"],
            "test_log_loss": champion_test_metric["log_loss"],
            "test_brier": champion_test_metric["brier"],
            "test_roc_auc": champion_test_metric["roc_auc"],
        }
    )
    threshold_rows.append(
        {
            "target": target_name,
            "model": "champion",
            "threshold": champion_threshold,
            "validation_f1": champion_validation_metric["f1"],
            "validation_auc": None
            if champion_validation_auc is None
            else round(champion_validation_auc, 6),
        }
    )
    metrics_rows.append(
        _metric_row(
            target=target_name,
            model_name="ensemble_weighted_auc",
            split="test",
            n_total=n_total,
            n_train=n_train,
            n_validation=n_validation,
            n_test=n_test,
            y_true=y_test,
            probs=list(ensemble_test_probs),
            threshold=ensemble_threshold,
        )
    )

    agreement = sum(
        1
        for idx in range(len(y_test))
        if len(
            {
                int(test_proba_map["xgb"][idx] >= threshold_map["xgb"]),
                int(test_proba_map["hist_gb"][idx] >= threshold_map["hist_gb"]),
                int(test_proba_map["mlp"][idx] >= threshold_map["mlp"]),
            }
        )
        == 1
    )

    ensemble_artifact = {
        "version": "v6.1",
        "target": target_name,
        "model_name": "ensemble_weighted_auc",
        "weights": ensemble_weights,
        "candidate_models": candidate_models,
        "threshold": ensemble_threshold,
        "calibration_info": {
            "base_models": "isotonic calibrated on validation split",
            "ensemble_probability_sources": ensemble_source_map,
            "weights_split": "validation",
            "threshold_split": "validation",
            "weight_formula": "validation AUC grid search over eligible models",
            "grid_step": grid_step,
            "candidate_auc_window": ENSEMBLE_AUC_WINDOW,
            "champion_min_auc_delta": CHAMPION_MIN_AUC_DELTA,
        },
        "split": {
            "train": n_train,
            "validation": n_validation,
            "test": n_test,
        },
    }
    joblib.dump(ensemble_artifact, OUT_DIR / f"{target_name}_ensemble.joblib")

    champion_artifact = {
        "version": "v6.1",
        "target": target_name,
        "model_name": "champion",
        "selected_model": champion_name,
        "threshold": champion_threshold,
        "candidate_models": candidate_models,
        "best_individual_model": best_individual_name,
        "best_individual_validation_auc": best_individual_validation_auc,
        "ensemble_validation_auc": ensemble_validation_auc,
        "champion_min_auc_delta": CHAMPION_MIN_AUC_DELTA,
        "selection_split": "validation",
    }
    joblib.dump(champion_artifact, OUT_DIR / f"{target_name}_champion.joblib")

    consensus = {
        "version": "v6.1",
        "target": target_name,
        "n_test": len(y_test),
        "agreement_rate_all_models": round(agreement / len(y_test), 6),
        "ensemble_weights": ensemble_weights,
        "ensemble_candidates": candidate_models,
        "champion": champion_name,
        "thresholds": {
            **threshold_map,
            "ensemble_weighted_auc": ensemble_threshold,
            "champion": champion_threshold,
        },
    }

    return {
        "metrics": metrics_rows,
        "thresholds": threshold_rows,
        "weights": weights_rows,
        "calibration": calibration_rows,
        "champion": champion_rows,
        "xgb_feature_importance": feature_importance_rows,
        "consensus": consensus,
        "n_rows": n_total,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[train-v6.1] building samples", flush=True)
    samples = _build_samples(DB_PATH)
    print(f"[train-v6.1] samples_complete={len(samples)}", flush=True)

    q3_rows, q3_support = _dataset_rows_with_support(samples, "q3")
    q4_rows, q4_support = _dataset_rows_with_support(samples, "q4")

    _write_csv(OUT_DIR / "q3_dataset.csv", q3_rows)
    _write_csv(OUT_DIR / "q4_dataset.csv", q4_rows)
    _write_csv(OUT_DIR / "league_support_q3.csv", q3_support["league"])
    _write_csv(OUT_DIR / "league_support_q4.csv", q4_support["league"])
    _write_csv(
        OUT_DIR / "league_bucket_support_q3.csv",
        q3_support["league_bucket"],
    )
    _write_csv(
        OUT_DIR / "league_bucket_support_q4.csv",
        q4_support["league_bucket"],
    )
    _write_csv(
        OUT_DIR / "home_team_bucket_support_q3.csv",
        q3_support["home_team_bucket"],
    )
    _write_csv(
        OUT_DIR / "home_team_bucket_support_q4.csv",
        q4_support["home_team_bucket"],
    )
    _write_csv(
        OUT_DIR / "away_team_bucket_support_q3.csv",
        q3_support["away_team_bucket"],
    )
    _write_csv(
        OUT_DIR / "away_team_bucket_support_q4.csv",
        q4_support["away_team_bucket"],
    )

    print("[train-v6.1] training q3", flush=True)
    q3_result = _train_target(samples, "q3")
    print("[train-v6.1] training q4", flush=True)
    q4_result = _train_target(samples, "q4")

    _write_csv(OUT_DIR / "q3_metrics.csv", q3_result["metrics"])
    _write_csv(OUT_DIR / "q4_metrics.csv", q4_result["metrics"])
    _write_csv(OUT_DIR / "thresholds.csv", q3_result["thresholds"] + q4_result["thresholds"])
    _write_csv(
        OUT_DIR / "ensemble_weights.csv",
        q3_result["weights"] + q4_result["weights"],
    )
    _write_csv(
        OUT_DIR / "calibration_diagnostics.csv",
        q3_result["calibration"] + q4_result["calibration"],
    )
    _write_csv(
        OUT_DIR / "champion_selection.csv",
        q3_result["champion"] + q4_result["champion"],
    )
    _write_csv(
        OUT_DIR / "xgb_feature_importance.csv",
        q3_result["xgb_feature_importance"] + q4_result["xgb_feature_importance"],
    )

    with (OUT_DIR / "q3_consensus.json").open("w", encoding="utf-8") as f:
        json.dump(q3_result["consensus"], f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "q4_consensus.json").open("w", encoding="utf-8") as f:
        json.dump(q4_result["consensus"], f, indent=2, ensure_ascii=False)

    print("[train-v6.1] done")
    print(f"[train-v6.1] samples_complete={len(samples)}")
    print(f"[train-v6.1] q3_rows={q3_result['n_rows']}")
    print(f"[train-v6.1] q4_rows={q4_result['n_rows']}")
    print(f"[train-v6.1] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()

"""Train an independent M27_V1 Q4 model focused on minute-27 signal quality.

This variant is intentionally isolated from V6.3 so feature changes can be
made freely. It keeps only the minute-27 snapshot and uses a denoised feature
set centered on:

- Q1/Q2 closed-quarter context
- Q3 partial state at minute 27
- Global score pressure / comeback context
- Recent momentum windows inside Q3
- A small set of graph trend features

It explicitly avoids:

- raw league one-hot features
- team bucket one-hot features
- duplicated score families
- weak parity-only graph features
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
OUT_DIR = ROOT / "training" / "model_outputs_m27_v1"
DYNAMIC_ROWS_CACHE = OUT_DIR / "dynamic_rows_cache.joblib"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
CV_N_SPLITS = 5
SNAPSHOT_MINUTE = 27


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
        return {
            "gp_count": 0,
            "gp_last": 0,
            "gp_slope_3m": 0,
            "gp_slope_5m": 0,
        }
    slope_3m = values[-1] - values[-4] if len(values) >= 4 else values[-1] - values[0]
    slope_5m = values[-1] - values[-6] if len(values) >= 6 else values[-1] - values[0]
    return {
        "gp_count": len(values),
        "gp_last": values[-1],
        "gp_slope_3m": slope_3m,
        "gp_slope_5m": slope_5m,
    }


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


def _build_m27_v1_features(sample: v6.MatchSample, match_data: dict) -> dict:
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
        "halftime_trailer_cutting_in_q3": halftime_trailer_cutting_in_q3,
        "trailing_now_recent_run_3m": trailing_now_recent_run_3m,
        "trailing_now_recent_run_2m": trailing_now_recent_run_2m,
        "trailing_now_is_home": int(current_trailing_side == "home"),
        "trailing_now_is_away": int(current_trailing_side == "away"),
        "trailing_now_deficit_abs": abs(score_diff),
        "halftime_deficit_abs": abs(ht_diff),
    }
    out.update(graph)
    out.update(recent_3m)
    out.update(recent_2m)
    return out


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
            desc=f"[m27_v1] Construyendo samples dinamicos (desde {source})",
            unit="partido",
        ):
            if sample.target_q4 is None:
                continue
            if preloaded is not None:
                match_data = preloaded.get(str(sample.match_id))
            else:
                match_data = v6.db_mod.get_match(conn, str(sample.match_id))
            if not match_data or not _window_is_eligible_m27(match_data):
                continue
            rows.append({
                "features": _build_m27_v1_features(sample, match_data),
                "target": int(sample.target_q4),
                "dt": sample.dt,
                "match_id": sample.match_id,
                "snapshot_minute": SNAPSHOT_MINUTE,
            })
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
        print(f"[m27_v1] Cargando dynamic_rows desde cache: {DYNAMIC_ROWS_CACHE}")
        return joblib.load(DYNAMIC_ROWS_CACHE)
    rows = _build_dynamic_samples(samples, preloaded=preloaded)
    DYNAMIC_ROWS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rows, DYNAMIC_ROWS_CACHE)
    print(f"[m27_v1] dynamic_rows guardados en cache: {DYNAMIC_ROWS_CACHE}")
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


def _isotonic_calibrate(model, x_val: np.ndarray, y_val: list[int], x_test: np.ndarray) -> list[float]:
    raw_val = model.predict_proba(x_val)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_val, y_val)
        return list(iso.predict(raw_test))
    except Exception:
        return list(raw_test)


def _timeseries_cv(x_all: np.ndarray, y: list[int], n_splits: int = CV_N_SPLITS) -> list[dict]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(splitter.split(x_all), total=n_splits, desc="[m27_v1] CV temporal", unit="fold")
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
                model_name=f"m27_v1_{model_name}",
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
            model_name="m27_v1_champion_q4_ensemble_avg",
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


def _train(samples: list[v6.MatchSample], preloaded: dict[str, dict] | None, force_rebuild: bool) -> dict:
    print("[m27_v1] Construyendo samples dinamicos...")
    start = time.perf_counter()
    dynamic_rows = _load_or_build_dynamic_samples(samples, preloaded=preloaded, force_rebuild=force_rebuild)
    elapsed = time.perf_counter() - start
    print(f"[m27_v1] Total samples: {len(dynamic_rows)} ({elapsed:.1f}s)")
    if len(dynamic_rows) < 300:
        raise RuntimeError(f"[m27_v1] Muy pocas filas: {len(dynamic_rows)}")

    train_rows, val_rows, test_rows, split_info = _split_rows_temporal_by_match(dynamic_rows)
    n_total = len(dynamic_rows)
    n_train = len(train_rows)
    n_val = len(val_rows)
    n_test = len(test_rows)
    if min(n_train, n_val, n_test) == 0:
        raise RuntimeError("[m27_v1] Split vacio")

    y_train = [row["target"] for row in train_rows]
    y_val = [row["target"] for row in val_rows]
    y_test = [row["target"] for row in test_rows]
    if len(set(y_train)) < 2 or len(set(y_val)) < 2 or len(set(y_test)) < 2:
        raise RuntimeError("[m27_v1] Clases insuficientes en algun split")

    x_train_dict = [row["features"] for row in train_rows]
    x_val_dict = [row["features"] for row in val_rows]
    x_test_dict = [row["features"] for row in test_rows]

    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(x_train_dict)
    x_val = vectorizer.transform(x_val_dict)
    x_test = vectorizer.transform(x_test_dict)

    print(
        "[m27_v1] split filas: "
        f"train={n_train} val={n_val} test={n_test} | "
        f"matches: {split_info['matches_train']}/{split_info['matches_val']}/{split_info['matches_test']}"
    )

    x_cv = np.vstack([x_train, x_val])
    y_cv = y_train + y_val
    print("[m27_v1] Ejecutando validacion temporal (CV)...")
    cv_rows = _timeseries_cv(x_cv, y_cv) if len(y_cv) > CV_N_SPLITS + 1 else []
    if cv_rows:
        print(f"[m27_v1] CV listo: {len(cv_rows)} filas de metricas")
    else:
        print("[m27_v1] CV omitido por tamano insuficiente")

    metrics_rows: list[dict] = []
    models = _make_models()
    proba_map: dict[str, list[float]] = {}
    proba_map_cal: dict[str, list[float]] = {}

    print("[m27_v1] Entrenando modelos holdout...")
    for model_name, model in tqdm(models.items(), desc="[m27_v1] Entrenando modelos", unit="modelo"):
        model.fit(x_train, y_train)
        probs_test = list(model.predict_proba(x_test)[:, 1])
        proba_map[model_name] = probs_test
        metrics_rows.append(_metric_row(f"m27_v1_{model_name}", "test", n_total, n_train, n_val, n_test, y_test, probs_test))

        probs_val = list(model.predict_proba(x_val)[:, 1])
        metrics_rows.append(_metric_row(f"m27_v1_{model_name}", "val", n_total, n_train, n_val, n_test, y_val, probs_val))

        probs_cal = _isotonic_calibrate(model, x_val, y_val, x_test)
        proba_map_cal[model_name] = probs_cal
        metrics_rows.append(_metric_row(f"m27_v1_{model_name}_cal", "test", n_total, n_train, n_val, n_test, y_test, probs_cal))

        artifact = {
            "version": "m27_v1",
            "target": "q4",
            "snapshot_minute": SNAPSHOT_MINUTE,
            "model_name": f"m27_v1_{model_name}",
            "vectorizer": vectorizer,
            "model": model,
            "trained_rows": n_total,
            "feature_count": len(vectorizer.feature_names_),
            "split": {"train": n_train, "val": n_val, "test": n_test},
            "split_matches": split_info,
        }
        joblib.dump(artifact, OUT_DIR / f"q4_m27_v1_{model_name}.joblib")
        print(f"[m27_v1] artifact guardado: q4_m27_v1_{model_name}.joblib")

    champion_probs = [
        (proba_map["xgb"][index] + proba_map["hist_gb"][index]) / 2.0
        for index in range(len(y_test))
    ]
    metrics_rows.append(_metric_row("m27_v1_champion_q4_ensemble_avg", "test", n_total, n_train, n_val, n_test, y_test, champion_probs))

    champion_probs_cal = [
        (proba_map_cal["xgb"][index] + proba_map_cal["hist_gb"][index]) / 2.0
        for index in range(len(y_test))
    ]
    metrics_rows.append(_metric_row("m27_v1_champion_q4_ensemble_avg_cal", "test", n_total, n_train, n_val, n_test, y_test, champion_probs_cal))

    champion_artifact = {
        "version": "m27_v1",
        "target": "q4",
        "snapshot_minute": SNAPSHOT_MINUTE,
        "model_name": "m27_v1_champion_q4_ensemble_avg",
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
                "q1/q2 closed-quarter context",
                "halftime state",
                "current global score state at minute 27",
                "q3 partial state",
                "recent 3m and 2m run windows",
                "graph last value and short slopes",
                "pressure/comeback flags without leakage",
            ],
            "removed": [
                "raw league one-hot",
                "team bucket one-hot",
                "duplicated score_3q_* family",
                "weak parity-only graph features",
            ],
        },
    }
    joblib.dump(champion_artifact, OUT_DIR / "q4_m27_v1_champion.joblib")
    print("[m27_v1] artifact guardado: q4_m27_v1_champion.joblib")

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
    parser = argparse.ArgumentParser(description="Entrenar modelo independiente M27_V1 para Q4")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignorar cache de dynamic_rows y reconstruir desde cero",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_bar = tqdm(total=5, desc="[m27_v1] Pipeline", unit="fase")
    print("[m27_v1] Construyendo samples base con pipeline V6...")
    start = time.perf_counter()
    samples, match_data_cache = _build_base_samples_and_data(DB_PATH)
    print(f"[m27_v1] Pipeline base listo: {len(samples)} samples en {time.perf_counter() - start:.1f}s")
    pipeline_bar.update(1)

    result = _train(samples, preloaded=match_data_cache, force_rebuild=args.rebuild_cache)
    pipeline_bar.update(1)

    print("[m27_v1] Guardando metricas CSV...")
    _write_csv(OUT_DIR / "q4_metrics.csv", result["metrics"])
    _write_csv(OUT_DIR / "q4_cv_metrics.csv", result["cv_rows"])
    pipeline_bar.update(1)

    summary = {
        "version": "m27_v1",
        "target": "q4_only",
        "snapshot_minute": SNAPSHOT_MINUTE,
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
    print("[m27_v1] Guardando run_summary.json...")
    with (OUT_DIR / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    pipeline_bar.update(1)

    print("[m27_v1] Finalizando...")
    pipeline_bar.update(1)
    pipeline_bar.close()
    print("[m27_v1] done")
    print(f"[m27_v1] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()

"""Train V6.3 predictors for Q4 winner (Q3 descartado).

V6.3 extends V6.2 with:

- Solo Q4 (Q3 descartado completamente).
- Ventanas dinamicas de features alineadas con el monitoreo en vivo:
        SNAPSHOT_WINDOWS_Q4 = [27, 30] minutos.
    Cada partido se representa como un sample por ventana elegible.
    Entrenamiento multi-modelo por snapshot para alinear monitor/reporte.
- Split 70 / 15 / 15 (train / validation / test) temporal.
- Cross-validation temporal (TimeSeriesSplit, 5 folds) para verificar
  estabilidad en el tiempo.
- Calibracion isotonica post-hoc (variant _cal) ademas del modelo raw.
- Mantiene filtro de ligas de V6.2 (name exclusion + signal pruning).
- Exporta A/B vs V6 y vs V6.2 con las mismas columnas de metricas.
- Exporta resumen de distribucion de ventanas por partido.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
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
import xgboost as xgb
from tqdm import tqdm

import train_q3_q4_models_v6 as v6
import train_q3_q4_models_v6_2 as v62
import infer_match as infer_live

ROOT = v6.ROOT
DB_PATH = v6.DB_PATH
OUT_DIR = ROOT / "training" / "model_outputs_v6_3"
DYNAMIC_ROWS_CACHE = OUT_DIR / "dynamic_rows_cache.joblib"
BASELINE_V6_DIR = ROOT / "training" / "model_outputs_v6"
BASELINE_V62_DIR = ROOT / "training" / "model_outputs_v6_2"
LEAGUE_EXCLUSION_CONFIG_PATH = ROOT / "training" / "v6_2_league_name_exclusions.json"

# Split temporal 70/15/15
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15  (resto)

# Ventanas dinamicas de inferencia Q4 en minutos de juego.
# Alineadas con el monitor: empieza a vigilar en minuto 27, cutoff default 36.
SNAPSHOT_WINDOWS_Q4: list[int] = [27, 30]

# Parametros de filtro de liga (mismos que V6.2)
LEAGUE_MIN_TRAIN_ROWS = 30
LEAGUE_MIN_EFFECT_ABS_DIFF = 0.015
LEAGUE_OTHER_TOKEN = "LEAGUE_OTHER_SIGNAL_WEAK"

# v6.3 raw runs are intended to be fully unfiltered unless explicitly changed.
APPLY_LEAGUE_NAME_FILTER = False

# CV temporal
CV_N_SPLITS = 5

# Late snapshot windows are only useful if the match is still close.
TRAIN_LATE_WINDOW_MAX_MARGIN = {
    27: 999,
    30: 6,
    33: 8,
    36: 10,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_auc(y_true: list[int], probs: list[float]) -> float | None:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return None


def _metric_row(
    target: str,
    model_name: str,
    split: str,
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    y_true: list[int],
    probs: list[float],
) -> dict:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    row = {
        "target": target,
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


# ── Ventanas dinamicas ─────────────────────────────────────────────────────────

def _graph_stats_upto(graph_points: list[dict], max_minute: int) -> dict:
    """Identico al de V6 pero expuesto directamente aqui para claridad."""
    points = [p for p in graph_points if int(p.get("minute", 0)) <= max_minute]
    values = [int(p.get("value", 0)) for p in points]
    if not values:
        return {
            "gp_count": 0,
            "gp_last": 0,
            "gp_peak_home": 0,
            "gp_peak_away": 0,
            "gp_area_home": 0,
            "gp_area_away": 0,
            "gp_area_diff": 0,
            "gp_mean_abs": 0.0,
            "gp_swings": 0,
            "gp_slope_3m": 0,
            "gp_slope_5m": 0,
        }
    area_home = sum(max(v, 0) for v in values)
    area_away = sum(max(-v, 0) for v in values)
    mean_abs = sum(abs(v) for v in values) / len(values)
    slope_3m = values[-1] - values[-4] if len(values) >= 4 else values[-1] - values[0]
    slope_5m = values[-1] - values[-6] if len(values) >= 6 else values[-1] - values[0]
    return {
        "gp_count": len(values),
        "gp_last": values[-1],
        "gp_peak_home": max(values),
        "gp_peak_away": abs(min(values)),
        "gp_area_home": area_home,
        "gp_area_away": area_away,
        "gp_area_diff": area_home - area_away,
        "gp_mean_abs": mean_abs,
        "gp_swings": v6._count_sign_swings(values),
        "gp_slope_3m": slope_3m,
        "gp_slope_5m": slope_5m,
    }


def _build_q4_features_at_window(sample: v6.MatchSample, match_data: dict, snapshot_minute: int) -> dict:
    """Construye features Q4 live-like reutilizando features invariantes ya calculadas.

    Evita el costo dominante de volver a consultar buckets globales de ligas/equipos
    y prior win rates por equipo para cada snapshot.
    """
    q1h, q1a = v6._quarter_points(match_data, "Q1")
    q2h, q2a = v6._quarter_points(match_data, "Q2")

    est_home, est_away = infer_live._score_upto(match_data, snapshot_minute)
    ht_home = int(q1h or 0) + int(q2h or 0)
    ht_away = int(q1a or 0) + int(q2a or 0)
    q3_home = max(0, int(est_home) - ht_home)
    q3_away = max(0, int(est_away) - ht_away)

    out = {
        "league": sample.features_q4.get("league", ""),
        "league_bucket": sample.features_q4.get("league_bucket", "LEAGUE_OTHER"),
        "gender_bucket": sample.features_q4.get("gender_bucket", "men_or_open"),
        "home_team_bucket": sample.features_q4.get("home_team_bucket", "TEAM_OTHER"),
        "away_team_bucket": sample.features_q4.get("away_team_bucket", "TEAM_OTHER"),
        "home_prior_wr": sample.features_q4.get("home_prior_wr", 0.0),
        "away_prior_wr": sample.features_q4.get("away_prior_wr", 0.0),
        "prior_wr_diff": sample.features_q4.get("prior_wr_diff", 0.0),
        "prior_wr_sum": sample.features_q4.get("prior_wr_sum", 0.0),
        "q1_diff": sample.features_q4.get("q1_diff", 0),
        "q2_diff": sample.features_q4.get("q2_diff", 0),
        "cutoff_minute": int(snapshot_minute),
        "score_est_home": int(est_home),
        "score_est_away": int(est_away),
        "score_est_diff": int(est_home) - int(est_away),
        "q3_diff": q3_home - q3_away,
        "q3_total": q3_home + q3_away,
        "score_3q_home": int(est_home),
        "score_3q_away": int(est_away),
        "score_3q_diff": int(est_home) - int(est_away),
    }
    out.update(_graph_stats_upto(match_data.get("graph_points", []), snapshot_minute))
    out.update(infer_live._pbp_stats_upto_minute(match_data, snapshot_minute))
    return out


def _window_is_eligible(match_data: dict, snapshot_minute: int) -> bool:
    gp_count = len([
        p for p in (match_data.get("graph_points") or [])
        if int(p.get("minute", 0)) <= int(snapshot_minute)
    ])
    pbp_count = len(infer_live._pbp_events_upto(match_data, int(snapshot_minute)))

    thr = infer_live._sufficiency_thresholds("q4", int(snapshot_minute))
    if gp_count < int(thr["min_graph_points"]) or pbp_count < int(thr["min_pbp_events"]):
        return False

    if int(snapshot_minute) > 27:
        game_home, game_away = infer_live._score_upto(match_data, int(snapshot_minute))
        margin = abs(int(game_home) - int(game_away))
        if margin > int(TRAIN_LATE_WINDOW_MAX_MARGIN.get(int(snapshot_minute), 6)):
            return False

    return True


def _build_base_samples_and_data(db_path: Path) -> tuple[list[v6.MatchSample], dict[str, dict]]:
    """Run v6._build_samples and capture match_data as a side effect.

    Returns (samples, {match_id: match_data}) so that _build_dynamic_samples
    can skip a second full DB scan (~25K x 4 SQL queries).
    """
    _captured: dict[str, dict] = {}
    _orig = v6.db_mod.get_match

    def _capturing(conn, match_id: str) -> dict | None:
        result = _orig(conn, match_id)
        if result is not None:
            _captured[str(match_id)] = result
        return result

    v6.db_mod.get_match = _capturing
    try:
        samples = v6._build_samples(db_path)
    finally:
        v6.db_mod.get_match = _orig
    return samples, _captured


def _build_dynamic_samples(
    base_samples: list[v6.MatchSample],
    preloaded: dict[str, dict] | None = None,
) -> list[dict]:
    """Por cada partido genera un sample por cada snapshot elegible.

    Retorna lista de dicts con keys: features, target, dt, match_id, snapshot_minute.
    Incluye solo ventanas que cumplan cobertura minima de datos y, para snapshots
    tardios, el partido siga relativamente cerrado.
    """
    out: list[dict] = []
    # Only open a DB connection when match_data is not preloaded
    conn = None if preloaded is not None else v6.db_mod.get_conn(str(DB_PATH))
    if conn is not None:
        v6.db_mod.init_db(conn)
    source = "cache" if preloaded is not None else "DB"
    try:
        for s in tqdm(
            base_samples,
            desc=f"[v6.3] Construyendo samples dinamicos Q4 (desde {source})",
            unit="partido",
        ):
            if s.target_q4 is None:
                continue

            if preloaded is not None:
                match_data = preloaded.get(str(s.match_id))
            else:
                match_data = v6.db_mod.get_match(conn, str(s.match_id))
            if not match_data:
                continue

            for snap in SNAPSHOT_WINDOWS_Q4:
                if not _window_is_eligible(match_data, snap):
                    continue
                feat = _build_q4_features_at_window(s, match_data, snap)
                out.append({
                    "features": feat,
                    "target": int(s.target_q4),
                    "dt": s.dt,
                    "match_id": s.match_id,
                    "snapshot_minute": snap,
                })
    finally:
        if conn is not None:
            conn.close()
    # Ordenar temporalmente
    out.sort(key=lambda r: (r["dt"], r["match_id"], r["snapshot_minute"]))
    return out


def _load_or_build_dynamic_samples(
    samples: list[v6.MatchSample],
    preloaded: dict[str, dict] | None = None,
    force_rebuild: bool = False,
) -> list[dict]:
    """Load dynamic_rows from disk cache if available; build and save otherwise.

    Args:
        samples: V6 MatchSamples (only needed when building from scratch).
        preloaded: Optional pre-loaded match_data dict (skips second DB scan).
        force_rebuild: Ignore existing cache and rebuild unconditionally.
    """
    if not force_rebuild and DYNAMIC_ROWS_CACHE.exists():
        print(f"[v6.3] Cargando dynamic_rows desde cache: {DYNAMIC_ROWS_CACHE}")
        return joblib.load(DYNAMIC_ROWS_CACHE)

    rows = _build_dynamic_samples(samples, preloaded=preloaded)
    DYNAMIC_ROWS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rows, DYNAMIC_ROWS_CACHE)
    print(f"[v6.3] dynamic_rows guardados en cache: {DYNAMIC_ROWS_CACHE}")
    return rows


# ── Filtro de ligas (mismo que V6.2) ───────────────────────────────────────────

def _compute_league_signal(
    train_rows: list[dict],
) -> tuple[set[str], list[dict]]:
    y_train = [r["target"] for r in train_rows]
    leagues = [str(r["features"].get("league", "")) for r in train_rows]
    global_pos_rate = float(np.mean(y_train)) if y_train else 0.0
    stats: dict[str, dict] = {}
    for league, y in zip(leagues, y_train):
        rec = stats.setdefault(league, {"league": league, "train_rows": 0, "positives": 0})
        rec["train_rows"] += 1
        rec["positives"] += int(y)
    keep: set[str] = set()
    out_rows: list[dict] = []
    for league, rec in sorted(stats.items(), key=lambda kv: (-kv[1]["train_rows"], kv[0])):
        train_rows = int(rec["train_rows"])
        pos_rate = float(rec["positives"]) / float(train_rows) if train_rows else 0.0
        abs_effect = abs(pos_rate - global_pos_rate)
        if train_rows < LEAGUE_MIN_TRAIN_ROWS:
            cls = "weak_support"
        elif abs_effect < LEAGUE_MIN_EFFECT_ABS_DIFF:
            cls = "weak_effect"
        else:
            cls = "keep"
            keep.add(league)
        out_rows.append({
            "target": "q4",
            "league": league,
            "train_rows": train_rows,
            "train_pos_rate": round(pos_rate, 6),
            "global_train_pos_rate": round(global_pos_rate, 6),
            "abs_effect_vs_global": round(abs_effect, 6),
            "signal_class": cls,
        })
    return keep, out_rows


def _split_rows_temporal_by_match(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Temporal split by match_id to avoid snapshot leakage across splits."""
    if not rows:
        return [], [], [], {
            "matches_total": 0,
            "matches_train": 0,
            "matches_val": 0,
            "matches_test": 0,
        }

    match_first_dt: dict[str, object] = {}
    for r in rows:
        mid = str(r["match_id"])
        dt = r["dt"]
        prev = match_first_dt.get(mid)
        if prev is None or dt < prev:
            match_first_dt[mid] = dt

    ordered_match_ids = [
        mid
        for mid, _ in sorted(match_first_dt.items(), key=lambda kv: (kv[1], kv[0]))
    ]

    n_matches = len(ordered_match_ids)
    n_train_matches = int(n_matches * TRAIN_RATIO)
    n_val_matches = int(n_matches * VAL_RATIO)

    train_mids = set(ordered_match_ids[:n_train_matches])
    val_mids = set(ordered_match_ids[n_train_matches:n_train_matches + n_val_matches])
    test_mids = set(ordered_match_ids[n_train_matches + n_val_matches:])

    train_rows = [r for r in rows if str(r["match_id"]) in train_mids]
    val_rows = [r for r in rows if str(r["match_id"]) in val_mids]
    test_rows = [r for r in rows if str(r["match_id"]) in test_mids]

    for block in (train_rows, val_rows, test_rows):
        block.sort(key=lambda r: (r["dt"], r["match_id"], r["snapshot_minute"]))

    split_info = {
        "matches_total": n_matches,
        "matches_train": len(train_mids),
        "matches_val": len(val_mids),
        "matches_test": len(test_mids),
    }
    return train_rows, val_rows, test_rows, split_info


def _apply_league_filter(
    x_dict: list[dict],
    keep_leagues: set[str],
) -> list[dict]:
    filtered = []
    for row in x_dict:
        rec = dict(row)
        lg = str(rec.get("league", ""))
        if lg not in keep_leagues:
            rec["league"] = LEAGUE_OTHER_TOKEN
            rec["league_bucket"] = LEAGUE_OTHER_TOKEN
        filtered.append(rec)
    return filtered


def _exclude_rows_by_league_name(
    rows: list[dict],
    rules: list[dict[str, str]],
) -> tuple[list[dict], list[dict], dict]:
    kept: list[dict] = []
    excluded: list[dict] = []
    for r in rows:
        league = str(r["features"].get("league", ""))
        league_lc = league.lower()
        hit: dict[str, str] | None = None
        for rule in rules:
            if rule["pattern_lc"] in league_lc:
                hit = rule
                break
        if hit is None:
            kept.append(r)
        else:
            excluded.append({
                "match_id": r["match_id"],
                "snapshot_minute": r["snapshot_minute"],
                "league": league,
                "exclude_category": hit["category"],
                "exclude_pattern": hit["pattern"],
            })
    total = len(rows)
    excl = len(excluded)
    summary = {
        "target": "q4",
        "rows_before_exclusion": total,
        "rows_excluded_by_name": excl,
        "rows_after_exclusion": len(kept),
        "exclude_ratio": round(float(excl) / float(total), 6) if total else 0.0,
        "active_patterns": len(rules),
    }
    return kept, excluded, summary


# ── Modelos ────────────────────────────────────────────────────────────────────

def _make_models() -> dict:
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
    """Calibracion isotonica post-hoc: fit en val, predict en test."""
    raw_val = model.predict_proba(x_val)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_val, y_val)
        return list(iso.predict(raw_test))
    except Exception:
        return list(raw_test)


# ── CV temporal ────────────────────────────────────────────────────────────────

def _timeseries_cv(
    x_all: np.ndarray,
    y: list[int],
    n_splits: int = CV_N_SPLITS,
) -> list[dict]:
    tss = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(tss.split(x_all), total=n_splits, desc="[v6.3] CV temporal", unit="fold")
    ):
        x_tr = x_all[train_idx]
        x_te = x_all[test_idx]
        y_tr = [y[i] for i in train_idx]
        y_te = [y[i] for i in test_idx]
        if len(set(y_te)) < 2:
            continue
        fold_models = _make_models()
        proba_map: dict[str, list[float]] = {}
        for mn, model in fold_models.items():
            model.fit(x_tr, y_tr)
            probs = list(model.predict_proba(x_te)[:, 1])
            proba_map[mn] = probs
            preds = [1 if p >= 0.5 else 0 for p in probs]
            row: dict = {
                "target": "q4",
                "model": mn,
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": round(float(accuracy_score(y_te, preds)), 6),
                "f1": round(float(f1_score(y_te, preds, zero_division=0)), 6),
                "log_loss": round(float(log_loss(y_te, probs, labels=[0, 1])), 6),
                "brier": round(float(brier_score_loss(y_te, probs)), 6),
            }
            auc = _safe_auc(y_te, probs)
            row["roc_auc"] = None if auc is None else round(auc, 6)
            rows.append(row)
        # Champion ensemble Q4
        ens = [
            (proba_map["xgb"][i] + proba_map["hist_gb"][i]) / 2.0
            for i in range(len(y_te))
        ]
        preds_ens = [1 if p >= 0.5 else 0 for p in ens]
        ens_row: dict = {
            "target": "q4",
            "model": "champion_q4_ensemble_avg",
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "accuracy": round(float(accuracy_score(y_te, preds_ens)), 6),
            "f1": round(float(f1_score(y_te, preds_ens, zero_division=0)), 6),
            "log_loss": round(float(log_loss(y_te, ens, labels=[0, 1])), 6),
            "brier": round(float(brier_score_loss(y_te, ens)), 6),
        }
        auc = _safe_auc(y_te, ens)
        ens_row["roc_auc"] = None if auc is None else round(auc, 6)
        rows.append(ens_row)
    return rows


def _print_cv_summary(cv_rows: list[dict]) -> None:
    by_model: dict[str, dict[str, list[float]]] = {}
    for row in cv_rows:
        m = row["model"]
        if m not in by_model:
            by_model[m] = {"accuracy": [], "roc_auc": [], "log_loss": [], "brier": []}
        for metric in ("accuracy", "roc_auc", "log_loss", "brier"):
            val = row.get(metric)
            if val is not None:
                by_model[m][metric].append(float(val))

    print(f"\n{'=' * 72}")
    print(f"  CV Temporal (TimeSeriesSplit, {CV_N_SPLITS} folds) — Q4")
    print(f"{'=' * 72}")
    print(f"  {'Model':<35} {'AUC':>12} {'Accuracy':>12} {'LogLoss':>10} {'Brier':>8}")
    print(f"  {'-' * 78}")
    for mn in sorted(by_model.keys()):
        if mn not in by_model:
            continue
        d = by_model[mn]

        def fmt(key: str) -> str:
            vals = d.get(key, [])
            if not vals:
                return "         N/A"
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            return f"{mean:.3f}±{std:.3f}"

        print(f"  {mn:<35} {fmt('roc_auc'):>12} {fmt('accuracy'):>12} {fmt('log_loss'):>10} {fmt('brier'):>8}")
    print()


# ── Entrenamiento principal ────────────────────────────────────────────────────

def _train_q4(
    samples: list[v6.MatchSample],
    league_name_rules: list[dict[str, str]],
    preloaded: dict[str, dict] | None = None,
    force_rebuild: bool = False,
) -> dict:
    print("[v6.3] Construyendo samples dinamicos Q4...")
    t0 = time.perf_counter()
    dynamic_rows = _load_or_build_dynamic_samples(
        samples, preloaded=preloaded, force_rebuild=force_rebuild
    )
    elapsed = time.perf_counter() - t0
    print(f"[v6.3] Total samples con ventanas dinamicas: {len(dynamic_rows)}  ({elapsed:.1f}s)")

    if APPLY_LEAGUE_NAME_FILTER and league_name_rules:
        dynamic_rows, league_excl_rows, league_excl_summary = _exclude_rows_by_league_name(
            dynamic_rows, league_name_rules
        )
        print(f"[v6.3] Tras exclusion por nombre de liga: {len(dynamic_rows)}")
    else:
        league_excl_rows = []
        league_excl_summary = {
            "target": "q4",
            "rows_before_exclusion": len(dynamic_rows),
            "rows_excluded_by_name": 0,
            "rows_after_exclusion": len(dynamic_rows),
            "exclude_ratio": 0.0,
            "active_patterns": 0,
            "filters_enabled": False,
        }

    n_total = len(dynamic_rows)
    if n_total < 300:
        raise RuntimeError(f"[v6.3] Muy pocas filas: {n_total}")
    metrics_rows: list[dict] = []
    cv_rows: list[dict] = []
    league_signal_rows: list[dict] = []
    split_rows_summary: list[dict] = []
    keep_leagues_union: set[str] = set()

    for snap in SNAPSHOT_WINDOWS_Q4:
        rows_snap = [r for r in dynamic_rows if int(r["snapshot_minute"]) == int(snap)]
        if len(rows_snap) < 200:
            print(f"[v6.3] Snapshot {snap}: insuficiente ({len(rows_snap)}), se omite")
            continue

        train_rows, val_rows, test_rows, split_info = _split_rows_temporal_by_match(rows_snap)
        n_snap_total = len(rows_snap)
        n_train = len(train_rows)
        n_val = len(val_rows)
        n_test = len(test_rows)

        if min(n_train, n_val, n_test) == 0:
            print(f"[v6.3] Snapshot {snap}: split vacio, se omite")
            continue

        y_train = [r["target"] for r in train_rows]
        y_val = [r["target"] for r in val_rows]
        y_test = [r["target"] for r in test_rows]
        if len(set(y_train)) < 2 or len(set(y_val)) < 2 or len(set(y_test)) < 2:
            print(f"[v6.3] Snapshot {snap}: clases insuficientes en algun split, se omite")
            continue

        if APPLY_LEAGUE_NAME_FILTER and league_name_rules:
            keep_leagues, signal_rows = _compute_league_signal(train_rows)
            for sr in signal_rows:
                sr["snapshot_minute"] = snap
            league_signal_rows.extend(signal_rows)
            keep_leagues_union.update(keep_leagues)
            x_train_dict = _apply_league_filter([r["features"] for r in train_rows], keep_leagues)
            x_val_dict = _apply_league_filter([r["features"] for r in val_rows], keep_leagues)
            x_test_dict = _apply_league_filter([r["features"] for r in test_rows], keep_leagues)
        else:
            keep_leagues = set()
            x_train_dict = [r["features"] for r in train_rows]
            x_val_dict = [r["features"] for r in val_rows]
            x_test_dict = [r["features"] for r in test_rows]

        vec = DictVectorizer(sparse=False)
        x_train = vec.fit_transform(x_train_dict)
        x_val = vec.transform(x_val_dict)
        x_test = vec.transform(x_test_dict)

        print(
            f"[v6.3] Snapshot {snap} split filas: train={n_train} val={n_val} test={n_test} | "
            f"matches: {split_info['matches_train']}/{split_info['matches_val']}/{split_info['matches_test']}"
        )

        # CV temporal (diagnostico) usando train+val para respetar orden temporal.
        x_cv = np.vstack([x_train, x_val])
        y_cv = y_train + y_val
        if len(y_cv) > CV_N_SPLITS + 1:
            cv_rows_snap = _timeseries_cv(x_cv, y_cv)
            for rr in cv_rows_snap:
                rr["model"] = f"m{snap}_{rr['model']}"
                rr["snapshot_minute"] = snap
            cv_rows.extend(cv_rows_snap)

        models = _make_models()
        proba_map: dict[str, list[float]] = {}
        proba_map_cal: dict[str, list[float]] = {}

        for mn, model in tqdm(models.items(), desc=f"[v6.3] Entrenando modelos Q4 m{snap}", unit="modelo"):
            model.fit(x_train, y_train)

            probs_test = list(model.predict_proba(x_test)[:, 1])
            proba_map[mn] = probs_test
            metrics_rows.append(_metric_row(
                target="q4", model_name=f"m{snap}_{mn}", split="test",
                n_total=n_snap_total, n_train=n_train, n_val=n_val, n_test=n_test,
                y_true=y_test, probs=probs_test,
            ))

            probs_val = list(model.predict_proba(x_val)[:, 1])
            metrics_rows.append(_metric_row(
                target="q4", model_name=f"m{snap}_{mn}", split="val",
                n_total=n_snap_total, n_train=n_train, n_val=n_val, n_test=n_test,
                y_true=y_val, probs=probs_val,
            ))

            probs_cal = _isotonic_calibrate(model, x_val, y_val, x_test)
            proba_map_cal[mn] = probs_cal
            metrics_rows.append(_metric_row(
                target="q4", model_name=f"m{snap}_{mn}_cal", split="test",
                n_total=n_snap_total, n_train=n_train, n_val=n_val, n_test=n_test,
                y_true=y_test, probs=probs_cal,
            ))

            artifact = {
                "version": "v6.3",
                "target": "q4",
                "snapshot_minute": snap,
                "model_name": f"m{snap}_{mn}",
                "vectorizer": vec,
                "model": model,
                "league_filter": {
                    "enabled": bool(APPLY_LEAGUE_NAME_FILTER and league_name_rules),
                    "min_train_rows": LEAGUE_MIN_TRAIN_ROWS,
                    "min_abs_effect_diff": LEAGUE_MIN_EFFECT_ABS_DIFF,
                    "other_token": LEAGUE_OTHER_TOKEN,
                    "kept_leagues": sorted(keep_leagues),
                },
                "snapshot_windows": SNAPSHOT_WINDOWS_Q4,
                "trained_rows": n_snap_total,
                "feature_count": len(vec.feature_names_),
                "split": {"train": n_train, "val": n_val, "test": n_test},
                "split_matches": split_info,
            }
            joblib.dump(artifact, OUT_DIR / f"q4_m{snap}_{mn}.joblib")

        champion_probs = [
            (proba_map["xgb"][i] + proba_map["hist_gb"][i]) / 2.0
            for i in range(len(y_test))
        ]
        champion_name = f"m{snap}_champion_q4_ensemble_avg"
        metrics_rows.append(_metric_row(
            target="q4", model_name=champion_name, split="test",
            n_total=n_snap_total, n_train=n_train, n_val=n_val, n_test=n_test,
            y_true=y_test, probs=champion_probs,
        ))

        champion_probs_cal = [
            (proba_map_cal["xgb"][i] + proba_map_cal["hist_gb"][i]) / 2.0
            for i in range(len(y_test))
        ]
        metrics_rows.append(_metric_row(
            target="q4", model_name=f"{champion_name}_cal", split="test",
            n_total=n_snap_total, n_train=n_train, n_val=n_val, n_test=n_test,
            y_true=y_test, probs=champion_probs_cal,
        ))

        champion_artifact = {
            "version": "v6.3",
            "target": "q4",
            "snapshot_minute": snap,
            "model_name": champion_name,
            "vectorizer": vec,
            "models": models,
            "champion_strategy": "avg_prob_xgb_hist_gb",
            "league_filter": {
                "enabled": bool(APPLY_LEAGUE_NAME_FILTER and league_name_rules),
                "min_train_rows": LEAGUE_MIN_TRAIN_ROWS,
                "min_abs_effect_diff": LEAGUE_MIN_EFFECT_ABS_DIFF,
                "other_token": LEAGUE_OTHER_TOKEN,
                "kept_leagues": sorted(keep_leagues),
            },
            "snapshot_windows": SNAPSHOT_WINDOWS_Q4,
            "split": {"train": n_train, "val": n_val, "test": n_test},
            "split_matches": split_info,
        }
        joblib.dump(champion_artifact, OUT_DIR / f"q4_m{snap}_champion.joblib")

        split_rows_summary.append({
            "snapshot_minute": snap,
            "rows_total": n_snap_total,
            "rows_train": n_train,
            "rows_val": n_val,
            "rows_test": n_test,
            "matches_total": split_info["matches_total"],
            "matches_train": split_info["matches_train"],
            "matches_val": split_info["matches_val"],
            "matches_test": split_info["matches_test"],
        })

    if cv_rows:
        _print_cv_summary(cv_rows)

    # Distribucion de ventanas
    snap_counts = {s: 0 for s in SNAPSHOT_WINDOWS_Q4}
    for r in dynamic_rows:
        snap_counts[r["snapshot_minute"]] = snap_counts.get(r["snapshot_minute"], 0) + 1
    window_dist_rows = [
        {"snapshot_minute": k, "count": v, "ratio": round(v / n_total, 4)}
        for k, v in sorted(snap_counts.items())
    ]

    return {
        "metrics": metrics_rows,
        "cv_rows": cv_rows,
        "league_signal": league_signal_rows,
        "league_excl_rows": league_excl_rows,
        "league_excl_summary": league_excl_summary,
        "window_distribution": window_dist_rows,
        "n_total": n_total,
        "n_train": sum(int(r.get("rows_train", 0)) for r in split_rows_summary),
        "n_val": sum(int(r.get("rows_val", 0)) for r in split_rows_summary),
        "n_test": sum(int(r.get("rows_test", 0)) for r in split_rows_summary),
        "feature_names": [],
        "keep_leagues": keep_leagues_union,
        "split_rows_summary": split_rows_summary,
    }


# ── Comparacion A/B ────────────────────────────────────────────────────────────

def _ab_rows(
    baseline_rows: list[dict],
    v63_rows: list[dict],
    baseline_label: str,
    v63_label: str,
    model_map: dict[str, str],
) -> list[dict]:
    """Genera filas A/B solo para el split 'test' de V6.3."""
    by_model_base: dict[str, dict[str, dict]] = {}
    for row in baseline_rows:
        by_model_base.setdefault(row["target"], {})[row["model"]] = row

    out: list[dict] = []
    for row in v63_rows:
        if row.get("split") != "test":
            continue
        target = row["target"]
        model = row["model"]
        base_model = model_map.get(model)
        if base_model is None:
            continue
        base = by_model_base.get(target, {}).get(base_model)
        if not base:
            continue
        out.append({
            "target": target,
            f"{baseline_label}_model_ref": base_model,
            f"{v63_label}_model": model,
            f"{baseline_label}_accuracy": base.get("accuracy"),
            f"{v63_label}_accuracy": row["accuracy"],
            "delta_accuracy": round(float(row["accuracy"]) - float(base["accuracy"]), 6),
            f"{baseline_label}_f1": base.get("f1"),
            f"{v63_label}_f1": row["f1"],
            "delta_f1": round(float(row["f1"]) - float(base["f1"]), 6),
            f"{baseline_label}_log_loss": base.get("log_loss"),
            f"{v63_label}_log_loss": row["log_loss"],
            "delta_log_loss": round(float(row["log_loss"]) - float(base["log_loss"]), 6),
            f"{baseline_label}_brier": base.get("brier"),
            f"{v63_label}_brier": row["brier"],
            "delta_brier": round(float(row["brier"]) - float(base["brier"]), 6),
            f"{baseline_label}_roc_auc": base.get("roc_auc"),
            f"{v63_label}_roc_auc": row["roc_auc"],
            "delta_roc_auc": round(float(row["roc_auc"]) - float(base["roc_auc"]), 6)
            if row.get("roc_auc") is not None and base.get("roc_auc") is not None
            else None,
        })
    out.sort(key=lambda r: r[f"{v63_label}_model"])
    return out


# ── Reporte Markdown ───────────────────────────────────────────────────────────

def _write_markdown_report(
    result: dict,
    ab_vs_v6: list[dict],
    ab_vs_v62: list[dict],
    league_name_rules: list[dict[str, str]],
) -> None:
    lines: list[str] = []
    lines += [
        "# V6.3 Report", "",
        "## Config",
        f"- version: v6.3",
        f"- target: q4 only (Q3 descartado)",
        f"- snapshot_windows_q4: {SNAPSHOT_WINDOWS_Q4}",
        f"- split: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int((1-TRAIN_RATIO-VAL_RATIO)*100)} (train/val/test) temporal",
        f"- cv_splits: {CV_N_SPLITS} (TimeSeriesSplit)",
        f"- calibration: isotonica post-hoc (fit en val, predict en test)",
        f"- models: xgb, hist_gb",
        f"- champion_q4: avg_prob(xgb, hist_gb)",
        f"- league_min_train_rows: {LEAGUE_MIN_TRAIN_ROWS}",
        f"- league_min_effect_abs_diff: {LEAGUE_MIN_EFFECT_ABS_DIFF}",
        f"- league_other_token: {LEAGUE_OTHER_TOKEN}",
        f"- league_name_exclusion_patterns: {len(league_name_rules)}",
        "",
    ]

    # Window distribution
    lines += [
        "## Distribucion de Ventanas Dinamicas (Q4)",
        "| snapshot_minute | count | ratio |",
        "|---:|---:|---:|",
    ]
    for r in result["window_distribution"]:
        lines.append(f"| {r['snapshot_minute']} | {r['count']} | {r['ratio']:.4f} |")
    lines.append("")

    # League exclusion
    es = result["league_excl_summary"]
    lines += [
        "## Exclusion de Ligas por Nombre",
        "| rows_before | rows_excluded | rows_after | exclude_ratio |",
        "|---:|---:|---:|---:|",
        f"| {es['rows_before_exclusion']} | {es['rows_excluded_by_name']} | {es['rows_after_exclusion']} | {es['exclude_ratio']:.4f} |",
        "",
    ]

    # Metricas holdout
    lines += [
        "## Metricas Holdout (V6.3)",
        "| split | model | accuracy | f1 | log_loss | brier | roc_auc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["metrics"]:
        auc = row["roc_auc"]
        auc_s = f"{auc:.6f}" if auc is not None else "N/A"
        lines.append(
            f"| {row['split']} | {row['model']} | {row['accuracy']:.6f} | {row['f1']:.6f} | {row['log_loss']:.6f} | {row['brier']:.6f} | {auc_s} |"
        )
    lines.append("")

    # CV summary
    lines += [
        "## Cross-Validation Temporal",
        "| model | fold | n_train | n_test | accuracy | f1 | log_loss | brier | roc_auc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["cv_rows"]:
        auc = row.get("roc_auc")
        auc_s = f"{auc:.6f}" if auc is not None else "N/A"
        lines.append(
            f"| {row['model']} | {row['fold']} | {row['n_train']} | {row['n_test']} | {row['accuracy']:.6f} | {row['f1']:.6f} | {row['log_loss']:.6f} | {row['brier']:.6f} | {auc_s} |"
        )
    lines.append("")

    # A/B vs V6
    lines += [
        "## A/B vs V6",
        "| v6_ref | v6.3_model | d_acc | d_f1 | d_log_loss | d_brier | d_auc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ab_vs_v6:
        dauc = row.get("delta_roc_auc")
        dauc_s = f"{dauc:+.6f}" if dauc is not None else "N/A"
        lines.append(
            f"| {row.get('v6_model_ref')} | {row.get('v63_model')} | {row['delta_accuracy']:+.6f} | {row['delta_f1']:+.6f} | {row['delta_log_loss']:+.6f} | {row['delta_brier']:+.6f} | {dauc_s} |"
        )
    lines.append("")

    # A/B vs V6.2
    lines += [
        "## A/B vs V6.2",
        "| v6.2_ref | v6.3_model | d_acc | d_f1 | d_log_loss | d_brier | d_auc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ab_vs_v62:
        dauc = row.get("delta_roc_auc")
        dauc_s = f"{dauc:+.6f}" if dauc is not None else "N/A"
        lines.append(
            f"| {row.get('v62_model_ref')} | {row.get('v63_model')} | {row['delta_accuracy']:+.6f} | {row['delta_f1']:+.6f} | {row['delta_log_loss']:+.6f} | {row['delta_brier']:+.6f} | {dauc_s} |"
        )

    (OUT_DIR / "V6_3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar modelos V6.3 Q4")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignorar cache de dynamic_rows y reconstruir desde cero",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    league_name_rules: list[dict[str, str]] = []

    print("[v6.3] Construyendo samples con pipeline V6 (capturando match_data)...")
    t_start = time.perf_counter()
    samples, match_data_cache = _build_base_samples_and_data(DB_PATH)
    t_v6 = time.perf_counter() - t_start
    print(f"[v6.3] Pipeline V6 listo: {len(samples)} samples en {t_v6:.1f}s")

    result = _train_q4(
        samples,
        league_name_rules,
        preloaded=match_data_cache,
        force_rebuild=args.rebuild_cache,
    )

    # Persistir metricas
    _write_csv(OUT_DIR / "q4_metrics.csv", result["metrics"])
    _write_csv(OUT_DIR / "q4_cv_metrics.csv", result["cv_rows"])
    _write_csv(OUT_DIR / "league_signal_q4.csv", result["league_signal"])
    _write_csv(OUT_DIR / "league_name_exclusions_q4.csv", result["league_excl_rows"])
    _write_csv(OUT_DIR / "league_name_exclusion_summary.csv", [result["league_excl_summary"]])
    _write_csv(OUT_DIR / "window_distribution.csv", result["window_distribution"])

    # A/B vs V6
    baseline_v6_q4: list[dict] = []
    v6_q4_path = BASELINE_V6_DIR / "q4_metrics.csv"
    if v6_q4_path.exists():
        with v6_q4_path.open("r", encoding="utf-8", newline="") as f:
            baseline_v6_q4 = list(csv.DictReader(f))

    # Mapa: nombre v6.3 -> nombre en baseline V6
    model_map_v6 = {
        "m27_xgb": "xgb",
        "m27_hist_gb": "hist_gb",
        "m27_champion_q4_ensemble_avg": "ensemble_avg_prob",
        "m27_xgb_cal": "xgb",
        "m27_hist_gb_cal": "hist_gb",
        "m27_champion_q4_ensemble_avg_cal": "ensemble_avg_prob",
    }
    ab_vs_v6 = _ab_rows(
        baseline_v6_q4, result["metrics"],
        baseline_label="v6", v63_label="v63",
        model_map=model_map_v6,
    )
    _write_csv(OUT_DIR / "ab_comparison_v6_vs_v6_3.csv", ab_vs_v6)

    # A/B vs V6.2
    baseline_v62_q4: list[dict] = []
    v62_q4_path = BASELINE_V62_DIR / "q4_metrics.csv"
    if v62_q4_path.exists():
        with v62_q4_path.open("r", encoding="utf-8", newline="") as f:
            baseline_v62_q4 = list(csv.DictReader(f))

    model_map_v62 = {
        "m27_xgb": "xgb",
        "m27_hist_gb": "hist_gb",
        "m27_champion_q4_ensemble_avg": "champion_q4_blend_xgb_0.6_hist_0.4",
        "m27_xgb_cal": "xgb",
        "m27_hist_gb_cal": "hist_gb",
        "m27_champion_q4_ensemble_avg_cal": "champion_q4_blend_xgb_0.6_hist_0.4",
    }
    ab_vs_v62 = _ab_rows(
        baseline_v62_q4, result["metrics"],
        baseline_label="v62", v63_label="v63",
        model_map=model_map_v62,
    )
    _write_csv(OUT_DIR / "ab_comparison_v6_2_vs_v6_3.csv", ab_vs_v62)

    # Reporte Markdown
    _write_markdown_report(result, ab_vs_v6, ab_vs_v62, league_name_rules)

    # run_summary.json
    summary = {
        "version": "v6.3",
        "baseline_versions": ["v6", "v6.2"],
        "target": "q4_only",
        "split": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": round(1.0 - TRAIN_RATIO - VAL_RATIO, 2),
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "n_test": result["n_test"],
            "n_total": result["n_total"],
        },
        "snapshot_windows_q4": SNAPSHOT_WINDOWS_Q4,
        "calibration": "isotonic_posthoc_fit_on_val",
        "cv_splits": CV_N_SPLITS,
        "league_filter": {
            "min_train_rows": LEAGUE_MIN_TRAIN_ROWS,
            "min_abs_effect_diff": LEAGUE_MIN_EFFECT_ABS_DIFF,
            "other_token": LEAGUE_OTHER_TOKEN,
        },
        "league_name_exclusion": result["league_excl_summary"],
        "window_distribution": result["window_distribution"],
    }
    with (OUT_DIR / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[v6.3] done")
    print(f"[v6.3] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()

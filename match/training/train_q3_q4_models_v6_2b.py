"""Train V6.2b predictors for quarter winners (Q3, Q4).

V6.2b is a conservative iteration over V6:
- keeps temporal 80/20 split exactly like V6
- prunes only features pre-labeled as no_signal in V6 diagnostics
- keeps the V6 model family (xgb, hist_gb, mlp)
- always exports A/B comparison against V6 baseline metrics
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
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

import train_q3_q4_models_v6 as v6

ROOT = v6.ROOT
DB_PATH = v6.DB_PATH
OUT_DIR = ROOT / "training" / "model_outputs_v6_2b"
BASELINE_DIR = ROOT / "training" / "model_outputs_v6"


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_no_signal_feature_specs() -> dict:
    real_rows = _read_csv_rows(BASELINE_DIR / "real_features_no_signal.csv")
    league_rows = _read_csv_rows(BASELINE_DIR / "league_features_no_signal.csv")
    other_rows = _read_csv_rows(BASELINE_DIR / "other_categorical_features_no_signal.csv")

    no_signal_features: set[str] = set()
    for row in real_rows + league_rows + other_rows:
        feature = str(row.get("feature", "")).strip()
        if feature:
            no_signal_features.add(feature)

    numeric_drop: set[str] = {f for f in no_signal_features if "=" not in f}
    categorical_drop_pairs: set[tuple[str, str]] = set()
    for feat in no_signal_features:
        if "=" in feat:
            key, value = feat.split("=", 1)
            categorical_drop_pairs.add((key, value))

    categorical_fallback = {
        "league": "LEAGUE_OTHER",
        "league_bucket": "LEAGUE_OTHER",
        "home_team_bucket": "TEAM_OTHER",
        "away_team_bucket": "TEAM_OTHER",
        "gender_bucket": "men_or_open",
    }

    return {
        "all_no_signal_features": sorted(no_signal_features),
        "numeric_drop": numeric_drop,
        "categorical_drop_pairs": categorical_drop_pairs,
        "categorical_fallback": categorical_fallback,
    }


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
    n_total: int,
    n_train: int,
    n_test: int,
    y_true: list[int],
    probs: list[float],
) -> dict:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    row = {
        "target": target,
        "model": model_name,
        "samples_total": n_total,
        "samples_train": n_train,
        "samples_test": n_test,
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1": round(float(f1_score(y_true, preds)), 6),
        "precision": round(float(precision_score(y_true, preds)), 6),
        "recall": round(float(recall_score(y_true, preds)), 6),
        "log_loss": round(float(log_loss(y_true, probs, labels=[0, 1])), 6),
        "brier": round(float(brier_score_loss(y_true, probs)), 6),
    }
    auc = _safe_auc(y_true, probs)
    row["roc_auc"] = None if auc is None else round(auc, 6)
    return row


def _apply_no_signal_pruning(
    x_dict: list[dict],
    no_signal_specs: dict,
) -> tuple[list[dict], dict]:
    numeric_drop: set[str] = no_signal_specs["numeric_drop"]
    categorical_drop_pairs: set[tuple[str, str]] = no_signal_specs["categorical_drop_pairs"]
    categorical_fallback: dict[str, str] = no_signal_specs["categorical_fallback"]

    filtered: list[dict] = []
    numeric_dropped = 0
    categorical_collapsed = 0

    for row in x_dict:
        rec = dict(row)

        for key in numeric_drop:
            if key in rec:
                rec.pop(key, None)
                numeric_dropped += 1

        for cat_key, fallback in categorical_fallback.items():
            cat_val = str(rec.get(cat_key, ""))
            if (cat_key, cat_val) in categorical_drop_pairs:
                rec[cat_key] = fallback
                categorical_collapsed += 1

        filtered.append(rec)

    summary = {
        "rows_total": len(x_dict),
        "numeric_feature_keys_dropped": len(numeric_drop),
        "categorical_feature_pairs_flagged": len(categorical_drop_pairs),
        "numeric_drop_events": numeric_dropped,
        "categorical_collapse_events": categorical_collapsed,
    }
    return filtered, summary


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
        "mlp": v6.MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            n_iter_no_change=20,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
    }


def _train_target(samples: list[v6.MatchSample], target_name: str, no_signal_specs: dict) -> dict:
    if target_name == "q3":
        rows = [s for s in samples if s.target_q3 is not None]
        rows = sorted(rows, key=lambda item: item.dt)
        x_raw = [s.features_q3 for s in rows]
        y = [int(s.target_q3) for s in rows]
    else:
        rows = [s for s in samples if s.target_q4 is not None]
        rows = sorted(rows, key=lambda item: item.dt)
        x_raw = [s.features_q4 for s in rows]
        y = [int(s.target_q4) for s in rows]

    n_total = len(rows)
    if n_total < 200:
        raise RuntimeError(f"Not enough rows for {target_name}. Got {n_total}.")

    n_train = int(n_total * 0.8)
    n_test = n_total - n_train

    x_filtered, filter_summary = _apply_no_signal_pruning(x_raw, no_signal_specs)

    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(x_filtered)
    x_train = x_all[:n_train]
    x_test = x_all[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    models = _make_models()
    proba_map: dict[str, list[float]] = {}
    metrics_rows: list[dict] = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        probs = list(model.predict_proba(x_test)[:, 1])
        proba_map[model_name] = probs

        metrics_rows.append(
            _metric_row(
                target=target_name,
                model_name=model_name,
                n_total=n_total,
                n_train=n_train,
                n_test=n_test,
                y_true=y_test,
                probs=probs,
            )
        )

        artifact = {
            "version": "v6.2b",
            "target": target_name,
            "model_name": model_name,
            "vectorizer": vec,
            "model": model,
            "no_signal_pruning": {
                "numeric_drop": sorted(no_signal_specs["numeric_drop"]),
                "categorical_pairs_count": len(no_signal_specs["categorical_drop_pairs"]),
            },
            "trained_rows": n_total,
            "feature_count": len(vec.feature_names_),
        }
        joblib.dump(artifact, OUT_DIR / f"{target_name}_{model_name}.joblib")

    # Champion strategy by target.
    if target_name == "q3":
        champion_probs = proba_map["xgb"]
        champion_name = "champion_q3_xgb"
    else:
        champion_probs = [
            (proba_map["xgb"][i] + proba_map["hist_gb"][i] + proba_map["mlp"][i]) / 3.0
            for i in range(len(y_test))
        ]
        champion_name = "champion_q4_ensemble_avg_prob"

    metrics_rows.append(
        _metric_row(
            target=target_name,
            model_name=champion_name,
            n_total=n_total,
            n_train=n_train,
            n_test=n_test,
            y_true=y_test,
            probs=champion_probs,
        )
    )

    champion_artifact = {
        "version": "v6.2b",
        "target": target_name,
        "model_name": champion_name,
        "vectorizer": vec,
        "models": models,
        "champion_strategy": {
            "q3": "xgb_only",
            "q4": "ensemble_avg_prob_of_xgb_hist_gb_mlp",
        },
        "no_signal_pruning": {
            "numeric_drop": sorted(no_signal_specs["numeric_drop"]),
            "categorical_pairs_count": len(no_signal_specs["categorical_drop_pairs"]),
        },
    }
    joblib.dump(champion_artifact, OUT_DIR / f"{target_name}_champion.joblib")

    return {
        "target": target_name,
        "metrics": metrics_rows,
        "filter_summary": {
            "target": target_name,
            **filter_summary,
            "n_total": n_total,
            "n_train": n_train,
            "n_test": n_test,
            "features_after_vectorizer": len(vec.feature_names_),
            "features_no_signal_total": len(no_signal_specs["all_no_signal_features"]),
        },
    }


def _ab_comparison_rows(v6_rows: list[dict], v62_rows: list[dict]) -> list[dict]:
    by_target_v6: dict[str, dict[str, dict]] = {}
    for row in v6_rows:
        by_target_v6.setdefault(row["target"], {})[row["model"]] = row

    out: list[dict] = []
    for row in v62_rows:
        target = row["target"]
        model = row["model"]

        if model.startswith("champion_q3"):
            base_model = "xgb"
        elif model.startswith("champion_q4"):
            base_model = "ensemble_avg_prob"
        else:
            base_model = model

        base = by_target_v6.get(target, {}).get(base_model)
        if not base:
            continue

        comp = {
            "target": target,
            "v6_model_reference": base_model,
            "v6_2b_model": model,
            "v6_accuracy": base["accuracy"],
            "v6_2b_accuracy": row["accuracy"],
            "delta_accuracy": round(float(row["accuracy"]) - float(base["accuracy"]), 6),
            "v6_f1": base["f1"],
            "v6_2b_f1": row["f1"],
            "delta_f1": round(float(row["f1"]) - float(base["f1"]), 6),
            "v6_log_loss": base["log_loss"],
            "v6_2b_log_loss": row["log_loss"],
            "delta_log_loss": round(float(row["log_loss"]) - float(base["log_loss"]), 6),
            "v6_brier": base["brier"],
            "v6_2b_brier": row["brier"],
            "delta_brier": round(float(row["brier"]) - float(base["brier"]), 6),
            "v6_roc_auc": base["roc_auc"],
            "v6_2b_roc_auc": row["roc_auc"],
            "delta_roc_auc": round(float(row["roc_auc"]) - float(base["roc_auc"]), 6),
        }
        out.append(comp)

    out.sort(key=lambda r: (r["target"], r["v6_2b_model"]))
    return out


def _write_markdown_report(
    q3_result: dict,
    q4_result: dict,
    ab_rows: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# V6.2b Report")
    lines.append("")
    lines.append("## Config")
    lines.append("- pruning_mode: no_signal_only_from_v6_permutation_importance")
    lines.append("- split: temporal 80/20 (same as v6)")
    lines.append("- models: xgb, hist_gb, mlp")
    lines.append("- champion_q3: xgb")
    lines.append("- champion_q4: ensemble_avg_prob (xgb+hist_gb+mlp)")
    lines.append("")

    lines.append("## No-Signal Pruning Summary")
    lines.append("| target | features_no_signal_total | numeric_drop_events | categorical_collapse_events | features_after_vectorizer |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in [q3_result["filter_summary"], q4_result["filter_summary"]]:
        lines.append(
            f"| {r['target']} | {r['features_no_signal_total']} | {r['numeric_drop_events']} | {r['categorical_collapse_events']} | {r['features_after_vectorizer']} |"
        )
    lines.append("")

    lines.append("## Holdout Metrics (V6.2b)")
    lines.append("| target | model | accuracy | f1 | log_loss | brier | roc_auc |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in q3_result["metrics"] + q4_result["metrics"]:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['accuracy']:.6f} | {row['f1']:.6f} | {row['log_loss']:.6f} | {row['brier']:.6f} | {row['roc_auc']:.6f} |"
        )
    lines.append("")

    lines.append("## A/B vs V6")
    lines.append("| target | v6_ref | v6.2b_model | d_acc | d_f1 | d_log_loss | d_brier | d_auc |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in ab_rows:
        lines.append(
            f"| {row['target']} | {row['v6_model_reference']} | {row['v6_2b_model']} | {row['delta_accuracy']:+.6f} | {row['delta_f1']:+.6f} | {row['delta_log_loss']:+.6f} | {row['delta_brier']:+.6f} | {row['delta_roc_auc']:+.6f} |"
        )

    (OUT_DIR / "V6_2B_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    no_signal_specs = _load_no_signal_feature_specs()

    print("[v6.2b] building samples using v6 feature pipeline...")
    samples = v6._build_samples(DB_PATH)

    print("[v6.2b] training q3...")
    q3_result = _train_target(samples, "q3", no_signal_specs)

    print("[v6.2b] training q4...")
    q4_result = _train_target(samples, "q4", no_signal_specs)

    v62_metrics = q3_result["metrics"] + q4_result["metrics"]
    _write_csv(OUT_DIR / "q3_metrics.csv", [r for r in q3_result["metrics"] if r["target"] == "q3"])
    _write_csv(OUT_DIR / "q4_metrics.csv", [r for r in q4_result["metrics"] if r["target"] == "q4"])
    _write_csv(OUT_DIR / "no_signal_pruning_summary.csv", [q3_result["filter_summary"], q4_result["filter_summary"]])

    baseline_q3 = []
    baseline_q4 = []
    with (BASELINE_DIR / "q3_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        baseline_q3 = list(csv.DictReader(f))
    with (BASELINE_DIR / "q4_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        baseline_q4 = list(csv.DictReader(f))

    v6_rows = baseline_q3 + baseline_q4
    ab_rows = _ab_comparison_rows(v6_rows, v62_metrics)
    _write_csv(OUT_DIR / "ab_comparison_v6_vs_v6_2b.csv", ab_rows)
    _write_markdown_report(q3_result, q4_result, ab_rows)

    summary = {
        "version": "v6.2b",
        "baseline_version": "v6",
        "split": "temporal_80_20",
        "pruning_mode": "no_signal_only",
        "features_no_signal_total": len(no_signal_specs["all_no_signal_features"]),
        "q3": q3_result["filter_summary"],
        "q4": q4_result["filter_summary"],
    }
    with (OUT_DIR / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[v6.2b] done")
    print(f"[v6.2b] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()

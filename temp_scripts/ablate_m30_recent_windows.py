import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
TRAINING = ROOT / "match" / "training"
sys.path.insert(0, str(TRAINING))

import train_q4_m30_v1 as m30_v1_train  # noqa: E402


ABLATE_PREFIXES = ("recent_4m_", "recent_2m_")
ABLATE_EXACT = {"trailing_now_recent_run_4m", "trailing_now_recent_run_2m"}


def _strip_recent_features(features: dict) -> dict:
    return {
        key: value
        for key, value in features.items()
        if key not in ABLATE_EXACT
        and not any(key.startswith(prefix) for prefix in ABLATE_PREFIXES)
    }


samples, preloaded = m30_v1_train._build_base_samples_and_data(m30_v1_train.DB_PATH)
dynamic_rows = m30_v1_train._load_or_build_dynamic_samples(
    samples,
    preloaded=preloaded,
    force_rebuild=False,
)
train_rows, val_rows, test_rows, split_info = m30_v1_train._split_rows_temporal_by_match(
    dynamic_rows
)

ablated_train = [
    {**row, "features": _strip_recent_features(row["features"])}
    for row in train_rows
]
ablated_val = [
    {**row, "features": _strip_recent_features(row["features"])}
    for row in val_rows
]
ablated_test = [
    {**row, "features": _strip_recent_features(row["features"])}
    for row in test_rows
]

y_train = [row["target"] for row in ablated_train]
y_val = [row["target"] for row in ablated_val]
y_test = [row["target"] for row in ablated_test]

vectorizer = DictVectorizer(sparse=False)
x_train = vectorizer.fit_transform([row["features"] for row in ablated_train])
x_val = vectorizer.transform([row["features"] for row in ablated_val])
x_test = vectorizer.transform([row["features"] for row in ablated_test])

models = m30_v1_train._make_models()
metrics_rows = []
proba_map = {}
proba_map_cal = {}
for model_name, model in models.items():
    model.fit(x_train, y_train)
    probs_test = list(model.predict_proba(x_test)[:, 1])
    probs_val = list(model.predict_proba(x_val)[:, 1])
    proba_map[model_name] = probs_test
    metrics_rows.append(
        m30_v1_train._metric_row(
            f"m30_recent_ablate_{model_name}",
            "test",
            len(dynamic_rows),
            len(ablated_train),
            len(ablated_val),
            len(ablated_test),
            y_test,
            probs_test,
        )
    )
    metrics_rows.append(
        m30_v1_train._metric_row(
            f"m30_recent_ablate_{model_name}",
            "val",
            len(dynamic_rows),
            len(ablated_train),
            len(ablated_val),
            len(ablated_test),
            y_val,
            probs_val,
        )
    )
    probs_cal = m30_v1_train._isotonic_calibrate(model, x_val, y_val, x_test)
    proba_map_cal[model_name] = probs_cal
    metrics_rows.append(
        m30_v1_train._metric_row(
            f"m30_recent_ablate_{model_name}_cal",
            "test",
            len(dynamic_rows),
            len(ablated_train),
            len(ablated_val),
            len(ablated_test),
            y_test,
            probs_cal,
        )
    )

ensemble_test = [
    (proba_map["xgb"][index] + proba_map["hist_gb"][index]) / 2.0
    for index in range(len(y_test))
]
ensemble_test_cal = [
    (proba_map_cal["xgb"][index] + proba_map_cal["hist_gb"][index]) / 2.0
    for index in range(len(y_test))
]
metrics_rows.append(
    m30_v1_train._metric_row(
        "m30_recent_ablate_champion_q4_ensemble_avg",
        "test",
        len(dynamic_rows),
        len(ablated_train),
        len(ablated_val),
        len(ablated_test),
        y_test,
        ensemble_test,
    )
)
metrics_rows.append(
    m30_v1_train._metric_row(
        "m30_recent_ablate_champion_q4_ensemble_avg_cal",
        "test",
        len(dynamic_rows),
        len(ablated_train),
        len(ablated_val),
        len(ablated_test),
        y_test,
        ensemble_test_cal,
    )
)

current_metrics_path = m30_v1_train.OUT_DIR / "q4_metrics.csv"
current_lines = current_metrics_path.read_text(encoding="utf-8").splitlines()
current_header = current_lines[0].split(",")
current_rows = []
for line in current_lines[1:]:
    values = line.split(",")
    current_rows.append(dict(zip(current_header, values)))
current_lookup = {
    (row["model"], row["split"]): row for row in current_rows
}

compare_targets = [
    ("m30_v1_xgb", "m30_recent_ablate_xgb", "test"),
    ("m30_v1_hist_gb", "m30_recent_ablate_hist_gb", "test"),
    ("m30_v1_champion_q4_ensemble_avg", "m30_recent_ablate_champion_q4_ensemble_avg", "test"),
    ("m30_v1_champion_q4_ensemble_avg_cal", "m30_recent_ablate_champion_q4_ensemble_avg_cal", "test"),
]
new_lookup = {(row["model"], row["split"]): row for row in metrics_rows}
comparisons = []
for base_model, ablated_model, split in compare_targets:
    base_row = current_lookup[(base_model, split)]
    new_row = new_lookup[(ablated_model, split)]
    comparisons.append(
        {
            "baseline_model": base_model,
            "ablated_model": ablated_model,
            "split": split,
            "baseline_accuracy": float(base_row["accuracy"]),
            "ablated_accuracy": float(new_row["accuracy"]),
            "delta_accuracy": float(new_row["accuracy"] - float(base_row["accuracy"])),
            "baseline_roc_auc": float(base_row["roc_auc"]),
            "ablated_roc_auc": float(new_row["roc_auc"]),
            "delta_roc_auc": float(new_row["roc_auc"] - float(base_row["roc_auc"])),
            "baseline_log_loss": float(base_row["log_loss"]),
            "ablated_log_loss": float(new_row["log_loss"]),
            "delta_log_loss": float(new_row["log_loss"] - float(base_row["log_loss"])),
        }
    )

result = {
    "split_info": split_info,
    "ablation_scope": {
        "removed_prefixes": list(ABLATE_PREFIXES),
        "removed_exact": sorted(ABLATE_EXACT),
        "feature_count_after_ablation": int(len(vectorizer.feature_names_)),
    },
    "metrics": metrics_rows,
    "comparisons_vs_current": comparisons,
}

print(json.dumps(result, ensure_ascii=False, indent=2))

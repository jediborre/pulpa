import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
TRAINING = ROOT / "match" / "training"
sys.path.insert(0, str(TRAINING))

import train_q4_m30_v1 as m30_v1_train  # noqa: E402


def _feature_names(vectorizer):
    if hasattr(vectorizer, "get_feature_names_out"):
        return list(vectorizer.get_feature_names_out())
    return list(vectorizer.feature_names_)


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _corr(values, targets):
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(values) < 2:
        return None
    if np.std(values) == 0 or np.std(targets) == 0:
        return None
    corr = np.corrcoef(values, targets)[0, 1]
    if math.isnan(corr):
        return None
    return float(corr)


def _feature_family(name: str) -> str:
    if name.startswith("q1_q2_q3_partial_combo="):
        return "combo_q1_q2_q3_partial"
    if name.startswith("q3_partial_current_combo="):
        return "combo_q3_partial_current"
    if name.startswith("halftime_q3_partial_combo="):
        return "combo_halftime_q3_partial"
    if name.startswith("halftime_current_combo="):
        return "combo_halftime_current"
    if "recent_" in name:
        return "recent_window"
    if "trailing_now_" in name or "current_trailer_" in name:
        return "current_trailer_pressure"
    if (
        "halftime_trailer_" in name
        or "halftime_leader_" in name
        or "lead_flip_" in name
    ):
        return "halftime_recovery"
    if name.startswith("q3_partial_"):
        return "q3_partial"
    if (
        "quarter_minutes" in name
        or "q3_elapsed" in name
        or "q3_remaining" in name
        or "q4_start" in name
        or "q3_start" in name
    ):
        return "snapshot_phase"
    if name.startswith("score_est_") or name.startswith("current_"):
        return "current_state"
    if name.startswith("q1_") or name.startswith("q2_"):
        return "early_quarters"
    if name.startswith("prior_wr") or name.endswith("prior_wr"):
        return "prior_strength"
    if name.startswith("gp_"):
        return "graph"
    if name.startswith("gender_bucket"):
        return "meta"
    return "other"


artifact = joblib.load(m30_v1_train.OUT_DIR / "q4_m30_v1_champion.joblib")
samples, preloaded = m30_v1_train._build_base_samples_and_data(
    m30_v1_train.DB_PATH
)
dynamic_rows = m30_v1_train._build_dynamic_samples(
    samples,
    preloaded=preloaded,
)
_, _, test_rows, split_info = m30_v1_train._split_rows_temporal_by_match(
    dynamic_rows
)

current_vectorizer = DictVectorizer(sparse=False)
X_test = current_vectorizer.fit_transform(
    [row["features"] for row in test_rows]
)
feature_names = _feature_names(current_vectorizer)
y_test = np.asarray([row["target"] for row in test_rows], dtype=int)

artifact_names = set(_feature_names(artifact["vectorizer"]))
current_names = set(feature_names)
feature_presence_check = {
    "added_expected_present": {
            "big_halftime_lead_now_close": (
                "big_halftime_lead_now_close" in current_names
            ),
            "halftime_trailer_strong_recovery": (
                "halftime_trailer_strong_recovery" in current_names
            ),
    },
    "removed_expected_absent": {
        "q1_q2_q3_partial_combo=*": not any(
            name.startswith("q1_q2_q3_partial_combo=")
            for name in current_names
        ),
        "q3_partial_current_combo=*": not any(
            name.startswith("q3_partial_current_combo=")
            for name in current_names
        ),
    },
    "artifact_only_count": int(len(artifact_names - current_names)),
    "current_only_count": int(len(current_names - artifact_names)),
}

feature_rows = []
for idx, name in enumerate(feature_names):
    values = X_test[:, idx]
    corr = _corr(values, y_test)
    if corr is None:
        continue
    feature_rows.append(
        {
            "feature": name,
            "family": _feature_family(name),
            "corr": float(corr),
            "abs_corr": float(abs(corr)),
            "non_zero_rate": float(np.mean(values != 0)),
            "mean_target1": _safe_float(np.mean(values[y_test == 1])),
            "mean_target0": _safe_float(np.mean(values[y_test == 0])),
        }
    )

corr_df = pd.DataFrame(feature_rows).sort_values("abs_corr", ascending=False)

family_summary = []
for family, sub in corr_df.groupby("family"):
    top5 = sub.head(5)
    family_summary.append(
        {
            "family": family,
            "feature_count": int(len(sub)),
            "top_abs_corr": float(sub["abs_corr"].max()),
            "mean_top5_abs_corr": float(top5["abs_corr"].mean()),
            "top_features": top5[
                ["feature", "corr", "abs_corr"]
            ].to_dict(orient="records"),
        }
    )
family_summary = sorted(
    family_summary,
    key=lambda row: row["top_abs_corr"],
    reverse=True,
)

combo_focus = corr_df[
    corr_df["family"].isin(
        [
            "combo_q1_q2_q3_partial",
            "combo_q3_partial_current",
            "combo_halftime_q3_partial",
            "combo_halftime_current",
        ]
    )
]
recovery_focus = corr_df[
    corr_df["family"].isin(
        [
            "halftime_recovery",
            "current_trailer_pressure",
            "current_state",
            "snapshot_phase",
            "q3_partial",
            "recent_window",
            "graph",
        ]
    )
]

specific_prefixes = {}
for prefix in [
    "q1_q2_q3_partial_combo=",
    "q3_partial_current_combo=",
    "halftime_q3_partial_combo=",
    "halftime_current_combo=",
]:
    sub = corr_df[corr_df["feature"].str.startswith(prefix)]
    specific_prefixes[prefix] = sub.head(15).to_dict(orient="records")

result = {
    "split_info": split_info,
    "feature_presence_check": feature_presence_check,
    "top_overall": corr_df.head(40).to_dict(orient="records"),
    "family_summary": family_summary,
    "top_combo_features": combo_focus.head(30).to_dict(orient="records"),
    "top_recovery_pressure_features": recovery_focus.head(40).to_dict(
        orient="records"
    ),
    "specific_combo_prefixes": specific_prefixes,
}

print(json.dumps(result, ensure_ascii=False, indent=2))

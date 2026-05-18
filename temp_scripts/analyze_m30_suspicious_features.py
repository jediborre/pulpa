import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
TRAINING = ROOT / "match" / "training"
sys.path.insert(0, str(TRAINING))

import train_q4_m30_v1 as m30_v1_train  # noqa: E402


def _feature_names(vectorizer):
    if hasattr(vectorizer, "get_feature_names_out"):
        return list(vectorizer.get_feature_names_out())
    return list(vectorizer.feature_names_)


def _corr(values, targets):
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(values) < 2 or np.std(values) == 0 or np.std(targets) == 0:
        return None
    corr = np.corrcoef(values, targets)[0, 1]
    if math.isnan(corr):
        return None
    return float(corr)


def _family(name: str) -> str:
    if name.startswith("halftime_q3_partial_combo="):
        return "combo_halftime_q3_partial"
    if name.startswith("halftime_current_combo="):
        return "combo_halftime_current"
    if name.startswith("q3_partial_"):
        return "q3_partial"
    if name.startswith("recent_"):
        return "recent_window"
    if (
        name.startswith("halftime_trailer_")
        or name.startswith("halftime_leader_")
        or name.startswith("lead_flip_")
    ):
        return "halftime_recovery"
    if (
        name.startswith("score_est_")
        or name.startswith("current_")
        or name.startswith("trailing_now_")
    ):
        return "current_state"
    if name.startswith("q1_") or name.startswith("q2_"):
        return "early_quarters"
    if "prior_wr" in name:
        return "prior_strength"
    if name.startswith("gp_"):
        return "graph"
    if "quarter_minutes" in name or "q3_elapsed" in name or "q3_remaining" in name:
        return "snapshot_phase"
    return "other"


artifact = joblib.load(m30_v1_train.OUT_DIR / "q4_m30_v1_champion.joblib")
dynamic_rows = joblib.load(m30_v1_train.DYNAMIC_ROWS_CACHE)
_, _, test_rows, _ = m30_v1_train._split_rows_temporal_by_match(dynamic_rows)

vectorizer = artifact["vectorizer"]
feature_names = _feature_names(vectorizer)
X_test = vectorizer.transform([row["features"] for row in test_rows])
y_test = np.asarray([row["target"] for row in test_rows], dtype=int)

xgb = artifact["models"]["xgb"]
importances = getattr(xgb, "feature_importances_", None)
if importances is None:
    raise RuntimeError("xgb feature_importances_ not available")

rows = []
for idx, name in enumerate(feature_names):
    values = X_test[:, idx]
    corr = _corr(values, y_test)
    abs_corr = 0.0 if corr is None else abs(float(corr))
    importance = float(importances[idx])
    rows.append(
        {
            "feature": name,
            "family": _family(name),
            "importance": importance,
            "corr": corr,
            "abs_corr": abs_corr,
            "importance_rank_score": importance,
            "suspicious_score": importance / max(abs_corr, 0.005),
            "non_zero_rate": float(np.mean(values != 0)),
        }
    )

df = pd.DataFrame(rows).sort_values("importance", ascending=False)

suspicious = df[
    (df["importance"] >= 0.008)
    & (df["abs_corr"] <= 0.025)
].sort_values(["suspicious_score", "importance"], ascending=[False, False])

family_mass = (
    df.groupby("family", as_index=False)
    .agg(
        feature_count=("feature", "count"),
        importance_sum=("importance", "sum"),
        mean_abs_corr=("abs_corr", "mean"),
        max_abs_corr=("abs_corr", "max"),
    )
    .sort_values("importance_sum", ascending=False)
)

result = {
    "top_importance": df.head(35).to_dict(orient="records"),
    "suspicious_high_importance_low_corr": suspicious.head(40).to_dict(
        orient="records"
    ),
    "family_importance_mass": family_mass.to_dict(orient="records"),
}

print(json.dumps(result, ensure_ascii=False, indent=2))

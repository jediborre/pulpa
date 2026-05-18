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


def _is_halftime_transition(name: str) -> bool:
    return (
        name.startswith("halftime_trailer_")
        or name.startswith("halftime_leader_")
        or name == "lead_flip_from_halftime"
        or name == "halftime_to_current_margin_delta"
        or name == "abs_margin_delta_from_halftime"
    )


artifact = joblib.load(m30_v1_train.OUT_DIR / "q4_m30_v1_champion.joblib")
cached = joblib.load(m30_v1_train.DYNAMIC_ROWS_CACHE)
if isinstance(cached, dict):
    dynamic_rows = cached["rows"]
else:
    dynamic_rows = cached
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
    if not _is_halftime_transition(name):
        continue
    values = X_test[:, idx]
    corr = _corr(values, y_test)
    abs_corr = 0.0 if corr is None else abs(float(corr))
    importance = float(importances[idx])
    rows.append(
        {
            "feature": name,
            "importance": importance,
            "corr": corr,
            "abs_corr": abs_corr,
            "suspicious_score": importance / max(abs_corr, 0.005),
            "non_zero_rate": float(np.mean(values != 0)),
        }
    )

df = pd.DataFrame(rows).sort_values(
    ["suspicious_score", "importance"],
    ascending=[False, False],
)

result = {
    "top_by_suspicious_score": df.to_dict(orient="records"),
    "top_by_importance": df.sort_values("importance", ascending=False).to_dict(
        orient="records"
    ),
}

print(json.dumps(result, ensure_ascii=False, indent=2))

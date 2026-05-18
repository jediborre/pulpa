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

import report_m_v1_roi as rep  # noqa: E402
import train_q4_m27_v1 as m27_v1_train  # noqa: E402
import train_q4_m30_v1 as m30_v1_train  # noqa: E402
import train_q3_q4_models_v6 as v6  # noqa: E402


def _feature_names(vectorizer):
    if hasattr(vectorizer, "get_feature_names_out"):
        return list(vectorizer.get_feature_names_out())
    return list(vectorizer.feature_names_)


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _cohen_d(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return None
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos_var = pos.var(ddof=1)
    neg_var = neg.var(ddof=1)
    pooled_num = ((len(pos) - 1) * pos_var) + ((len(neg) - 1) * neg_var)
    pooled_den = len(pos) + len(neg) - 2
    if pooled_den <= 0:
        return None
    pooled_std = math.sqrt(max(pooled_num / pooled_den, 0.0))
    if pooled_std == 0:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled_std)


def _point_biserial(values, targets):
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(values) < 2 or values.std() == 0 or targets.std() == 0:
        return None
    corr = np.corrcoef(values, targets)[0, 1]
    if math.isnan(corr):
        return None
    return float(corr)


def _series_accuracy(probs, targets):
    picks = (np.asarray(probs, dtype=float) >= 0.5).astype(int)
    return float((picks == np.asarray(targets, dtype=int)).mean())


def _pick_strength(probs):
    probs = np.asarray(probs, dtype=float)
    return np.maximum(probs, 1.0 - probs)


_, _, test_rows = rep._prepare_splits(force_rebuild=False, only_date=None)
match_ids = [str(sample.match_id) for sample in test_rows]

m27_art = joblib.load(rep.M27_V1_CHAMPION_PATH)
m30_art = joblib.load(rep.M30_V1_CHAMPION_PATH)

m27_names = _feature_names(m27_art["vectorizer"])
m30_names = _feature_names(m30_art["vectorizer"])

m27_xgb_imp = getattr(m27_art["models"]["xgb"], "feature_importances_", None)
m30_xgb_imp = getattr(m30_art["models"]["xgb"], "feature_importances_", None)

m27_top = []
if m27_xgb_imp is not None:
    pairs = sorted(zip(m27_names, m27_xgb_imp), key=lambda x: float(x[1]), reverse=True)
    m27_top = [
        {"feature": name, "importance": float(imp)}
        for name, imp in pairs[:25]
    ]

m30_top = []
if m30_xgb_imp is not None:
    pairs = sorted(zip(m30_names, m30_xgb_imp), key=lambda x: float(x[1]), reverse=True)
    m30_top = [
        {"feature": name, "importance": float(imp)}
        for name, imp in pairs[:25]
    ]

conn = v6.db_mod.get_conn(str(v6.DB_PATH))
v6.db_mod.init_db(conn)
rows27 = []
rows30 = []
qdur_rows = []
for sample in test_rows:
    match_data = v6.db_mod.get_match(conn, str(sample.match_id))
    if not match_data:
        continue
    qdur = m30_v1_train._infer_regulation_quarter_minutes(match_data)
    feat27 = m27_v1_train._build_m27_v1_features(sample, match_data)
    feat30 = m30_v1_train._build_m30_v1_features(sample, match_data)
    base = {
        "match_id": str(sample.match_id),
        "target_q4": int(sample.target_q4),
        "league": str(sample.features_q4.get("league", "")),
        "quarter_minutes": float(qdur),
    }
    rows27.append(base | feat27)
    rows30.append(base | feat30)
    qdur_rows.append(base)
conn.close()

df27 = pd.DataFrame(rows27)
df30 = pd.DataFrame(rows30)
qdur_df = pd.DataFrame(qdur_rows)

p27 = rep._predict_m_v1_probs_raw(
    test_rows,
    "m27_v1",
    rep.M27_V1_CHAMPION_PATH,
    m27_v1_train._build_m27_v1_features,
    int(m27_v1_train.SNAPSHOT_MINUTE),
)
p30 = rep._predict_m_v1_probs_raw(
    test_rows,
    "m30_v1",
    rep.M30_V1_CHAMPION_PATH,
    m30_v1_train._build_m30_v1_features,
    int(m30_v1_train.SNAPSHOT_MINUTE),
)

pred_df = qdur_df.copy()
pred_df["p27"] = p27
pred_df["p30"] = p30
pred_df["ps27"] = _pick_strength(pred_df["p27"])
pred_df["ps30"] = _pick_strength(pred_df["p30"])
pred_df["pick27"] = (pred_df["p27"] >= 0.5).astype(int)
pred_df["pick30"] = (pred_df["p30"] >= 0.5).astype(int)
pred_df["hit27"] = (pred_df["pick27"] == pred_df["target_q4"]).astype(int)
pred_df["hit30"] = (pred_df["pick30"] == pred_df["target_q4"]).astype(int)
pred_df["delta_ps30_27"] = pred_df["ps30"] - pred_df["ps27"]
pred_df["qdur_bucket"] = pred_df["quarter_minutes"].map(lambda x: "12m" if x >= 11 else "10m")

qdur_summary = []
for bucket, sub in pred_df.groupby("qdur_bucket"):
    qdur_summary.append(
        {
            "qdur_bucket": bucket,
            "matches": int(len(sub)),
            "m27_acc_raw": _series_accuracy(sub["p27"], sub["target_q4"]),
            "m30_acc_raw": _series_accuracy(sub["p30"], sub["target_q4"]),
            "m27_pick_strength_mean": float(sub["ps27"].mean()),
            "m30_pick_strength_mean": float(sub["ps30"].mean()),
            "delta_acc_m30_minus_m27": _series_accuracy(sub["p30"], sub["target_q4"]) - _series_accuracy(sub["p27"], sub["target_q4"]),
            "delta_pick_strength_m30_minus_m27": float(sub["delta_ps30_27"].mean()),
            "only_m27_correct": int(((sub["hit27"] == 1) & (sub["hit30"] == 0)).sum()),
            "only_m30_correct": int(((sub["hit30"] == 1) & (sub["hit27"] == 0)).sum()),
        }
    )

m30_q3_cols = [col for col in df30.columns if "q3_partial" in col]
q3_partial_stats = []
for col in m30_q3_cols:
    values = pd.to_numeric(df30[col], errors="coerce")
    usable = values.notna()
    values = values[usable]
    targets = df30.loc[usable, "target_q4"]
    pos = values[targets == 1]
    neg = values[targets == 0]
    q3_partial_stats.append(
        {
            "feature": col,
            "mean_target1": _safe_float(pos.mean() if len(pos) else None),
            "mean_target0": _safe_float(neg.mean() if len(neg) else None),
            "std_all": _safe_float(values.std()),
            "cohen_d": _safe_float(_cohen_d(pos, neg)),
            "point_biserial": _safe_float(_point_biserial(values, targets)),
            "non_null": int(usable.sum()),
            "unique_values": int(values.nunique(dropna=True)),
            "importance_xgb": _safe_float(
                float(m30_xgb_imp[m30_names.index(col)]) if m30_xgb_imp is not None and col in m30_names else None
            ),
        }
    )
q3_partial_stats = sorted(
    q3_partial_stats,
    key=lambda row: abs(row["cohen_d"]) if row["cohen_d"] is not None else -1,
    reverse=True,
)

q3_partial_shift = []
for col in m30_q3_cols:
    values = pd.to_numeric(df30[col], errors="coerce")
    ten = values[df30["quarter_minutes"] < 11].dropna()
    twelve = values[df30["quarter_minutes"] >= 11].dropna()
    q3_partial_shift.append(
        {
            "feature": col,
            "mean_10m": _safe_float(ten.mean() if len(ten) else None),
            "mean_12m": _safe_float(twelve.mean() if len(twelve) else None),
            "cohen_d_10m_vs_12m": _safe_float(_cohen_d(twelve, ten)),
            "importance_xgb": _safe_float(
                float(m30_xgb_imp[m30_names.index(col)]) if m30_xgb_imp is not None and col in m30_names else None
            ),
        }
    )
q3_partial_shift = sorted(
    q3_partial_shift,
    key=lambda row: abs(row["cohen_d_10m_vs_12m"]) if row["cohen_d_10m_vs_12m"] is not None else -1,
    reverse=True,
)

shared_cols = sorted((set(df27.columns) & set(df30.columns)) - {"match_id", "target_q4", "league", "quarter_minutes"})
shared_q3_cols = [col for col in shared_cols if "q3_partial" in col]
shared_q3_compare = []
for col in shared_q3_cols:
    vals27 = pd.to_numeric(df27[col], errors="coerce")
    vals30 = pd.to_numeric(df30[col], errors="coerce")
    sub27 = pd.DataFrame({"value": vals27, "target": df27["target_q4"]}).dropna()
    sub30 = pd.DataFrame({"value": vals30, "target": df30["target_q4"]}).dropna()
    shared_q3_compare.append(
        {
            "feature": col,
            "m27_cohen_d": _safe_float(
                _cohen_d(
                    sub27.loc[sub27["target"] == 1, "value"],
                    sub27.loc[sub27["target"] == 0, "value"],
                )
            ),
            "m30_cohen_d": _safe_float(
                _cohen_d(
                    sub30.loc[sub30["target"] == 1, "value"],
                    sub30.loc[sub30["target"] == 0, "value"],
                )
            ),
            "m27_point_biserial": _safe_float(
                _point_biserial(sub27["value"], sub27["target"])
            ),
            "m30_point_biserial": _safe_float(
                _point_biserial(sub30["value"], sub30["target"])
            ),
        }
    )
shared_q3_compare = sorted(
    shared_q3_compare,
    key=lambda row: abs((row["m30_cohen_d"] or 0.0) - (row["m27_cohen_d"] or 0.0)),
    reverse=True,
)

result = {
    "artifact_overview": {
        "m27_feature_count": int(m27_art.get("feature_count", len(m27_names))),
        "m30_feature_count": int(m30_art.get("feature_count", len(m30_names))),
        "m27_top_xgb_features": m27_top,
        "m30_top_xgb_features": m30_top,
    },
    "quarter_duration_summary": sorted(qdur_summary, key=lambda row: row["qdur_bucket"]),
    "q3_partial_m30_signal_top": q3_partial_stats[:20],
    "q3_partial_10m_12m_shift_top": q3_partial_shift[:20],
    "shared_q3_partial_compare_top": shared_q3_compare[:20],
}

print(json.dumps(result, ensure_ascii=False, indent=2))

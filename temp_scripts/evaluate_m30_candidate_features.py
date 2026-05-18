import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
TRAINING = ROOT / "match" / "training"
sys.path.insert(0, str(TRAINING))

import train_q4_m30_v1 as m30_v1_train  # noqa: E402


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
    if len(values) < 2 or np.std(values) == 0 or np.std(targets) == 0:
        return None
    corr = np.corrcoef(values, targets)[0, 1]
    if math.isnan(corr):
        return None
    return float(corr)


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


def _bucket_abs_margin(value):
    value = abs(float(value))
    if value <= 3:
        return "01_03"
    if value <= 7:
        return "04_07"
    return "08_plus"


def _sign_label(value):
    value = float(value)
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return "zero"


def _leader_recent_pressure(score_diff, recent_diff):
    if score_diff > 0:
        return -float(recent_diff)
    if score_diff < 0:
        return float(recent_diff)
    return 0.0


def _trailer_recent_edge(trailing_side, recent_diff):
    if trailing_side == "home":
        return float(recent_diff)
    if trailing_side == "away":
        return -float(recent_diff)
    return 0.0


def _reachable_flag(abs_score_diff, halftime_deficit_abs):
    return int(
        halftime_deficit_abs >= 8 and abs_score_diff <= 7
    )


def _family(name):
    if "pressure" in name:
        return "pressure"
    if "recovery" in name or "comeback" in name:
        return "recovery"
    if "reachable" in name or "alive" in name:
        return "reachability"
    if "state" in name:
        return "state_interaction"
    return "other"


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
base_df = pd.DataFrame(
    [row["features"] | {"target": row["target"]} for row in test_rows]
)

feature_presence_check = {
    "big_halftime_lead_now_close": (
        "big_halftime_lead_now_close" in base_df.columns
    ),
    "halftime_trailer_strong_recovery": (
        "halftime_trailer_strong_recovery" in base_df.columns
    ),
    "q1_q2_q3_partial_combo": "q1_q2_q3_partial_combo" in base_df.columns,
    "q3_partial_current_combo": "q3_partial_current_combo" in base_df.columns,
}

candidates = {}
for _, row in base_df.iterrows():
    score_diff = float(row["score_est_diff"])
    halftime_diff = float(row["halftime_diff"])
    halftime_deficit_abs = float(row["halftime_deficit_abs"])
    trailing_now_deficit_abs = float(row["trailing_now_deficit_abs"])
    recent_2m_points_diff = float(row["recent_2m_points_diff"])
    recent_4m_points_diff = float(row["recent_4m_points_diff"])
    trailing_side = row["current_trailing_side"]
    leader_side = row["current_leader"]
    halftime_trailing_side = row["halftime_trailing_side"]

    leader_pressure_2m = _leader_recent_pressure(
        score_diff,
        recent_2m_points_diff,
    )
    leader_pressure_4m = _leader_recent_pressure(
        score_diff,
        recent_4m_points_diff,
    )
    trailer_edge_2m = _trailer_recent_edge(
        trailing_side,
        recent_2m_points_diff,
    )
    trailer_edge_4m = _trailer_recent_edge(
        trailing_side,
        recent_4m_points_diff,
    )
    margin_compression = abs(halftime_diff) - abs(score_diff)
    home_margin_change = score_diff - halftime_diff
    recovery_ratio = (
        margin_compression / halftime_deficit_abs
        if halftime_deficit_abs > 0
        else 0.0
    )
    abs_score_diff = abs(score_diff)
    current_margin_bucket = _bucket_abs_margin(score_diff)
    halftime_deficit_bucket = _bucket_abs_margin(halftime_deficit_abs)

    row_candidates = {
        "leader_pressure_2m": leader_pressure_2m,
        "leader_pressure_4m": leader_pressure_4m,
        "trailer_edge_2m": trailer_edge_2m,
        "trailer_edge_4m": trailer_edge_4m,
        "leader_pressure_2m_per_margin": (
            leader_pressure_2m / (abs_score_diff + 1.0)
        ),
        "leader_pressure_4m_per_margin": (
            leader_pressure_4m / (abs_score_diff + 1.0)
        ),
        "trailer_edge_2m_per_margin": trailer_edge_2m / (abs_score_diff + 1.0),
        "trailer_edge_4m_per_margin": trailer_edge_4m / (abs_score_diff + 1.0),
        "margin_compression": margin_compression,
        "margin_compression_ratio": recovery_ratio,
        "home_margin_change": home_margin_change,
        "leader_under_pressure_close": int(
            abs_score_diff <= 7 and leader_pressure_2m > 0
        ),
        "leader_under_pressure_big": int(
            abs_score_diff >= 8 and leader_pressure_2m > 0
        ),
        "close_game_with_trailer_momentum": int(
            abs_score_diff <= 7 and trailer_edge_2m > 0
        ),
        "reachable_trailer_with_momentum": int(
            _reachable_flag(abs_score_diff, halftime_deficit_abs)
            and trailer_edge_2m > 0
        ),
        "reachable_trailer_without_momentum": int(
            _reachable_flag(abs_score_diff, halftime_deficit_abs)
            and trailer_edge_2m <= 0
        ),
        "big_halftime_lead_now_close": int(
            halftime_deficit_abs >= 8 and abs_score_diff <= 7
        ),
        "big_halftime_lead_still_big": int(
            halftime_deficit_abs >= 8 and abs_score_diff >= 8
        ),
        "halftime_trailer_positive_recovery": int(
            halftime_trailing_side != "tied" and margin_compression > 0
        ),
        "halftime_trailer_strong_recovery": int(
            halftime_trailing_side != "tied" and recovery_ratio >= 0.5
        ),
        "halftime_trailer_full_neutralization": int(
            halftime_trailing_side != "tied" and abs_score_diff == 0
        ),
        "halftime_leader_under_pressure_close": int(
            leader_side != "tied"
            and leader_pressure_2m > 0
            and abs_score_diff <= 7
        ),
        "halftime_leader_under_pressure_big": int(
            leader_side != "tied"
            and leader_pressure_2m > 0
            and abs_score_diff >= 8
        ),
        "recovery_state": (
            f"hd{halftime_deficit_bucket}"
            f"__cm{current_margin_bucket}"
            f"__mom{_sign_label(trailer_edge_2m)}"
        ),
        "pressure_state": (
            f"cm{current_margin_bucket}"
            f"__lp{_sign_label(leader_pressure_2m)}"
        ),
        "trailer_state": (
            f"trail_{trailing_side}"
            f"__cm{current_margin_bucket}"
            f"__mom{_sign_label(trailer_edge_2m)}"
        ),
    }

    for key, value in row_candidates.items():
        candidates.setdefault(key, []).append(value)

cand_df = pd.DataFrame(candidates)
y = base_df["target"].astype(int)

results = []
for col in cand_df.columns:
    series = cand_df[col]
    if series.dtype == object:
        dummies = pd.get_dummies(series, prefix=col)
        for dummy_col in dummies.columns:
            values = dummies[dummy_col].astype(float)
            corr = _corr(values, y)
            if corr is None:
                continue
            pos = values[y == 1]
            neg = values[y == 0]
            results.append(
                {
                    "feature": dummy_col,
                    "source_candidate": col,
                    "family": _family(col),
                    "corr": float(corr),
                    "abs_corr": float(abs(corr)),
                    "cohen_d": _safe_float(_cohen_d(pos, neg)),
                    "non_zero_rate": float(values.mean()),
                    "mean_target1": _safe_float(pos.mean()),
                    "mean_target0": _safe_float(neg.mean()),
                }
            )
    else:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        corr = _corr(values, y)
        if corr is None:
            continue
        pos = values[y == 1]
        neg = values[y == 0]
        results.append(
            {
                "feature": col,
                "source_candidate": col,
                "family": _family(col),
                "corr": float(corr),
                "abs_corr": float(abs(corr)),
                "cohen_d": _safe_float(_cohen_d(pos, neg)),
                "non_zero_rate": float(np.mean(values != 0)),
                "mean_target1": _safe_float(pos.mean()),
                "mean_target0": _safe_float(neg.mean()),
            }
        )

res_df = pd.DataFrame(results).sort_values("abs_corr", ascending=False)
strong_df = res_df[
    (res_df["abs_corr"] >= 0.03) | (res_df["cohen_d"].abs() >= 0.06)
]
family_summary = []
for family, sub in res_df.groupby("family"):
    family_summary.append(
        {
            "family": family,
            "feature_count": int(len(sub)),
            "top_abs_corr": float(sub["abs_corr"].max()),
            "mean_top5_abs_corr": float(sub.head(5)["abs_corr"].mean()),
            "top_features": sub.head(10).to_dict(orient="records"),
        }
    )
family_summary = sorted(
    family_summary,
    key=lambda row: row["top_abs_corr"],
    reverse=True,
)

result = {
    "split_info": split_info,
    "feature_presence_check": feature_presence_check,
    "top_candidates_overall": res_df.head(40).to_dict(orient="records"),
    "strong_candidates_only": strong_df.head(40).to_dict(orient="records"),
    "family_summary": family_summary,
}

print(json.dumps(result, ensure_ascii=False, indent=2))

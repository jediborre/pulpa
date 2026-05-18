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
    if value <= 1:
        return "01"
    if value <= 3:
        return "02_03"
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


def _window_events(match_data, end_minute, window_minutes):
    events = m30_v1_train._pbp_events_upto_m30(match_data, end_minute)
    start_minute = max(0.0, float(end_minute) - float(window_minutes))
    return [
        event
        for event in events
        if float(event.get("_global_min", 0.0)) >= start_minute
    ]


def _event_points(events, side):
    total = 0
    for event in events:
        if event.get("team") == side:
            total += int(event.get("points", 0) or 0)
    return total


def _max_run(events, side):
    best = 0
    run = 0
    for event in events:
        team = event.get("team")
        pts = int(event.get("points", 0) or 0)
        if team == side and pts > 0:
            run += pts
            if run > best:
                best = run
        elif team in {"home", "away"}:
            run = 0
    return best


def _last_scoring_side(events):
    for event in reversed(events):
        team = event.get("team")
        pts = int(event.get("points", 0) or 0)
        if team in {"home", "away"} and pts > 0:
            return team
    return "none"


def _trailer_edge(trailing_side, home_points, away_points):
    if trailing_side == "home":
        return float(home_points - away_points)
    if trailing_side == "away":
        return float(away_points - home_points)
    return 0.0


def _family(name):
    if "close_state" in name or "boundary_state" in name:
        return "boundary_state"
    if "pressure" in name or "edge" in name:
        return "pressure"
    if "run" in name or "stalled" in name or "last_scoring" in name:
        return "closing_run"
    if "flip" in name or "neutralized" in name or "recovery" in name:
        return "recovery"
    if "10m" in name or "boundary" in name:
        return "boundary_gate"
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

base_df = pd.DataFrame(test_rows)
candidate_rows = []
for row in test_rows:
    features = row["features"]
    match_data = preloaded.get(str(row["match_id"])) or {}
    quarter_minutes = float(
        m30_v1_train._infer_regulation_quarter_minutes(match_data)
    )
    q4_start_minute = quarter_minutes * 3.0
    is_10m_boundary = int(abs(q4_start_minute - 30.0) < 1e-9)

    score_diff = float(features.get("score_est_diff", 0.0))
    abs_score_diff = abs(score_diff)
    halftime_diff = float(features.get("halftime_diff", 0.0))
    halftime_deficit_abs = float(features.get("halftime_deficit_abs", 0.0))
    current_leader = str(features.get("current_leader", "tied"))
    trailing_side = str(features.get("current_trailing_side", "tied"))
    halftime_leader = str(features.get("halftime_leader", "tied"))
    halftime_trailing_side = str(
        features.get("halftime_trailing_side", "tied")
    )
    q2_winner = str(features.get("q2_winner", "tied"))
    q3_partial_leader = str(features.get("q3_partial_leader", "tied"))
    home_prior_wr = float(features.get("home_prior_wr", 0.0))
    away_prior_wr = float(features.get("away_prior_wr", 0.0))

    close_90s = _window_events(match_data, 30.0, 1.5)
    close_120s = _window_events(match_data, 30.0, 2.0)
    close_60s = _window_events(match_data, 30.0, 1.0)

    close90_home = _event_points(close_90s, "home")
    close90_away = _event_points(close_90s, "away")
    close120_home = _event_points(close_120s, "home")
    close120_away = _event_points(close_120s, "away")
    close60_home = _event_points(close_60s, "home")
    close60_away = _event_points(close_60s, "away")

    trailer_edge_90s = _trailer_edge(trailing_side, close90_home, close90_away)
    trailer_edge_120s = _trailer_edge(
        trailing_side,
        close120_home,
        close120_away,
    )
    trailer_edge_60s = _trailer_edge(trailing_side, close60_home, close60_away)

    leader_side = "tied"
    if current_leader == "home":
        leader_side = "home"
    elif current_leader == "away":
        leader_side = "away"

    leader90_points = _event_points(close_90s, leader_side)
    leader120_points = _event_points(close_120s, leader_side)
    trailer_run_90s = _max_run(close_90s, trailing_side)
    trailer_run_120s = _max_run(close_120s, trailing_side)
    leader_run_90s = _max_run(close_90s, leader_side)
    last_scoring_90s = _last_scoring_side(close_90s)
    last_scoring_60s = _last_scoring_side(close_60s)

    close_margin_bucket = _bucket_abs_margin(abs_score_diff)
    halftime_bucket = _bucket_abs_margin(halftime_deficit_abs)
    trailer_momentum_sign = _sign_label(trailer_edge_90s)
    halftime_trailer_has_prior_edge = 0
    if halftime_trailing_side == "home":
        halftime_trailer_has_prior_edge = int(home_prior_wr > away_prior_wr)
    elif halftime_trailing_side == "away":
        halftime_trailer_has_prior_edge = int(away_prior_wr > home_prior_wr)
    halftime_trailer_won_q2 = int(
        halftime_trailing_side != "tied" and q2_winner == halftime_trailing_side
    )
    halftime_trailer_won_q3_partial = int(
        halftime_trailing_side != "tied"
        and q3_partial_leader == halftime_trailing_side
    )
    biglead_flip_or_tie = int(
        halftime_deficit_abs >= 8
        and (current_leader != halftime_leader or current_leader == "tied")
    )

    candidate_rows.append(
        {
            "target": int(row["target"]),
            "is_10m_boundary_snapshot": is_10m_boundary,
            "q3_close_one_possession_10m": int(
                is_10m_boundary and abs_score_diff <= 3
            ),
            "q3_close_tied_10m": int(is_10m_boundary and score_diff == 0),
            "q3_close_lead_flip_10m": int(
                is_10m_boundary
                and halftime_leader not in {"tied", current_leader}
                and current_leader != "tied"
            ),
            "q3_close_big_halftime_lead_now_one_possession_10m": int(
                is_10m_boundary
                and halftime_deficit_abs >= 8
                and abs_score_diff <= 3
            ),
            "q3_close_trailer_edge_60s_10m": (
                trailer_edge_60s if is_10m_boundary else 0.0
            ),
            "q3_close_trailer_edge_90s_10m": (
                trailer_edge_90s if is_10m_boundary else 0.0
            ),
            "q3_close_trailer_edge_120s_10m": (
                trailer_edge_120s if is_10m_boundary else 0.0
            ),
            "q3_close_trailer_pressure_per_margin_90s_10m": (
                trailer_edge_90s / (abs_score_diff + 1.0)
                if is_10m_boundary
                else 0.0
            ),
            "q3_close_trailer_pressure_per_margin_120s_10m": (
                trailer_edge_120s / (abs_score_diff + 1.0)
                if is_10m_boundary
                else 0.0
            ),
            "q3_close_trailer_run_90s_10m": (
                trailer_run_90s if is_10m_boundary else 0.0
            ),
            "q3_close_trailer_run_120s_10m": (
                trailer_run_120s if is_10m_boundary else 0.0
            ),
            "q3_close_leader_run_90s_10m": (
                leader_run_90s if is_10m_boundary else 0.0
            ),
            "q3_close_leader_stalled_90s_10m": int(
                is_10m_boundary and leader90_points == 0 and trailer_edge_90s > 0
            ),
            "q3_close_leader_stalled_120s_10m": int(
                is_10m_boundary and leader120_points == 0 and trailer_edge_120s > 0
            ),
            "q3_close_trailer_last_scoring_90s_10m": int(
                is_10m_boundary and trailing_side != "tied" and last_scoring_90s == trailing_side
            ),
            "q3_close_trailer_last_scoring_60s_10m": int(
                is_10m_boundary and trailing_side != "tied" and last_scoring_60s == trailing_side
            ),
            "q3_close_trailer_outscored_by4_120s_10m": int(
                is_10m_boundary and trailer_edge_120s >= 4
            ),
            "q3_close_trailer_outscored_by6_120s_10m": int(
                is_10m_boundary and trailer_edge_120s >= 6
            ),
            "q3_close_trailer_run_ge4_90s_10m": int(
                is_10m_boundary and trailer_run_90s >= 4
            ),
            "q3_close_trailer_run_ge6_120s_10m": int(
                is_10m_boundary and trailer_run_120s >= 6
            ),
            "q3_close_trailer_momentum_one_possession_10m": int(
                is_10m_boundary and abs_score_diff <= 3 and trailer_edge_90s > 0
            ),
            "q3_close_biglead_neutralized_with_momentum_10m": int(
                is_10m_boundary
                and halftime_deficit_abs >= 8
                and abs_score_diff <= 3
                and trailer_edge_90s > 0
            ),
            "q3_close_flip_or_tie_after_biglead_10m": int(
                is_10m_boundary and biglead_flip_or_tie
            ),
            "q3_close_biglead_now_04_07_10m": int(
                is_10m_boundary
                and halftime_deficit_abs >= 8
                and 4 <= abs_score_diff <= 7
            ),
            "q3_close_biglead_now_tied_10m": int(
                is_10m_boundary
                and halftime_deficit_abs >= 8
                and current_leader == "tied"
            ),
            "q3_close_biglead_flip_or_tie_with_prior_edge_10m": int(
                is_10m_boundary
                and biglead_flip_or_tie
                and halftime_trailer_has_prior_edge == 1
            ),
            "q3_close_biglead_flip_or_tie_without_prior_edge_10m": int(
                is_10m_boundary
                and biglead_flip_or_tie
                and halftime_trailer_has_prior_edge == 0
            ),
            "q3_close_biglead_flip_or_tie_and_q2_trailer_win_10m": int(
                is_10m_boundary
                and biglead_flip_or_tie
                and halftime_trailer_won_q2 == 1
            ),
            "q3_close_biglead_flip_or_tie_and_q3_trailer_win_10m": int(
                is_10m_boundary
                and biglead_flip_or_tie
                and halftime_trailer_won_q3_partial == 1
            ),
            "q3_close_recovery_state_10m": (
                f"hd{halftime_bucket}"
                f"__cm{close_margin_bucket}"
                f"__mom{trailer_momentum_sign}"
                if is_10m_boundary
                else "inactive"
            ),
            "q3_close_boundary_state_10m": (
                f"lead{current_leader}"
                f"__trail{trailing_side}"
                f"__cm{close_margin_bucket}"
                f"__flip{int(halftime_leader not in {'tied', current_leader} and current_leader != 'tied')}"
                if is_10m_boundary
                else "inactive"
            ),
            "q3_close_halftime_trailer_state_10m": (
                f"httrail{halftime_trailing_side}"
                f"__cm{close_margin_bucket}"
                f"__mom{trailer_momentum_sign}"
                if is_10m_boundary
                else "inactive"
            ),
            "q3_close_biglead_recovery_state_10m": (
                f"prior{halftime_trailer_has_prior_edge}"
                f"__q2{halftime_trailer_won_q2}"
                f"__q3{halftime_trailer_won_q3_partial}"
                f"__cm{close_margin_bucket}"
                if is_10m_boundary and halftime_deficit_abs >= 8
                else "inactive"
            ),
        }
    )

cand_df = pd.DataFrame(candidate_rows)
y = cand_df["target"].astype(int)

results = []
for col in cand_df.columns:
    if col == "target":
        continue
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
    (res_df["abs_corr"] >= 0.03)
    | (res_df["cohen_d"].abs() >= 0.06)
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
    "top_candidates_overall": res_df.head(60).to_dict(orient="records"),
    "strong_candidates_only": strong_df.head(60).to_dict(orient="records"),
    "family_summary": family_summary,
}

print(json.dumps(result, ensure_ascii=False, indent=2))

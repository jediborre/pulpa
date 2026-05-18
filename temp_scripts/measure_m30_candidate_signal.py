"""Measure signal strength of proposed M30_V1 candidate features.

Compares candidates vs current features across:
- abs_corr (absolute correlation with Q4 target)
- cohen_d (effect size)
- non_zero_rate (coverage)
- Separated by 10m vs 12m game duration
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
TRAINING = ROOT / "match" / "training"
sys.path.insert(0, str(TRAINING))

import infer_match as infer_live
import train_q4_m30_v1 as m30  # noqa: E402


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


def _sign_label(value):
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return "zero"


def _margin_trend(halftime_diff, score_diff, halftime_leader, current_leader):
    """Categorize how the margin evolved from halftime to minute-30."""
    if current_leader == "tied":
        return "tied"
    if halftime_leader == "tied":
        return "leader_emerged"
    # same leader
    if halftime_leader == current_leader:
        if abs(score_diff) > abs(halftime_diff):
            return "extending"
        else:
            return "shrinking"
    # leader flipped
    return "flipped"


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
        elif team in ("home", "away"):
            run = 0
    return best


def _last_k_events_net(events, k):
    """Net points (home - away) in last K scoring events."""
    tail = events[-k:] if len(events) >= k else events
    home_pts = sum(int(e.get("points", 0) or 0) for e in tail if e.get("team") == "home")
    away_pts = sum(int(e.get("points", 0) or 0) for e in tail if e.get("team") == "away")
    return home_pts - away_pts


def _q3_pbp_events(match_data, quarter_minutes):
    """Get all Q3 play-by-play events."""
    pbp = match_data.get("play_by_play", {}) or {}
    q3_start = quarter_minutes * 2.0
    q4_start = quarter_minutes * 3.0
    events = []
    for quarter_label, plays in pbp.items():
        q_idx = infer_live._quarter_index(str(quarter_label))
        if q_idx is None or q_idx != 3:
            continue
        for play in plays or []:
            rem_sec = infer_live._clock_to_seconds(str(play.get("time", "")))
            if rem_sec is None or rem_sec > int(quarter_minutes * 60.0):
                continue
            elapsed_in_q = quarter_minutes - (rem_sec / 60.0)
            global_min = q3_start + elapsed_in_q
            if global_min <= m30.SNAPSHOT_MINUTE + 1e-9:
                event = dict(play)
                event["_global_min"] = global_min
                events.append(event)
    events.sort(key=lambda e: float(e.get("_global_min", 0.0)))
    return events


def measure_signals():
    print("[measure] Building base samples...")
    samples, preloaded = m30._build_base_samples_and_data(m30.DB_PATH)

    print("[measure] Building dynamic rows...")
    dynamic_rows = m30._build_dynamic_samples(samples, preloaded=preloaded)

    _, _, test_rows, split_info = m30._split_rows_temporal_by_match(dynamic_rows)
    print(f"[measure] Test rows: {len(test_rows)}, matches: {split_info}")

    # Build current feature matrix + compute candidates
    all_features = []
    for row in test_rows:
        feat = row["features"]
        match_data = preloaded.get(str(row["match_id"])) or {}
        quarter_minutes = float(m30._infer_regulation_quarter_minutes(match_data))
        q4_start_minute = quarter_minutes * 3.0
        is_10m = int(abs(quarter_minutes - 10.0) < 1e-9)
        is_12m = int(abs(quarter_minutes - 12.0) < 1e-9)

        score_diff = float(feat.get("score_est_diff", 0.0))
        halftime_diff = float(feat.get("halftime_diff", 0.0))
        halftime_leader = str(feat.get("halftime_leader", "tied"))
        current_leader = str(feat.get("current_leader", "tied"))
        halftime_trailing_side = str(feat.get("halftime_trailing_side", "tied"))
        trailing_side = str(feat.get("current_trailing_side", "tied"))
        q3_partial_home = float(feat.get("q3_partial_home", 0.0))
        q3_partial_away = float(feat.get("q3_partial_away", 0.0))
        q3_partial_total = q3_partial_home + q3_partial_away
        ht_total = float(feat.get("halftime_total", 0.0))
        q3_partial_diff = float(feat.get("q3_partial_diff", 0.0))
        home_prior_wr = float(feat.get("home_prior_wr", 0.0))
        away_prior_wr = float(feat.get("away_prior_wr", 0.0))
        halftime_deficit_abs = abs(halftime_diff)

        # --- CANDIDATE FEATURES ---

        # F1: margin_trend_category
        trend = _margin_trend(halftime_diff, score_diff, halftime_leader, current_leader)

        # F2: close_game_flag
        close_game = int(abs(score_diff) <= 3)

        # F3: q3_acceleration (Q3 pace / halftime pace per-minute)
        q3_elapsed = max(0.0, min(float(m30.SNAPSHOT_MINUTE), q4_start_minute) - quarter_minutes * 2.0)
        ht_per_min = ht_total / (quarter_minutes * 2.0) if quarter_minutes > 0 else 1.0
        q3_per_min = q3_partial_total / q3_elapsed if q3_elapsed > 0 else ht_per_min
        q3_accel = q3_per_min / ht_per_min if ht_per_min > 0 else 1.0

        # F4: is_10m_game / is_12m_game
        is_10m_flag = int(is_10m)
        is_12m_flag = int(is_12m)

        # F5: q3_is_complete (Q3 finished at snapshot)
        q3_is_complete = int(is_10m)

        # F6: q3_completion_ratio at snapshot
        q3_completion = float(feat.get("q3_partial_completion", 0.0))

        # F7: last 5 events net points (from PBP)
        q3_pbp = _q3_pbp_events(match_data, quarter_minutes)
        last_5_net = _last_k_events_net(q3_pbp, 5)

        # F8: last 3 events net points
        last_3_net = _last_k_events_net(q3_pbp, 3)

        # F9: q3_run_dominance_ratio
        home_max_run = _max_run(q3_pbp, "home")
        away_max_run = _max_run(q3_pbp, "away")
        max_run = max(home_max_run, away_max_run)
        run_dominance = max_run / q3_partial_total if q3_partial_total > 0 else 0.0

        # F10: halftime_to_q3_margin_delta_signed
        signed_delta = (abs(score_diff) - halftime_deficit_abs) * (1 if score_diff >= 0 else -1)

        # F11: score_est_diff * is_10m interaction
        score_diff_10m = score_diff if is_10m else 0.0

        # F12: halftime_diff * is_12m interaction
        halftime_diff_12m = halftime_diff if is_12m else 0.0

        # F13: q3_partial_diff * is_10m interaction
        q3_partial_diff_10m = q3_partial_diff if is_10m else 0.0

        # F14: last team to score in Q3
        last_scoring_team = "none"
        for ev in reversed(q3_pbp):
            team = ev.get("team")
            pts = int(ev.get("points", 0) or 0)
            if team in ("home", "away") and pts > 0:
                last_scoring_team = team
                break

        # F15: Q3 total points (pace indicator for Q4)
        q3_total_points = q3_partial_total

        # F16: did the halftime trailer also trail at end of Q3? (persistent trailing)
        persistent_trailer = int(
            trailing_side != "tied"
            and halftime_trailing_side != "tied"
            and trailing_side == halftime_trailing_side
        )

        # F17: Q3 margin bucket * 10m interaction
        q3_margin_bucket = "01_03" if abs(q3_partial_diff) <= 3 else ("04_07" if abs(q3_partial_diff) <= 7 else "08_plus")

        # F18: prior_wr_diff * close_game interaction
        prior_diff = home_prior_wr - away_prior_wr
        prior_diff_close = prior_diff if close_game else 0.0

        # F19: blowout recovery chance
        # halftime deficit >= 8 but now within 7
        blowout_recovery = int(halftime_deficit_abs >= 8 and abs(score_diff) <= 7)

        # F20: which team has momentum (last 3 events net sign)
        momentum_side = "home" if last_3_net > 0 else ("away" if last_3_net < 0 else "neutral")

        # F21: close_game * trailer_has_momentum
        trailer_momentum_close = int(
            close_game
            and trailing_side != "tied"
            and ((trailing_side == "home" and last_3_net > 0) or (trailing_side == "away" and last_3_net < 0))
        )

        # F22: q3_leader_extended (did the Q3 winner extend the halftime lead or start new lead?)
        # Q3 leader extended relative to halftime: q3_partial_diff aligned with overall score_diff direction
        q3_leader_aligned = int(
            (q3_partial_diff > 0 and score_diff > 0)
            or (q3_partial_diff < 0 and score_diff < 0)
        )

        candidates = {
            "candidate_margin_trend_leader_emerged": int(trend == "leader_emerged"),
            "candidate_margin_trend_extending": int(trend == "extending"),
            "candidate_margin_trend_shrinking": int(trend == "shrinking"),
            "candidate_margin_trend_flipped": int(trend == "flipped"),
            "candidate_margin_trend_tied": int(trend == "tied"),
            "candidate_close_game": close_game,
            "candidate_q3_acceleration": q3_accel,
            "candidate_is_10m_game": is_10m_flag,
            "candidate_is_12m_game": is_12m_flag,
            "candidate_q3_is_complete": q3_is_complete,
            "candidate_q3_completion_ratio": q3_completion,
            "candidate_last_5_events_net": last_5_net,
            "candidate_last_3_events_net": last_3_net,
            "candidate_q3_run_dominance_ratio": run_dominance,
            "candidate_halftime_to_q3_signed_delta": signed_delta,
            "candidate_score_diff_10m": score_diff_10m,
            "candidate_halftime_diff_12m": halftime_diff_12m,
            "candidate_q3_partial_diff_10m": q3_partial_diff_10m,
            "candidate_last_scoring_home_in_q3": int(last_scoring_team == "home"),
            "candidate_last_scoring_away_in_q3": int(last_scoring_team == "away"),
            "candidate_q3_total_points": q3_total_points,
            "candidate_persistent_trailer": persistent_trailer,
            "candidate_q3_margin_01_03": int(q3_margin_bucket == "01_03"),
            "candidate_q3_margin_04_07": int(q3_margin_bucket == "04_07"),
            "candidate_q3_margin_08_plus": int(q3_margin_bucket == "08_plus"),
            "candidate_prior_diff_close": prior_diff_close,
            "candidate_blowout_recovery": blowout_recovery,
            "candidate_momentum_home": int(momentum_side == "home"),
            "candidate_momentum_away": int(momentum_side == "away"),
            "candidate_trailer_momentum_close": trailer_momentum_close,
            "candidate_q3_leader_aligned": q3_leader_aligned,
        }
        all_features.append({
            "target": int(row["target"]),
            **feat,
            **candidates,
        })

    df = pd.DataFrame(all_features)
    y = df["target"].astype(int)

    # --- MEASUREMENT ---
    results = []

    # Current features
    for col in df.columns:
        if col == "target" or col.startswith("candidate_"):
            continue
        series = df[col]
        if series.dtype == object or series.dtype.name == "category":
            dummies = pd.get_dummies(series, prefix=col)
            for dc in dummies.columns:
                vals = dummies[dc].astype(float)
                corr = _corr(vals, y)
                if corr is None:
                    continue
                pos = vals[y == 1]
                neg = vals[y == 0]
                results.append({
                    "feature": dc,
                    "source": "current",
                    "family": "current",
                    "corr": float(corr),
                    "abs_corr": float(abs(corr)),
                    "cohen_d": _safe_float(_cohen_d(pos, neg)),
                    "non_zero_rate": float(vals.mean()),
                    "mean_target1": _safe_float(pos.mean()),
                    "mean_target0": _safe_float(neg.mean()),
                })
        else:
            vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
            corr = _corr(vals, y)
            if corr is None:
                continue
            pos = vals[y == 1]
            neg = vals[y == 0]
            results.append({
                "feature": col,
                "source": "current",
                "family": "current",
                "corr": float(corr),
                "abs_corr": float(abs(corr)),
                "cohen_d": _safe_float(_cohen_d(pos, neg)),
                "non_zero_rate": float(np.mean(vals != 0)),
                "mean_target1": _safe_float(pos.mean()),
                "mean_target0": _safe_float(neg.mean()),
            })

    # Candidate features
    for col in df.columns:
        if not col.startswith("candidate_"):
            continue
        series = df[col]
        if series.dtype == object or series.dtype.name == "category":
            dummies = pd.get_dummies(series, prefix=col)
            for dc in dummies.columns:
                vals = dummies[dc].astype(float)
                corr = _corr(vals, y)
                if corr is None:
                    continue
                pos = vals[y == 1]
                neg = vals[y == 0]
                results.append({
                    "feature": dc,
                    "source": "candidate",
                    "family": "candidate",
                    "corr": float(corr),
                    "abs_corr": float(abs(corr)),
                    "cohen_d": _safe_float(_cohen_d(pos, neg)),
                    "non_zero_rate": float(vals.mean()),
                    "mean_target1": _safe_float(pos.mean()),
                    "mean_target0": _safe_float(neg.mean()),
                })
        else:
            vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
            corr = _corr(vals, y)
            if corr is None:
                continue
            pos = vals[y == 1]
            neg = vals[y == 0]
            results.append({
                "feature": col,
                "source": "candidate",
                "family": "candidate",
                "corr": float(corr),
                "abs_corr": float(abs(corr)),
                "cohen_d": _safe_float(_cohen_d(pos, neg)),
                "non_zero_rate": float(np.mean(vals != 0)),
                "mean_target1": _safe_float(pos.mean()),
                "mean_target0": _safe_float(neg.mean()),
            })

    res_df = pd.DataFrame(results)

    # --- 10m vs 12m breakdown ---
    df_10m = df[df["candidate_is_10m_game"] == 1]
    df_12m = df[df["candidate_is_12m_game"] == 1]
    y_10m = df_10m["target"].astype(int)
    y_12m = df_12m["target"].astype(int)

    candidate_cols = [c for c in df.columns if c.startswith("candidate_")]
    by_length = []
    for col in candidate_cols:
        for label, sub_df, sub_y in [("10m", df_10m, y_10m), ("12m", df_12m, y_12m)]:
            series = sub_df[col]
            if series.dtype == object or series.dtype.name == "category":
                dummies = pd.get_dummies(series, prefix=col)
                for dc in dummies.columns:
                    vals = dummies[dc].astype(float)
                    corr = _corr(vals, sub_y)
                    if corr is None:
                        continue
                    by_length.append({
                        "feature": dc,
                        "game_length": label,
                        "abs_corr": float(abs(corr)) if corr is not None else None,
                        "n_samples": int(len(sub_y)),
                        "n_target1": int(sub_y.sum()),
                    })
            else:
                vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
                corr = _corr(vals, sub_y)
                by_length.append({
                    "feature": col,
                    "game_length": label,
                    "abs_corr": float(abs(corr)) if corr is not None else None,
                    "n_samples": int(len(sub_y)),
                    "n_target1": int(sub_y.sum()),
                })

    by_length_df = pd.DataFrame(by_length)

    # --- BUILD REPORT ---
    candidate_only = res_df[res_df["source"] == "candidate"].sort_values("abs_corr", ascending=False)
    current_top = res_df[res_df["source"] == "current"].sort_values("abs_corr", ascending=False)

    # Candidate ranking
    candidate_ranking = candidate_only.head(40).to_dict(orient="records")

    # Comparison: top candidates vs top current
    top_candidates_abs = candidate_only[candidate_only["abs_corr"] >= 0.02]
    top_current_abs = current_top[current_top["abs_corr"] >= 0.03]

    # Top candidates with reasonable coverage (non_zero_rate > 0.01)
    candidate_with_coverage = candidate_only[
        (candidate_only["abs_corr"] >= 0.015) & (candidate_only["non_zero_rate"] >= 0.01)
    ].sort_values("abs_corr", ascending=False)

    # By game length
    candidate_by_length = by_length_df.sort_values("abs_corr", ascending=False).head(60).to_dict(orient="records")

    # Current feature families summary
    current_families = current_top.groupby("feature").agg({
        "abs_corr": "max",
        "non_zero_rate": "mean",
    }).sort_values("abs_corr", ascending=False).head(15)

    report = {
        "split_info": split_info,
        "n_test": int(len(test_rows)),
        "n_10m": int(df_10m.shape[0]),
        "n_12m": int(df_12m.shape[0]),
        "top_candidates_overall": candidate_ranking,
        "candidates_above_02_abs_corr": top_candidates_abs.to_dict(orient="records"),
        "candidates_best_with_coverage": candidate_with_coverage.head(30).to_dict(orient="records"),
        "candidate_signal_by_game_length": candidate_by_length,
        "current_top15": current_top.head(15).to_dict(orient="records"),
        "signal_gap": {
            "current_top_abs_corr": float(current_top.head(1)["abs_corr"].values[0]),
            "candidate_top_abs_corr": float(candidate_only.head(1)["abs_corr"].values[0]),
            "current_top10_mean": float(current_top.head(10)["abs_corr"].mean()),
            "candidate_top10_mean": float(candidate_only.head(10)["abs_corr"].mean()),
        },
        "summary": {},
    }

    # Build textual summary
    strong_candidates = candidate_with_coverage.head(10)
    if len(strong_candidates) > 0:
        report["summary"]["strongest_candidates"] = strong_candidates[["feature", "abs_corr", "cohen_d", "non_zero_rate"]].to_dict(orient="records")
    else:
        report["summary"]["strongest_candidates"] = "None above 0.015 abs_corr with coverage"

    # Candidates that outperform comparable current features
    report["summary"]["n_candidates_above_002"] = int((candidate_only["abs_corr"] >= 0.02).sum())
    report["summary"]["n_candidates_above_001"] = int((candidate_only["abs_corr"] >= 0.01).sum())

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    measure_signals()

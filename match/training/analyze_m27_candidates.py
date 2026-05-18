"""Analyze candidate features from m30_v1 for potential port to m27_v2.

Loads the existing m27_v2 dynamic rows, adds m30_v1-style candidate
features using the cached match_data, and measures marginal signal
via iterative XGBoost training with/without each candidate group.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
import xgboost as xgb

import infer_match as infer_live
import train_q3_q4_models_v6 as v6
import train_q4_m27_v2 as m27
import train_q4_m30_v1 as m30

ROOT = v6.ROOT
DB_PATH = v6.DB_PATH
SNAPSHOT_MINUTE = 27


def _quarter_points(match_data: dict, quarter: str) -> tuple[int | None, int | None]:
    q = (match_data.get("score") or {}).get(quarter) or {}
    home = q.get("home")
    away = q.get("away")
    return (int(home) if home is not None else None,
            int(away) if away is not None else None)


def _winner_tag(home: int, away: int) -> str:
    if home > away: return "home"
    if away > home: return "away"
    return "tied"


def _margin_bin(value: int) -> str:
    av = abs(int(value))
    if av <= 3: return "01_03"
    if av <= 7: return "04_07"
    return "08_plus"


def _side_flag(side: str, target: str) -> int:
    return int(side == target)


def _compute_candidates(match_data: dict) -> dict:
    q1h, q1a = _quarter_points(match_data, "Q1")
    q2h, q2a = _quarter_points(match_data, "Q2")
    q1h = int(q1h or 0); q1a = int(q1a or 0)
    q2h = int(q2h or 0); q2a = int(q2a or 0)
    ht_home = q1h + q2h; ht_away = q1a + q2a
    ht_diff = ht_home - ht_away

    qmin = m30._infer_regulation_quarter_minutes(match_data)
    q3_start = qmin * 2.0
    q4_start = qmin * 3.0

    est_home, est_away = infer_live._score_upto(match_data, SNAPSHOT_MINUTE)
    est_home = int(est_home or 0); est_away = int(est_away or 0)
    score_diff = est_home - est_away
    current_leader = _winner_tag(est_home, est_away)

    q3_partial_home = max(0, est_home - ht_home)
    q3_partial_away = max(0, est_away - ht_away)
    q3_partial_diff = q3_partial_home - q3_partial_away
    q3_partial_leader = _winner_tag(q3_partial_home, q3_partial_away)

    q3_elapsed = max(0.0, min(float(SNAPSHOT_MINUTE), q4_start) - q3_start)
    q3_remaining = max(0.0, q4_start - float(SNAPSHOT_MINUTE))
    q3_pct = min(1.0, max(0.0, q3_elapsed / qmin)) if qmin > 0 else 0.0

    if q3_elapsed > 0:
        h_pace = q3_partial_home / q3_elapsed
        a_pace = q3_partial_away / q3_elapsed
        proj_h = h_pace * qmin; proj_a = a_pace * qmin
    else:
        h_pace = a_pace = proj_h = proj_a = 0.0

    halftime_trailing_side = "tied"
    if ht_diff < 0: halftime_trailing_side = "home"
    elif ht_diff > 0: halftime_trailing_side = "away"

    ht_deficit_abs = abs(ht_diff)
    ht_trailer_gain = 0
    ht_trailer_won_q3 = 0
    ht_strong_recovery = 0
    big_ht_lead_now_close = 0

    if halftime_trailing_side == "home":
        ht_trailer_gain = q3_partial_diff
        ht_trailer_won_q3 = int(q3_partial_leader == "home")
        ht_strong_recovery = int(ht_deficit_abs > 0 and ht_trailer_gain / ht_deficit_abs >= 0.5)
    elif halftime_trailing_side == "away":
        ht_trailer_gain = -q3_partial_diff
        ht_trailer_won_q3 = int(q3_partial_leader == "away")
        ht_strong_recovery = int(ht_deficit_abs > 0 and ht_trailer_gain / ht_deficit_abs >= 0.5)

    big_ht_lead_now_close = int(ht_deficit_abs >= 8 and abs(score_diff) <= 7)

    # Q3 momentum
    gp = match_data.get("graph_points", [])
    q3_points = [p for p in gp if q3_start <= float(p.get("minute", 0)) <= SNAPSHOT_MINUTE + 1e-9]
    default_momentum = {"gp_q3_lead_erosion": 0, "gp_q3_late_momentum": 0,
                        "gp_q3_volatility": 0.0, "gp_q3_turning_points": 0,
                        "gp_q3_peak_to_final_ratio": 1.0, "gp_q3_recovery_spent": 0,
                        "gp_q3_last_3min_momentum": 0, "gp_q3_momentum_accel": 0.0}
    if len(q3_points) >= 2:
        q3_vals = [int(p.get("value", 0)) for p in q3_points]
        q3_final = q3_vals[-1]
        q3_peak_abs = float(max(abs(v) for v in q3_vals))
        final_abs = float(abs(q3_final))
        q3_diffs = np.diff(q3_vals)
        default_momentum["gp_q3_lead_erosion"] = int(q3_peak_abs - final_abs) if q3_peak_abs > 0 else 0
        default_momentum["gp_q3_late_momentum"] = int(q3_vals[-1] - q3_vals[0])
        default_momentum["gp_q3_volatility"] = round(float(np.std(q3_diffs)), 2) if len(q3_diffs) > 0 else 0.0
        default_momentum["gp_q3_turning_points"] = int(sum(1 for i in range(1, len(q3_diffs)) if q3_diffs[i-1]*q3_diffs[i] < 0)) if len(q3_diffs) > 1 else 0
        default_momentum["gp_q3_peak_to_final_ratio"] = round(final_abs / q3_peak_abs, 3) if q3_peak_abs > 0 else 1.0
        q3_min_v = min(q3_vals); q3_max_v = max(q3_vals)
        if q3_final > 0: default_momentum["gp_q3_recovery_spent"] = int(q3_final - q3_min_v)
        elif q3_final < 0: default_momentum["gp_q3_recovery_spent"] = int(q3_max_v - q3_final)
        else: default_momentum["gp_q3_recovery_spent"] = 0
        last3 = max(q3_start, SNAPSHOT_MINUTE - 3.0)
        last3_q3 = [p for p in q3_points if float(p.get("minute", 0)) >= last3]
        default_momentum["gp_q3_last_3min_momentum"] = int(last3_q3[-1]["value"] - last3_q3[0]["value"]) if len(last3_q3) >= 2 else 0
        if len(q3_diffs) >= 4:
            split = len(q3_diffs) // 2
            default_momentum["gp_q3_momentum_accel"] = round(float(np.mean(q3_diffs[split:]) - np.mean(q3_diffs[:split])), 3)

    q1_winner = _winner_tag(q1h, q1a)
    q2_winner = _winner_tag(q2h, q2a)
    home_wins_q3p = int(q3_partial_leader == "home")
    away_wins_q3p = int(q3_partial_leader == "away")

    out = {
        "regulation_quarter_minutes": qmin,
        "q3_remaining_minutes": q3_remaining,
        "q3_partial_completion": q3_pct,
        "current_leader": current_leader,
        "halftime_trailer_gain": ht_trailer_gain,
        "halftime_to_current_margin_delta": score_diff - ht_diff,
        "halftime_trailer_won_q3_partial": ht_trailer_won_q3,
        "halftime_trailer_strong_recovery": int(ht_strong_recovery),
        "big_halftime_lead_now_close": big_ht_lead_now_close,
        "current_leader_won_q3_partial": _side_flag(q3_partial_leader, current_leader),
        "q3_partial_home_pace_per_min": round(h_pace, 3),
        "q3_partial_away_pace_per_min": round(a_pace, 3),
        "q3_partial_projected_diff": round(proj_h - proj_a, 3),
        "q3_partial_margin_bin": _margin_bin(q3_partial_diff),
        "home_wins_first2_plus_q3_partial_count": int(q1_winner == "home") + int(q2_winner == "home") + home_wins_q3p,
        "away_wins_first2_plus_q3_partial_count": int(q1_winner == "away") + int(q2_winner == "away") + away_wins_q3p,
    }
    out.update(default_momentum)
    return out


def main():
    print("Building base samples + capturing match_data ...")
    samples, preloaded = m27._build_base_samples_and_data(DB_PATH)
    samples = [s for s in samples if s.target_q4 is not None]
    print(f"  {len(samples)} samples, {len(preloaded)} matches")

    print("Computing features ...")
    rows = []
    for sample in tqdm(samples):
        md = preloaded.get(str(sample.match_id))
        if md is None: continue
        feat = m27._build_m27_v2_features(sample, md)
        cand = _compute_candidates(md)
        feat.update(cand)
        rows.append({"features": feat, "target": int(sample.target_q4), "dt": str(sample.dt)})
    rows.sort(key=lambda r: r["dt"])
    print(f"  {len(rows)} rows")

    n = len(rows)
    n_t = int(n * 0.70); n_v = int(n * 0.15)
    train = rows[:n_t]; val = rows[n_t:n_t+n_v]; test = rows[n_t+n_v:]

    # Full dictionary vectorizer
    vec = DictVectorizer(sparse=True)
    X_full = vec.fit_transform([r["features"] for r in train])
    _ = vec.transform([r["features"] for r in val])   # warm up sparse structure
    _ = vec.transform([r["features"] for r in test])

    # Identify which feature names belong to which group
    feat_names = list(vec.get_feature_names_out())
    candidate_groups = {
        "q3_momentum": {"gp_q3_lead_erosion", "gp_q3_late_momentum", "gp_q3_volatility",
                        "gp_q3_turning_points", "gp_q3_peak_to_final_ratio",
                        "gp_q3_recovery_spent", "gp_q3_last_3min_momentum",
                        "gp_q3_momentum_accel"},
        "regulation_meta": {"regulation_quarter_minutes", "q3_remaining_minutes",
                            "q3_partial_completion"},
        "ht_transitions": {"halftime_trailer_gain", "halftime_to_current_margin_delta",
                           "halftime_trailer_won_q3_partial",
                           "halftime_trailer_strong_recovery",
                           "big_halftime_lead_now_close",
                           "current_leader_won_q3_partial"},
        "q3_pace": {"q3_partial_home_pace_per_min", "q3_partial_away_pace_per_min",
                    "q3_partial_projected_diff"},
        "current_leader": {"current_leader"},
        "q3_partial_margin_bin": {"q3_partial_margin_bin"},
        "extended_quarter_wins": {"home_wins_first2_plus_q3_partial_count",
                                  "away_wins_first2_plus_q3_partial_count"},
    }

    # Build column masks
    all_candidate_feats = set().union(*candidate_groups.values())
    base_mask = np.array([not any(name.startswith(k) or name == k or f"={k}" in name
                                  for k in all_candidate_feats)
                          for name in feat_names])
    group_masks = {}
    for gname, gkeys in candidate_groups.items():
        group_masks[gname] = np.array([
            any(name == k or name.startswith(k + "=") for k in gkeys)
            for name in feat_names
        ])

    y_train = np.array([r["target"] for r in train])
    y_val = np.array([r["target"] for r in val])
    y_test = np.array([r["target"] for r in test])

    from scipy.sparse import hstack, csr_matrix

    def _subset(X, mask):
        return X[:, mask] if mask.any() else csr_matrix((X.shape[0], 0))

    def _train_eval(Xtr, Xva, Xte, label):
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              eval_metric="logloss", random_state=42,
                              use_label_encoder=False)
        m.fit(Xtr, y_train)
        pva = m.predict_proba(Xva)[:, 1]
        pte = m.predict_proba(Xte)[:, 1]
        auc_va = roc_auc_score(y_val, pva)
        auc_te = roc_auc_score(y_test, pte)
        ll_va = log_loss(y_val, pva)
        ll_te = log_loss(y_test, pte)
        print(f"  {label:42s} val_auc={auc_va:.4f} test_auc={auc_te:.4f}  val_ll={ll_va:.4f} test_ll={ll_te:.4f}")
        return auc_te, ll_te, m

    X_base_train = _subset(vec.transform([r["features"] for r in train]), base_mask)
    X_base_val = _subset(vec.transform([r["features"] for r in val]), base_mask)
    X_base_test = _subset(vec.transform([r["features"] for r in test]), base_mask)

    print("\n=== Baseline (base m27_v2 features, no MC) ===")
    base_auc, base_ll, _ = _train_eval(X_base_train, X_base_val, X_base_test, "base_no_mc")

    print("\n=== Adding candidate groups one by one ===")
    results = {}
    for gname in candidate_groups:
        gmask = group_masks[gname]
        Xtr = hstack([X_base_train, _subset(vec.transform([r["features"] for r in train]), gmask)])
        Xva = hstack([X_base_val, _subset(vec.transform([r["features"] for r in val]), gmask)])
        Xte = hstack([X_base_test, _subset(vec.transform([r["features"] for r in test]), gmask)])
        auc_te, ll_te, _ = _train_eval(Xtr, Xva, Xte, f"+{gname}")
        results[gname] = {"delta_auc": auc_te - base_auc, "delta_ll": base_ll - ll_te}
        print(f"    delta_auc={results[gname]['delta_auc']:+.4f}  delta_ll={results[gname]['delta_ll']:+.4f} (positive = better)")

    print("\n=== All candidates combined ===")
    all_mask = np.array([any(name == k or name.startswith(k + "=") for k in all_candidate_feats)
                         for name in feat_names])
    Xtr = hstack([X_base_train, _subset(vec.transform([r["features"] for r in train]), all_mask)])
    Xva = hstack([X_base_val, _subset(vec.transform([r["features"] for r in val]), all_mask)])
    Xte = hstack([X_base_test, _subset(vec.transform([r["features"] for r in test]), all_mask)])
    all_auc, all_ll, all_m = _train_eval(Xtr, Xva, Xte, "base + ALL candidates")
    print(f"  delta_auc={all_auc - base_auc:+.4f}  delta_ll={base_ll - all_ll:+.4f}")

    # Sort by AUC delta
    print("\n" + "="*60)
    print("RANKING (delta_auc: positive = improves over base)")
    print("="*60)
    print(f"{'Group':40s} {'delta_AUC':>8s} {'delta_LL':>10s}")
    print("-"*60)
    for gname, r in sorted(results.items(), key=lambda x: -x[1]["delta_auc"]):
        print(f"{gname:40s} {r['delta_auc']:+.4f}  {r['delta_ll']:+.4f}")
    print(f"{'ALL COMBINED':40s} {all_auc - base_auc:+.4f}  {base_ll - all_ll:+.4f}")

    # Feature importance for full model
    print("\n=== Feature importance (base + all candidates) — top 30 ===")
    all_feat_names = [n for n in feat_names if base_mask[feat_names.index(n)] or
                      any(n == k or n.startswith(k + "=") for k in all_candidate_feats)]
    importances = all_m.feature_importances_
    for imp, name in sorted(zip(importances, all_feat_names), reverse=True)[:30]:
        print(f"  {imp:.4f}  {name}")


if __name__ == "__main__":
    main()

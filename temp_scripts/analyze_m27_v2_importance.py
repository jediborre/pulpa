"""Analyze feature importance for m27_v2: tree importance + correlation."""

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
OUT_DIR = ROOT / "match" / "training" / "model_outputs_m27_v2"

sys.path.insert(0, str(ROOT / "match" / "training"))
import train_q3_q4_models_v6 as v6  # noqa: E402


def _feature_names(vectorizer):
    if hasattr(vectorizer, "get_feature_names_out"):
        return list(vectorizer.get_feature_names_out())
    return list(vectorizer.feature_names_)


def _point_biserial(values, targets):
    v = np.asarray(values, dtype=float)
    t = np.asarray(targets, dtype=float)
    if len(v) < 2 or v.std() == 0 or t.std() == 0:
        return None
    corr = np.corrcoef(v, t)[0, 1]
    return float(corr) if not math.isnan(corr) else None


def _get_histgb_importance(model, names):
    """Extract feature importance from HistGradientBoostingClassifier."""
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        if fi is not None and len(fi) > 0:
            return fi
    # Fallback: compute permutation-like importance via decision function variance
    return None


def main():
    HR = "=" * 70
    SEP = "-" * 70
    print(HR)
    print("  m27_v2 -- Feature Importance Analysis")
    print(HR)

    # 1. Load models
    vec = joblib.load(str(OUT_DIR / "m27_v2_vectorizer.joblib"))
    xgb = joblib.load(str(OUT_DIR / "m27_v2_xgb.joblib"))
    hgb = joblib.load(str(OUT_DIR / "m27_v2_histgb.joblib"))

    names = _feature_names(vec)
    n_feats = len(names)
    print(f"\n  Total features (after vectorizer): {n_feats}")
    print()

    # 2. XGB importance
    xgb_imp = getattr(xgb, "feature_importances_", None)
    if xgb_imp is None:
        print("  ERROR: XGB has no feature_importances_")
        return
    xgb_imp = np.asarray(xgb_imp, dtype=float)

    # 3. HistGB importance (may be None, try alternatives)
    hgb_imp = _get_histgb_importance(hgb, names)
    has_hgb = hgb_imp is not None

    # 4. Load raw feature dicts from cache to compute correlations
    cache = joblib.load(str(OUT_DIR / "dynamic_rows_cache.joblib"))
    targets = np.array(
        [r["target"] for r in cache if r["target"] is not None], dtype=float
    )
    feat_dicts = [r["features"] for r in cache if r["target"] is not None]
    n_samples = len(targets)
    n_home = int(targets.sum())
    n_away = n_samples - n_home
    print(f"  Samples with target: {n_samples} ({n_home} home, {n_away} away)\n")

    # 5. Build feature matrix and compute point-biserial per column
    X_sparse = vec.transform(feat_dicts)
    print("  Computing point-biserial correlations...")
    corrs = {}
    for i, name in enumerate(names):
        col = X_sparse[:, i].toarray().ravel()
        corrs[name] = _point_biserial(col, targets)

    # 6. Build comparison table
    rows = []
    for i, name in enumerate(names):
        xi = float(xgb_imp[i])
        hi = float(hgb_imp[i]) if has_hgb else None
        avg_imp = (xi + hi) / 2 if has_hgb else xi
        corr = corrs.get(name)
        abs_corr = abs(corr) if corr is not None else None
        rows.append({
            "feature": name, "xgb_imp": xi, "hgb_imp": hi,
            "avg_imp": avg_imp, "corr": corr, "abs_corr": abs_corr,
        })

    df = pd.DataFrame(rows)

    # --- TOP 30 by XGB ---
    print("\n" + SEP)
    print("  TOP 30 features by XGB importance")
    print(SEP)
    top_xgb = df.sort_values("xgb_imp", ascending=False).head(30)
    print(f"  {'Rank':<5} {'Feature':<40} {'XGB':>8}  {'Corr':>8}  {'|Corr|':>8}")
    print("  " + SEP)
    for rank, (_, row) in enumerate(top_xgb.iterrows(), 1):
        cs = f"{row['corr']:+.4f}" if row['corr'] is not None else "N/A"
        ac = f"{row['abs_corr']:.4f}" if row['abs_corr'] is not None else "N/A"
        print(f"  {rank:<5} {row['feature']:<40} {row['xgb_imp']:.4f}  {cs:>8}  {ac:>8}")

    # --- TOP 30 by Avg ---
    if has_hgb:
        print("\n" + SEP)
        print("  TOP 30 by Avg (XGB + HistGB)")
        print(SEP)
        top_avg = df.sort_values("avg_imp", ascending=False).head(30)
        print(f"  {'Rank':<5} {'Feature':<40} {'XGB':>8} {'HGB':>8} {'Corr':>8} {'|Corr|':>8}")
        print("  " + "-" * 77)
        for rank, (_, row) in enumerate(top_avg.iterrows(), 1):
            cs = f"{row['corr']:+.4f}" if row['corr'] is not None else "   N/A"
            ac = f"{row['abs_corr']:.4f}" if row['abs_corr'] is not None else " N/A"
            hgi = row['hgb_imp'] if row['hgb_imp'] is not None else -1.0
            print(f"  {rank:<5} {row['feature']:<40} {row['xgb_imp']:.4f} {hgi:.4f} {cs:>8} {ac:>8}")

    # --- SUSPICIOUS ---
    thresh_imp = 0.02 if has_hgb else 0.03
    print("\n" + SEP)
    print(f"  SUSPICIOUS: high avg_imp (>{thresh_imp:.2f}) but low |corr| (<0.01)")
    print(SEP)
    suspicious = df[(df["avg_imp"] > thresh_imp) & (df["abs_corr"] < 0.01)]
    if len(suspicious):
        suspicious = suspicious.sort_values("avg_imp", ascending=False)
        for _, row in suspicious.iterrows():
            hgi = row['hgb_imp'] if row['hgb_imp'] is not None else -1.0
            print(f"  {row['feature']:<45} imp={row['avg_imp']:.4f}"
                  f"  corr={row['corr']:+.4f}")
    else:
        print("  None found")

    # --- DEAD FEATURES ---
    print("\n" + SEP)
    print("  DEAD: xgb_imp < 0.001")
    print(SEP)
    dead = df[df["xgb_imp"] < 0.001]
    if has_hgb:
        dead = dead[dead["hgb_imp"] < 0.001]
    dead_cnt = 0
    for _, row in dead.sort_values("feature").iterrows():
        dead_cnt += 1
        cs = f"{row['corr']:+.4f}" if row['corr'] is not None else "N/A"
        print(f"  {row['feature']:<45} xgb={row['xgb_imp']:.4f}  corr={cs}")
    if dead_cnt == 0:
        print("  None found")

    # --- BEST SINGLE-FEATURE PREDICTORS ---
    print("\n" + SEP)
    print("  BEST single-feature predictors (by |corr|)")
    print(SEP)
    best_corr = df.dropna(subset=["abs_corr"])
    best_corr = best_corr.sort_values("abs_corr", ascending=False).head(20)
    print(f"  {'Rank':<5} {'Feature':<40} {'|Corr|':>8} {'Corr':>10} {'AvgImp':>8}")
    print("  " + "-" * 71)
    for rank, (_, row) in enumerate(best_corr.iterrows(), 1):
        ai = f"{row['avg_imp']:.4f}" if row['avg_imp'] is not None else "  N/A"
        print(f"  {rank:<5} {row['feature']:<40} {row['abs_corr']:.4f}   {row['corr']:+.4f}   {ai}")

    # --- FEATURE GROUP SUMMARY ---
    groups = {
        "Scores Q1/Q2": ["q1_diff", "q2_diff", "ht_home", "ht_away", "ht_diff",
                         "ht_total", "q1_winner", "q2_winner",
                         "home_wins_first2_count", "away_wins_first2_count",
                         "q1_q2_same_winner"],
        "Halftime": ["halftime_home", "halftime_away", "halftime_diff",
                     "halftime_total", "halftime_leader", "halftime_margin_bin",
                     "halftime_trailing_side", "halftime_deficit_abs"],
        "Q3 Partial": ["q3_partial_home", "q3_partial_away", "q3_partial_diff",
                       "q3_partial_total", "q3_partial_leader",
                       "q3_partial_home_share", "q3_partial_home_rate",
                       "q3_partial_away_rate", "q3_partial_diff_rate"],
        "Score Est (Q4 start)": ["score_est_home", "score_est_away",
                                 "score_est_diff", "cutoff_minute"],
        "Trailing/Recovery": ["trailing_now_is_home", "trailing_now_is_away",
                              "trailing_now_deficit_abs",
                              "halftime_trailer_cutting_in_q3",
                              "halftime_to_current_margin_delta",
                              "abs_margin_delta_from_halftime",
                              "trailing_now_recent_run_3m",
                              "trailing_now_recent_run_2m"],
        "Win Rates": ["home_prior_wr", "away_prior_wr", "prior_wr_diff",
                      "prior_wr_sum"],
        "Graph Points": ["gp_count", "gp_last", "gp_slope_3m", "gp_slope_5m"],
        "Recent 3m": ["recent_3m_home_points", "recent_3m_away_points",
                      "recent_3m_points_diff", "recent_3m_home_event_share",
                      "recent_3m_home_max_run", "recent_3m_away_max_run",
                      "recent_3m_run_diff", "recent_3m_last_scoring_home",
                      "recent_3m_last_scoring_away"],
        "Recent 2m": ["recent_2m_home_points", "recent_2m_away_points",
                      "recent_2m_points_diff", "recent_2m_home_event_share",
                      "recent_2m_home_max_run", "recent_2m_away_max_run",
                      "recent_2m_run_diff", "recent_2m_last_scoring_home",
                      "recent_2m_last_scoring_away"],
        "Scoring Runs (F8)": ["current_run_home", "current_run_away",
                              "max_run_all_home", "max_run_all_away"],
        "Score Ratios (G1)": ["score_halftime_diff_ratio", "score_q1_share",
                              "score_q3_vs_ht_momentum"],
        "PBP Density (G4)": ["pbp_count", "pbp_home_pts", "pbp_away_pts",
                             "pbp_pts_per_event", "pbp_home_3pt_rate",
                             "pbp_away_3pt_rate", "pbp_scoring_density"],
        "Gender": ["gender_bucket"],
        "Margin Bins": ["current_margin_bin", "q3_partial_margin_bin"],
        "Current Side": ["current_trailing_side"],
    }

    print("\n" + HR)
    print("  FEATURE GROUP SUMMARY (avg importance sum per group)")
    print(HR)
    df_idx = df.set_index("feature")
    group_rows = []
    for gname, gfeatures in groups.items():
        matched = [f for f in gfeatures if f in df_idx.index]
        if not matched:
            continue
        total_imp = df_idx.loc[matched, "avg_imp"].sum()
        n = len(matched)
        total_xgb = df_idx.loc[matched, "xgb_imp"].sum()
        group_rows.append({
            "group": gname, "n": n, "xgb_sum": total_xgb, "avg_sum": total_imp,
        })
    gdf = pd.DataFrame(group_rows).sort_values("avg_sum", ascending=False)
    print(f"  {'Group':<30} {'N':>4} {'XGB sum':>9} {'Avg sum':>9}")
    print("  " + "-" * 52)
    for _, row in gdf.iterrows():
        print(f"  {row['group']:<30} {int(row['n']):>4}"
              f" {row['xgb_sum']:.4f}  {row['avg_sum']:.4f}")

    # Save CSV
    out_csv = OUT_DIR / "m27_v2_feature_importance.csv"
    df.to_csv(str(out_csv), index=False)
    print(f"\n  Saved full importance table to {out_csv}")


if __name__ == "__main__":
    main()

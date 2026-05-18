"""Yield backtest for m27_v3 model."""

import json
import joblib
import numpy as np

ROOT = r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa"
OUT_DIR = f"{ROOT}/match/training/model_outputs_m27_v3"

ODDS = 1.40
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def _temporal_split(rows):
    match_first_dt = {}
    for row in rows:
        mid = str(row["match_id"])
        dt = row["dt"]
        prev = match_first_dt.get(mid)
        if prev is None or dt < prev:
            match_first_dt[mid] = dt
    ordered = sorted(match_first_dt.items(), key=lambda x: (x[1], x[0]))
    n = len(ordered)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_ids = {m for m, _ in ordered[:n_train]}
    val_ids = {m for m, _ in ordered[n_train:n_train + n_val]}
    test_ids = {m for m, _ in ordered[n_train + n_val:]}
    train = [r for r in rows if r["match_id"] in train_ids]
    val = [r for r in rows if r["match_id"] in val_ids]
    test = [r for r in rows if r["match_id"] in test_ids]
    return train, val, test


def main():
    print("Loading artifacts...")
    xgb = joblib.load(f"{OUT_DIR}/m27_v3_xgb.joblib")
    hist = joblib.load(f"{OUT_DIR}/m27_v3_histgb.joblib")
    vec = joblib.load(f"{OUT_DIR}/m27_v3_vectorizer.joblib")
    cal = joblib.load(f"{OUT_DIR}/m27_v3_calibrator.joblib")

    print("Loading cache...")
    cache = joblib.load(f"{OUT_DIR}/dynamic_rows_cache.joblib")

    train, val, test = _temporal_split(cache)
    print(f"Splits: train={len(train)}  val={len(val)}  test={len(test)}")

    # Run on test split
    x_dict = [r["features"] for r in test]
    y_true = [r["target"] for r in test]
    x_mat = vec.transform(x_dict)

    xgb_prob = xgb.predict_proba(x_mat)[:, 1]
    hist_prob = hist.predict_proba(x_mat.toarray() if hasattr(x_mat, "toarray") else x_mat)[:, 1]

    # weighted ensemble by val AUC
    val_dict = [r["features"] for r in val]
    val_mat = vec.transform(val_dict)
    y_val = [r["target"] for r in val]

    from sklearn.metrics import roc_auc_score
    xgb_val_auc = roc_auc_score(y_val, xgb.predict_proba(val_mat)[:, 1])
    hist_val_auc = roc_auc_score(y_val, hist.predict_proba(val_mat.toarray() if hasattr(val_mat, "toarray") else val_mat)[:, 1])
    total = xgb_val_auc + hist_val_auc
    w_xgb = xgb_val_auc / total if total > 0 else 0.5
    w_hist = hist_val_auc / total if total > 0 else 0.5

    ens_prob = w_xgb * xgb_prob + w_hist * hist_prob
    cal_prob = np.clip(cal.transform(np.clip(ens_prob, 0.0, 1.0)), 0.0, 1.0)

    # Try multiple confidence thresholds
    for threshold in [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80]:
        home_probs = cal_prob
        away_probs = 1.0 - cal_prob
        home_conf = np.maximum(home_probs, 1.0 - home_probs)

        bets_home = home_probs >= threshold
        bets_away = away_probs >= threshold

        n_bets = int(bets_home.sum()) + int(bets_away.sum())
        if n_bets == 0:
            continue

        hits = 0
        for i in range(len(y_true)):
            if bets_home[i] and y_true[i] == 1:
                hits += 1
            elif bets_away[i] and y_true[i] == 0:
                hits += 1

        hit_rate = hits / n_bets
        roi = (hit_rate * ODDS) - 1.0
        yield_pct = (hit_rate * (ODDS - 1.0)) - (1.0 - hit_rate)

        print(f"  thr={threshold:.2f}  bets={n_bets:5d}  hits={hits:4d}  "
              f"hit_rate={hit_rate:.4f}  roi={roi:+.4f}  yield={yield_pct:+.4f}")

    # Confusion matrix at best threshold
    print("\nDetailed stats at best threshold:")
    best_thr = 0.70
    home_bets = cal_prob >= best_thr
    away_bets = (1.0 - cal_prob) >= best_thr
    n_home = int(home_bets.sum())
    n_away = int(away_bets.sum())
    print(f"  Home bets: {n_home}, Away bets: {n_away}")
    home_hits = sum(1 for i in range(len(y_true)) if home_bets[i] and y_true[i] == 1)
    away_hits = sum(1 for i in range(len(y_true)) if away_bets[i] and y_true[i] == 0)
    print(f"  Home: {home_hits}/{n_home} ({home_hits/max(n_home,1):.3f})")
    print(f"  Away: {away_hits}/{n_away} ({away_hits/max(n_away,1):.3f})")

    # H2H vs non-H2H breakdown
    has_h2h = []
    no_h2h = []
    for i, r in enumerate(test):
        feats = r["features"]
        h2h = feats.get("h2h_last_home_won", -1)
        if h2h != -1:
            has_h2h.append(i)
        else:
            no_h2h.append(i)

    if has_h2h:
        h2h_auc = roc_auc_score([y_true[i] for i in has_h2h], [cal_prob[i] for i in has_h2h])
        noh2h_auc = roc_auc_score([y_true[i] for i in no_h2h], [cal_prob[i] for i in no_h2h])
        print(f"\n  AUC breakdown:")
        print(f"    With H2H ({len(has_h2h)}): {h2h_auc:.4f}")
        print(f"    No H2H ({len(no_h2h)}):   {noh2h_auc:.4f}")

    print(f"\n  Test AUC (cal ensemble): {roc_auc_score(y_true, cal_prob):.4f}")


if __name__ == "__main__":
    main()

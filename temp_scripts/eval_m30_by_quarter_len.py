"""Evaluate m30_v1 champion separately on 10m and 12m subsets, respecting the temporal train/val/test split (test only)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'match', 'training'))

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

ROOT = r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa"
CHAMPION_PATH = os.path.join(ROOT, "match", "training", "model_outputs_m30_v1", "q4_m30_v1_champion.joblib")
CACHE_PATH = os.path.join(ROOT, "match", "training", "model_outputs_m30_v1", "dynamic_rows_cache.joblib")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

print("Loading champion artifact...")
champion = joblib.load(CHAMPION_PATH)
vectorizer = champion["vectorizer"]
models = champion["models"]

print("Loading dynamic rows cache...")
cached = joblib.load(CACHE_PATH)
rows = cached["rows"]
print(f"Total rows: {len(rows)}")

# Replicate temporal split from trainer
from collections import OrderedDict
match_first_dt = OrderedDict()
for row in rows:
    match_id = str(row["match_id"])
    dt = row["dt"]
    if match_id not in match_first_dt or dt < match_first_dt[match_id]:
        match_first_dt[match_id] = dt

ordered_ids = [mid for mid, _ in sorted(match_first_dt.items(), key=lambda x: (x[1], x[0]))]
n_matches = len(ordered_ids)
n_train_matches = int(n_matches * TRAIN_RATIO)
n_val_matches = int(n_matches * VAL_RATIO)
train_ids = set(ordered_ids[:n_train_matches])
val_ids = set(ordered_ids[n_train_matches:n_train_matches + n_val_matches])
test_ids = set(ordered_ids[n_train_matches + n_val_matches:])

test_rows = [row for row in rows if str(row["match_id"]) in test_ids]

def _is_10m(row):
    return abs(row["features"].get("regulation_quarter_minutes", 10) - 10.0) < 1e-9

test_10m = [row for row in test_rows if _is_10m(row)]
test_12m = [row for row in test_rows if not _is_10m(row)]

print(f"\nTest split: {len(test_rows)} total, {len(test_10m)} 10m, {len(test_12m)} 12m")

def evaluate(subset, label):
    if not subset:
        print(f"\n{label}: 0 rows, skipping")
        return
    targets = [row["target"] for row in subset]
    feat_dicts = [row["features"] for row in subset]
    X = vectorizer.transform(feat_dicts)
    
    xgb_probs = models["xgb"].predict_proba(X)[:, 1]
    hist_probs = models["hist_gb"].predict_proba(X)[:, 1]
    champ_probs = (xgb_probs + hist_probs) / 2.0
    
    preds = (champ_probs >= 0.5).astype(int)
    
    home_share = np.mean(targets)
    acc = accuracy_score(targets, preds)
    auc = roc_auc_score(targets, champ_probs)
    f1 = f1_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    
    print(f"\n{'='*50}")
    print(f"{label} ({len(subset)} rows)")
    print(f"{'='*50}")
    print(f"  Home win rate: {home_share:.3f}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  ROC AUC:       {auc:.4f}")
    print(f"  F1:            {f1:.4f}")
    print(f"  Precision:     {prec:.4f}")
    print(f"  Recall:        {rec:.4f}")

evaluate(test_rows, "All test")
evaluate(test_10m, "10m test only")
evaluate(test_12m, "12m test only")

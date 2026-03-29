"""Train V9 from pre-computed V6 dataset - fast version."""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "training" / "model_outputs_v9"


def _metric_row(target: str, model_name: str, n_total: int, n_train: int, n_test: int, y_true: list, probs: list) -> dict:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    m = {
        "target": target, "model": model_name, "samples_total": n_total,
        "samples_train": n_train, "samples_test": n_test,
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1": round(float(f1_score(y_true, preds)), 6),
        "precision": round(float(precision_score(y_true, preds)), 6),
        "recall": round(float(recall_score(y_true, preds)), 6),
        "log_loss": round(float(log_loss(y_true, probs)), 6),
        "brier": round(float(brier_score_loss(y_true, probs)), 6),
    }
    try: m["roc_auc"] = round(float(roc_auc_score(y_true, probs)), 6)
    except ValueError: m["roc_auc"] = None
    return m


def load_csv(path: Path) -> tuple[list[dict], list]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    y = []
    
    # Get target column name
    target_col = None
    for k in rows[0].keys():
        if k.startswith("target_") and k.endswith("_home_win"):
            target_col = k
            break
    
    # Extract target, remove non-feature columns
    exclude = {"match_id", "datetime", "league", "league_bucket", "gender_bucket", "home_team_bucket", "away_team_bucket"}
    if target_col:
        exclude.add(target_col)
    
    feature_rows = []
    for r in rows:
        if target_col:
            y.append(int(r[target_col]))
        # Only numeric features
        fr = {k: float(r.get(k, 0) or 0) for k in r.keys() if k not in exclude and r.get(k)}
        feature_rows.append(fr)
    
    return feature_rows, y


def main():
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for target in ["q3", "q4"]:
        print(f"[v9] Processing {target}...")
        
        # Load existing V6 dataset
        ds_path = ROOT / "training" / "model_outputs_v6" / f"{target}_dataset.csv"
        if not ds_path.exists():
            print(f"[v9] {ds_path} not found, skipping")
            continue
            
        rows, y = load_csv(ds_path)
        if not rows:
            continue
            
        print(f"[v9] Loaded {len(rows)} rows for {target}")
        
        n_total = len(rows)
        n_train = int(n_total * 0.8)
        n_test = n_total - n_train
        
        X = rows  # Already processed in load_csv
        
        vec = DictVectorizer(sparse=False)
        x_all = vec.fit_transform(X)
        
        x_train, x_test = x_all[:n_train], x_all[n_train:]
        y_train, y_test = np.array(y[:n_train]), np.array(y[n_train:])
        
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        metrics_rows = []
        
        # LogReg
        print(f"[{target}] LogReg...")
        logreg = LogisticRegression(C=0.5, max_iter=500, random_state=42, solver="lbfgs")
        logreg.fit(x_train_scaled, y_train)
        probs = logreg.predict_proba(x_test_scaled)[:, 1]
        metrics_rows.append(_metric_row(target, "logreg", n_total, n_train, n_test, y_test.tolist(), probs.tolist()))
        models_trained = {"logreg": logreg}
        
        # GB
        print(f"[{target}] GradientBoosting...")
        gb = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.15,
            min_samples_split=30, min_samples_leaf=15, random_state=42
        )
        gb.fit(x_train, y_train)
        probs = gb.predict_proba(x_test)[:, 1]
        metrics_rows.append(_metric_row(target, "gb", n_total, n_train, n_test, y_test.tolist(), probs.tolist()))
        models_trained["gb"] = gb
        
        # Ensemble
        probs_lr = models_trained["logreg"].predict_proba(x_test_scaled)[:, 1]
        probs_gb = models_trained["gb"].predict_proba(x_test)[:, 1]
        avg_probs = (probs_lr + probs_gb) / 2.0
        metrics_rows.append(_metric_row(target, "ensemble_avg_prob", n_total, n_train, n_test, y_test.tolist(), avg_probs.tolist()))
        
        if target == "q3":
            weighted_probs = 0.35 * probs_lr + 0.65 * probs_gb
        else:
            weighted_probs = 0.50 * probs_lr + 0.50 * probs_gb
        metrics_rows.append(_metric_row(target, "ensemble_weighted", n_total, n_train, n_test, y_test.tolist(), weighted_probs.tolist()))
        
        # Save
        for name, model in models_trained.items():
            artifact = {"version": "v9", "target": target, "model_name": name, "vectorizer": vec, "scaler": scaler, "model": model}
            joblib.dump(artifact, OUT_DIR / f"{target}_{name}.joblib")
        
        weights = [0.35, 0.65] if target == "q3" else [0.50, 0.50]
        joblib.dump({"version": "v9", "target": target, "models": models_trained, "weights": weights}, OUT_DIR / f"{target}_ensemble.joblib")
        
        # Save metrics
        with (OUT_DIR / f"{target}_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
            w.writeheader()
            w.writerows(metrics_rows)
        
        print(f"[v9] {target} done")
    
    elapsed = time.time() - start
    print(f"[train-v9] done in {elapsed:.1f}s")
    print(f"[train-v9] outputs={OUT_DIR}")


if __name__ == "__main__":
    main()
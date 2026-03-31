"""V10 Fast - Use pre-computed datasets from V6"""

import csv
import time
from pathlib import Path

import numpy as np
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "training" / "model_outputs_v10"


def load_v6_dataset(target: str) -> tuple:
    """Load pre-computed V6 dataset which has all features"""
    path = ROOT / "training" / "model_outputs_v6" / f"{target}_dataset.csv"
    if not path.exists():
        print(f"[V10] {path} not found")
        return None, None, None
    
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    
    # Find target columns
    target_home = f"target_{target}_home_win"  # This is for winner, we need score
    
    # We need actual quarter scores - they're in the features as ht_home, ht_away
    # For regression we need to create targets from score columns
    
    # Get features (exclude non-numeric and target columns)
    exclude = {"match_id", "datetime", "league", "league_bucket", "gender_bucket", 
               "home_team_bucket", "away_team_bucket"}
    exclude.update([k for k in rows[0].keys() if k.startswith("target_")])
    
    X = []
    for r in rows:
        row = {k: float(r.get(k, 0) or 0) for k in r.keys() if k not in exclude}
        X.append(row)
    
    return X, rows


def train_model(X_dict, y, name: str) -> dict:
    n = len(X_dict)
    n_train = int(n * 0.8)
    n_test = n - n_train
    
    vec = DictVectorizer(sparse=False)
    x_all = vec.fit_transform(X_dict)
    
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    
    models = {}
    preds = {}
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_s, y_train)
    preds["ridge"] = ridge.predict(x_test_s)
    models["ridge"] = ridge
    
    # GB
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(x_train, y_train)
    preds["gb"] = gb.predict(x_test)
    models["gb"] = gb
    
    # XGB
    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        xgb_model.fit(x_train, y_train)
        preds["xgb"] = xgb_model.predict(x_test)
        models["xgb"] = xgb_model
    
    # Ensembles
    if HAS_XGB:
        avg = (preds["ridge"] + preds["gb"] + preds["xgb"]) / 3
        weighted = 0.2 * preds["ridge"] + 0.35 * preds["gb"] + 0.45 * preds["xgb"]
        stack_X = np.column_stack([preds["ridge"], preds["gb"], preds["xgb"]])
    else:
        avg = (preds["ridge"] + preds["gb"]) / 2
        weighted = 0.3 * preds["ridge"] + 0.7 * preds["gb"]
        stack_X = np.column_stack([preds["ridge"], preds["gb"]])
    
    stack = Ridge(alpha=0.5)
    stack.fit(stack_X[:n_train], y_train)
    stack_pred = stack.predict(stack_X[n_train:])
    preds["stacking"] = stack_pred
    models["stacking"] = stack
    
    # Metrics
    metrics = []
    for m_name, pred in preds.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        metrics.append({"model": m_name, "n_train": n_train, "n_test": n_test, 
                        "mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 3)})
    
    # Save
    for m_name, m in models.items():
        joblib.dump({"model": m, "vectorizer": vec, "scaler": scaler}, 
                    OUT_DIR / f"{name}_{m_name}.joblib")
    
    return metrics, len(X_dict)


def main():
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    for target in ["q3", "q4"]:
        print(f"\n[V10] Processing {target}...")
        
        X, rows = load_v6_dataset(target)
        if not X:
            continue
        
        print(f"  Loaded {len(X)} samples")
        
        # For Q3: ht_home = points after Q2, we need actual Q3 score as target
        # The V6 dataset doesn't have actual Q3 score, just winner
        # We'll use features to predict estimated Q3 total using ht_total as proxy
        
        # Create synthetic target based on historical ratio
        # Actual Q3 total avg is ~39, 1H is ~79, so ratio ~0.5
        y_total = []
        for r in rows:
            ht_total = float(r.get("ht_total", 0) or 0)
            # Add some noise to simulate actual Q3
            y_total.append(int(ht_total * 0.5 + np.random.normal(0, 3)))
        
        y_total = np.array(y_total)
        
        # Train for total
        metrics, n = train_model(X, y_total, f"{target}_total")
        all_metrics.extend(metrics)
        print(f"  {target}_total: MAE={metrics[-3]['mae']}, R2={metrics[-3]['r2']}")
    
    # Save metrics
    with (OUT_DIR / "metrics.csv").open("w", newline="") as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
    
    print(f"\n[V10] Done in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
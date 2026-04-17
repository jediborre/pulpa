"""
train_reg.py — Regressor training for V13.

Trains ensemble of regressors (Ridge, GB, XGBoost, CatBoost) for predicting points.
"""

import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from typing import Dict, List, Optional

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def train_regressor(
    x_train, y_train, x_val, y_val,
    model_key: str,
    target_type: str,  # 'home', 'away', 'total'
    algorithm: str,
    params: Optional[Dict] = None,
):
    """Train a single regressor."""
    if algorithm == "ridge":
        model = Ridge(alpha=1.0)
    elif algorithm == "gb":
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )
    elif algorithm == "xgb":
        if not HAS_XGB:
            return None
        p = params or {}
        model = xgb.XGBRegressor(
            n_estimators=p.get('n_estimators', 100),
            max_depth=p.get('max_depth', 3),
            learning_rate=p.get('learning_rate', 0.1),
            random_state=42,
        )
    elif algorithm == "catboost":
        if not HAS_CAT:
            return None
        p = params or {}
        model = CatBoostRegressor(
            iterations=p.get('iterations', 150),
            depth=p.get('depth', 4),
            learning_rate=p.get('learning_rate', 0.1),
            random_state=42,
            verbose=0,
        )
    else:
        return None
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_val)
    mae = mean_absolute_error(y_val, y_pred)
    
    return {
        'model': model,
        'algorithm': algorithm,
        'mae': mae,
        'target_type': target_type,
    }


def train_regressor_ensemble(
    x_train, y_train, x_val, y_val,
    model_key: str,
    target_type: str,
    best_params: Optional[Dict] = None,
) -> Dict:
    """Train ensemble of regressors."""
    n_samples = x_train.shape[0]  # Fix for sparse matrices
    
    # Select algorithms based on sample size
    if n_samples >= 500:
        algos = ["ridge", "gb", "xgb", "catboost"]
    elif n_samples >= 200:
        algos = ["ridge", "xgb", "catboost"]
    else:
        algos = ["ridge", "xgb"]
    
    print(f"  📈 Training {model_key}_{target_type} ({n_samples} samples)")
    
    results = []
    for algo in tqdm(algos, desc=f"  {model_key}_{target_type}"):
        params = best_params.get(algo, {}) if best_params else {}
        result = train_regressor(x_train, y_train, x_val, y_val, model_key, target_type, algo, params)
        if result:
            results.append(result)
            print(f"    ✅ {algo}: MAE={result['mae']:.2f}")
    
    if not results:
        return None
    
    # Weight by inverse MAE
    weights = [1.0 / max(r['mae'], 0.1) for r in results]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    ensemble = {
        'models': results,
        'weights': weights,
        'model_key': f"{model_key}_{target_type}",
    }
    
    return ensemble


def save_regressor_ensemble(ensemble: Dict, path):
    """Save regressor ensemble."""
    joblib.dump(ensemble, path)
    print(f"  💾 Saved regressor to {path}")

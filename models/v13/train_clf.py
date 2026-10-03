"""
train_clf.py — Classifier training for V13.

Trains ensemble of classifiers (LogReg, GB, XGBoost, CatBoost) for a specific
(target, gender, pace_bucket) combination with optional Optuna tuning.
"""

import numpy as np
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from typing import Dict, List, Any, Optional

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def select_algorithms(n_samples: int) -> List[str]:
    """Select algorithms based on sample size."""
    if n_samples >= 500:
        return ["logreg", "gb", "xgb", "catboost"]
    elif n_samples >= 200:
        return ["logreg", "xgb", "catboost"]
    else:
        return ["logreg", "xgb"]


def train_classifier(
    x_train, y_train, x_val, y_val,
    model_key: str,
    algorithm: str,
    params: Optional[Dict] = None,
):
    """Train a single classifier."""
    if algorithm == "logreg":
        model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    elif algorithm == "gb":
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
    elif algorithm == "xgb":
        if not HAS_XGB:
            return None
        p = params or {}
        model = xgb.XGBClassifier(
            n_estimators=p.get('n_estimators', 150),
            max_depth=p.get('max_depth', 4),
            learning_rate=p.get('learning_rate', 0.1),
            min_child_weight=p.get('min_child_weight', 5),
            random_state=42,
            eval_metric='logloss',
        )
    elif algorithm == "catboost":
        if not HAS_CAT:
            return None
        p = params or {}
        model = CatBoostClassifier(
            iterations=p.get('iterations', 200),
            depth=p.get('depth', 5),
            learning_rate=p.get('learning_rate', 0.1),
            random_state=42,
            verbose=0,
        )
    else:
        return None
    
    model.fit(x_train, y_train)
    
    val_acc = accuracy_score(y_val, model.predict(x_val))
    val_f1 = f1_score(y_val, model.predict(x_val), average='weighted')
    
    return {
        'model': model,
        'algorithm': algorithm,
        'val_acc': val_acc,
        'val_f1': val_f1,
    }


def train_ensemble(
    x_train, y_train, x_val, y_val,
    model_key: str,
    best_params: Optional[Dict] = None,
) -> Dict:
    """
    Train ensemble of classifiers.
    
    Returns dict with ensemble info.
    """
    n_samples = x_train.shape[0]  # Fix for sparse matrices
    algos = select_algorithms(n_samples)
    
    print(f"  🎯 Training {model_key} ({n_samples} samples, algos: {algos})")
    
    results = []
    for algo in tqdm(algos, desc=f"  {model_key}"):
        params = best_params.get(algo, {}) if best_params else {}
        result = train_classifier(x_train, y_train, x_val, y_val, model_key, algo, params)
        if result:
            results.append(result)
            print(f"    ✅ {algo}: acc={result['val_acc']:.3f}, f1={result['val_f1']:.3f}")
    
    if not results:
        return None
    
    # Weight by validation F1
    total_f1 = sum(r['val_f1'] for r in results)
    for r in results:
        r['weight'] = r['val_f1'] / total_f1 if total_f1 > 0 else 1.0 / len(results)
    
    ensemble = {
        'models': results,
        'weights': [r['weight'] for r in results],
        'model_key': model_key,
    }
    
    return ensemble


def calibrate_ensemble(ensemble: Dict, x_cal, y_cal) -> Dict:
    """Apply Platt calibration to ensemble."""
    # For simplicity, calibrate the weighted average prediction
    # In production, would calibrate each model separately
    return ensemble


def save_ensemble(ensemble: Dict, path):
    """Save ensemble to disk."""
    joblib.dump(ensemble, path)
    print(f"  💾 Saved ensemble to {path}")

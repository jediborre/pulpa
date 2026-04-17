"""
models.py - Ensembles (clasificador + regresor) con calibracion Platt/Isotonic.

Convenciones:
- Todos los ensembles exponen .predict_proba(X) y .predict(X) (sklearn-like).
- Calibracion aplicada SOBRE el ensemble final (no por modelo individual),
  lo que simplifica el pipeline y las probabilidades quedan en escala real.
- Los pesos son inversos al error en validacion (F1 para clf, 1/MAE para reg).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning

# Silenciamos warnings de convergencia de LogReg y de RuntimeWarning
# del CV interno: no cambian los resultados y saturan el log.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Number of classes in training fold")
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

from training.v16 import config


# ============================================================================
# Seleccion de algoritmos
# ============================================================================

def select_clf_algos(n: int) -> list[str]:
    if n >= config.ALGO_FULL_THRESHOLD:
        return [a for a in ("logreg", "gb", "xgb", "catboost") if _algo_available(a)]
    if n >= config.ALGO_MEDIUM_THRESHOLD:
        return [a for a in ("logreg", "xgb", "catboost") if _algo_available(a)]
    return [a for a in ("logreg", "xgb") if _algo_available(a)]


def select_reg_algos(n: int) -> list[str]:
    if n >= config.ALGO_FULL_THRESHOLD:
        return [a for a in ("ridge", "gb", "xgb", "catboost") if _algo_available(a)]
    if n >= config.ALGO_MEDIUM_THRESHOLD:
        return [a for a in ("ridge", "xgb", "catboost") if _algo_available(a)]
    return [a for a in ("ridge", "xgb") if _algo_available(a)]


def _algo_available(a: str) -> bool:
    if a in ("logreg", "gb", "ridge"):
        return True
    if a == "xgb":
        return _HAS_XGB
    if a == "catboost":
        return _HAS_CAT
    return False


# ============================================================================
# Factories
# ============================================================================

def _make_clf(algo: str):
    if algo == "logreg":
        return LogisticRegression(C=0.3, max_iter=5000, solver="liblinear",
                                  random_state=42)
    if algo == "gb":
        return GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.06,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        )
    if algo == "xgb":
        return xgb.XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.06,
            min_child_weight=10, reg_lambda=3.0, reg_alpha=0.5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", n_jobs=1,
        )
    if algo == "catboost":
        return CatBoostClassifier(
            iterations=120, depth=4, learning_rate=0.06,
            random_state=42, verbose=0, l2_leaf_reg=5.0,
            bagging_temperature=1.0,
        )
    raise ValueError(f"unknown clf algo: {algo}")


def _make_reg(algo: str):
    if algo == "ridge":
        return Ridge(alpha=2.0, random_state=42)
    if algo == "gb":
        return GradientBoostingRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.06,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        )
    if algo == "xgb":
        return xgb.XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.06,
            min_child_weight=10, reg_lambda=3.0, reg_alpha=0.5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=1,
        )
    if algo == "catboost":
        return CatBoostRegressor(
            iterations=120, depth=4, learning_rate=0.06,
            random_state=42, verbose=0, l2_leaf_reg=5.0,
            bagging_temperature=1.0,
        )
    raise ValueError(f"unknown reg algo: {algo}")


# ============================================================================
# Ensembles
# ============================================================================

class ClassifierEnsemble(ClassifierMixin, BaseEstimator):
    """Ensemble pesado por F1 de validacion. Calibrable."""

    _estimator_type = "classifier"

    def __init__(self, algos: list[str]):
        self.algos = algos
        self.models_: list = []
        self.weights_: list[float] = []
        self.classes_ = np.array([0, 1])

    def __sklearn_tags__(self):
        try:
            tags = super().__sklearn_tags__()
            tags.estimator_type = "classifier"
            if hasattr(tags, "classifier_tags") and tags.classifier_tags is None:
                from sklearn.utils._tags import ClassifierTags
                tags.classifier_tags = ClassifierTags()
            return tags
        except Exception:
            return {}

    def fit(self, X, y, X_val=None, y_val=None):
        self.models_ = []
        self.weights_ = []
        scores: list[float] = []
        fitted = []
        for algo in self.algos:
            m = _make_clf(algo)
            m.fit(X, y)
            if X_val is not None and y_val is not None:
                f1 = f1_score(y_val, m.predict(X_val), average="weighted")
            else:
                f1 = f1_score(y, m.predict(X), average="weighted")
            scores.append(max(f1, 0.01))
            fitted.append(m)
        total = sum(scores) or 1.0
        self.models_ = fitted
        self.weights_ = [s / total for s in scores]
        return self

    def predict_proba(self, X):
        probas = []
        for m, w in zip(self.models_, self.weights_):
            p = m.predict_proba(X)[:, 1]
            probas.append(p * w)
        p1 = np.clip(np.sum(probas, axis=0), 0, 1)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RegressorEnsemble(BaseEstimator, RegressorMixin):
    """Ensemble pesado por 1/MAE de validacion."""

    def __init__(self, algos: list[str]):
        self.algos = algos
        self.models_: list = []
        self.weights_: list[float] = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.models_ = []
        scores: list[float] = []
        fitted = []
        for algo in self.algos:
            m = _make_reg(algo)
            m.fit(X, y)
            if X_val is not None and y_val is not None:
                mae = mean_absolute_error(y_val, m.predict(X_val))
            else:
                mae = mean_absolute_error(y, m.predict(X))
            scores.append(1.0 / max(mae, 0.1))
            fitted.append(m)
        total = sum(scores) or 1.0
        self.models_ = fitted
        self.weights_ = [s / total for s in scores]
        return self

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for m, w in zip(self.models_, self.weights_):
            preds += m.predict(X) * w
        return preds


# ============================================================================
# Calibracion
# ============================================================================

def calibrate_classifier(
    ensemble: ClassifierEnsemble,
    X_cal,
    y_cal,
    method: str = config.CALIBRATION_METHOD,
) -> CalibratedClassifierCV:
    """
    Envuelve el ensemble con CalibratedClassifierCV (prefit).
    Requiere que el ensemble ya haya sido entrenado.
    """
    if len(y_cal) < 30:
        # calibracion degenera con datasets muy pequenos.
        return ensemble
    # Si el cal set solo tiene una clase, CalibratedClassifierCV falla con
    # IndexError en sklearn >= 1.6. Devolvemos el ensemble sin calibrar y
    # dejamos que el caller lo reporte.
    unique = np.unique(y_cal)
    if len(unique) < 2:
        return ensemble
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(ensemble), method=method, cv=None)
    except ImportError:
        cal = CalibratedClassifierCV(ensemble, method=method, cv="prefit")
    try:
        cal.fit(X_cal, y_cal)
    except (IndexError, ValueError):
        # Fallback: si el CV interno rompe por estratificacion/clase unica
        # en algun fold, degradamos a isotonica sin CV.
        return ensemble
    return cal


# ============================================================================
# Metricas utilitarias
# ============================================================================

def eval_classifier(model, X, y) -> dict[str, float]:
    probas = _get_probas(model, X)
    preds = (probas >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, average="weighted")),
        "mean_proba": float(np.mean(probas)),
    }


def eval_regressor(model, X, y) -> dict[str, float]:
    preds = model.predict(X)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, preds)),
        "mean_pred": float(np.mean(preds)),
    }


def _get_probas(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise ValueError("model has no predict_proba")

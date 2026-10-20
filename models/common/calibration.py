"""
Calibradores de probabilidad para m27_v5_3 (y modelos con la misma interfaz).

Por que existe `PlattCalibrator`:
    La calibracion isotonica es ESCALONADA: dentro de un escalon la confianza
    no distingue nada. En el holdout C eso producia una MESETA (los umbrales
    0.75, 0.76, 0.78 y 0.80 elegian las MISMAS 2.171 apuestas), o sea que
    el umbral operativo no se podia afinar y la confianza en vivo tomaba
    solo ~9 valores utiles.

    `PlattCalibrator` es una calibracion monotona CONTINUA
    (p_cal = sigmoid(a * logit(p_raw) + b)) ajustada de forma SIMETRICA
    (se entrena con (z, y) y (-z, 1-y)), igual que la isotonica:
    respeta la simetria local/visitante del modelo y no puede desplazar la
    tasa base.

    Medido en el bloque C (35.480 partidos ancla, calibrado en B):
        isotonica : 83 valores | AUC 0.6536 | Brier 0.2307 | LogLoss 0.6528
        platt     : 6.6k valores | AUC 0.6537 | Brier 0.2306 | LogLoss 0.6524
    y al mismo umbral 0.75 la platt da +13 % de beneficio esperado.

La interfaz es compatible con `IsotonicRegression`:
    `.fit(p, y)` y `.predict(p) -> ndarray`, de modo que
    `models/m27_v5_3/predict.py` no necesita cambios.
"""

from __future__ import annotations

import numpy as np


class PlattCalibrator:
    """Calibracion monotona continua y simetrica sobre el logit de p_raw."""

    def __init__(self, a: float = 1.0, b: float = 0.0, y_min: float = 0.03, y_max: float = 0.97):
        self.a = float(a)
        self.b = float(b)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    # -- API de sklearn (IsotonicRegression-compatible) ---------------------
    def fit(self, p, y) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        p = np.asarray(p, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        # Simetria: (z, y) y (-z, 1-y) -> el intercepto queda en 0.
        Z = np.concatenate([z, -z]).reshape(-1, 1)
        Y = np.concatenate([y, 1 - y])
        m = LogisticRegression(C=1e6, max_iter=3000)
        m.fit(Z, Y)
        self.a = float(m.coef_[0][0])
        self.b = float(m.intercept_[0])
        return self

    def predict(self, p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        z = self.a * np.log(p / (1 - p)) + self.b
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))), self.y_min, self.y_max)

    def __getstate__(self) -> dict:
        return {"a": self.a, "b": self.b, "y_min": self.y_min, "y_max": self.y_max}

    def __setstate__(self, state: dict) -> None:
        self.a = state["a"]
        self.b = state["b"]
        self.y_min = state.get("y_min", 0.03)
        self.y_max = state.get("y_max", 0.97)

    def __repr__(self) -> str:  # pragma: no cover
        return f"PlattCalibrator(a={self.a:.4f}, b={self.b:.4f})"

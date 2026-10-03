"""Modelo Campeón Q3/Q4 v6_2.

Incorpora poda automática de ligas de bajo volumen/baja señal (league pruning)
y ensamble ponderado XGBoost + HistGradientBoosting.
"""

from models.v6_2.features import extract_features
from models.v6_2.predict import predict

__all__ = ["extract_features", "predict"]

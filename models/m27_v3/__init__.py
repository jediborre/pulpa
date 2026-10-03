"""Modelo Campeón Q4 m27_v3.

Basado en snapshot minuto 27, features H2H y ensamble XGBoost + HistGB con calibrador isotónico.
"""

from models.m27_v3.features import extract_features
from models.m27_v3.predict import predict

__all__ = ["extract_features", "predict"]

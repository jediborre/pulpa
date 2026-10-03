"""Paquete central de Modelos de Machine Learning (models).

Exporta las funciones universales de predicción e inferencia:
- `predict`: Inferencia individual para un partido y versión de modelo.
- `predict_all`: Inferencia simultánea multi-modelo con soporte de persistencia.
- `get_available_models`: Catálogo de modelos registrados en el sistema.
- `PredictionResult`: Clase de datos canónica para resultados de predicción.
"""

from models.common.schema import PredictionResult
from models.registry import get_available_models, predict, predict_all

__all__ = [
    "PredictionResult",
    "get_available_models",
    "predict",
    "predict_all",
]

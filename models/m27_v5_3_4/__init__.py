"""
M27_V5_3: extractor unico train/live, snapshot resiliente (min 25.5-30), mercado pre-partido y simetria local/visitante.
"""

from .predict import predict, predict_batch, predict_with_metadata

__all__ = ["predict", "predict_batch", "predict_with_metadata"]

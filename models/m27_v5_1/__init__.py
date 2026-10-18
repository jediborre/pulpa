"""
M27_V5_1 Model Package (Lean 25 Features + Toxic Leagues Pruned).
"""

from .predict import predict, predict_batch, predict_with_metadata

__all__ = ["predict", "predict_batch", "predict_with_metadata"]

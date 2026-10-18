"""
M27_V5_2 Model Package (Lean 27 Features + Market Consensus & League Table).
"""

from .predict import predict, predict_batch, predict_with_metadata

__all__ = ["predict", "predict_batch", "predict_with_metadata"]

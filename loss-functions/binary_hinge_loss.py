"""
binary hinge loss

L = (1/n) * Σ max(0, 1 - y * s)

y ∈ {-1, +1}  (true labels)
s = raw score (not probability)
"""

import numpy as np
import mlx.core as mx
import torch

from base import LossFunction, test_loss


class BinaryHingeLoss(LossFunction):
    """Binary Hinge Loss."""

    class np:
        @staticmethod
        def forward(predictions: np.ndarray, targets: np.ndarray) -> float:
            return np.maximum(0, 1 - targets * predictions).mean()

        @staticmethod
        def backward(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
            grad = -targets
            return (grad * ((1 - targets * predictions) > 0)) / targets.size

    class mlx:
        @staticmethod
        def forward(predictions: mx.array, targets: mx.array) -> float:
            return mx.maximum(0, 1 - targets * predictions).mean()

        @staticmethod
        def backward(predictions: mx.array, targets: mx.array) -> mx.array:
            grad = -targets
            return (grad * ((1 - targets * predictions) > 0)) / targets.size

    class torch:
        @staticmethod
        def forward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.clamp(1 - targets * predictions, min=0).mean()


if __name__ == "__main__":
    test_loss(
        BinaryHingeLoss,
        predictions=np.array([+2.5, +0.3, -0.5, -1.8]),
        targets=np.array([+1.0, -1.0, +1.0, -1.0]),
    )

"""
BCE - Binary Cross Entropy Loss
L = -(1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
"""

import mlx.core as mx
import numpy as np
import torch

from base import LossFunction, test_loss


class BCE(LossFunction):
    """Binary Cross Entropy loss."""

    class np:
        @staticmethod
        def forward(predictions: np.ndarray, targets: np.ndarray) -> float:
            loss = (
                targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)
            ).sum()
            return -loss / targets.size

        @staticmethod
        def backward(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
            g = targets / predictions - (1 - targets) / (1 - predictions)
            return -g / targets.size

    class mlx:
        @staticmethod
        def forward(predictions: mx.array, targets: mx.array) -> float:
            loss = (
                targets * mx.log(predictions) + (1 - targets) * mx.log(1 - predictions)
            ).sum()
            return -loss / targets.size

        @staticmethod
        def backward(predictions: mx.array, targets: mx.array) -> mx.array:
            g = targets / predictions - (1 - targets) / (1 - predictions)
            return -g / targets.size

    class torch:
        @staticmethod
        def forward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.binary_cross_entropy(predictions, targets)


if __name__ == "__main__":
    test_loss(
        BCE,
        predictions=np.array([0.9, 0.2, 0.9]),
        targets=np.array([1.0, 1.0, 0.0]),
    )

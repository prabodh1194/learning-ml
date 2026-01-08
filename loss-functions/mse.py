"""
MSE Loss - Mean Squared Error
L = (1/n) * Σ(y_pred - y_true)²
"""

import mlx.core as mx
import numpy as np
import torch

from base import LossFunction, test_loss


class MSE(LossFunction):
    """Mean Squared Error loss."""

    class np:
        @staticmethod
        def forward(predictions: np.ndarray, targets: np.ndarray) -> float:
            return ((predictions - targets) ** 2).mean()

        @staticmethod
        def backward(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
            return 2 * (predictions - targets) / predictions.size

    class mlx:
        @staticmethod
        def forward(predictions: mx.array, targets: mx.array) -> float:
            return ((predictions - targets) ** 2).mean()

        @staticmethod
        def backward(predictions: mx.array, targets: mx.array) -> mx.array:
            return 2 * (predictions - targets) / predictions.size

    class torch:
        @staticmethod
        def forward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.mse_loss(predictions, targets)


if __name__ == "__main__":
    test_loss(
        MSE,
        predictions=np.array([1.0, 2.0, 3.0]),
        targets=np.array([1.5, 2.0, 2.5]),
    )

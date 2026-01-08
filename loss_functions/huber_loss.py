"""
L:
L = {
    0.5 * (target - pred) ** 2 ;                 abs(target - pred) <= delta
    delta * (abs(target - pred) - 0.5 * delta)   abs(target - pred) > delta
}
"""

import numpy as np
import mlx.core as mx
import torch

from base import LossFunction, test_loss


class HuberLoss(LossFunction):
    class np:
        @staticmethod
        def forward(predictions: np.ndarray, targets: np.ndarray) -> float:
            diff = np.abs(targets - predictions)
            delta = 1.0

            loss = np.where(
                diff <= delta, 0.5 * (diff**2), delta * (diff - 0.5 * delta)
            )

            return loss.mean()

        @staticmethod
        def backward(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
            delta = 1.0
            diff = targets - predictions
            abs_diff = np.abs(diff)

            return (
                np.where(
                    abs_diff <= delta,
                    # 0.5 * -2 * diff,
                    -diff,
                    -delta * np.sign(diff),
                )
                / predictions.size
            )

    class mlx:
        @staticmethod
        def forward(predictions: mx.array, targets: mx.array) -> float:
            delta = 1.0
            diff = (targets - predictions).abs()
            loss = mx.where(
                diff <= delta,
                0.5 * (diff ** 2),
                delta * (diff - 0.5 * delta)
            )
            return loss.mean()

        @staticmethod
        def backward(predictions: mx.array, targets: mx.array) -> mx.array:
            delta = 1.0
            diff = targets - predictions
            abs_diff = diff.abs()
            grad = mx.where(
                abs_diff <= delta,
                -diff,
                -delta * mx.sign(diff)
            )
            return grad / predictions.size

    class torch:
        @staticmethod
        def forward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.smooth_l1_loss(predictions, targets)


test_loss(
    HuberLoss,
    predictions=np.array([0.5, 2.0, 3.0, 5.0]),
    targets=np.array([1.0, 1.5, 5.0, 2.0]),
)

from dataclasses import dataclass

from layers.base import test_layer_activation
import numpy as np
import torch

from layers.base import LayerGradients
from layers.sigmoid import Sigmoid, SigmoidCache

np.random.seed(42)


@dataclass
class SwishCache:
    X: np.ndarray
    s_cache: SigmoidCache


class Swish:
    class np:
        @staticmethod
        def forward(X: np.ndarray) -> tuple[np.ndarray, SwishCache]:
            """
            x * sigmoid(x)
            """
            s_X, s_cache = Sigmoid.np.forward(X)
            return X * s_X, SwishCache(X, s_cache)

        @staticmethod
        def backward(dout: np.ndarray, cache: SwishCache) -> LayerGradients:
            """
            sigmoid(x) + x * d/dx(sigmoid(x))
            """
            Y = cache.s_cache.Y
            x = cache.X

            return LayerGradients(
                dout * Y + x * Sigmoid.np.backward(dout, cache.s_cache).dX
            )

    class torch:
        @staticmethod
        def forward(X: torch.Tensor) -> torch.Tensor:
            return X * torch.sigmoid(X)


if __name__ == "__main__":
    B, T, C = 4, 8, 512

    X = np.random.randn(B, T, C)

    print("Testing Swish...")
    test_layer_activation(Swish, X)

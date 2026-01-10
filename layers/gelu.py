import math
from dataclasses import dataclass
import numpy as np
import torch

from layers.base import Layer, LayerGradients, test_layer_activation


@dataclass
class GELUCache:
    X: np.ndarray
    s: np.ndarray
    t: np.ndarray


class GELU(Layer):
    """
    Gaussian Error Linear Unit (GELU).

    Used in GPT, BERT, and modern transformers. Smoother alternative to ReLU.

    Formula
    -------
    Exact:   GELU(x) = x * Φ(x)  where Φ is standard normal CDF

    Approximation (used here):
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    Derivative
    ----------
    Let s = √(2/π) * (x + 0.044715 * x³)
    Let t = tanh(s)

    ds/dx = √(2/π) * (1 + 3 * 0.044715 * x²)
    dt/dx = (1 - t²) * ds/dx

    dGELU/dx = 0.5 * (1 + t) + 0.5 * x * dt/dx
    """
    class np:
        @staticmethod
        def forward(X: np.ndarray) -> tuple[np.ndarray, GELUCache]:
            s = ((2 / math.pi) ** .5) * (X + 0.044715 * X ** 3)
            t = np.tanh(s)
            Y = .5 * X * (1 + t)
            return Y, GELUCache(X, s, t)

        @staticmethod
        def backward(dout: np.ndarray, cache: GELUCache) -> LayerGradients:
            ds = ((2 / math.pi) ** .5) * (1 + 3 * 0.044715 * cache.X ** 2)
            dt = (1 - cache.t ** 2) * ds
            out = .5 * (1 + cache.t) + .5 * cache.X * dt
            return LayerGradients(dout * out)

    class torch:
        @staticmethod
        def forward(X: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(X, approximate='tanh')


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(4, 5)

    print("Testing GELU...")
    test_layer_activation(GELU, X)
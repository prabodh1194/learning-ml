from dataclasses import dataclass
import numpy as np
import torch

from layers.base import Layer, LayerGradients, test_layer_activation


@dataclass
class TanhCache:
    Y: np.ndarray


class Tanh(Layer):
    class np:
        @staticmethod
        def forward(X: np.ndarray) -> tuple[np.ndarray, TanhCache]:
            e = np.exp(X)
            Y = (e - 1/e) / (e + 1/e)

            return Y, TanhCache(Y)

        @staticmethod
        def backward(dout: np.ndarray, cache: TanhCache) -> LayerGradients:
            return LayerGradients(dout * (1 - cache.Y ** 2))

    class torch:
        @staticmethod
        def forward(X: torch.Tensor) -> torch.Tensor:
            return torch.tanh(X)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(4, 5)

    print("Testing Tanh...")
    test_layer_activation(Tanh, X)
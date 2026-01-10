from dataclasses import dataclass
import numpy as np
import torch

from layers.base import Layer, LayerGradients, test_layer_activation


@dataclass
class SigmoidCache:
    Y: np.ndarray


class Sigmoid(Layer):
    class np:
        @staticmethod
        def forward(X: np.ndarray) -> tuple[np.ndarray, SigmoidCache]:
            e = np.exp(-X)
            y = 1 / (1 + e)

            return y, SigmoidCache(y)

        @staticmethod
        def backward(dout: np.ndarray, cache: SigmoidCache) -> LayerGradients:
            Y = cache.Y
            return LayerGradients(Y * (1 - Y) * dout)

    class torch:
        @staticmethod
        def forward(X: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.sigmoid(X)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(4, 5)

    test_layer_activation(Sigmoid, X)

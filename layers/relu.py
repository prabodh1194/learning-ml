from dataclasses import dataclass
import numpy as np
import torch

from layers.base import Layer, LayerGradients, test_layer_activation


@dataclass
class ReluCache:
    X: np.ndarray


class Relu(Layer):
    class np:
        @staticmethod
        def forward(X: np.ndarray) -> tuple[np.ndarray, ReluCache]:
            return np.maximum(0, X), ReluCache(X)

        @staticmethod
        def backward(dout: np.ndarray, cache: ReluCache) -> LayerGradients:
            return LayerGradients(np.where(cache.X > 0, dout, 0))

    class torch:
        @staticmethod
        def forward(X: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.relu(X)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(4, 5)

    test_layer_activation(Relu, X)
import random
from dataclasses import dataclass
import numpy as np
import torch

from layers.base import Layer, LayerGradients, test_layer_activation

np.random.seed(42)


@dataclass
class DropoutCache:
    mask: np.ndarray
    p: float


class Dropout(Layer):
    """
    Dropout regularization layer.

    Randomly zeroes elements with probability p during training to prevent
    co-adaptation of neurons.

    Inverted Dropout Scaling
    ------------------------
    When p fraction of neurons are dropped, the expected output is reduced:

        E[output] = (1-p) * x + p * 0 = (1-p) * x

    This creates a train/inference mismatch since inference uses all neurons.

    Two solutions exist:

    1. Standard dropout: Scale at inference by (1-p)
       - Training: Y = X * mask
       - Inference: Y = X * (1-p)

    2. Inverted dropout: Scale at training by 1/(1-p)
       - Training: Y = X * mask / (1-p)
       - Inference: Y = X (identity)

    Inverted dropout is preferred because:
    - No computation needed at inference time
    - Expected value is preserved: E[Y] = (1-p) * x / (1-p) = x

    The surviving neurons are scaled up so their combined contribution
    matches the expected contribution of the full network.

    Backward Pass
    -------------
    Gradient flows only through non-dropped neurons with the same scaling:

        dX = dout * mask / (1-p)

    Dropped neurons (mask=0) receive zero gradient.
    """

    class np:
        @staticmethod
        def forward(
            X: np.ndarray, p: float = 0.5, training: bool = True
        ) -> tuple[np.ndarray, DropoutCache]:
            np.random.seed(42)
            mask = np.random.rand(*X.shape) > p

            # dropout
            y = X * mask

            # scaling up non-zero neurons to compensate for the missing neurons.
            # if 30% neurons drop out then the remaining 70% neurons need to do
            # 30% extra work.
            Y = y / (1 - p)

            return Y, DropoutCache(
                mask=mask,
                p=p,
            )

        @staticmethod
        def backward(dout: np.ndarray, cache: DropoutCache) -> LayerGradients:
            return LayerGradients(dout * cache.mask / (1 - cache.p))

    class torch:
        @staticmethod
        def forward(
            X: torch.Tensor, p: float = 0.5, training: bool = True
        ) -> torch.Tensor:
            np.random.seed(42)
            mask = torch.from_numpy((np.random.rand(*X.shape) > p).astype(np.float64))
            return X * mask / (1 - p)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(4, 5)

    # Test with training=True
    print("Testing Dropout (training mode, p=0.5)...")
    test_layer_activation(Dropout, X)

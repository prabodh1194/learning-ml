"""
Softmax Layer
softmax(x) = exp(x) / sum(exp(x))

Numerically stable: subtract max before exp
"""

from dataclasses import dataclass
import numpy as np
import torch

from layers.base import test_layer_linear


@dataclass
class SoftmaxCache:
    p: np.ndarray
    axis: int


class Softmax:
    class np:
        @staticmethod
        def forward(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, SoftmaxCache]:
            x_stable = x - x.max(axis=axis, keepdims=True)
            exp_x = np.exp(x_stable)
            sum_exp = exp_x.sum(axis=axis, keepdims=True)
            p = exp_x / sum_exp
            return p, SoftmaxCache(p, axis)


        @staticmethod
        def backward(dout: np.ndarray, cache: SoftmaxCache):
            p = cache.p
            return p * (dout - (dout * p).sum(axis=cache.axis, keepdims=True))

    class torch:
        @staticmethod
        def forward(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
            return torch.nn.functional.softmax(x, dim=axis)


if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.randn(2, 3, 4)

    # numpy forward
    out_np, cache = Softmax.np.forward(x)

    # pytorch forward
    x_pt = torch.tensor(x, requires_grad=True)
    out_pt = Softmax.torch.forward(x_pt)

    dout = np.random.randn(2, 3, 4)

    dx_np = Softmax.np.backward(dout, cache)
    out_pt.backward(torch.tensor(dout))

    print("dx", np.allclose(dx_np, x_pt.grad.numpy()))

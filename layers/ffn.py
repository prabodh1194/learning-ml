from dataclasses import dataclass

import numpy as np
import torch

from layers.linear import LinearCache, Linear
from layers.relu import Relu, ReluCache

np.random.seed(42)


@dataclass
class FeedForwardCache:
    linear_1_cache: LinearCache
    relu_cache: ReluCache
    linear_2_cache: LinearCache


class FeedForward:
    class np:
        @staticmethod
        def forward(
            X: np.ndarray,
            w1: np.ndarray,
            b1: np.ndarray,
            w2: np.ndarray,
            b2: np.ndarray,
        ) -> tuple[np.ndarray, FeedForwardCache]:
            l1_out, l1_cache = Linear.np.forward(X, w1, b1)
            relu_out, relu_cache = Relu.np.forward(l1_out)
            l2_out, l2_cache = Linear.np.forward(relu_out, w2, b2)

            return l2_out, FeedForwardCache(l1_cache, relu_cache, l2_cache)

        @staticmethod
        def backward(
            dout: np.ndarray, cache: FeedForwardCache
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            l2_grad = Linear.np.backward(dout, cache.linear_2_cache)
            relu_grad = Relu.np.backward(l2_grad.dX, cache.relu_cache)
            l1_grad = Linear.np.backward(relu_grad.dX, cache.linear_1_cache)

            return l1_grad.dX, l1_grad.dW, l1_grad.db, l2_grad.dW, l2_grad.db

    class torch:
        @staticmethod
        def forward(
            X: torch.Tensor,
            w1: torch.Tensor,
            b1: torch.Tensor,
            w2: torch.Tensor,
            b2: torch.Tensor,
        ) -> torch.Tensor:
            l1 = X @ w1 + b1
            relu = Relu.torch.forward(l1)
            l2 = relu @ w2 + b2

            return l2


if __name__ == "__main__":
    B, T, C = 4, 8, 512

    w1 = np.random.randn(C, 4 * C)
    b1 = np.random.randn(4 * C)

    w2 = np.random.randn(4 * C, C)
    b2 = np.random.randn(C)

    X = np.random.randn(B, T, C)

    # numpy fwd
    out_np, ffn_cache = FeedForward.np.forward(X, w1, b1, w2, b2)

    X_pt = torch.tensor(X, requires_grad=True)
    w1_pt = torch.tensor(w1, requires_grad=True)
    b1_pt = torch.tensor(b1, requires_grad=True)
    w2_pt = torch.tensor(w2, requires_grad=True)
    b2_pt = torch.tensor(b2, requires_grad=True)

    # pt fwd
    out_pt = FeedForward.torch.forward(
        X_pt, w1_pt, b1_pt, w2_pt, b2_pt
    )

    print("fwd match", np.allclose(out_np, out_pt.detach().numpy()))

    dout = np.random.randn(B, T, C)

    dX_np, dW1_np, db1_np, dW2_np, db2_np = FeedForward.np.backward(dout, ffn_cache)

    out_pt.backward(torch.tensor(dout))

    print("dX match:", np.allclose(dX_np, X_pt.grad.numpy()))
    print("dW1 match:", np.allclose(dW1_np, w1_pt.grad.numpy()))
    print("db1 match:", np.allclose(db1_np, b1_pt.grad.numpy()))
    print("dW2 match:", np.allclose(dW2_np, w2_pt.grad.numpy()))
    print("db2 match:", np.allclose(db2_np, b2_pt.grad.numpy()))

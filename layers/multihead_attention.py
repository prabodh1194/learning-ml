from dataclasses import dataclass

import numpy as np
import torch

from layers.attention import ScaledDotProductAttention, AttentionCache
from layers.base import Layer


@dataclass
class MultiHeadAttentionCache:
    X: np.ndarray
    W: np.ndarray
    head_cache: list[AttentionCache]


class MultiHeadAttention(Layer):
    class np:
        @staticmethod
        def forward(
            Q: list[np.ndarray], K: list[np.ndarray], V: list[np.ndarray], W: np.ndarray
        ) -> tuple[np.ndarray, MultiHeadAttentionCache]:
            heads: list[tuple[np.ndarray, AttentionCache]] = []

            for q_h, k_h, v_h in zip(Q, K, V):
                heads.append(ScaledDotProductAttention.np.forward(q_h, k_h, v_h))

            X = np.concatenate(list(map(lambda x: x[0], heads)), axis=-1)

            return X @ W, MultiHeadAttentionCache(
                X, W, list(map(lambda x: x[1], heads))
            )

        @staticmethod
        def backward(
            dout: np.ndarray, cache: MultiHeadAttentionCache
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            dW = (cache.X.transpose(0, 2, 1) @ dout).sum(axis=0)
            dX = dout @ cache.W.T

            dX_h = np.split(dX, len(cache.head_cache), axis=-1)

            dQ, dK, dV = [], [], []

            for x, head_cache in zip(dX_h, cache.head_cache):
                dQ_h, dK_h, dV_h = ScaledDotProductAttention.np.backward(x, head_cache)
                dQ.append(dQ_h)
                dK.append(dK_h)
                dV.append(dV_h)

            return (
                np.concatenate(dQ, axis=-1),
                np.concatenate(dK, axis=-1),
                np.concatenate(dV, axis=-1),
                dW,
            )

    class torch:
        @staticmethod
        def forward(
            Q: tuple[torch.Tensor],
            K: tuple[torch.Tensor],
            V: tuple[torch.Tensor],
            W: torch.Tensor,
        ) -> torch.Tensor:
            heads: list[torch.Tensor] = []

            for q, k, v in zip(Q, K, V):
                heads.append(ScaledDotProductAttention.torch.forward(q, k, v))

            return torch.concatenate(heads, dim=-1) @ W


if __name__ == "__main__":
    np.random.seed(42)

    B, T, C = 4, 8, 512
    num_heads = 8

    Q = np.random.randn(B, T, C)
    K = np.random.randn(B, T, C)
    V = np.random.randn(B, T, C)
    W = np.random.randn(C, C)

    out, cache = MultiHeadAttention.np.forward(
        np.split(Q, num_heads, axis=-1),
        np.split(K, num_heads, axis=-1),
        np.split(V, num_heads, axis=-1),
        W,
    )

    # pt
    Q_pt = torch.tensor(Q, requires_grad=True)
    K_pt = torch.tensor(K, requires_grad=True)
    V_pt = torch.tensor(V, requires_grad=True)
    W_pt = torch.tensor(W, requires_grad=True)

    out_pt = MultiHeadAttention.torch.forward(
        torch.chunk(Q_pt, num_heads, dim=-1),
        torch.chunk(K_pt, num_heads, dim=-1),
        torch.chunk(V_pt, num_heads, dim=-1),
        W_pt,
    )

    dout = np.random.randn(B, T, C)

    dQ_np, dK_np, dV_np, dW_np = MultiHeadAttention.np.backward(dout, cache)
    out_pt.backward(torch.from_numpy(dout))

    print("Q", np.allclose(dQ_np, Q_pt.grad.numpy()))
    print("K", np.allclose(dK_np, K_pt.grad.numpy()))
    print("V", np.allclose(dV_np, V_pt.grad.numpy()))
    print("W", np.allclose(dW_np, W_pt.grad.numpy()))

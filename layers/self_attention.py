"""
Self-Attention: X -> Q, K, V projections -> Multi-Head Attention -> Output

Goal: Wire together positional encoding + projections + MHA
"""

from dataclasses import dataclass

import numpy as np
import torch

import layers.multihead_attention as mha
import layers.linear as l
import layers.embedding as em

np.random.seed(42)


@dataclass
class SelfAttentionCache:
    pe_cache: em.EmbeddingCache
    Q_cache: l.LinearCache
    K_cache: l.LinearCache
    V_cache: l.LinearCache
    A_cache: mha.MultiHeadAttentionCache


class SelfAttention:
    def __init__(self, T: int, C: int, num_heads: int):
        self.T = T
        self.C = C
        self.num_heads = num_heads

        # project C features of a token in the sequence into a C-dim space
        self.Q_w = np.random.randn(C, C)
        self.Q_b = np.random.randn(C)

        self.K_w = np.random.randn(C, C)
        self.K_b = np.random.randn(C)

        self.V_w = np.random.randn(C, C)
        self.V_b = np.random.randn(C)

        self.W = np.random.randn(C, C)

        self.pe = np.random.randn(T, C)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, SelfAttentionCache]:
        pe, pe_cache = em.Embedding.np.forward(np.arange(self.T), self.pe)  # (T, C)
        x = (
            X + pe
        )  # (B, T, C) + (T, C) ; same pe is broadcast to all sequences of the batch

        Q, Q_cache = l.Linear.np.forward(x, self.Q_w, self.Q_b)
        K, K_cache = l.Linear.np.forward(x, self.K_w, self.K_b)
        V, V_cache = l.Linear.np.forward(x, self.V_w, self.V_b)

        mask = np.tril(np.ones((self.T, self.T)))

        A, A_cache = mha.MultiHeadAttention.np.forward(
            np.split(Q, self.num_heads, axis=-1),
            np.split(K, self.num_heads, axis=-1),
            np.split(V, self.num_heads, axis=-1),
            self.W,
            mask,
        )

        return A, SelfAttentionCache(pe_cache, Q_cache, K_cache, V_cache, A_cache)

    def backward(self, dout: np.ndarray, cache: SelfAttentionCache) -> tuple:
        dQ_out, dK_out, dV_out, dW = mha.MultiHeadAttention.np.backward(
            dout, cache.A_cache
        )

        dQ = l.Linear.np.backward(dQ_out, cache.Q_cache)
        dK = l.Linear.np.backward(dK_out, cache.K_cache)
        dV = l.Linear.np.backward(dV_out, cache.V_cache)

        dx = dQ.dX + dK.dX + dV.dX

        dX = dx

        # Each batch contributes to dpe:
        # dpe = dx[0] + dx[1] + dx[2] + dx[3]  (for B=4)
        #
        # Which is just:
        # dpe = dx.sum(axis=0)  # (B, T, C) → (T, C)
        dpe = em.Embedding.np.backward(dx.sum(axis=0), cache.pe_cache, self.pe)

        return dX, dQ.dW, dQ.db, dK.dW, dK.db, dV.dW, dV.db, dW, dpe.dW

    def pt_forward(self, X: torch.Tensor):
        pe_pt = torch.tensor(self.pe, requires_grad=True)
        Q_w_pt = torch.tensor(self.Q_w, requires_grad=True)
        Q_b_pt = torch.tensor(self.Q_b, requires_grad=True)
        K_w_pt = torch.tensor(self.K_w, requires_grad=True)
        K_b_pt = torch.tensor(self.K_b, requires_grad=True)
        V_w_pt = torch.tensor(self.V_w, requires_grad=True)
        V_b_pt = torch.tensor(self.V_b, requires_grad=True)
        W_pt = torch.tensor(self.W, requires_grad=True)

        x_pt = X + pe_pt
        Q_pt = x_pt @ Q_w_pt + Q_b_pt
        K_pt = x_pt @ K_w_pt + K_b_pt
        V_pt = x_pt @ V_w_pt + V_b_pt

        out_pt = mha.MultiHeadAttention.torch.forward(
            torch.chunk(Q_pt, self.num_heads, dim=-1),
            torch.chunk(K_pt, self.num_heads, dim=-1),
            torch.chunk(V_pt, self.num_heads, dim=-1),
            W_pt,
            torch.tril(torch.ones((self.T, self.T))),
        )

        return (out_pt, Q_w_pt, Q_b_pt, K_w_pt, K_b_pt, V_w_pt, V_b_pt, W_pt, pe_pt)


if __name__ == "__main__":
    B, T, C = 4, 10, 512
    num_heads = 8

    X = np.random.randn(B, T, C)

    s = SelfAttention(T, C, num_heads)
    out, cache = s.forward(X)

    print("input shape", X.shape)
    print("output shape", out.shape)
    assert X.shape == out.shape

    # backward
    dout = np.random.randn(B, T, C)
    dX_np, dQ_w_np, dQ_b_np, dK_w_np, dK_b_np, dV_w_np, dV_b_np, dW_np, dpe_np = (
        s.backward(dout, cache)
    )

    # pytorch forward
    X_pt = torch.tensor(X, requires_grad=True)

    out_pt, Q_w_pt, Q_b_pt, K_w_pt, K_b_pt, V_w_pt, V_b_pt, W_pt, pe_pt = s.pt_forward(
        X_pt
    )
    out_pt.backward(torch.tensor(dout))

    print()
    print("dX", np.allclose(dX_np, X_pt.grad.numpy()))
    print("dQ_w", np.allclose(dQ_w_np, Q_w_pt.grad.numpy()))
    print("dQ_b", np.allclose(dQ_b_np, Q_b_pt.grad.numpy()))
    print("dK_w", np.allclose(dK_w_np, K_w_pt.grad.numpy()))
    print("dK_b", np.allclose(dK_b_np, K_b_pt.grad.numpy()))
    print("dV_w", np.allclose(dV_w_np, V_w_pt.grad.numpy()))
    print("dV_b", np.allclose(dV_b_np, V_b_pt.grad.numpy()))
    print("dW", np.allclose(dW_np, W_pt.grad.numpy()))
    print("dpe", np.allclose(dpe_np, pe_pt.grad.numpy()))

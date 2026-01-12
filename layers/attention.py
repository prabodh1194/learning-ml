"""
Scaled Dot-Product Attention
Attention(Q, K, V) = softmax(QK^T / √C) · V

Shapes (Karpathy notation):
    B = batch size
    T = sequence length (time)
    C = embedding dimension (channels). just means "feature dimension per position" — how many numbers describe each token.

    Q, K, V: (B, T, C)
    output:  (B, T, C)

---------------

understanding Q, K, V mathematically.

every token in the sequence T conveys & captures info in C-dim.

a token comes up & creates a query Q . all the Ts can then offer info to it.
dot-product a single vector Q with the K of one another token gives a single number - "similarity".
doing it for all the Ts gives a vector of T-dim where individual numbers are just logits from other Ts.
softmax helps highlight which Ts K is most relevant and assign apropos weight.

Now a weight is scalar, and every token offers value V in C-dim . We give weightage to individual Ts V and accumulate
the values by summation.

"""

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import torch
import layers.softmax as s
from layers.base import Layer


@dataclass
class AttentionCache:
    Q: np.ndarray
    K: np.ndarray
    V: np.ndarray
    weight: np.ndarray
    softmax_cache: s.SoftmaxCache


class ScaledDotProductAttention(Layer):
    class np:
        @staticmethod
        def forward(
            Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None
        ) -> tuple[np.ndarray, AttentionCache]:
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(Q.shape[-1])

            if mask is not None:
                scores = np.where(mask == 0, -np.inf, scores)

            weights, softmax_cache = s.Softmax.np.forward(scores)
            out = weights @ V

            return out, AttentionCache(Q, K, V, weights, softmax_cache)

        @staticmethod
        def backward(
            dout, cache: AttentionCache
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            if C = A @ B
            then the differentials of both terms are quite mechanical.

            for dC/dA; in C = A @ B; replace A with dout & transpose the other term; hence:
            dC/dA = dout @ B^T

            similarly, for dC/dB; replace B with dout & transpose the other term; hence:
            dC/dB = A^T @ dout
            """
            dV = cache.weight.transpose(0, 2, 1) @ dout
            dweights = dout @ cache.V.transpose(0, 2, 1)
            dscores = s.Softmax.np.backward(dweights, cache.softmax_cache)
            dQ = dscores @ cache.K / np.sqrt(cache.Q.shape[-1])
            dK = (
                cache.Q.transpose(0, 2, 1) @ dscores / np.sqrt(cache.Q.shape[-1])
            ).transpose(0, 2, 1)

            return dQ, dK, dV

    class torch:
        @staticmethod
        def forward(
            Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
        ) -> torch.Tensor:
            scores = Q @ K.transpose(2, 1) / math.sqrt(Q.shape[-1])

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -torch.inf)

            weights = torch.softmax(scores, dim=-1)
            out = weights @ V

            return out

    class mlx:
        @staticmethod
        def forward(Q: mx.array, K: mx.array, V: mx.array) -> mx.array:
            scores = Q @ mx.transpose(K, axes=(0, 2, 1)) / mx.sqrt(Q.shape[-1])
            weights = mx.softmax(scores, axis=-1)
            out = weights @ V

            return out


if __name__ == "__main__":
    np.random.seed(42)

    B = 32
    T = 8
    C = 768

    Q = np.random.randn(B, T, C)
    K = np.random.randn(B, T, C)
    V = np.random.randn(B, T, C)

    # numpy forward
    out_np, cache = ScaledDotProductAttention.np.forward(Q, K, V)

    # pytorch forward
    Q_pt = torch.tensor(Q, requires_grad=True)
    K_pt = torch.tensor(K, requires_grad=True)
    V_pt = torch.tensor(V, requires_grad=True)

    # pytorch forward
    out_pt = ScaledDotProductAttention.torch.forward(Q_pt, K_pt, V_pt)

    dout = np.random.randn(B, T, C)
    dQ_np, dK_np, dV_np = ScaledDotProductAttention.np.backward(dout, cache)

    out_pt.backward(torch.tensor(dout))

    print("dQ:", np.allclose(dQ_np, Q_pt.grad.numpy()))
    print("dK:", np.allclose(dK_np, K_pt.grad.numpy()))
    print("dV:", np.allclose(dV_np, V_pt.grad.numpy()))

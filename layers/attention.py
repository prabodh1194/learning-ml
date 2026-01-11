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

from dataclasses import dataclass

import numpy as np
import torch
import layers.softmax as s


@dataclass
class AttentionCache:
    Q: np.ndarray
    K: np.ndarray
    V: np.ndarray
    weight: np.ndarray
    softmax_cache: s.SoftmaxCache


class ScaledDotProductAttention:
    class np:
        @staticmethod
        def forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(Q.shape[-1])
            weights, softmax_cache = s.Softmax.np.forward(scores)
            out = weights @ V

            return out, AttentionCache(Q, K, V, weights, softmax_cache)

        @staticmethod
        def backward(dout, cache):
            # TODO
            pass

    class torch:
        @staticmethod
        def forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
            # TODO
            pass


if __name__ == "__main__":
    np.random.seed(42)

    B = 32
    T = 8
    C = 768

    Q = np.random.randn(B, T, C)
    K = np.random.randn(B, T, C)
    V = np.random.randn(B, T, C)

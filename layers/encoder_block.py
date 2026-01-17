"""
encoder block architecture:

1. input x
2. self attn [1_out]
3. layernorm [1_out + 2_out]
4. ffn [3_out]
5. layernorm [3_out + 4_out]

---
Architecture

Input X (B, T, C)
    │
    ├──────────────────┐
    ▼                  │
[Self-Attention]       │
    │                  │
    ▼                  │
  + ◄──────────────────┘  (residual)
    │
    ▼
[LayerNorm]
    │
    ├──────────────────┐
    ▼                  │
[FeedForward]          │
    │                  │
    ▼                  │
  + ◄──────────────────┘  (residual)
    │
    ▼
[LayerNorm]
    │
    ▼
Output (B, T, C)
---

Note: Using Post-Norm (original paper). Pre-Norm is an alternative.

"""

from dataclasses import dataclass

import numpy as np
import torch

import layers.self_attention as sa
import layers.layernorm as ln
import layers.ffn as ffn


np.random.seed(42)


@dataclass
class EncoderBlockCache:
    s_attn: sa.SelfAttention
    s_attn_cache: sa.SelfAttentionCache
    l_norm_1_cache: ln.LayerNormCache
    ffn_cache: ffn.FeedForwardCache
    l_norm_2_cache: ln.LayerNormCache


class EncoderBlock:
    class np:
        @staticmethod
        def forward(
            X: np.ndarray,
            s_attn: sa.SelfAttention,
            gamma_1: np.ndarray,
            beta_1: np.ndarray,
            w1: np.ndarray,
            b1: np.ndarray,
            w2: np.ndarray,
            b2: np.ndarray,
            gamma_2: np.ndarray,
            beta_2: np.ndarray,
        ) -> tuple[np.ndarray, EncoderBlockCache]:
            s_attn_out, s_attn_cache = s_attn.forward(X)

            l_norm_1_out, l_norm_1_cache = ln.LayerNorm.np.forward(
                X + s_attn_out, gamma_1, beta_1
            )

            ffn_out, ffn_cache = ffn.FeedForward.np.forward(
                l_norm_1_out, w1, b1, w2, b2
            )

            l_norm_2_out, l_norm_2_cache = ln.LayerNorm.np.forward(
                l_norm_1_out + ffn_out, gamma_2, beta_2
            )

            return l_norm_2_out, EncoderBlockCache(
                s_attn, s_attn_cache, l_norm_1_cache, ffn_cache, l_norm_2_cache
            )

        @staticmethod
        def backward(dout: np.ndarray, cache: EncoderBlockCache):
            l2_dout = ln.LayerNorm.np.backward(dout, cache.l_norm_2_cache)
            ffn_dout = ffn.FeedForward.np.backward(l2_dout.dX, cache.ffn_cache)
            l1_dout = ln.LayerNorm.np.backward(
                ffn_dout[0] + l2_dout.dX, cache.l_norm_1_cache
            )
            s_attn_dout = cache.s_attn.backward(l1_dout.dX, cache.s_attn_cache)

            dX = s_attn_dout[0] + l1_dout.dX

            return (
                dX,
                # LN1 grads (gamma, beta)
                l1_dout.dW,
                l1_dout.db,
                # FFN grads (dW1, db1, dW2, db2)
                ffn_dout[1],
                ffn_dout[2],
                ffn_dout[3],
                ffn_dout[4],
                # LN2 grads (gamma, beta)
                l2_dout.dW,
                l2_dout.db,
                # Self-attention grads (dQ_w, dQ_b, dK_w, dK_b, dV_w, dV_b, dW, dpe)
                *s_attn_dout[1:],
            )

    class torch:
        @staticmethod
        def forward(
            X: torch.Tensor,
            s_attn: sa.SelfAttention,
            gamma_1: torch.Tensor,
            beta_1: torch.Tensor,
            w1: torch.Tensor,
            b1: torch.Tensor,
            w2: torch.Tensor,
            b2: torch.Tensor,
            gamma_2: torch.Tensor,
            beta_2: torch.Tensor,
        ) -> tuple[torch.Tensor, tuple]:
            # Self-attention (returns output + param tensors for grad checking)
            s_attn_out, *s_attn_params = s_attn.pt_forward(X)

            # Residual + LayerNorm 1
            l_norm_1_out = ln.LayerNorm.torch.forward(X + s_attn_out, gamma_1, beta_1)

            # FFN
            ffn_out = ffn.FeedForward.torch.forward(l_norm_1_out, w1, b1, w2, b2)

            # Residual + LayerNorm 2
            l_norm_2_out = ln.LayerNorm.torch.forward(
                l_norm_1_out + ffn_out, gamma_2, beta_2
            )

            return l_norm_2_out, s_attn_params


if __name__ == "__main__":
    B, T, C = 32, 64, 512
    num_heads = 8
    s_attn = sa.SelfAttention(T, C, num_heads)

    X = np.random.randn(B, T, C)
    X_pt = torch.tensor(X, requires_grad=True)

    gamma_1 = np.random.randn(C)
    gamma_1_pt = torch.tensor(gamma_1, requires_grad=True)

    beta_1 = np.random.randn(C)
    beta_1_pt = torch.tensor(beta_1, requires_grad=True)

    w1_ffn = np.random.randn(C, 4 * C)
    b1_ffn = np.random.randn(4 * C)
    w1_ffn_pt = torch.tensor(w1_ffn, requires_grad=True)
    b1_ffn_pt = torch.tensor(b1_ffn, requires_grad=True)

    w2_ffn = np.random.randn(4 * C, C)
    b2_ffn = np.random.randn(C)
    w2_ffn_pt = torch.tensor(w2_ffn, requires_grad=True)
    b2_ffn_pt = torch.tensor(b2_ffn, requires_grad=True)

    gamma_2 = np.random.randn(C)
    gamma_2_pt = torch.tensor(gamma_2, requires_grad=True)

    beta_2 = np.random.randn(C)
    beta_2_pt = torch.tensor(beta_2, requires_grad=True)

    encoder_out_np, encoder_cache = EncoderBlock.np.forward(
        X,
        s_attn,
        gamma_1,
        beta_1,
        w1_ffn,
        b1_ffn,
        w2_ffn,
        b2_ffn,
        gamma_2,
        beta_2,
    )

    # PyTorch forward
    encoder_out_pt, s_attn_params_pt = EncoderBlock.torch.forward(
        X_pt,
        s_attn,
        gamma_1_pt,
        beta_1_pt,
        w1_ffn_pt,
        b1_ffn_pt,
        w2_ffn_pt,
        b2_ffn_pt,
        gamma_2_pt,
        beta_2_pt,
    )

    print(
        "Forward match:", np.allclose(encoder_out_np, encoder_out_pt.detach().numpy())
    )

    # Backward
    dout = np.random.randn(B, T, C)
    grads_np = EncoderBlock.np.backward(dout, encoder_cache)

    encoder_out_pt.backward(torch.tensor(dout))

    Q_w_pt, Q_b_pt, K_w_pt, K_b_pt, V_w_pt, V_b_pt, W_pt, pe_pt = s_attn_params_pt

    print("dX match:", np.allclose(grads_np[0], X_pt.grad.numpy()))
    print("dGamma1 match:", np.allclose(grads_np[1], gamma_1_pt.grad.numpy()))
    print("dBeta1 match:", np.allclose(grads_np[2], beta_1_pt.grad.numpy()))
    print("dW1_ffn match:", np.allclose(grads_np[3], w1_ffn_pt.grad.numpy()))
    print("db1_ffn match:", np.allclose(grads_np[4], b1_ffn_pt.grad.numpy()))
    print("dW2_ffn match:", np.allclose(grads_np[5], w2_ffn_pt.grad.numpy()))
    print("db2_ffn match:", np.allclose(grads_np[6], b2_ffn_pt.grad.numpy()))
    print("dGamma2 match:", np.allclose(grads_np[7], gamma_2_pt.grad.numpy()))
    print("dBeta2 match:", np.allclose(grads_np[8], beta_2_pt.grad.numpy()))
    print("dQ_w match:", np.allclose(grads_np[9], Q_w_pt.grad.numpy()))
    print("dQ_b match:", np.allclose(grads_np[10], Q_b_pt.grad.numpy()))
    print("dK_w match:", np.allclose(grads_np[11], K_w_pt.grad.numpy()))
    print("dK_b match:", np.allclose(grads_np[12], K_b_pt.grad.numpy()))
    print("dV_w match:", np.allclose(grads_np[13], V_w_pt.grad.numpy()))
    print("dV_b match:", np.allclose(grads_np[14], V_b_pt.grad.numpy()))
    print("dW match:", np.allclose(grads_np[15], W_pt.grad.numpy()))
    print("dpe match:", np.allclose(grads_np[16], pe_pt.grad.numpy()))

"""
RoPE is for position encoding. Just like sinusoidal & learned embedding.
Sinusoidal & learned embedding swear by the position index. e.g., these embedding specifically learn
that a token is at position X and some other token is at position Y.

In languages though, positions are relative, e.g. two tokens are Y - X distance apart.
RoPE looks to capture this relation.

e.g.:
- Take Q vector of token X & rotate by angle θ * x
- Take K vector of token Y & rotate by angle θ * y

Now, the ScaledDotProduct attention inherently captures relative positions now.

Cuz Q is already rotated by angle θ * x. Now the dot product rotates it to θ * y,
effectively rotating it by θ * (y - x).

- Learned & sinusoidal positions work by adding positioning embeddings into the content embedding.
  This makes this info inherently sticky due to the "mixing" op.
  ==> X + PE

- RoPE keeps the content embedding unchanged and instead adds a rotation info to it.
  This preserves content embedding as-is.
  ==> Rotate(X)

Self-Attention only focuses on Q & K for similarity matching. V is not impacted by improvements in PE,
hence RoPE is only applied on Q & K.
"""

from dataclasses import dataclass

import numpy as np
import torch

from layers.base import Layer, LayerGradients


@dataclass
class RoPECache:
    sin_theta: np.ndarray
    cos_theta: np.ndarray


class RoPE(Layer):
    class np:
        @staticmethod
        def forward(X: np.ndarray) -> tuple[np.ndarray, RoPECache]:
            B, T, C = X.shape

            # positions i & i + 1 will concern themselves with the
            # angles at ith-dim.
            theta = [10_000 ** (-2 * i / C) for i in range(C // 2)]

            m = np.arange(T)

            # m is a vector of values.
            # theta is a vector as well, but it'll be broadcast into the `m` dimension.
            angles = np.outer(m, theta)

            cos_theta = np.cos(angles)
            sin_theta = np.sin(angles)

            x_even = X[..., ::2]
            x_odd = X[..., 1::2]

            """
            For each position m and dimension pair (2i, 2i+1):

            θ_i = 10000^(-2i / C)

            [x_2i  ]      [cos(m·θ_i)  -sin(m·θ_i)] [x_2i  ]
            [      ]  =   [                       ] [      ]
            [x_2i+1]      [sin(m·θ_i)   cos(m·θ_i)] [x_2i+1]

            That's it. Each pair gets rotated by angle m · θ_i.
            """
            out_even = x_even * cos_theta - x_odd * sin_theta
            out_odd = x_even * sin_theta + x_odd * cos_theta

            out = np.empty_like(X)

            out[..., ::2] = out_even
            out[..., 1::2] = out_odd

            return out, RoPECache(sin_theta, cos_theta)

        @staticmethod
        def backward(dout: np.ndarray, cache: RoPECache) -> LayerGradients:
            x_even = dout[..., ::2]
            x_odd = dout[..., 1::2]

            '''
            x_2i contributes to the even term by cos(m_theta_i) & odd term by sin(m_theta_i);
            hence the backward is just accumulating the derivatives of these terms along with
            the respective dout gradients.
            
            since form is y = a*x; gradients flow as is scaled by the respective sin/cos terms.
            '''

            out_even = x_even * cache.cos_theta + x_odd * cache.sin_theta
            out_odd = -x_even * cache.sin_theta + x_odd * cache.cos_theta

            dX = np.empty_like(dout)
            dX[..., ::2] = out_even
            dX[..., 1::2] = out_odd

            return LayerGradients(dX)

    class torch:
        @staticmethod
        def forward(X: torch.Tensor) -> torch.Tensor:
            B, T, C = X.shape

            theta = torch.tensor([10_000 ** (-2 * i / C) for i in range(C // 2)])
            m = torch.arange(T)
            angles = torch.outer(m, theta)

            cos_theta = torch.cos(angles)
            sin_theta = torch.sin(angles)

            x_even = X[..., ::2]
            x_odd = X[..., 1::2]

            out_even = x_even * cos_theta - x_odd * sin_theta
            out_odd = x_even * sin_theta + x_odd * cos_theta

            out = torch.empty_like(X)
            out[..., ::2] = out_even
            out[..., 1::2] = out_odd

            return out


if __name__ == "__main__":
    from layers.base import test_layer_activation

    np.random.seed(42)
    B, T, C = 4, 8, 16
    X = np.random.randn(B, T, C)

    print("Testing RoPE...")
    test_layer_activation(RoPE, X)

    # Rotation should preserve vector length
    out_np, _ = RoPE.np.forward(X)
    norm_in = np.linalg.norm(X, axis=-1)
    norm_out = np.linalg.norm(out_np, axis=-1)
    assert np.allclose(norm_in, norm_out, rtol=1e-5), "Norm not preserved!"
    print("✓ Norm preserved (rotation is orthogonal)")

"""
SwiGLU FFN: A gated feedforward network that provides input-dependent
control over information flow. While attention dynamically filters
across tokens ("which positions matter?"), gating dynamically filters
across dimensions ("which features matter?"). The gate acts as a
learned, context-sensitive volume control — amplifying relevant
dimensions and suppressing irrelevant ones for each specific input.
"""

from dataclasses import dataclass

import numpy as np
import torch

from layers.base import Layer
from layers.swish import SwishCache, Swish


@dataclass
class SwiGLUCache:
    X: np.ndarray
    w_up: np.ndarray
    w_gate: np.ndarray
    y_gate: np.ndarray
    gate_act: np.ndarray
    y_up: np.ndarray
    y_preact: np.ndarray
    w_down: np.ndarray
    swish_cache: SwishCache


class SwiGLU(Layer):
    class np:
        @staticmethod
        def forward(
            X: np.ndarray, w_up: np.ndarray, w_gate: np.ndarray, w_down: np.ndarray
        ) -> tuple[np.ndarray, SwiGLUCache]:
            """
            y_up = X @ w_up

            y_gate_preact = X @ w_gate
            gate_act = swish(y_gate_preact)

            y_preact = y_up (gate) gate_act

            y = y_preact @ w_down
            """
            y_gate = X @ w_gate
            y_up = X @ w_up

            gate_act, swish_cache = Swish.np.forward(y_gate)
            y_preact = y_up * gate_act
            y = y_preact @ w_down

            return y, SwiGLUCache(
                X, w_up, w_gate, y_gate, gate_act, y_up, y_preact, w_down, swish_cache
            )

        @staticmethod
        def backward(dout: np.ndarray, cache: SwiGLUCache) -> tuple[np.ndarray, ...]:
            # backprop grad through the linear layer & obtain grads for:
            # - w_down
            # - y_preact
            dw_down = (cache.y_preact.transpose(0, 2, 1) @ dout).sum(0)
            d_y_preact = dout @ cache.w_down.T

            # backprop y_preact to the gating layer:
            # since this element wise multiply, this is simple.
            # - y_up then directly goes up to the linear layer
            # - gate_act backprops through swish layer.
            dy_up = d_y_preact * cache.gate_act
            dgate_act = d_y_preact * cache.y_up

            dX = dy_up @ cache.w_up.T
            dw_up = (cache.X.transpose(0, 2, 1) @ dy_up).sum(0)

            dy_gate_preact = Swish.np.backward(dgate_act, cache.swish_cache)
            dX += dy_gate_preact.dX @ cache.w_gate.T
            dw_gate = (cache.X.transpose(0, 2, 1) @ dy_gate_preact.dX).sum(0)

            return dw_down, dw_up, dw_gate, dX

    class torch:
        @staticmethod
        def forward(
            X: torch.Tensor,
            w_up: torch.Tensor,
            w_gate: torch.Tensor,
            w_down: torch.Tensor,
        ) -> torch.Tensor:
            y_gate = X @ w_gate
            y_up = X @ w_up
            gate_act = y_gate * torch.sigmoid(y_gate)  # swish
            y_preact = y_up * gate_act
            return y_preact @ w_down


if __name__ == "__main__":
    np.random.seed(42)
    B, T, C = 4, 8, 64

    """
    ffn uses: C -> 4C -> C; hence params are C * 4C + 4C * C = 8C^2
    swiglu ffn uses: 2(C -> 4C) -> C; hence params are 2 * (C * 4C) + 4C * C = 12C^2
    
    to reduce swiglu ffn params down 8C^2, we should do C -> 2.67C
    
    2 * (C * nC) + nC * C = 8C^2
    3nC^2 = 8C^2
    hence: n = 8/3
    """
    hidden = int(8 / 3) * C

    X = np.random.randn(B, T, C)
    w_up = np.random.randn(C, hidden)
    w_gate = np.random.randn(C, hidden)
    w_down = np.random.randn(hidden, C)

    # NumPy forward
    out_np, cache = SwiGLU.np.forward(X, w_up, w_gate, w_down)

    # PyTorch forward
    X_pt = torch.tensor(X, requires_grad=True)
    w_up_pt = torch.tensor(w_up, requires_grad=True)
    w_gate_pt = torch.tensor(w_gate, requires_grad=True)
    w_down_pt = torch.tensor(w_down, requires_grad=True)

    out_pt = SwiGLU.torch.forward(X_pt, w_up_pt, w_gate_pt, w_down_pt)

    print("Forward match:", np.allclose(out_np, out_pt.detach().numpy()))

    # Backward
    dout = np.random.randn(B, T, C)
    dw_down_np, dw_up_np, dw_gate_np, dX_np = SwiGLU.np.backward(dout, cache)

    out_pt.backward(torch.tensor(dout))

    print("dX match:", np.allclose(dX_np, X_pt.grad.numpy()))
    print("dw_up match:", np.allclose(dw_up_np, w_up_pt.grad.numpy()))
    print("dw_gate match:", np.allclose(dw_gate_np, w_gate_pt.grad.numpy()))
    print("dw_down match:", np.allclose(dw_down_np, w_down_pt.grad.numpy()))

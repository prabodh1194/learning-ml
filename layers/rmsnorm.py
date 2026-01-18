from dataclasses import dataclass

import numpy as np
import torch

from layers.base import Layer, LayerGradients


@dataclass
class RMSNormCache:
    y: np.ndarray
    X: np.ndarray
    rms_x: np.ndarray
    x_norm: np.ndarray
    gamma: np.ndarray


class RMSNorm(Layer):
    class np:
        @staticmethod
        def forward(
            X: np.ndarray, gamma: np.ndarray, eps: float = 1e-6
        ) -> tuple[np.ndarray, RMSNormCache]:
            """
            rms(x) = sqrt(mean(x^2) + eps)
            y = gamma * X / rms(X)
            """

            # X = (B, T, C)
            # np.square = (B, T, C)
            # np.mean = (B, T, 1)
            # rms_x = np.sqrt = (B, T, 1)
            rms_x = np.sqrt(np.mean(np.square(X), axis=-1, keepdims=True) + eps)

            x_norm = X * (rms_x**-1)

            # gamma = (C)
            y = gamma * x_norm

            return y, RMSNormCache(y, X, rms_x, x_norm, gamma)

        @staticmethod
        def backward(dout: np.ndarray, cache: RMSNormCache) -> LayerGradients:
            # deposit the grad across batches & sequences.
            dgamma = (dout * cache.x_norm).sum((0, 1))
            dx_norm = dout * cache.gamma

            # dX through x_norm = X / rms_x
            # Using the simplified formula:
            # dX = (1/rms) * (dx_norm - x_norm * mean(dx_norm * x_norm))
            dX = (1 / cache.rms_x) * (
                dx_norm
                - cache.x_norm * (dx_norm * cache.x_norm).mean(axis=-1, keepdims=True)
            )

            return LayerGradients(dX, dgamma)

    class torch:
        @staticmethod
        def forward(
            X: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6
        ) -> torch.Tensor:
            return torch.nn.functional.rms_norm(X, gamma.shape, gamma, eps)


if __name__ == "__main__":
    np.random.seed(42)
    B, T, C = 4, 8, 64

    X = np.random.randn(B, T, C)
    gamma = np.ones(C)  # init to ones

    # NumPy forward
    out_np, cache = RMSNorm.np.forward(X, gamma)

    # PyTorch forward
    X_pt = torch.tensor(X, requires_grad=True)
    gamma_pt = torch.tensor(gamma, requires_grad=True)

    out_pt = RMSNorm.torch.forward(X_pt, gamma_pt)

    print("Forward match:", np.allclose(out_np, out_pt.detach().numpy()))

    # Backward
    dout = np.random.randn(B, T, C)
    grads_np = RMSNorm.np.backward(dout, cache)

    out_pt.backward(torch.tensor(dout))

    print("dX match:", np.allclose(grads_np.dX, X_pt.grad.numpy()))
    print("dgamma match:", np.allclose(grads_np.dW, gamma_pt.grad.numpy()))

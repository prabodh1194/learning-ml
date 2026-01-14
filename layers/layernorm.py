from dataclasses import dataclass
import numpy as np
import torch

from layers.base import Layer, LayerGradients, test_layer_linear


@dataclass
class LayerNormCache:
    X: np.ndarray
    X_hat: np.ndarray  # normalized input
    gamma: np.ndarray
    std: np.ndarray  # sqrt(var + eps)
    eps: float


class LayerNorm(Layer):
    """
    Layer Normalization.

    Normalizes across features (last axis) for each sample independently.
    Used in transformers instead of BatchNorm.

    Formula
    -------
    μ = mean(x, axis=-1, keepdims=True)
    σ² = var(x, axis=-1, keepdims=True)
    x̂ = (x - μ) / √(σ² + ε)
    y = γ * x̂ + β

    Where γ (gamma) and β (beta) are learnable parameters.

    Backward Pass
    -------------
    Given dout = dL/dy:

    dβ = sum(dout, axis=0)                    # gradient for bias
    dγ = sum(dout * x̂, axis=0)               # gradient for scale

    For dx, the chain rule through normalization is tricky because
    μ and σ² both depend on all elements of x.

    Let N = number of features (last axis size)

    dx̂ = dout * γ

    dσ² = sum(dx̂ * (x - μ) * -0.5 * (σ² + ε)^(-3/2), axis=-1, keepdims=True)

    dμ = sum(dx̂ * -1/std, axis=-1, keepdims=True)
       + dσ² * sum(-2 * (x - μ), axis=-1, keepdims=True) / N

    dx = dx̂ / std + dσ² * 2 * (x - μ) / N + dμ / N

    Simplified form (equivalent):
        dx = (1/std) * (dx̂ - mean(dx̂) - x̂ * mean(dx̂ * x̂))
    """

    class np:
        @staticmethod
        def forward(
            X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
        ) -> tuple[np.ndarray, LayerNormCache]:
            mu = np.mean(X, axis=-1, keepdims=True)
            var = np.var(X, axis=-1, keepdims=True)
            std = np.sqrt(var + eps)

            x_hat = (X - mu) / std
            Y = gamma * x_hat + beta

            return Y, LayerNormCache(X, x_hat, gamma, std, eps)

        @staticmethod
        def backward(dout: np.ndarray, cache: LayerNormCache) -> LayerGradients:
            """
            loss is distributed equally to beta & gamma * x_hat
            in fact for a given sample, every dout linearly maps to the input x & accumulates across batches.
            """
            # Gradients for learnable parameters
            # For 3D input (B, T, C), sum over batch AND sequence
            if dout.ndim == 2:
                dbeta = dout.sum(axis=0)
                dgamma = (cache.X_hat * dout).sum(axis=0)
            else:  # 3D
                dbeta = dout.sum(axis=(0, 1))
                dgamma = (cache.X_hat * dout).sum(axis=(0, 1))

            # Gradient for input
            dX_hat = cache.gamma * dout

            # Simplified formula for dX
            # dX = (1/std) * (dX_hat - mean(dX_hat) - X_hat * mean(dX_hat * X_hat))
            dX = (1 / cache.std) * (
                dX_hat
                - dX_hat.mean(axis=-1, keepdims=True)  # mean effect
                - cache.X_hat
                * (dX_hat * cache.X_hat).mean(axis=-1, keepdims=True)  # std effect
            )

            return LayerGradients(dX, dgamma, dbeta)

    class torch:
        @staticmethod
        def forward(
            X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            return torch.nn.functional.layer_norm(X, X.shape[-1:], gamma, beta, eps)


if __name__ == "__main__":
    np.random.seed(42)
    # input: batch of 4, each with 5 features
    X = np.random.randn(4, 5)
    # learnable params: one per feature
    gamma = np.random.randn(5)
    beta = np.random.randn(5)

    print("Testing LayerNorm...")
    test_layer_linear(LayerNorm, X, gamma, beta)

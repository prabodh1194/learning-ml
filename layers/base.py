"""
Base interface for layers.

Provides:
- Layer: Protocol-based interface with np, torch nested classes
- LayerGradients: Container for gradients returned by backward pass
- test_layer: universal test driver for validating against PyTorch autograd

Key difference from loss functions:
- Forward returns (output, cache) - cache stores values needed for backward
- Backward takes (upstream_grad, cache) and returns LayerGradients
"""
from dataclasses import dataclass
from typing import Protocol, NamedTuple, Any
import numpy as np
import torch


@dataclass
class LayerGradients:
    """
    Gradients returned by backward pass.

    dX: gradient w.r.t. input (always required, for chain rule)
    dW: gradient w.r.t. weights (if layer has weights)
    db: gradient w.r.t. bias (if layer has bias)
    """
    dX: np.ndarray
    dW: np.ndarray | None = None
    db: np.ndarray | None = None


class NumpyLayerImpl(Protocol):
    """Protocol for numpy layer implementation."""

    @staticmethod
    def forward(X: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, Any]:
        """Forward pass. Returns (output, cache)."""
        ...

    @staticmethod
    def backward(dout: np.ndarray, cache: Any) -> LayerGradients:
        """Backward pass. Returns gradients."""
        ...


class TorchLayerImpl(Protocol):
    """Protocol for PyTorch implementation (forward only, autograd handles backward)."""

    @staticmethod
    def forward(X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass only - autograd computes gradients."""
        ...


class Layer(Protocol):
    """
    Protocol for layers.

    Subclasses must implement nested classes:
    - np: NumpyLayerImpl (forward, backward)
    - torch: TorchLayerImpl (forward only)
    """

    np: type[NumpyLayerImpl]
    torch: type[TorchLayerImpl]


def test_layer_linear(
    layer_cls: type[Layer],
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Test driver for layers with weights (Linear, etc).

    1. Runs forward/backward for numpy
    2. Runs torch forward + autograd
    3. Compares all gradients (dX, dW, db)
    """
    # NumPy forward/backward
    Y_np, cache = layer_cls.np.forward(X, W, b)
    dout = np.ones_like(Y_np)  # upstream gradient = 1 for testing
    grads_np = layer_cls.np.backward(dout, cache)

    print(f"NumPy output shape: {Y_np.shape}")
    print(f"NumPy dX shape: {grads_np.dX.shape}")
    print(f"NumPy dW shape: {grads_np.dW.shape}")
    print(f"NumPy db shape: {grads_np.db.shape}")
    print("=" * 40)

    # PyTorch (ground truth via autograd)
    X_pt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    W_pt = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b, dtype=torch.float64, requires_grad=True)

    Y_pt = layer_cls.torch.forward(X_pt, W_pt, b_pt)
    Y_pt.sum().backward()  # sum because dout = ones

    print(f"PyTorch output shape: {Y_pt.shape}")
    print(f"PyTorch dX shape: {X_pt.grad.shape}")
    print(f"PyTorch dW shape: {W_pt.grad.shape}")
    print(f"PyTorch db shape: {b_pt.grad.shape}")
    print("=" * 40)

    # Assertions
    assert np.allclose(grads_np.dX, X_pt.grad.numpy(), rtol=rtol, atol=atol), \
        f"dX mismatch!\n  Expected: {X_pt.grad.numpy()}\n  Got: {grads_np.dX}"
    print("✓ dX matches PyTorch!")

    assert np.allclose(grads_np.dW, W_pt.grad.numpy(), rtol=rtol, atol=atol), \
        f"dW mismatch!\n  Expected: {W_pt.grad.numpy()}\n  Got: {grads_np.dW}"
    print("✓ dW matches PyTorch!")

    assert np.allclose(grads_np.db, b_pt.grad.numpy(), rtol=rtol, atol=atol), \
        f"db mismatch!\n  Expected: {b_pt.grad.numpy()}\n  Got: {grads_np.db}"
    print("✓ db matches PyTorch!")

    # Also check forward pass
    assert np.allclose(Y_np, Y_pt.detach().numpy(), rtol=rtol, atol=atol), \
        f"Forward mismatch!"
    print("✓ Forward pass matches PyTorch!")


def test_layer_activation(
    layer_cls: type[Layer],
    X: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Test driver for activation layers (ReLU, Sigmoid, etc).

    These have no learnable parameters - only dX gradient.
    """
    # NumPy forward/backward
    Y_np, cache = layer_cls.np.forward(X)
    dout = np.ones_like(Y_np)
    grads_np = layer_cls.np.backward(dout, cache)

    print(f"NumPy output shape: {Y_np.shape}")
    print(f"NumPy dX shape: {grads_np.dX.shape}")
    print("=" * 40)

    # PyTorch (ground truth via autograd)
    X_pt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    Y_pt = layer_cls.torch.forward(X_pt)
    Y_pt.sum().backward()

    print(f"PyTorch output shape: {Y_pt.shape}")
    print(f"PyTorch dX shape: {X_pt.grad.shape}")
    print("=" * 40)

    # Assertions
    assert np.allclose(grads_np.dX, X_pt.grad.numpy(), rtol=rtol, atol=atol), \
        f"dX mismatch!\n  Expected: {X_pt.grad.numpy()}\n  Got: {grads_np.dX}"
    print("✓ dX matches PyTorch!")

    assert np.allclose(Y_np, Y_pt.detach().numpy(), rtol=rtol, atol=atol), \
        f"Forward mismatch!"
    print("✓ Forward pass matches PyTorch!")

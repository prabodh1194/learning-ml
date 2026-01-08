"""
Base interface for loss functions.

Provides:
- LossFunction: Protocol-based interface with np, mlx, torch nested classes
- test_loss: universal test driver for printing values & asserting equality
"""

from typing import Protocol
import numpy as np
import mlx.core as mx
import torch


class NumpyImpl(Protocol):
    """Protocol for numpy implementation."""

    @staticmethod
    def forward(predictions: np.ndarray, targets: np.ndarray) -> float: ...

    @staticmethod
    def backward(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray: ...


class MLXImpl(Protocol):
    """Protocol for MLX implementation."""

    @staticmethod
    def forward(predictions: mx.array, targets: mx.array) -> float: ...

    @staticmethod
    def backward(predictions: mx.array, targets: mx.array) -> mx.array: ...


class TorchImpl(Protocol):
    """Protocol for PyTorch implementation (forward only, autograd handles backward)."""

    @staticmethod
    def forward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...


class LossFunction(Protocol):
    """
    Protocol for loss functions.

    Subclasses must implement nested classes:
    - np: NumpyImpl (forward, backward)
    - mlx: MLXImpl (forward, backward)
    - torch: TorchImpl (forward only)
    """

    np: type[NumpyImpl]
    mlx: type[MLXImpl]
    torch: type[TorchImpl]


def test_loss(
    loss_cls: type[LossFunction],
    predictions: np.ndarray,
    targets: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Universal test driver for loss functions.

    1. Runs forward/backward for numpy & mlx
    2. Runs torch forward + autograd
    3. Prints all values
    4. Asserts equality
    """
    # NumPy
    loss_np = loss_cls.np.forward(predictions, targets)
    grad_np = loss_cls.np.backward(predictions, targets)
    print(f"NumPy   loss: {loss_np}")
    print(f"NumPy   grad: {grad_np}")
    print("=" * 10)

    # MLX
    pred_mlx = mx.array(predictions)
    targ_mlx = mx.array(targets)
    loss_mlx = loss_cls.mlx.forward(pred_mlx, targ_mlx)
    grad_mlx = loss_cls.mlx.backward(pred_mlx, targ_mlx)
    print(f"MLX     loss: {loss_mlx}")
    print(f"MLX     grad: {grad_mlx}")
    print("=" * 10)

    # PyTorch (ground truth via autograd)
    pred_pt = torch.tensor(predictions, dtype=torch.float64, requires_grad=True)
    targ_pt = torch.tensor(targets, dtype=torch.float64)
    loss_pt = loss_cls.torch.forward(pred_pt, targ_pt)
    loss_pt.backward()
    print(f"PyTorch loss: {loss_pt.item()}")
    print(f"PyTorch grad: {pred_pt.grad.numpy()}")
    print("=" * 10)

    # Assertions
    grad_pt = pred_pt.grad.numpy()

    assert np.allclose(
        grad_np, grad_pt, rtol=rtol, atol=atol
    ), f"NumPy grad mismatch!\n  Expected: {grad_pt}\n  Got: {grad_np}"
    print("✓ NumPy gradients match PyTorch!")

    assert np.allclose(
        np.asarray(grad_mlx), grad_pt, rtol=rtol, atol=atol
    ), f"MLX grad mismatch!\n  Expected: {grad_pt}\n  Got: {np.asarray(grad_mlx)}"
    print("✓ MLX gradients match PyTorch!")

    assert np.allclose(
        loss_np, loss_pt.item(), rtol=rtol, atol=atol
    ), f"NumPy loss mismatch!"
    assert np.allclose(
        float(loss_mlx), loss_pt.item(), rtol=rtol, atol=atol
    ), f"MLX loss mismatch!"
    print("✓ All losses match!")

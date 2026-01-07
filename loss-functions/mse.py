"""
MSE Loss - Mean Squared Error
L = (1/n) * Σ(y_pred - y_true)²
"""

import mlx.core as mx
import numpy as np
import torch

# =============================================================================
# NumPy & MLX Implementation
# =============================================================================


class _np:
    @staticmethod
    def forward(y_pred: np.ndarray, targets: np.ndarray) -> float:
        """Compute MSE loss."""
        sq_err = (y_pred - targets) ** 2
        return sq_err.mean()

    @staticmethod
    def backward(y_pred: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of MSE loss w.r.t y_pred."""
        return (2 * (y_pred - targets)) / y_pred.size


class _mlx:
    @staticmethod
    def forward(y_pred: mx.array, targets: mx.array) -> float:
        sq_err = (y_pred - targets) ** 2
        return sq_err.mean()

    @staticmethod
    def backward(y_pred: mx.array, targets: mx.array) -> mx.array:
        return (2 * (y_pred - targets)) / y_pred.size


# =============================================================================
# Test & Validation
# =============================================================================

if __name__ == "__main__":
    # Test data
    y_pred_np = np.array([1.0, 2.0, 3.0])
    targets_np = np.array([1.5, 2.0, 2.5])

    y_pred_mlx = mx.array([1.0, 2.0, 3.0])
    targets_mlx = mx.array([1.5, 2.0, 2.5])

    # NumPy forward & backward
    loss_np = _np.forward(y_pred_np, targets_np)
    grad_np = _np.backward(y_pred_np, targets_np)
    print(f"NumPy loss: {loss_np}")
    print(f"NumPy grad: {grad_np}")

    # MLX fwd & bwd
    loss_mlx = _mlx.forward(y_pred_mlx, targets_mlx)
    grad_mlx = _mlx.backward(y_pred_mlx, targets_mlx)
    print(f"mlx loss: {loss_mlx}")
    print(f"mlx grad: {grad_mlx}")

    # PyTorch validation
    y_pred_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    targets_pt = torch.tensor([1.5, 2.0, 2.5])
    loss_pt = torch.nn.functional.mse_loss(y_pred_pt, targets_pt)
    loss_pt.backward()
    print(f"PyTorch grad: {y_pred_pt.grad}")

    # Assert match
    assert np.allclose(grad_np, y_pred_pt.grad.numpy()), "np Gradients don't match!"
    print("✓ NumPy gradients match PyTorch!")

    assert np.allclose(
        np.asarray(grad_mlx), y_pred_pt.grad.numpy()
    ), "mlx Gradients don't match!"
    print("✓ MLX gradients match PyTorch!")

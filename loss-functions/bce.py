"""
BCE - Binary Cross Entropy Loss
L = -(1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
"""

import mlx.core as mx
import numpy as np
import torch


def forward(y_pred: np.ndarray | mx.array, targets: np.ndarray | mx.array) -> float:
    l = (targets * np.log(y_pred) + (1 - targets) * np.log(1 - y_pred)).sum()
    return -l / targets.size


def backward(
    y_pred: np.ndarray | mx.array, targets: np.ndarray | mx.array
) -> np.ndarray:
    g = targets / y_pred - (1 - targets) / (1 - y_pred)
    return -g / targets.size


if __name__ == "__main__":
    y_pred_np = np.array([0.9, 0.2, 0.9])
    targets_np = np.array([1.0, 1.0, 0.0])

    loss_np = forward(y_pred_np, targets_np)
    grad_np = backward(y_pred_np, targets_np)
    print(f"numpy loss: {loss_np}")
    print(f"numpy grad: {grad_np}")

    loss_mlx = forward(mx.array(y_pred_np), mx.array(targets_np))
    grad_mlx = backward(mx.array(y_pred_np), mx.array(targets_np))
    print(f"mlx loss: {loss_mlx}")
    print(f"mlx grad: {grad_mlx}")

    y_pred_pt = torch.tensor(y_pred_np, requires_grad=True)
    targets_pt = torch.tensor(targets_np)
    loss_pt = torch.nn.functional.binary_cross_entropy(y_pred_pt, targets_pt)
    loss_pt.backward()
    print(f"torch loss: {y_pred_pt.grad}")

    assert np.allclose(grad_np, y_pred_pt.grad.numpy()), "Gradient is not correct"
    print("✓ NumPy gradients match PyTorch!")

    assert np.allclose(
        np.asarray(grad_mlx), y_pred_pt.grad.numpy()
    ), "mlx Gradients don't match!"
    print("✓ MLX gradients match PyTorch!")

"""
binary hinge loss

L = (1/n) * Σ max(0, 1 - y * s)

y ∈ {-1, +1}  (true labels)
s = raw score (not probability)
"""

import numpy as np
import mlx.core as mx
import torch.nn.functional


def forward(
    y: np.ndarray | mx.array, s: np.ndarray | mx.array
) -> np.ndarray | mx.array:
    return np.maximum(0, 1 - y * s).mean()


def backward(
    y: np.ndarray | mx.array, s: np.ndarray | mx.array
) -> np.ndarray | mx.array:
    grad = -y

    return (grad * ((1 - y * s) > 0)) / y.size


if __name__ == "__main__":
    y = np.array(
        [
            +1,  # Email 0: actually spam
            -1,  # Email 1: actually not spam
            +1,  # Email 2: actually spam
            -1,  # Email 3: actually not spam
        ]
    )

    s = np.array(
        [
            +2.5,  # Email 0: strongly predicts spam → CORRECT ✅
            +0.3,  # Email 1: weakly predicts spam → WRONG ❌
            -0.5,  # Email 2: predicts not spam → WRONG ❌
            -1.8,  # Email 3: strongly predicts not spam → CORRECT ✅
        ]
    )

    loss_np = forward(y, s)
    grad_np = backward(y, s)

    print("numpy loss: ", loss_np)
    print("numpy grad: ", grad_np)

    loss_mlx = forward(mx.array(y), mx.array(s))
    grad_mlx = backward(mx.array(y), mx.array(s))

    print("mlx loss: ", loss_mlx)
    print("mlx grad: ", grad_mlx)

    y_pt = torch.tensor(y)
    s_pt = torch.tensor(s, requires_grad=True)

    loss_pt = torch.clamp(1 - y_pt * s_pt, min=0).mean()
    loss_pt.backward()

    print("torch loss: ", loss_pt.item())
    print("torch grad: ", s_pt.grad)

    assert np.allclose(grad_np, s_pt.grad.numpy()), "gradient is not correct"
    print("✓ NumPy gradients match PyTorch!")

    assert np.allclose(
        np.asarray(grad_mlx), s_pt.grad.numpy()
    ), "mlx Gradients don't match!"
    print("✓ MLX gradients match PyTorch!")

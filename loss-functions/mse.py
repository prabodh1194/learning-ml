import torch
import numpy as np

np.random.seed(42)


def forward(y_pred: np.ndarray, targets: np.ndarray) -> float:
    sq_err = (y_pred - targets) ** 2
    loss = sq_err.mean()
    return loss

def backward(y_pred: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return (2 * (y_pred - targets)) / y_pred.size

print(forward(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.5])))
n = backward(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.5]))
print(n)

y_pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
targets = torch.tensor([1.5, 2.0, 2.5], requires_grad=True)
loss = torch.nn.functional.mse_loss(y_pred, targets)

loss.backward()

print(y_pred.grad)

assert np.allclose(y_pred.grad.numpy(), n)


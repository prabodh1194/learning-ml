import numpy as np
import torch


class MOERouter:
    class np:
        @staticmethod
        def forward(X: np.ndarray): ...

        @staticmethod
        def backward(dout: np.ndarray): ...

    class torch:
        @staticmethod
        def forward(X: torch.Tensor, W: torch.Tensor):
            """
            X: (B, T, C)
            W: (N, C) -- N neurons for the number of experts, each having C number of weights
            """
            logits = X @ W.T

            return logits


if __name__ == "__main__":
    B, T, C = 4, 8, 10
    num_experts = 4

    X_np = np.random.randn(B, T, C)
    W_router_np = np.random.randn(num_experts, C)

    X_torch = torch.tensor(X_np)
    W_torch = torch.tensor(W_router_np, requires_grad=True)

    MOERouter.torch.forward(X_torch, W_torch)

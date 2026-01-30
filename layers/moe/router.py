import numpy as np
import torch
from torch import nn


class MOERouter(nn.Module):
    def __init__(self, num_experts: int, C: int):
        super().__init__()
        self.W = torch.randn(num_experts, C, requires_grad=True)

    def forward(self, X: torch.Tensor):
        """
        X: (B, T, C)
        W: (N, C) -- N neurons for the number of experts, each having C number of weights

        each token is supposed to be activated by few of the N "experts"

        """
        logits = X @ self.W.T

        scores = torch.softmax(logits, dim=-1)
        experts = scores.topk(2, dim=-1)

        return experts.values / experts.values.sum(
            dim=-1, keepdim=True
        ), experts.indices


if __name__ == "__main__":
    B, T, C = 4, 8, 10
    num_experts = 4

    X_np = np.random.randn(B, T, C)
    X_torch = torch.tensor(X_np, dtype=torch.float)

    weights, indices = MOERouter(num_experts, C).forward(X_torch)

    assert torch.isclose(weights.sum(), torch.tensor(B * T, dtype=torch.float)), (
        "all scores must sum to 1"
    )

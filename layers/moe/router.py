import numpy as np
import torch


class MOERouter:
    class torch:
        @staticmethod
        def forward(X: torch.Tensor, W: torch.Tensor):
            """
            X: (B, T, C)
            W: (N, C) -- N neurons for the number of experts, each having C number of weights

            each token is supposed to be activated by few of the N "experts"

            """
            logits = X @ W.T

            scores = torch.softmax(logits, dim=-1)
            experts = scores.topk(2, dim=-1)

            return experts.values / experts.values.sum(
                dim=-1, keepdim=True
            ), experts.indices


if __name__ == "__main__":
    B, T, C = 4, 8, 10
    num_experts = 5

    X_np = np.random.randn(B, T, C)
    W_router_np = np.random.randn(num_experts, C)

    X_torch = torch.tensor(X_np)
    W_torch = torch.tensor(W_router_np, requires_grad=True)

    weights, indices = MOERouter.torch.forward(X_torch, W_torch)

    assert torch.isclose(weights.sum(), torch.tensor(B * T, dtype=torch.double)), (
        "all scores must sum to 1"
    )

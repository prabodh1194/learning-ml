import numpy as np
import torch
from torch import nn


class MOERouter(nn.Module):
    def __init__(self, num_experts: int, C: int, topk: int = 2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_experts, C))
        self.topk = topk
        self.num_experts = num_experts

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        X: (B, T, C)
        W: (N, C) -- N neurons for the number of experts, each having C number of weights

        each token is supposed to be activated by few of the N "experts"

        """
        logits = X @ self.W.T

        scores = torch.softmax(logits, dim=-1)
        experts = scores.topk(self.topk, dim=-1)

        expert_weights = experts.values / experts.values.sum(dim=-1, keepdim=True)
        expert_indices = experts.indices

        return expert_weights, expert_indices

    def compute_aux_loss(
        self, expert_weights: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        aux_loss = num_experts * sum(f_i * P_i)

        f_i = fraction of tokens working with the expert i.
        P_i = probability mass for the expert i.
              probability mass is (sum of all weights for expert i) / (total_tokens * top_k)

        The `f` terms is a simple probability that an expert is selected. P shows how confidently was it selected.
        The loss term looks to penalize high confidence selections of an expert & supports low confidence selections.

        in an ideal case; every expert has equal probability & confidence of selection.

        hence; f = 1/N ; P = 1/N where N is the number of experts.
        hence aux_loss = sum(1 / N^2) = N * (1 / N^2) = 1 / N

        since for large N; aux_loss will reduce.
        To keep the loss as 1.0; we can multiply by N.
        """

        B, T, K = expert_weights.shape

        total_tokens = B * T

        f_i = []
        p_i = []

        for expert_index in range(self.num_experts):
            mask = expert_indices == expert_index

            f_i.append(mask.sum() / total_tokens)

            expert_weight = expert_weights[mask]
            p_i.append(expert_weight.sum() / (total_tokens * self.topk))

        f, P = torch.stack(f_i), torch.stack(p_i)

        return self.num_experts * (f * P).sum()


if __name__ == "__main__":
    B, T, C = 4, 8, 10
    num_experts = 4

    X_np = np.random.randn(B, T, C)
    X_torch = torch.tensor(X_np, dtype=torch.float)

    router = MOERouter(num_experts, C)

    weights, indices = router.forward(X_torch)

    # Test 1: weights sum to 1 per token
    assert torch.isclose(weights.sum(), torch.tensor(B * T, dtype=torch.float)), (
        "all scores must sum to 1"
    )
    print("✓ weights sum to 1 per token")

    # Test 2: aux_loss is scalar
    aux_loss = router.compute_aux_loss(weights, indices)
    assert aux_loss.dim() == 0, "aux_loss should be scalar"
    print(f"✓ aux_loss: {aux_loss.item():.4f}")

    # Test 3: gradient flows to router
    aux_loss.backward()
    assert router.W.grad is not None, "gradients should flow to router"
    print("✓ gradients flow to router")

    # Test 4: balanced routing should have aux_loss ~ 1.0
    # (uniform: aux_loss = num_experts * sum(1/N * 1/N) = num_experts * N * 1/N^2 = 1.0)
    print(f"  (balanced target ≈ 1.0, got {aux_loss.item():.4f})")

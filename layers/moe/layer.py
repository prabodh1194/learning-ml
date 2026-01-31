from pprint import pprint

import torch
from torch import nn

from layers.moe.expert import ExpertArray
from layers.moe.router import MOERouter


class MOELayer(nn.Module):
    def __init__(self, num_experts: int, dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim

        self.router = MOERouter(num_experts, dim)
        self.experts = ExpertArray(num_experts, dim)

    def forward(self, X: torch.Tensor):
        expert_weights, expert_indices = self.router.forward(X)
        B, T, C = X.shape
        output = torch.zeros(B, T, C)

        for expert_idx in range(self.num_experts):
            # solve for; did a token pick this expert in any of its k slots?
            mask = expert_indices == expert_idx  # (B, T, K);
            weights = expert_weights[mask]
            tokens_mask = mask.any(dim=-1)  # (B, T) true if a token picked this expert
            out = weights.unsqueeze(-1) * self.experts.forward(
                X[tokens_mask], expert_idx
            )
            output[tokens_mask] += out

        stats = self.get_load_balance_stats(expert_indices)

        pprint(stats)

        return output

    def get_load_balance_stats(self, expert_indices: torch.Tensor) -> dict:
        # Count how many tokens each expert got
        # Return dict with metrics
        counts = []
        for expert_idx in range(self.num_experts):
            mask = expert_indices == expert_idx
            tokens_mask = mask.any(dim=-1)
            counts.append(tokens_mask.sum().item())

        counts = torch.tensor(counts, dtype=torch.float)
        means = counts.mean()
        stds = counts.std()

        return {
            "expert_counts": counts.tolist(),
            "cv": (stds / means).item() if means > 0 else 0,
            "load_imbalance": (counts.max() / means).item() if means > 0 else 0,
        }


if __name__ == "__main__":
    B, T, C = 4, 8, 10
    layer = MOELayer(num_experts=4, dim=C)

    X = torch.randn(B, T, C, requires_grad=True)

    out = layer.forward(X)
    assert out.shape == X.shape, f"Shape mismatch: {out.shape} vs {X.shape}"
    print("Output shape:", out.shape)
    print("MoE layer working!")

    loss = out.sum()
    loss.backward()

    print("grad flows:", X.grad is not None)

from pprint import pprint

import torch
from torch import nn

from layers.moe.expert import ExpertArray, Expert
from layers.moe.router import MOERouter


class MOELayer(nn.Module):
    def __init__(self, num_shared_experts: int, num_routed_experts: int, dim: int):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.dim = dim

        self.router = MOERouter(self.num_routed_experts, dim)

        self.shared_experts = nn.ModuleList(
            list(map(lambda _: Expert(dim), range(num_shared_experts)))
        )
        self.routed_experts = ExpertArray(self.num_routed_experts, dim)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_weights, expert_indices = self.router.forward(X)
        aux_loss = self.router.compute_aux_loss(expert_weights, expert_indices)

        B, T, C = X.shape

        shared_output = sum(expert(X) for expert in self.shared_experts)
        routed_output = torch.zeros(B, T, C)

        for expert_idx in range(self.num_routed_experts):
            # solve for; did a token pick this expert in any of its k slots?
            mask = expert_indices == expert_idx  # (B, T, K);
            weights = expert_weights[mask]
            tokens_mask = mask.any(dim=-1)  # (B, T) true if a token picked this expert
            out = weights.unsqueeze(-1) * self.routed_experts.forward(
                X[tokens_mask], expert_idx
            )
            routed_output[tokens_mask] += out

        stats = self.get_load_balance_stats(expert_indices)

        pprint(stats)

        return shared_output + routed_output, aux_loss

    def get_load_balance_stats(self, expert_indices: torch.Tensor) -> dict:
        # Count how many tokens each expert got
        # Return dict with metrics
        counts = []
        for expert_idx in range(self.num_routed_experts):
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
    print("Testing Complete MoE Layer")
    print("=" * 50)

    B, T, C = 4, 16, 256
    layer = MOELayer(num_shared_experts=2, num_routed_experts=8, dim=C)

    X = torch.randn(B, T, C, requires_grad=True)
    out, aux_loss = layer.forward(X)

    print(f"  Shape preserved: {out.shape == X.shape}")
    print(f"  Output shape: {out.shape}")

    # Gradient test
    loss = out.sum() + aux_loss * 0.01

    loss.backward()
    print(f"  Gradients flow: {X.grad is not None}")

    # Check gradients reach router and experts
    has_router_grad = layer.router.W.grad is not None
    has_expert_grad = any(p.grad is not None for p in layer.routed_experts.parameters())
    has_shared_grad = all(p.grad is not None for p in layer.shared_experts.parameters())
    print(f"  Shared experts have gradients: {has_shared_grad}")

    print(f"  Router has gradients: {has_router_grad}")
    print(f"  Experts have gradients: {has_expert_grad}")

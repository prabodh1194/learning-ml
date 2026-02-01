"""
                    Token X (B, T, C)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ Shared   │    │ Shared   │    │  Router  │
        │ Expert 0 │    │ Expert 1 │    │  (topK)  │
        └────┬─────┘    └────┬─────┘    └────┬─────┘
             │               │               │
             │  ALL TOKENS   │          indices, weights
             │               │               │
             ▼               ▼               ▼
        ┌─────────────────────┐    ┌─────────────────────────────┐
        │   shared_output     │    │      Routed Dispatch        │
        │   (dense path)      │    │    (sparse, top-K only)     │
        └──────────┬──────────┘    └──────────────┬──────────────┘
                   │                              │
                   │     ┌────────────────────────┘
                   │     │
                   │     ▼
                   │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
                   │  │ E0  │ E1  │ E2  │ E3  │ E4  │ E5  │ E6  │ E7  │
                   │  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
                   │     │     │     │     │     │     │     │     │
                   │     └─────┴─────┴──┬──┴─────┴─────┴─────┴─────┘
                   │                    │
                   │                    ▼
                   │              ┌───────────┐
                   │              │  Weighted │
                   │              │   Combine │
                   │              └─────┬─────┘
                   │                    │
                   │                    ▼
                   │            routed_output
                   │              (sparse)
                   │                    │
                   └────────┬───────────┘
                            │
                            ▼
                      ┌───────────┐
                      │    ADD    │
                      └─────┬─────┘
                            │
                            ▼
                    Output (B, T, C)


SHARED PATH (Dense):              ROUTED PATH (Sparse):
─────────────────────             ─────────────────────
• ALL tokens                      • Top-K experts per token
• Always active                   • Gate-weighted outputs
• Common patterns                 • Specialized patterns
• 1-2 experts                     • 64-256 experts

"""

import torch
from torch import nn

from layers.moe.expert import ExpertArray, Expert
from layers.moe.router import MOERouter


class MOELayer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_segments: int,
        num_shared_experts: int,
        num_routed_experts: int,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.dim = dim

        self.shared_experts = nn.ModuleList(
            list(map(lambda _: Expert(self.dim), range(num_shared_experts)))
        )

        self.router = MOERouter(self.num_routed_experts, self.dim // self.num_segments)
        self.routed_experts = ExpertArray(
            self.num_routed_experts, self.dim // self.num_segments
        )

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ """
        B, T, C = X.shape

        segments = X.view(B, T, self.num_segments, -1).transpose(1, 2)

        shared_out, routed_out, aux_loss = self.segmented_forward(X, segments)

        out = shared_out + routed_out.transpose(1, 2).view(B, T, C)

        return out, aux_loss

    def segmented_forward(
        self, X: torch.Tensor, X_segment: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        X is segmented input
        """
        shared_output = sum(expert(X) for expert in self.shared_experts)

        expert_weights, expert_indices = self.router.forward(X_segment)
        aux_loss = self.router.compute_aux_loss(expert_weights, expert_indices)

        routed_output = torch.zeros_like(X_segment)

        for expert_idx in range(self.num_routed_experts):
            # solve for; did a token pick this expert in any of its k slots?
            mask = expert_indices == expert_idx  # (B, T, K);
            weights = expert_weights[mask]
            tokens_mask = mask.any(dim=-1)  # (B, T) true if a token picked this expert
            out = weights.unsqueeze(-1) * self.routed_experts.forward(
                X_segment[tokens_mask], expert_idx
            )
            routed_output[tokens_mask] += out

        stats = self.get_load_balance_stats(expert_indices)

        print(f"  Expert counts: {stats['expert_counts']}")
        print(f"  CV: {stats['cv']:.3f}, Load imbalance: {stats['load_imbalance']:.2f}")

        return shared_output, routed_output, aux_loss

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

    # Config
    B, T, C = 4, 16, 256
    num_segments = 8
    num_shared_experts = 2
    num_routed_experts = 8
    top_k = 2

    print(f"\nConfig: B={B}, T={T}, C={C}")
    print(
        f"Segments: {num_segments}, Shared experts: {num_shared_experts}, Routed experts: {num_routed_experts}"
    )
    print(f"Segment dim: {C // num_segments}, Top-K: {top_k}")

    layer = MOELayer(
        num_segments=num_segments,
        num_shared_experts=num_shared_experts,
        num_routed_experts=num_routed_experts,
        dim=C,
    )

    X = torch.randn(B, T, C, requires_grad=True)
    out, aux_loss = layer.forward(X)

    # Shape test
    print("\nShape Test:")
    print(f"  Input:  {X.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Shape preserved: {out.shape == X.shape}")

    # Load balance explanation
    total_routing_decisions = B * num_segments * T * top_k
    print("\nLoad Balance Stats:")
    print(
        f"  Total routing decisions: {B} × {num_segments} × {T} × {top_k} = {total_routing_decisions}"
    )
    print("  (CV < 0.1 = well balanced, load_imbalance = 1.0 = perfect)")

    # Gradient test
    loss = out.sum()  # + aux_loss * 0.01
    loss.backward()

    has_router_grad = layer.router.W.grad is not None
    has_expert_grad = any(p.grad is not None for p in layer.routed_experts.parameters())
    has_shared_grad = all(p.grad is not None for p in layer.shared_experts.parameters())
    has_bias_grad = layer.router.expert_bias.grad is not None

    print("\nGradient Flow:")
    print(f"  Input:          {X.grad is not None}")
    print(f"  Router:         {has_router_grad}")
    print(f"  Router bias:    {has_bias_grad}")
    print(f"  Shared experts: {has_shared_grad}")
    print(f"  Routed experts: {has_expert_grad}")

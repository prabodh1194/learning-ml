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

        return output


if __name__ == "__main__":
    B, T, C = 4, 8, 10
    layer = MOELayer(num_experts=4, dim=C)

    X = torch.randn(B, T, C)

    out = layer.forward(X)
    assert out.shape == X.shape, f"Shape mismatch: {out.shape} vs {X.shape}"
    print("Output shape:", out.shape)
    print("MoE layer working!")

import torch
from torch import nn

from llama.swiglu import SwiGLU

Expert = SwiGLU


class ExpertArray(nn.Module):
    def __init__(self, num_experts: int, dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.expert_array = nn.ModuleList()
        for i in range(num_experts):
            self.expert_array.append(Expert(dim))

    def forward(self, X: torch.Tensor, expert_idx: int) -> torch.Tensor:
        return self.expert_array[expert_idx].forward(X)


if __name__ == "__main__":
    B, T, C = 4, 8, 10
    num_experts = 4

    expert_array = ExpertArray(num_experts, C)

    assert expert_array.forward(torch.randn(B, T, C), 2).shape == (B, T, C)

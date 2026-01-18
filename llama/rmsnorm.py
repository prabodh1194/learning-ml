import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x_2 = X**2
        x_norm = torch.sqrt(torch.mean(x_2, dim=-1, keepdim=True) + self.eps)

        out = self.gamma * X / x_norm

        return out

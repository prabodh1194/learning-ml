import torch
from torch import nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        self.hidden_dim = int(8 * dim / 3)

        self.w_up = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w_gate = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w_down = nn.Linear(self.hidden_dim, self.dim, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x_gate = F.silu(self.w_gate(X))
        x_up = self.w_up(X)

        gate = x_gate * x_up

        out = self.w_down(gate)

        return out

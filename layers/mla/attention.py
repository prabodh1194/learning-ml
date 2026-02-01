import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, C: int, C_latent: int, num_heads: int):
        super().__init__()

        self.C = C
        self.C_latent = C_latent
        self.num_heads = num_heads

        self.d_head = self.C // self.num_heads
        self.d_head_latent = self.C_latent // self.num_heads

        self.WK_c = nn.Linear(self.C, self.C_latent, bias=False)
        self.WK_u = nn.Linear(self.C_latent, self.C, bias=False)

        self.WV_c = nn.Linear(self.C, self.C_latent, bias=False)
        self.WV_u = nn.Linear(self.C_latent, self.C, bias=False)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        *_, C = X.shape

        assert C == self.C, f"didn't get the full dimension {self.C}. got {C} instead."

        XK_latent = self.WK_c(X)
        K = self.WK_u(XK_latent)

        XV_latent = self.WV_c(X)
        V = self.WV_u(XV_latent)

        return K, V

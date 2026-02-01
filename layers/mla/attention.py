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

        self.W_Q = nn.Linear(self.C, self.C)

        self.WK_c = nn.Linear(self.C, self.C_latent, bias=False)
        self.WK_u = nn.Linear(self.C_latent, self.C, bias=False)

        self.WV_c = nn.Linear(self.C, self.C_latent, bias=False)
        self.WV_u = nn.Linear(self.C_latent, self.C, bias=False)

    def forward(
        self, X: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        *_, C = X.shape

        assert C == self.C, f"didn't get the full dimension {self.C}. got {C} instead."

        K_latent = self.WK_c(X)
        V_latent = self.WV_c(X)

        if cache is not None:
            k_c, v_c = cache
            K_latent = torch.cat([k_c, K_latent], dim=-2)
            V_latent = torch.cat([v_c, V_latent], dim=-2)

        K = self.WK_u(K_latent)
        V = self.WV_u(V_latent)

        return K, V, (K_latent, V_latent)

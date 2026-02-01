import math

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
        self.W_O = nn.Linear(self.C, self.C)

        self.WK_c = nn.Linear(self.C, self.C_latent, bias=False)
        self.WK_u = nn.Linear(self.C_latent, self.C, bias=False)

        self.WV_c = nn.Linear(self.C, self.C_latent, bias=False)
        self.WV_u = nn.Linear(self.C_latent, self.C, bias=False)

    def forward(
        self, X: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = X.shape

        assert C == self.C, f"didn't get the full dimension {self.C}. got {C} instead."

        K_latent = self.WK_c(X)
        V_latent = self.WV_c(X)

        if cache is not None:
            k_c, v_c = cache
            K_latent = torch.cat([k_c, K_latent], dim=-2)
            V_latent = torch.cat([v_c, V_latent], dim=-2)

        K = self.WK_u(K_latent)
        V = self.WV_u(V_latent)
        Q = self.W_Q(X)

        # split into heads
        # (B, T, C) -> (B, num_heads, T, d_head)
        Q_h = Q.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        K_h = K.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        V_h = V.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)

        # attention
        scores = Q_h @ K_h.transpose(-2, -1) / math.sqrt(self.d_head)

        if T != 1:
            mask = torch.tril(torch.ones(T, T))
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = scores.softmax(dim=-1) @ V_h
        out = attn.transpose_(1, 2).contiguous().view(B, -1, C)

        return self.W_O(out), (K_latent, V_latent)

"""
A DeepSeek block combines MLA and MoE:

Input
  │
  ├──► RMSNorm ──► MLA ──► + ◄── (residual)
  │                        │
  └────────────────────────┘
                           │
  ┌────────────────────────┘
  │
  ├──► RMSNorm ──► MoE ──► + ◄── (residual)
  │                        │
  └────────────────────────┘
                           │
                        Output

"""

import torch
from torch import nn

from layers.mla.attention import Attention
from layers.moe.layer import MOELayer
from llama.rmsnorm import RMSNorm


class DeepseekBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_latent: int,
        num_heads: int,
        context_length: int,
        num_segments: int,
        num_shared_experts: int,
        num_routed_experts: int,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(
            C=dim,
            C_latent=dim_latent,
            num_heads=num_heads,
            context_length=context_length,
        )

        self.ffn_norm = RMSNorm(dim)
        self.ffn = MOELayer(
            dim=dim,
            num_segments=num_segments,
            num_shared_experts=num_shared_experts,
            num_routed_experts=num_routed_experts,
        )

    def forward(
        self, X: torch.Tensor, cache: tuple | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        X_attn_norm = self.attn_norm(X)
        X_attn, cache = self.attn(X_attn_norm, cache)
        X = X_attn + X

        X_ffn, aux_loss = self.ffn(self.ffn_norm(X))
        X = X_ffn + X

        return X, aux_loss, cache

import torch
from torch import nn

from llama.gqa import GQA
from llama.rmsnorm import RMSNorm
from llama.rope import RoPE
from llama.swiglu import SwiGLU


class LLaMABlock(nn.Module):
    """
    Pre-norm transformer block with residual connections:

    x ──┬── attn_norm ── attn ──┬── + ──┬── ffn_norm ── ffn ──┬── + ── out
        │                       │       │                     │
        └───────────────────────┘       └─────────────────────┘
             residual                        residual
    """

    def __init__(
        self,
        *,
        dim: int,
        context_length: int,
        num_head: int,
        num_kv_head: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.dim = dim

        # rope works on Q, K, V heads whose dim is standard.
        self.rope = RoPE(dim=dim // num_head, context_length=context_length)

        self.attn_norm = RMSNorm(dim)
        self.attn = GQA(self.rope, dim, num_head, num_kv_head)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        h, cache = self.attn(self.attn_norm(x), start_pos, kv_cache)
        x = x + h

        x = x + self.ffn(self.ffn_norm(x))

        return x, cache

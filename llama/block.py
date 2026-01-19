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

    def __init__(self, dim: int, max_seq_len: int, num_head: int, num_kv_head: int):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # rope works on Q, K, V heads whose dim is standard.
        self.rope = RoPE(dim // num_head, max_seq_len)

        self.attn_norm = RMSNorm(dim)
        self.attn = GQA(self.rope, dim, num_head, num_kv_head)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

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

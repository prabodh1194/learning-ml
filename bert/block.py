import torch
from torch import nn


class BertBlock(nn.Module):
    """
    Pre-norm transformer block — same pattern as LLaMABlock:

    x ──┬── LayerNorm ── Attention ──┬── + ──┬── LayerNorm ── FFN ──┬── + ── out
        │                            │       │                      │
        └────────────────────────────┘       └──────────────────────┘
                  residual                          residual

    FFN: Linear(d_model, ffn_dim) → GELU → Linear(ffn_dim, d_model) → Dropout
    (BERT uses GELU, not SwiGLU like LLaMA)
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        # TODO: LayerNorm, BertAttention, FFN (two Linears + GELU + Dropout)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, d_model)
        Returns: (B, seq_len, d_model)
        """
        # TODO: attention with residual
        # TODO: FFN with residual
        pass

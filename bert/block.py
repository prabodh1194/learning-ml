import torch
from torch import nn

from bert.attention import BertAttention


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
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

        self.attn = BertAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, d_model)
        Returns: (B, seq_len, d_model)
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))

        return x

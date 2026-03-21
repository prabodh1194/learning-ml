import math

import torch
from torch import nn


class BertAttention(nn.Module):
    """
    Standard multi-head attention — NO causal mask.

    Compare to LLaMA's GQA (gqa.py):
      - GQA has causal mask (tril), RoPE, KV cache, grouped KV heads
      - BERT has NONE of that — just plain multi-head attention

    Every token attends to every other token equally.

    Q, K, V = W_q(x), W_k(x), W_v(x)
    scores  = (Q @ K^T) / sqrt(d_head)
    attn    = softmax(scores)
    out     = W_o(attn @ V)
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, d_model)
        Returns: (B, seq_len, d_model)

        Hint: reshape to (B, n_heads, seq_len, d_head) for the matmul
        """
        B, T, C = x.shape

        # project Q, K, V
        Q, K, V = self.w_q(x), self.w_k(x), self.w_v(x)

        # reshape to multi-head
        q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot product (NO MASK!)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

        # softmax, matmul with V
        out = scores.softmax(dim=-1) @ v

        # reshape back, project through W_o
        return self.w_o(out.transpose(1, 2).reshape(B, T, C))

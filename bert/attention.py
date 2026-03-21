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
        # TODO: store d_model, n_heads, d_head = d_model // n_heads
        # TODO: create W_q, W_k, W_v, W_o as nn.Linear
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, d_model)
        Returns: (B, seq_len, d_model)

        Hint: reshape to (B, n_heads, seq_len, d_head) for the matmul
        """
        # TODO: project Q, K, V
        # TODO: reshape to multi-head
        # TODO: scaled dot product (NO MASK!)
        # TODO: softmax, matmul with V
        # TODO: reshape back, project through W_o
        pass

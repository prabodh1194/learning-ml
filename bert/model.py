import torch
from torch import nn

from bert.embedding import BertEmbedding
from bert.block import BertBlock


class BERT(nn.Module):
    """
    tokens ─► BertEmbedding ─► [Block 0] ─► [Block 1] ─► ... ─► [Block N] ─► out
     (B,T)      (B,T,d_model)                                                (B,T,d_model)

    Compare to LLaMA:
      - No KV cache (BERT processes everything at once)
      - No LM head here (heads are separate — see head.py)
      - Embedding has token_type_ids

    Tiny config for fast iteration:
      d_model=128, n_heads=4, n_layers=4, ffn_dim=512, vocab_size=30522
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = BertEmbedding(vocab_size, d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList(
            [BertBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids:      (B, seq_len)
        token_type_ids: (B, seq_len)

        Returns: (B, seq_len, d_model) — hidden states for ALL tokens
        """
        x = self.embed(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

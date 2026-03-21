import torch
from torch import nn


class BertEmbedding(nn.Module):
    """
    Three embeddings summed together, then LayerNorm + Dropout:

    token_embed(input_ids)        ──┐
    position_embed(positions)     ──┼── sum ── LayerNorm ── Dropout ── out
    token_type_embed(token_types) ──┘

    Compare to LLaMA which only has token + position (via RoPE).
    BERT uses LEARNED position embeddings, not rotary.

    Args:
        vocab_size:   30522 for bert-base (WordPiece)
        d_model:      embedding dimension (128 for tiny, 768 for base)
        max_seq_len:  max positions (512 for bert-base)
        dropout:      dropout rate
    """

    def __init__(
        self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1
    ):
        super().__init__()
        #   token_embed:      (vocab_size, d_model)
        #   position_embed:   (max_seq_len, d_model)
        #   token_type_embed: (2, d_model)  — only 2: sentence A=0, sentence B=1
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(max_seq_len, d_model)
        self.token_type_embed = nn.Embedding(2, d_model)

        # LayerNorm(d_model) and Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids:      (B, seq_len) — token indices
        token_type_ids: (B, seq_len) — 0s and 1s (default: all zeros if None)

        Returns: (B, seq_len, d_model)
        """
        input_embedded = self.token_embed(input_ids)
        position_embedded = self.position_embed(
            torch.arange(input_embedded.size(1)).to(input_embedded.device)
        )
        token_type_embedded = self.token_type_embed(token_type_ids)

        embeds = input_embedded + position_embedded + token_type_embedded
        layer_norm = self.ln(embeds)
        dropout = self.dropout(layer_norm)

        return dropout

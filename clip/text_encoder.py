import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_head: int, hidden_dim: int):
        super().__init__()

        self.T = 77
        self.C = d_model

        self.embedding = nn.Embedding(vocab_size, self.C)
        self.position_embedding = nn.Parameter(torch.randn(1, self.T, self.C))
        self.mask = torch.nn.Transformer.generate_square_subsequent_mask(77)

        _encoder_layer = nn.TransformerEncoderLayer(
            self.C,
            nhead=n_head,
            dim_feedforward=hidden_dim,
        )
        norm = nn.LayerNorm(self.C)

        self.encoder = nn.TransformerEncoder(
            _encoder_layer,
            num_layers=12,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x - (B, T)

        """
        ViT (bidirectional):
          [CLS] patch1 patch2 patch3
            ↕     ↕      ↕      ↕       ← every token sees EVERY other token
          CLS sees everything → use CLS as the summary

        CLIP text encoder (causal):
          [BOS]  a   photo  of   a   dog  [EOS]
            →    →    →     →    →    →     →     ← each token only sees LEFT

          BOS sees: nothing
          "a" sees: BOS
          "dog" sees: BOS, a, photo, of, a, dog
          EOS sees: EVERYTHING before it         ← the ONLY token that saw it all!

        Because of the causal mask, EOS is the only token that has attended to the entire sequence. It's the natural summary point.

        CLS in bidirectional  = "I can see everything"     ← use me
        EOS in causal         = "I'm the last, I saw everything" ← use me
        BOS in causal         = "I see nothing yet"         ← useless as summary

        It's a consequence of the masking direction. If you can only look left, the rightmost token knows the most.
        """
        eos_idx = x.argmax(dim=-1)

        x = self.embedding(x)  # x - (B, T, C)
        x = x + self.position_embedding
        x = self.encoder(x, mask=self.mask)
        x = x[torch.arange(x.shape[0]), eos_idx]

        return x

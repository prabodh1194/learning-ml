"""
CLIP — Contrastive Language-Image Pretraining

Wires everything together:

  ┌──────────┐         ┌────────────┐         ┌────────────┐
  │  Image   │────────▶│ ImageEncoder│────────▶│ Projection │──▶ image_embed (B, embed_dim)
  └──────────┘         └────────────┘         └────────────┘
                                                                    │
                                                                    ▼
                                                              similarity matrix
                                                                    ▲
  ┌──────────┐         ┌────────────┐         ┌────────────┐       │
  │  Text    │────────▶│ TextEncoder │────────▶│ Projection │──▶ text_embed  (B, embed_dim)
  └──────────┘         └────────────┘         └────────────┘
"""

import torch
from torch import nn

from clip.image_encoder import ImageEncoder
from clip.text_encoder import TextEncoder
from clip.projection import Projection


class CLIP(nn.Module):
    def __init__(
        self,
        # image encoder (ViT) params
        input_C: int,
        patch_size: int,
        image_d_model: int,
        image_seq_len: int,
        image_mlp_dim: int,
        image_n_heads: int,
        image_n_layers: int,
        # text encoder params
        vocab_size: int,
        text_d_model: int,
        text_n_heads: int,
        text_mlp_dim: int,
        # shared projection dim
        embed_dim: int,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            input_C=input_C,
            P=patch_size,
            C=image_d_model,
            T=image_seq_len,
            mlp_dim=image_mlp_dim,
            n_heads=image_n_heads,
            n_layers=image_n_layers,
            num_classes=1,  # unused — ImageEncoder skips classify
        )

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=text_d_model,
            n_head=text_n_heads,
            hidden_dim=text_mlp_dim,
        )

        self.image_projection = Projection(image_d_model, embed_dim)
        self.text_projection = Projection(text_d_model, embed_dim)

    def forward(
        self, images: torch.Tensor, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embed = self.image_projection(self.image_encoder(images))
        text_embed = self.text_projection(self.text_encoder(token_ids))
        return image_embed, text_embed

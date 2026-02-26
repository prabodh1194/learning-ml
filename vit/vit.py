import torch
from torch import nn

from vit.patch_embed import extract_patches, embed


def prefix_cls(embedded_patches: torch.Tensor, cls_token: nn.Parameter) -> torch.Tensor:
    """
    Prepend a learnable [CLS] token to the patch sequence.

    (B, N, d_model) → (B, N+1, d_model)

    CLS is a single nn.Parameter(1, 1, d_model) shared across all images.
    It acts as a "blank form" — every image gets the same starting CLS,
    but attention fills it in differently based on each image's patches.

    Memory walkthrough:

        nn.Parameter at 0xAAA        ← one learnable vector
            │
            ▼ expand(B, -1, -1)      ← still 0xAAA, just a view (no copy)
            │
            ▼ torch.cat              ← COPY happens here (new memory)
            │
            ├─ image[0] CLS at 0xBBB ── attention ──▶ "cat-like" vector
            └─ image[1] CLS at 0xCCC ── attention ──▶ "car-like" vector
            │
            ▼ backward               ← gradients from all images sum back to 0xAAA

    Why shared and not per-image?
        Same reason nn.Linear weights are shared across the batch.
        CLS learns to be a good question ("what's in this image?"),
        attention produces a different answer per image.

    Why randn and not zeros?
        Zeros = symmetric init = slow to learn (no signal to differentiate patches).
        randn = breaks symmetry from step 1.
    """

    B, N, d_model = embedded_patches.shape

    return torch.cat((cls_token.expand(B, -1, -1), embedded_patches), dim=1)


def add_pos_embed(embedded_patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos_embed = nn.Parameter(
        torch.randn((1, embedded_patches.shape[1], embedded_patches.shape[2]))
    )

    return embedded_patches + pos_embed, pos_embed


def encode(
    x: torch.Tensor, n_layers: int, n_heads: int, d_model: int, hidden_dim: int
) -> torch.Tensor:
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=hidden_dim, batch_first=True
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    return encoder(x)


def classify(
    x: torch.Tensor,
    norm: nn.LayerNorm,
    head: nn.Linear,
) -> torch.Tensor:
    """
    Take the CLS token output (position 0) and classify it:

    Input:  (B, 65, d_model)
                │
                ▼ slice [:, 0, :]     ← grab just the CLS token
            (B, d_model)
                │
                ▼ LayerNorm
            (B, d_model)
                │
                ▼ Linear(d_model, 10)  ← 10 classes for CIFAR-10
            (B, 10)                    ← logits (raw scores per class)
    """

    cls = x[:, 0, :]

    return head(norm(cls))


class ViT(nn.Module):
    def __init__(
        self,
        input_C: int,
        P: int,
        C: int,
        T: int,
        mlp_dim: int,
        n_heads: int,
        n_layers: int,
        num_classes: int,
    ):
        super().__init__()
        self.P = P
        self.C = C
        self.T = T

        # the input image has a certain number of channels
        patch_C = input_C * self.P * self.P

        self.patch_embed_layer = nn.Linear(patch_C, self.C)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.C))
        self.pos_embed = nn.Parameter(torch.randn((1, self.T, self.C)))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.C, nhead=n_heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        self.classifier_norm = nn.LayerNorm(self.C)
        self.classifier_head = nn.Linear(self.C, num_classes)

    def forward(self, images: torch.Tensor):
        x = extract_patches(images, self.P)
        x = embed(x, self.patch_embed_layer)
        x = prefix_cls(x, self.cls_token)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = classify(
            x,
            self.classifier_norm,
            self.classifier_head,
        )

        return x

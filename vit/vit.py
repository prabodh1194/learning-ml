import torch
from torch import nn


def prefix_cls(embedded_patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    cls_token = nn.Parameter(torch.randn(1, 1, d_model)).expand(B, -1, -1)

    return torch.cat((cls_token, embedded_patches), dim=1), cls_token


def add_pos_embed(embedded_patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos_embed = nn.Parameter(torch.randn((1, embedded_patches.shape[1], embedded_patches.shape[2])))

    return embedded_patches + pos_embed, pos_embed

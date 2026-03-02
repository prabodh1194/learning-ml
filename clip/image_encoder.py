from typing import override

import torch
from vit.model import ViT


class ImageEncoder(ViT):
    @override
    @property
    def skip_classify(self) -> bool:
        return True

    @override
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = super().forward(images)  # (B, T, C)
        return self.classifier_norm(x[:, 0, :])  # (B, C)

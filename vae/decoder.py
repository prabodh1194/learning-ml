"""
Encoder:                          Decoder:
  image (784)                       z (2)
      │                               │
      ▼                               ▼
  Linear(784, 256) + ReLU         Linear(2, 256) + ReLU
      │                               │
      ├──▶ μ (2)                      ▼
      └──▶ log σ² (2)            Linear(256, 784) + Sigmoid
                                      │
                                      ▼
                                  image (784)   ← pixel values in [0, 1]

The whole thing together:
  image (784) ──▶ ENCODER ──▶ μ, log σ² ──▶ REPARAM ──▶ z (2) ──▶ DECODER ──▶ image (784)
                  compress        sample              decompress
                              ┌──────────┐
                  BIG → small │ bottleneck│ small → BIG
                              └──────────┘

Why Sigmoid at the end? MNIST pixels are 0-1, so the output must be clamped to that range.

"""

import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.lm_1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.lm_2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.lm_1(x))
        x = torch.sigmoid(self.lm_2(x))

        return x

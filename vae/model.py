"""
Full VAE forward pass:

    image (B, 784)
        │
        ▼
    ┌─────────┐
    │ Encoder │ → μ, log σ²
    └─────────┘
        │
        ▼
    ┌──────────────┐
    │Reparameterize│ → z = μ + σ * ε
    └──────────────┘
        │
        ▼
    ┌─────────┐
    │ Decoder │ → reconstructed image (B, 784)
    └─────────┘
"""

import torch

from vae.decoder import Decoder
from vae.encoder import Encoder
from vae.reparameterize import reparameterize


class VAE:
    def __init__(
        self, *, input_dim: int, latent_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        self.encoder = Encoder(
            input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim
        )

        self.decoder = Decoder(
            latent_dim=latent_dim, output_dim=output_dim, hidden_dim=hidden_dim
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encoder(image)
        x = reparameterize(log_var, mu)
        x = self.decoder(x)

        return x

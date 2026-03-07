"""
VAE Loss = Reconstruction Loss + β * KL Divergence

Part 1: Reconstruction (recon) Loss
  "Did you rebuild the image correctly?"
  = BCE(output_pixels, input_pixels)
  Wants: output to be IDENTICAL to input

Part 2: KL Divergence
  "Is your latent space tidy?"
  = how far is q(z|x) from N(0,1)?
  Wants: all encodings to be near a standard normal

The TENSION:

  Recon only:   perfect copies, but latent space is chaos
                → can't generate new images (holes everywhere)

  KL only:      everything maps to N(0,1), all outputs look the same
                → generates, but all images are identical blobs

  Both:         slightly blurry copies, BUT smooth latent space
                → can sample ANYWHERE and get valid images!

                recon ←──── β ────→ KL
                accurate          organized

The KL term has a closed-form solution for Gaussians (no estimation needed):

KL(q(z|x) || N(0,1)) = -0.5 * Σ(1 + log σ² - μ² - σ²)

In code:
  kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

Why does this formula work?

If μ=0 and log σ²=0 (meaning σ²=1):
  KL = -0.5 * (1 + 0 - 0 - 1) = 0     ← perfect! already N(0,1)

If μ=5 (mean far from 0):
  KL = -0.5 * (1 + 0 - 25 - 1) = 12.5  ← big penalty! come back!

If log σ²=3 (variance too wide):
  KL = -0.5 * (1 + 3 - 0 - 20) = 8.0   ← big penalty! tighten up!
"""

import torch


class VAELoss:
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = torch.nn.functional.binary_cross_entropy(
            reconstructed, original, reduction="sum"
        )
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum()

        return recon_loss + kl * self.beta

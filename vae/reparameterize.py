"""
This is the cleverest part of VAE. The problem:

Encoder outputs: μ = [0.5, -0.3],  log σ² = [-1.2, -0.8]

You WANT to sample z from this distribution: z ~ N(μ, σ²)
But sampling is RANDOM — backprop can't flow through randomness!

         μ, σ ──▶ SAMPLE z ──▶ decoder ──▶ loss
                    ↑
               random operation
               ❌ no gradient!

The trick — move the randomness to the side:

INSTEAD of:   z = sample from N(μ, σ²)         ← can't differentiate

DO THIS:      ε = sample from N(0, 1)           ← random, but NOT in the graph
              z = μ + σ * ε                     ← deterministic math! ✅

         μ ────────────┐
                       ├──▶ z = μ + σ * ε ──▶ decoder ──▶ loss
         σ ────────────┘                ↑
                                 ε ~ N(0,1)   ← external randomness
                                 (no grad needed)

Gradients flow through μ and σ just fine — they're normal multiplies and adds.
The randomness (ε) is just a fixed input, like data.

The math for getting σ from log σ²:

log σ² = -1.2
σ²     = exp(-1.2) = 0.30
σ      = exp(-1.2 / 2) = exp(0.5 * log_var) = 0.55
"""

import torch


def reparameterize(*, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # 1. compute σ from log σ²
    # 2. sample ε ~ N(0, 1) with torch.randn_like
    # 3. return μ + σ * ε
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)

    return mu + eps * std

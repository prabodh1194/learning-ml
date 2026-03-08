"""
This function computes three things:

Step 1:  β_t — linearly spaced from 0.0001 to 0.02
         "how much noise to ADD at step t"

         t=0        t=500       t=999
         0.0001 ───────────── 0.02
         tiny                  more

Step 2:  α_t = 1 - β_t
         "how much image KEEPS at step t"

         t=0        t=500       t=999
         0.9999 ───────────── 0.98

Step 3:  ᾱ_t = α_1 × α_2 × ... × α_t
         "how much image SURVIVES after ALL steps up to t"
         (cumulative product)

         t=0        t=500       t=999
         ~1.0 ────────────── ~0.0

Think of it like this:

β = "what % of the cat do I erase THIS step?"
α = "what % of the cat REMAINS after THIS step?"
ᾱ = "what % of the cat REMAINS after ALL steps so far?"
"""

import torch


def linear_noise_schedule(T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # return beta, alpha, alpha_bar
    beta = torch.linspace(1e-4, 2e-2, T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=-1)

    return beta, alpha, alpha_bar

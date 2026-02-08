"""
What is LoRA? (ELI5)

The problem: TinyLlama has 1.1 BILLION parameters. Updating all of them:
- Takes forever
- Needs tons of memory
- Easy to break the model

LoRA's Big Idea: Don't update the original weights. Add tiny "adapter" weights instead!

NORMAL FINE-TUNING:
┌─────────────────────────────────────┐
│  Original W (huge: 2048 × 2048)     │
│  = 4 million numbers to update!     │
│                                     │
│  Training updates ALL of W          │
│  → Slow, memory-hungry              │
└─────────────────────────────────────┘

LoRA FINE-TUNING:
┌─────────────────────────────────────────┐
│  Original W (FROZEN - don't touch!)     │
│           +                             │
│  A (tiny: 2048 × 8) × B (tiny: 8 × 2048)│
│  = Only 32K numbers to update!          │
│                                         │
│  Training updates ONLY A and B          │
│  → Fast, lightweight                    │
└─────────────────────────────────────────┘

The math trick:

Before LoRA:     y = Wx

After LoRA:      y = Wx + (BA)x
                     │     │
                 frozen   trainable
                          (rank r = 8)

Why does this work?
- Most "knowledge" is already in W (frozen)
- We just need small adjustments (A and B)
- Low rank (r=8) is enough for fine-tuning!

Real numbers for TinyLlama:
- Total params: 1.1B
- LoRA params: ~2.8M (only 0.25%!)
- That's like updating 1 page instead of a 400-page book
"""

import torch
from torch import nn


class LoRALinear(nn.Module):
    """
    Wraps an existing Linear layer with LoRA adapters.

    Forward: y = W @ x + (B @ A) @ x * scaling

    Only A and B are trainable. Original W is frozen.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        # A: projects down to low rank
        # B: projects back up
        # A is initialized with small random
        # B is initialized with zeros
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

        # Freeze base layer:
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)

        # original output
        base_output = self.base_layer(x)  # (B, T, out_dim)

        # LoRA adjustment: x @ A^T @ B^T * scaling
        # x @ A^T: (B, T, in_dim) @ (in_dim, rank) -> (B, T, rank)
        # ... @ B^T: (B, T, rank) @ (rank, out_dim) -> (B, T, out_dim)
        # scaling decides how much lora participates in the output
        """
          Visual:

  x ─────────────────┬──► base_layer(x) ───────────────┬──► out
  (B, T, in_dim)     │      (B, T, out_dim)            │    (B, T, out_dim)
                     │                                 │
                     └──► @ A^T ──► @ B^T ──► * scale ─┘
                          (B,T,r)   (B,T,out_dim)

  Where r = rank (e.g., 8)

        """

        lora_out = (x @ self.A.T @ self.B.T) * self.scaling  # (B, T, out_dim)

        return base_output + lora_out  # (B, T, out_dim)


if __name__ == "__main__":
    # Test LoRA layer
    base = nn.Linear(2048, 2048, bias=False)
    lora = LoRALinear(base, rank=8, alpha=16)

    # Count params
    total = sum(p.numel() for p in lora.parameters())
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable %: {100 * trainable / total:.2f}%")

    # Test forward
    x = torch.randn(2, 10, 2048)  # (B=2, T=10, dim=2048)
    y = lora(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

"""

Image (B, 3, 32, 32)          Timestep t (B,)
    │                              │
    ▼                              ▼
    │                    sinusoidal_embedding(t, 256)  → (B, 256)
    │                              │
    │                    Linear(256→256) + ReLU        → (B, 256)
    │                              │
    ▼                              │ (time travels alongside)
┌──────────────────────────────────────┐
│ Down Block 1                         │
│   Conv2d(3→64, 3×3, pad=1) + ReLU    │  (B, 64, 32, 32)
│   Conv2d(64→64, 3×3, pad=1) + ReLU   │  (B, 64, 32, 32) ─── skip1
│   MaxPool2d(2)                       │  (B, 64, 16, 16)
└──────────────────────────────────────┘
    │                                              │
    ▼                                              │
┌──────────────────────────────────────┐           │
│ Down Block 2                         │           │
│   Conv2d(64→128, 3×3, pad=1) + ReLU  │  (B,128, 16, 16)
│   Conv2d(128→128, 3×3, pad=1)+ ReLU  │  (B,128, 16, 16) ─── skip2
│   MaxPool2d(2)                       │  (B,128, 8, 8)
└──────────────────────────────────────┘
    │                                              │
    ▼                                              │
┌──────────────────────────────────────┐           │
│ Bottleneck                           │           │
│   Conv2d(128→256, 3×3, pad=1)+ ReLU  │  (B,256, 8, 8)
│   Conv2d(256→128, 3×3, pad=1)+ ReLU  │  (B,128, 8, 8)
│                                      │           │
│   + time injection:                  │           │
│     Linear(256→128)                  │  (B,128)  │
│     reshape to (B,128,1,1)           │           │
│     add to feature map               │  (B,128, 8, 8)
│     (broadcasts across all pixels)   │           │
└──────────────────────────────────────┘           │
    │                                              │
    ▼                                              │
┌──────────────────────────────────────┐           │
│ Up Block 1                           │           │
│   Upsample(scale=2)                  │  (B,128, 16, 16)
│   cat(↑, skip2) along channels       │  (B,256, 16, 16)
│   Conv2d(256→64, 3×3, pad=1) + ReLU  │  (B, 64, 16, 16)
│   Conv2d(64→64, 3×3, pad=1) + ReLU   │  (B, 64, 16, 16)
└──────────────────────────────────────┘
    │                                              │
    ▼                                              │
┌──────────────────────────────────────┐           │
│ Up Block 2                           │           │
│   Upsample(scale=2)                  │  (B, 64, 32, 32)
│   cat(↑, skip1) along channels       │  (B,128, 32, 32)
│   Conv2d(128→64, 3×3, pad=1) + ReLU  │  (B, 64, 32, 32)
│   Conv2d(64→64, 3×3, pad=1) + ReLU   │  (B, 64, 32, 32)
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ Final                                │
│   Conv2d(64→3, 1×1)                  │  (B, 3, 32, 32)
└──────────────────────────────────────┘
    │
    ▼
Predicted noise (B, 3, 32, 32)

The cat(↑, skip) is torch.cat([upsampled, skip], dim=1) — gluing channels together. That's why Up Block 1 goes
from 128 to 256 channels momentarily (128 from upsample + 128 from skip2).

The final Conv2d(64→3, 1×1) is a 1×1 conv — no spatial mixing, just maps 64 channels back to 3 (RGB noise).

What is Conv2d?

A small sliding window that scans across the image:

3×3 kernel (the "window")        slides across the image
┌───────┐
│ w w w │    →  →  →
│ w w w │    step by step
│ w w w │    across every position
└───────┘
    ↓
one output number = sum of (pixel × weight) in the window

Think of it like: "look at a 3×3 patch, decide what pattern is here."

What does a Down Block do?

Two convolutions (detect patterns) then shrink the grid:

Input: (B, in_ch, H, W)
    │
    ▼
Conv2d 3×3  →  detect patterns
    │
    ▼
Conv2d 3×3  →  detect more patterns
    │
    ▼
MaxPool2d(2) →  shrink H,W by half

Output: (B, out_ch, H/2, W/2)

In PyTorch

nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
          ^^^^^^^^^^^  ^^^^^^^^^^^^
          how many ch  how many ch
          coming IN    going OUT

  padding=1 keeps H,W the same after conv
  (without padding, 3×3 kernel would shrink by 2 pixels)

nn.MaxPool2d(2)
  takes every 2×2 block → keeps the max → halves H and W

### how is the time embedding used?

  t = 500
    │
    ▼
  sinusoidal_embedding     ← FIXED (no learnable params, just math)
    │                         sin/cos at different frequencies
    │                         same input always gives same output
    ▼
  (B, 256) vector
    │
    ▼
  Linear + ReLU            ← LEARNED (backprop updates these weights)
    │                         learns "what about the timestep matters
    ▼                          for denoising?"
  (B, 256) vector
    │
    ▼
  time_proj Linear         ← LEARNED (projects to 128 channels)
    │
    ▼
  add into bottleneck

  The sinusoidal part just gives each timestep a unique, fixed fingerprint. The MLP on top learns what to do with it.

  Same split as in your transformer — positional encoding was fixed sin/cos, and the network learned how to use it.

### about skips & convs

- Bigger models can have more convs / block. every layer conv can detect finer detail
- skip is just defined as the layer penultimate to maxpool.

You can stack as many convs as you want before the pool and use the last one as the skip:

  conv1 → relu
  conv2 → relu
  conv3 → relu
  conv4 → relu → skip
  MaxPool → next block

  More convs = the network detects more complex patterns at that resolution before moving on. But the tradeoff:

  More convs per block:
    + richer features
    - more parameters
    - slower training
    - risk of overfitting on small datasets

  For CIFAR-10 (32×32, 50k images), 2 convs per block is the sweet spot. Bigger models (like for 256×256 images) often use 3-4 convs per block because they have more data and more spatial detail to process.

### scaling law

More resolution = more levels to the U:

  Our tiny U-Net (32×32 CIFAR-10):
    32 → 16 → 8              2 down blocks, 2 up blocks

  Medium U-Net (256×256):
    256 → 128 → 64 → 32 → 16     4 down blocks, 4 up blocks

  Large U-Net (512×512):
    512 → 256 → 128 → 64 → 32 → 16    5 down blocks, 5 up blocks

  Each level halves the spatial size, so you need enough levels to compress the image down to a small bottleneck. And at each level, they also use more channels:

  Ours:       3 → 64 → 128          (tiny)
  Stable Diffusion:  320 → 640 → 1280 → 1280   (massive)

  More blocks + more channels + more convs per block = bigger model = better quality but way more compute.

"""

import torch
from torch import nn

from diffusion.time_embedding import sinusoidal_embedding


class DownBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))  # skip
        return self.max_pool(x), x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, *, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: B, C, H, W
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


class UNet(nn.Module):
    def __init__(self, *, in_ch: int = 3, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim

        # time embedding: sinusoidal → Linear → ReLU → Linear → (B, 128)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.ReLU(),
            nn.Linear(self.time_dim, 128),
        )

        # down1: DownBlock(3, 64)
        # down2: DownBlock(64, 128)
        self.down1 = DownBlock(in_channels=3, out_channels=64)
        self.down2 = DownBlock(in_channels=64, out_channels=128)

        # bottleneck: Conv2d(128, 256, 3, pad=1), Conv2d(256, 128, 3, pad=1)
        self.bot1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bot2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # up1: UpBlock(256, 64)    ← 128 + 128(skip2) = 256
        # up2: UpBlock(128, 64)    ← 64 + 64(skip1) = 128
        self.up1 = UpBlock(256, 64)
        self.up2 = UpBlock(128, 64)

        # final: Conv2d(64, 3, kernel_size=1)
        self.final = nn.Conv2d(64, in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1. time embedding
        t_emb = sinusoidal_embedding(t, self.time_dim)  # (B, 256)
        t_emb = self.time_mlp(t_emb)  # (B, 128)

        # 2. down path:
        x, skip1 = self.down1(x)  # (B, 64, 16, 16), skip1=(B, 64, 32, 32)
        x, skip2 = self.down2(x)  # (B, 128, 8, 8),  skip2=(B, 128, 16, 16)

        #
        # 3. bottleneck + inject time
        x = torch.relu(self.bot1(x))  # (B, 256, 8, 8)
        x = torch.relu(self.bot2(x))  # (B, 128, 8, 8)
        x = x + t_emb[:, :, None, None]  # broadcasts over H, W

        # 4. up path
        x = self.up1(x=x, skip=skip2)  # (B, 64, 16, 16)
        x = self.up2(x=x, skip=skip1)  # (B, 64, 32, 32)

        # 5. final conv
        return self.final(x)  # (B, 3, 32, 32)

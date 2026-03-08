"""
Image (B, 3, 32, 32)
    │
    ▼
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

"""

import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
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

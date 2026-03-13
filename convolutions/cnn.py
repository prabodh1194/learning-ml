import torch
from torch import nn

"""
arch:

Input: (B, 1, 28, 28)     ← batch of MNIST images
           │
      Conv2d(1→32, 3x3, pad=1) + BatchNorm + ReLU
           │
      (B, 32, 28, 28)        ← 32 feature maps, same spatial size
           │
      Conv2d(32→64, 3x3, pad=1) + BatchNorm + ReLU
           │
      (B, 64, 28, 28)        ← 64 feature maps
           │
      MaxPool2d(2)            ← halve spatial size
           │
      (B, 64, 14, 14)
           │
      Conv2d(64→128, 3x3, pad=1) + BatchNorm + ReLU
           │
      (B, 128, 14, 14)       ← 128 feature maps
           │
      MaxPool2d(2)            ← halve again
           │
      (B, 128, 7, 7)
           │
      Flatten                 ← 128 * 7 * 7 = 6272
           │
      Linear(6272→10)         ← 10 digit classes

  What's new here

  - BatchNorm2d: normalizes each channel to mean=0, std=1. Helps training converge faster. Goes between Conv and ReLU.
  - MaxPool2d(2): takes the max of each 2×2 block → halves the size. Simpler alternative to stride=2 conv.
  - Flatten: squash the 3D feature maps into 1D so a Linear layer can classify.
"""


class MNIST(nn.Module):
    def __init__(self, image_size=28):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.batch_norm_1 = nn.BatchNorm2d(self.conv_1.out_channels)

        self.conv_2 = nn.Conv2d(
            in_channels=self.conv_1.out_channels,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.batch_norm_2 = nn.BatchNorm2d(self.conv_2.out_channels)

        self.max_pool_3 = nn.MaxPool2d(2)

        self.conv_4 = nn.Conv2d(
            in_channels=self.conv_2.out_channels,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.batch_norm_4 = nn.BatchNorm2d(self.conv_4.out_channels)

        self.max_pool_5 = nn.MaxPool2d(2)

        # 2 maxpools halve the size of the edge twice; hence edge goes from 28 -> 7.
        # hence pixels reduce from (28 * 28) -> (7 * 7)
        _flatten_size = self.conv_4.out_channels * (image_size // (2 * 2)) ** 2
        self.lin_6 = nn.Linear(_flatten_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.batch_norm_1(self.conv_1(x)))
        x = torch.relu(self.batch_norm_2(self.conv_2(x)))
        x = self.max_pool_3(x)
        x = torch.relu(self.batch_norm_4(self.conv_4(x)))
        x = self.max_pool_5(x)
        x = torch.flatten(x, 1)
        x = self.lin_6(x)

        return x

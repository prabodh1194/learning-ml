import torch
from torch import nn

from diffusion.time_embedding import sinusoidal_embedding
from diffusion.unet import DownBlock, UpBlock


class UNet(nn.Module):
    def __init__(self, *, in_ch: int = 4, time_dim: int = 256):
        super().__init__()

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.ReLU(),
            nn.Linear(self.time_dim, 128),
        )

        self.down1 = DownBlock(in_channels=in_ch, out_channels=64)

        self.bot1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bot2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.up1 = UpBlock(128, 64)

        self.final = nn.Conv2d(64, in_ch, kernel_size=3, padding=1)

    def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1. time embedding
        t_emb = sinusoidal_embedding(t, self.time_dim)  # (B, 256)
        t_emb = self.time_mlp(t_emb)  # (B, 128)

        x, skip1 = self.down1(x)

        x = torch.relu(self.bot1(x))
        x = torch.relu(self.bot2(x))
        x = x + t_emb[:, :, None, None]
        x = self.up1(x=x, skip=skip1)

        x = self.final(x)

        return x

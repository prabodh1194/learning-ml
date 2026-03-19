import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # (C=3, H, W) -> (C=32, H'=H//2, W'=W//2)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
        )

        # (C=32, H', W') -> (C=64, H'//2, W'//2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        )

        self.conv_mu = nn.Conv2d(
            in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1
        )
        self.conv_log_var = nn.Conv2d(
            in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        return mu, log_var

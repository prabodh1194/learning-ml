import torch
from torch import nn
from torch.nn import functional as F

from vae.reparameterize import reparameterize


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


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # (C=4, H, W) -> (C=64, H'=2 * H, W'=2 * W)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=4,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        # (C=64, H', W') -> (C=32, H''=2 * H', W'' = 2 * W')
        self.conv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))

        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(image)
        x = reparameterize(mu=mu, log_var=log_var)
        return self.decoder(x), mu, log_var

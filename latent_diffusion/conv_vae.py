import torch
from torch import nn
from torch.nn import functional as F

from vae.reparameterize import reparameterize


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # (3, 32, 32) → process at 32x32
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(64)
        # downsample → (64, 16, 16)
        self.down1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # process at 16x16
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)
        # downsample → (128, 8, 8)
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        # process at 8x8
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)

        # parallel heads: (256, 8, 8) → (8, 8, 8)
        self.conv_mu = nn.Conv2d(256, 8, kernel_size=3, padding=1)
        self.conv_log_var = nn.Conv2d(256, 8, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1a(self.conv1a(image)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.down1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.down2(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))

        return self.conv_mu(x), self.conv_log_var(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # (8, 8, 8) → process at 8x8
        self.conv1a = nn.Conv2d(8, 256, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(256)
        self.conv1b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(256)

        # upsample → (256, 16, 16)
        self.up1 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # process at 16x16
        self.conv2a = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(128)

        # upsample → (128, 32, 32)
        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # process at 32x32
        self.conv3a = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(64)

        # final → (3, 32, 32)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1a(self.conv1a(z)))
        x = F.relu(self.bn1b(self.conv1b(x)))

        x = self.up1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))

        x = self.up2(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))

        return torch.tanh(self.conv_out(x))


class ConvVAE(nn.Module):
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


def conv_vae_loss(
    *,
    orig_image: torch.Tensor,
    cons_image: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    recon_loss = F.mse_loss(cons_image, orig_image)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_loss

import torch
from torch import nn
from torch.nn import functional as F

from vae.reparameterize import reparameterize


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # (3, 32, 32) -> (64, 16, 16)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        # (64, 16, 16) -> (128, 8, 8)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # (128, 8, 8) -> (256, 4, 4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # parallel heads: (256, 4, 4) -> (4, 4, 4)
        self.conv_mu = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv_log_var = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # (4, 4, 4) -> (256, 8, 8)
        self.conv1 = nn.ConvTranspose2d(
            4, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # (256, 8, 8) -> (128, 16, 16)
        self.conv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # (128, 16, 16) -> (64, 32, 32)
        self.conv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # (64, 32, 32) -> (3, 32, 32)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.tanh(self.conv_out(x))

        return x


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
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_loss

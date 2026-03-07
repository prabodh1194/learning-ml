"""
Generate new digits from a trained VAE checkpoint.

Usage:
  PYTHONPATH=$PWD uv run python vae/generate.py vae/checkpoints/vae_final.pt
"""

import sys
import logging

import torch
from torchvision.utils import save_image

from vae.model import VAE

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def generate(checkpoint_path: str, n: int = 16):
    device = "mps"

    model = VAE(
        input_dim=784,
        latent_dim=2,
        hidden_dim=256,
        output_dim=784,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        z = torch.randn(n, 2).to(device)
        generated = model.decoder(z).view(-1, 1, 28, 28)

    out_path = "vae/outputs/generated.png"
    save_image(generated, out_path, nrow=4)
    log.info(f"saved {n} generated digits → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: uv run python vae/generate.py <checkpoint_path>")
        sys.exit(1)

    generate(sys.argv[1])

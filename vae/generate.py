"""
Generate new digits from a trained VAE checkpoint.

Usage:
  PYTHONPATH=$PWD uv run python vae/generate.py vae/checkpoints/vae_final.pt

⏺ The key insight: after training, you only need the decoder.

  During TRAINING:
    real image → encoder → μ, log σ² → reparameterize → z → decoder → reconstructed image
    \_________________________________________/          \_________________________/
          "learn what z should look like"                "learn to draw from z"

  During GENERATION:
    z ~ N(0,1) → decoder → NEW image
    \_________/   \____________________/
    random noise   decoder already knows
                   how to draw from z!

  Why does this work? Because of the KL term in the loss:

  KL forces the encoder to output distributions close to N(0, 1)

  So during training, the decoder learned to handle z values
  that look like N(0, 1) samples.

  After training:
    - The decoder has seen thousands of z values ~ N(0, 1)
    - It learned: z near [-2, 1] → draw a "7"
    - It learned: z near [1, -1] → draw a "0"
    - etc.

  So if you just SAMPLE from N(0, 1) and feed it to the decoder,
  it already knows what to do!

  Look at your latent space plot — it confirms this:

                z[1]
                 ↑
            9    |    7
         5       |
      ───────────┼──────────→ z[0]
            4    |
         3    6  |  0   1
                 |    2

  When you sample z = [-2, 1], the decoder sees
  "this is where 9s live" → draws a 9

  When you sample z = [1, -1], the decoder sees
  "this is where 0s live" → draws a 0

  Without the KL term, the encoder could map digits to arbitrary locations (like μ=500 for "3" and μ=-1000 for "7"). Then sampling from N(0,1) would land you in empty space — the decoder would never have seen
  those z values during training.

  Without KL:  encoder maps to random locations
               N(0,1) samples land in HOLES → garbage output

  With KL:     encoder maps near N(0,1)
               N(0,1) samples land where training data lives → valid digits!

  That's the whole trick — KL makes the latent space match the distribution you'll sample from at generation time.
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

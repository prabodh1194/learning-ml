"""
Training loop for DDPM on CIFAR-10.

The training step is dead simple:

  1. grab a clean image
  2. pick a random timestep t
  3. add noise with forward_diffusion  →  get (x_t, actual_noise)
  4. ask U-Net: "predict the noise"    →  get predicted_noise
  5. loss = MSE(predicted_noise, actual_noise)
  6. backprop

That's it. The model learns to predict what noise was added.

Usage:
  PYTHONPATH=$PWD uv run python diffusion/train.py
"""

import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATASETS_CACHE
from diffusion.forward import forward_diffusion
from diffusion.noise_schedule import linear_noise_schedule
from diffusion.unet import UNet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def train():
    device = "mps"
    T = 1000
    B = 128

    # noise schedule — precompute once
    beta, alpha, alpha_bar = linear_noise_schedule(T)
    alpha_bar = alpha_bar.to(device)

    # data — CIFAR-10, normalized to [0, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] → [-1,1]
        ]
    )
    train_set = datasets.CIFAR10(
        root=DATASETS_CACHE, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=B, shuffle=True)
    total_batches = len(train_loader)

    # model
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("diffusion/outputs", exist_ok=True)
    os.makedirs("diffusion/checkpoints", exist_ok=True)

    log_file = open("diffusion/training_log.csv", "w")
    log_file.write("step,epoch,timestamp,loss\n")

    for epoch in range(50):
        model.train()
        for step, (images, _) in enumerate(train_loader):
            images = images.to(device)  # (B, 3, 32, 32)

            # 1. pick random timesteps for each image in the batch
            t = torch.randint(0, T, (images.shape[0],), device=device)

            # 2. add noise
            x_t, noise = forward_diffusion(x_0=images, t=t, alpha_bar=alpha_bar)

            # 3. predict the noise
            predicted_noise = model(x_t, t.float())

            # 4. loss = how wrong was the prediction?
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            # 5. backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            now = datetime.now().strftime("%H:%M:%S")

            log_file.write(f"{step},{epoch},{now},{loss.item():.6f}\n")
            log_file.flush()

            if step % 100 == 0:
                log.info(
                    f"[{now}] epoch {epoch} | step {step}/{total_batches} | "
                    f"loss: {loss.item():.4f}"
                )

            global_step = epoch * total_batches + step
            if global_step > 0 and global_step % 300 == 0:
                ckpt_path = f"diffusion/checkpoints/ddpm_step_{global_step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": step,
                    },
                    ckpt_path,
                )
                log.info(f"checkpoint saved: {ckpt_path}")

        log.info(f"epoch {epoch} done | loss: {loss.item():.4f}")

    log_file.close()

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 49,
            "step": total_batches - 1,
        },
        "diffusion/checkpoints/ddpm_final.pt",
    )
    log.info("final checkpoint saved: diffusion/checkpoints/ddpm_final.pt")
    log.info("Training complete!")


if __name__ == "__main__":
    train()

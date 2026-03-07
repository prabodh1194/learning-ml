"""
Training loop for VAE on MNIST.

After training:
  1. Generate 16 new digits from pure noise
  2. Plot the 2D latent space colored by digit class
"""

import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from vae.model import VAE
from vae.loss import VAELoss

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def generate_digits(model: VAE, device: str):
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, 2).to(device)
        generated = model.decoder(z).view(-1, 1, 28, 28)
        save_image(generated, "vae/outputs/generated.png", nrow=4)
        log.info("saved generated digits → vae/outputs/generated.png")


def plot_latent_space(model: VAE, device: str):
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(
        root="./datasets_cache", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    all_mu, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 784).to(device)
            mu, _ = model.encoder(images)
            all_mu.append(mu.cpu())
            all_labels.append(labels)

    all_mu = torch.cat(all_mu).numpy()
    all_labels = torch.cat(all_labels).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap="tab10", s=1, alpha=0.5
    )
    plt.colorbar(scatter, label="digit")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("VAE Latent Space — MNIST")
    plt.savefig("vae/outputs/latent_space.png", dpi=150)
    log.info("saved latent space plot → vae/outputs/latent_space.png")


def train():
    device = "mps"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 255] → [0, 1]
        ]
    )

    train_set = datasets.MNIST(
        root="./datasets_cache", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    total_batches = len(train_loader)

    model = VAE(
        input_dim=784,  # 28 * 28
        latent_dim=2,  # 2D for visualization
        hidden_dim=256,
        output_dim=784,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = VAELoss(beta=1.0)

    os.makedirs("vae/outputs", exist_ok=True)
    os.makedirs("vae/checkpoints", exist_ok=True)

    log_file = open("vae/training_log.csv", "w")
    log_file.write("step,epoch,timestamp,loss,recon_loss,kl_loss\n")

    for epoch in range(20):
        for step, (images, _) in enumerate(train_loader):
            images = images.view(-1, 784).to(
                device
            )  # flatten (B, 1, 28, 28) → (B, 784)

            reconstructed, mu, log_var = model(images)
            loss = loss_fn.forward(reconstructed, images, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            now = datetime.now().strftime("%H:%M:%S")

            # log individual loss components
            with torch.no_grad():
                recon = torch.nn.functional.binary_cross_entropy(
                    reconstructed, images, reduction="sum"
                ).item()
                kl = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum()).item()

            log_file.write(
                f"{step},{epoch},{now},{loss.item():.4f},{recon:.4f},{kl:.4f}\n"
            )
            log_file.flush()

            if step % 100 == 0:
                log.info(
                    f"[{now}] epoch {epoch} | step {step}/{total_batches} | "
                    f"loss: {loss.item():.1f} | recon: {recon:.1f} | kl: {kl:.1f}"
                )

            global_step = epoch * total_batches + step
            if global_step > 0 and global_step % 300 == 0:
                ckpt_path = f"vae/checkpoints/vae_step_{global_step}.pt"
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

        log.info(f"epoch {epoch} done | loss: {loss.item():.1f}")

    log_file.close()

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 19,
            "step": total_batches - 1,
        },
        "vae/checkpoints/vae_final.pt",
    )
    log.info("final checkpoint saved: vae/checkpoints/vae_final.pt")

    generate_digits(model, device)
    plot_latent_space(model, device)

    log.info("Training complete!")


if __name__ == "__main__":
    train()

import logging
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from vit.model import ViT

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def train():
    device = "mps"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # [0,255] → [0,1] float
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                # → [-1, 1] range
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./datasets_cache", train=True, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    total_batches = len(train_loader)

    model = ViT(
        input_C=3,  # RGB
        P=4,  # patch size
        C=64,  # embedding dim
        T=65,  # 64 patches + 1 CLS
        mlp_dim=256,  # FFN expansion
        n_heads=4,
        n_layers=4,
        num_classes=10,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    log_file = open("training_log.csv", "w")
    log_file.write("step,epoch,timestamp,loss,accuracy\n")

    for epoch in range(20):
        correct, total = 0, 0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            acc = correct / total

            now = datetime.now().strftime("%H:%M:%S")

            log_file.write(f"{step},{epoch},{now},{loss.item():.4f},{acc:.4f}\n")
            log_file.flush()

            if step % 50 == 0:
                log.info(
                    f"[{now}] epoch {epoch} | step {step}/{total_batches} | loss: {loss.item():.4f} | acc: {acc:.4f}"
                )

        log.info(f"epoch {epoch} done | accuracy: {100 * correct / total:.2f}%")

    log_file.close()
    log.info("Training complete!")


if __name__ == "__main__":
    train()

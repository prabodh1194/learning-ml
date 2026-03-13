import logging
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from config import DATASETS_CACHE
from convolutions.cnn import MNIST

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def train():
    device = "mps"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_set = torchvision.datasets.MNIST(
        root=DATASETS_CACHE, train=True, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    total_batches = len(train_loader)

    model = MNIST().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    log_file = open("training_log.csv", "w")
    log_file.write("step,epoch,timestamp,loss,accuracy\n")

    for epoch in range(10):
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

            if step % 100 == 0:
                log.info(
                    f"[{now}] epoch {epoch} | step {step}/{total_batches} | loss: {loss.item():.4f} | acc: {acc:.4f}"
                )

        epoch_acc = 100 * correct / total
        log.info(f"epoch {epoch} done | accuracy: {epoch_acc:.2f}%")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "accuracy": epoch_acc,
            },
            f"convolutions/checkpoints/cnn_epoch_{epoch}.pt",
        )
        log.info(f"checkpoint saved: convolutions/checkpoints/cnn_epoch_{epoch}.pt")

    log_file.close()
    log.info("Training complete!")


if __name__ == "__main__":
    train()

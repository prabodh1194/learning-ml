"""
Training loop for CLIP on Flickr30k.

Uses HuggingFace CLIPTokenizer for text tokenization (BPE, same family as GPT).
Flickr30k: 30k images, 5 captions each — small enough to iterate fast on M3.
"""

import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from datasets import load_dataset

from clip.model import CLIP
from clip.loss import ClipLoss

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class Flickr30kDataset(Dataset):
    """
    Wraps HuggingFace Flickr30k into a PyTorch Dataset.
    Each item returns (image_tensor, token_ids).
    Picks one random caption per image (out of 5).
    """

    def __init__(self, split: str = "test"):
        self.ds = load_dataset("nlphuji/flickr30k", split=split)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = self.transform(item["image"].convert("RGB"))

        # pick first caption (out of 5)
        caption = item["caption"][0]
        tokens = self.tokenizer(
            caption,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        token_ids = tokens["input_ids"].squeeze(0)  # (77,)

        return image, token_ids


def train():
    device = "mps"

    dataset = Flickr30kDataset()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    total_batches = len(train_loader)

    # small config for toy training on M3
    model = CLIP(
        input_C=3,
        patch_size=4,
        image_d_model=64,
        image_seq_len=65,  # (32/4)^2 + 1 CLS = 65
        image_mlp_dim=256,
        image_n_heads=4,
        image_n_layers=4,
        vocab_size=49152,
        text_d_model=64,
        text_n_heads=4,
        text_mlp_dim=256,
        embed_dim=64,
    ).to(device)

    loss_fn = ClipLoss().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()), lr=3e-4
    )

    log_file = open("clip/training_log.csv", "w")
    log_file.write("step,epoch,timestamp,loss,diag_sim,offdiag_sim\n")

    os.makedirs("clip/checkpoints", exist_ok=True)

    for epoch in range(20):
        for step, (images, token_ids) in enumerate(train_loader):
            images, token_ids = images.to(device), token_ids.to(device)

            image_embed, text_embed = model(images, token_ids)
            loss = loss_fn(image_embed, text_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log similarity stats
            with torch.no_grad():
                sim = image_embed @ text_embed.T
                diag = sim.diag().mean().item()
                offdiag = (sim.sum() - sim.diag().sum()) / (
                    sim.shape[0] * (sim.shape[0] - 1)
                )
                offdiag = offdiag.item()

            now = datetime.now().strftime("%H:%M:%S")
            log_file.write(
                f"{step},{epoch},{now},{loss.item():.4f},{diag:.4f},{offdiag:.4f}\n"
            )
            log_file.flush()

            if step % 10 == 0:
                log.info(
                    f"[{now}] epoch {epoch} | step {step}/{total_batches} | "
                    f"loss: {loss.item():.4f} | diag_sim: {diag:.4f} | offdiag_sim: {offdiag:.4f}"
                )

            global_step = epoch * total_batches + step
            if global_step > 0 and global_step % 300 == 0:
                ckpt_path = f"clip/checkpoints/clip_step_{global_step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "loss_fn": loss_fn.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": step,
                    },
                    ckpt_path,
                )
                log.info(f"checkpoint saved: {ckpt_path}")

        log.info(f"epoch {epoch} done | loss: {loss.item():.4f}")

    log_file.close()
    log.info("Training complete!")


if __name__ == "__main__":
    train()

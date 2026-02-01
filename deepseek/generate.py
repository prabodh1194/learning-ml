"""
Train DeepSeek on TinyShakespeare and generate text.
"""

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from deepseek.model import Deepseek

# --- Config ---
DATA_PATH = "../data/tinyshakespeare/input.txt"
CONTEXT_LEN = 64
BATCH_SIZE = 256
EPOCHS = 3
LR = 3e-4
AUX_LOSS_WEIGHT = 0.01
DEVICE = "mps"


# --- Dataset (from LLaMA) ---
class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int = 512):
        self.text = text
        self.block_size = block_size

        self.chars = sorted(set(text))

        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        self.data = [self.stoi[c] for c in self.text]
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.itos[i] for i in indices)


# --- Training ---
def train(model, dataset, device, epochs=3, batch_size=64, lr=3e-4, aux_weight=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_batches = len(loader)

    print(f"Training on {device}: {total_batches} batches/epoch")

    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(loader):
            batch_start = time.time()
            x, y = x.to(device), y.to(device)

            logits, aux_loss, _ = model(x, caches=None)
            ce_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss = ce_loss + aux_weight * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_time = time.time() - batch_start

            if batch_idx % 10 == 0:
                print(
                    f"[{epoch + 1}] [{batch_idx + 1}/{total_batches}] "
                    f"loss={loss.item():.3f} ce={ce_loss.item():.3f} aux={aux_loss.item():.3f} "
                    f"| {batch_time * 1000:.0f}ms"
                )

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}, loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")

    total_time = time.time() - train_start
    print(f"Training complete in {total_time:.2f}s")


# --- Generation ---
@torch.no_grad()
def generate(model, dataset, prompt, max_tokens=200):
    model.eval()
    tokens = dataset.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    logits, _, caches = model(tokens, caches=None)

    for _ in range(max_tokens):
        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        tokens = torch.cat([tokens, next_token], dim=1)
        logits, _, caches = model(next_token, caches=caches)

    return dataset.decode(tokens[0].tolist())


if __name__ == "__main__":
    with open(DATA_PATH, "r") as f:
        text = f.read()

    dataset = CharDataset(text, block_size=CONTEXT_LEN)

    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Training on {len(dataset)} samples")
    print(f"Device: {DEVICE}")
    print()

    model = Deepseek(
        vocab_size=dataset.vocab_size,
        dim=64,
        num_layers=4,
        dim_latent=32,
        num_heads=4,
        context_length=CONTEXT_LEN,
        num_segments=4,
        num_shared_experts=1,
        num_routed_experts=4,
    ).to(DEVICE)

    train(
        model,
        dataset,
        DEVICE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        aux_weight=AUX_LOSS_WEIGHT,
    )

    print("\n--- Generation ---")
    print(generate(model, dataset, "First Citizen:\n"))

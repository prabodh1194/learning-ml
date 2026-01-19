import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import time

from llama.llama import LLaMA


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int = 512):
        self.text = text
        self.block_size = block_size

        self.chars = sorted(set(text))

        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        self.data = [self.stoi[c] for c in self.text]
        self.vocab_size = len(self.chars)

        self.block_size = block_size

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


def train(
    model: nn.Module,
    dataset: CharDataset,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        for x, y in loader:
            logits, _ = model(x)

            # logits is B, T, C
            # y is B, T
            # ce works on logits - (N, C) & y - (N, )
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}, loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")

    total_time = time.time() - train_start
    print(f"Training complete in {total_time:.2f}s")


if __name__ == "__main__":
    with open("data/tinyshakespeare/input.txt") as f:
        text = f.read()

    dataset = CharDataset(text, block_size=64)

    model = LLaMA(
        n_layers=12,
        vocab_size=dataset.vocab_size,
        dim=128,
        max_seq_len=64,
        num_head=4,
        num_kv_head=2,
    )

    train(model, dataset)

    prompt = "ROMEO:"
    prompt_tokens = torch.tensor([dataset.encode(prompt)])  # (1, T); B = 1
    output = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.8)
    print(dataset.decode(output[0].tolist()))

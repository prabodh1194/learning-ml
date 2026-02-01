import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import time

from llama import logger
from llama.model import LLaMA


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
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    aux_weight: float = 0.01,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_batches = len(loader)

    logger.info(f"Training on {device}: {total_batches} batches/epoch")

    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(loader):
            # x, y is (B, T)
            x, y = x.to(device), y.to(device)
            logits, aux_loss, _ = model(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            loss = ce_loss + aux_weight * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"[{epoch + 1}] [{batch_idx + 1}/{total_batches}] loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1}, loss: {avg_loss:.4f}, time: {epoch_time:.2f}s")

        # save checkpoint
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch + 1}.pt")
        logger.info(f"Saved checkpoint: checkpoints/epoch_{epoch + 1}.pt")

    total_time = time.time() - train_start
    logger.info(f"Training complete in {total_time:.2f}s")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # hyperparams:
    B = 256
    T = 64
    C = 128
    context_length = 64
    torch.manual_seed(42)

    assert context_length >= T, "context length must always be larger than T"

    with open("../data/tinyshakespeare/input.txt") as f:
        text = f.read()

    # this is T = 64
    dataset = CharDataset(text, block_size=T)

    # C is going to be 32 * 4 = 128
    model = LLaMA(
        n_layers=6,
        vocab_size=dataset.vocab_size,
        dim=C,
        context_length=context_length,
        num_head=4,
        num_kv_head=2,
    ).to(device)

    # B = 32
    train(model, dataset, device, batch_size=B)

    prompt = "ROMEO:"
    prompt_tokens = torch.tensor([dataset.encode(prompt)]).to(device)
    print(prompt, end="", flush=True)
    model.generate(
        prompt_tokens,
        max_new_tokens=100,
        temperature=0.8,
        decode_fn=lambda i: dataset.itos[i],
    )

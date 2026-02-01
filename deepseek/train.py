"""Train DeepSeek on TinyShakespeare."""

import torch

from deepseek.model import Deepseek
from llama.train import CharDataset, train

# --- Config ---
DATA_PATH = "../data/tinyshakespeare/input.txt"
CONTEXT_LEN = 64
B, T, C = 256, 64, 128
EPOCHS = 3
LR = 3e-4
AUX_LOSS_WEIGHT = 0.01
DEVICE = "mps"


if __name__ == "__main__":
    device = torch.device(DEVICE)

    with open(DATA_PATH, "r") as f:
        text = f.read()

    dataset = CharDataset(text, block_size=T)

    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Training on {len(dataset)} samples")
    print(f"Device: {device}")
    print()

    model = Deepseek(
        vocab_size=dataset.vocab_size,
        dim=C,
        num_layers=4,
        dim_latent=32,
        num_heads=4,
        context_length=CONTEXT_LEN,
        num_segments=4,
        num_shared_experts=1,
        num_routed_experts=4,
    ).to(device)

    train(
        model,
        dataset,
        device,
        epochs=EPOCHS,
        batch_size=B,
        lr=LR,
        aux_weight=AUX_LOSS_WEIGHT,
    )

"""
Train DeepSeek on TinyShakespeare and generate text.
"""

import torch
import torch.nn.functional as F

from deepseek.model import Deepseek

# --- Config ---
DATA_PATH = "../data/tinyshakespeare/input.txt"
CONTEXT_LEN = 64
BATCH_SIZE = 32
TRAIN_STEPS = 1000
LR = 3e-4
AUX_LOSS_WEIGHT = 0.01
DEVICE = "mps"

# --- Data ---
with open(DATA_PATH, "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}


def encode(s: str) -> list:
    return [char_to_idx[c] for c in s]


def decode(ids: list) -> str:
    return "".join(idx_to_char[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)


def get_batch():
    ix = torch.randint(len(data) - CONTEXT_LEN, (BATCH_SIZE,))
    x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# --- Model ---
model = Deepseek(
    vocab_size=vocab_size,
    dim=64,
    num_layers=4,
    dim_latent=32,
    num_heads=4,
    context_length=CONTEXT_LEN,
    num_segments=4,
    num_shared_experts=1,
    num_routed_experts=4,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# --- Training ---
def train():
    model.train()
    for step in range(TRAIN_STEPS):
        x, y = get_batch()

        logits, aux_loss, _ = model(x, caches=None)
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss = ce_loss + AUX_LOSS_WEIGHT * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"step {step}: loss={loss.item():.3f}, ce={ce_loss.item():.3f}, aux={aux_loss.item():.3f}"
            )


# --- Generation ---
@torch.no_grad()
def generate(prompt, max_tokens=200):
    model.eval()
    tokens = encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    logits, _, caches = model(tokens, caches=None)

    for _ in range(max_tokens):
        next_logits = logits[:, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        tokens = torch.cat([tokens, next_token], dim=1)
        logits, _, caches = model(next_token, caches=caches)

    return decode(tokens[0].tolist())


if __name__ == "__main__":
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {len(data)} chars")
    print(f"Device: {DEVICE}")
    print()

    train()

    print("\n--- Generation ---")
    print(generate("First Citizen:\n"))

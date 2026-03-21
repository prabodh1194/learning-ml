"""
MLM pre-training loop for BERT on WikiText-2.

Goal: watch the loss drop over a few hundred steps to confirm MLM works.
NOT training to convergence — just mechanism understanding.

Usage:
    uv run python -m bert.train_mlm
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast
from datasets import load_dataset

from bert.model import BERT
from bert.mlm import MLMHead, mlm_mask

# --- Config ---
TINY_CONFIG = dict(
    vocab_size=30522,
    d_model=128,
    n_heads=4,
    n_layers=4,
    ffn_dim=512,
    max_seq_len=128,
)
BATCH_SIZE = 32
LR = 3e-4
MAX_STEPS = 500
MASK_PROB = 0.15
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class WikiTextMLMDataset(Dataset):
    """
    Tokenizes WikiText-2 into fixed-length chunks for MLM.
    All text is concatenated into one long sequence, then chopped into max_seq_len chunks.
    """

    def __init__(self, tokenizer, max_seq_len: int, split: str = "train"):
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

        all_ids = []
        for row in dataset:
            text = row["text"].strip()
            if text:
                all_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])

        self.chunks = [
            all_ids[i : i + max_seq_len]
            for i in range(0, len(all_ids) - max_seq_len, max_seq_len)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.chunks[idx], dtype=torch.long),
            "token_type_ids": torch.zeros(len(self.chunks[idx]), dtype=torch.long),
        }


def train():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    mask_token_id = tokenizer.mask_token_id

    dataset = WikiTextMLMDataset(tokenizer, max_seq_len=TINY_CONFIG["max_seq_len"])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    bert = BERT(**TINY_CONFIG).to(DEVICE)
    mlm_head = MLMHead(TINY_CONFIG["d_model"], TINY_CONFIG["vocab_size"]).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(bert.parameters()) + list(mlm_head.parameters()), lr=LR
    )
    loss_fn = nn.CrossEntropyLoss()  # ignores -100 labels by default

    step = 0
    bert.train()
    mlm_head.train()

    vocab_size = TINY_CONFIG["vocab_size"]

    for epoch in range(100):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)

            masked_tokens, labels = mlm_mask(
                input_ids, vocab_size, mask_token_id, MASK_PROB
            )

            hidden = bert(masked_tokens, token_type_ids)
            logits = mlm_head(hidden)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"step {step:4d} | loss {loss.item():.4f}")

            step += 1
            if step >= MAX_STEPS:
                break

        if step >= MAX_STEPS:
            break

    print(f"Done! Final loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()

"""
Fine-tune pretrained BERT on SNLI for Natural Language Inference.

This is the classic BERT fine-tuning recipe:
  1. Load pretrained bert-base-uncased (already understands language from MLM)
  2. Attach a classification head: Linear(768, 3)
  3. Fine-tune on SNLI sentence pairs → entailment / contradiction / neutral

Usage:
    uv run python -m bert.train_nli
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel
from datasets import load_dataset, tqdm

# --- Config ---
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 3
MAX_SEQ_LEN = 128
NUM_CLASSES = 3  # entailment, neutral, contradiction
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class SNLIDataset(Dataset):
    """
    SNLI sentence pairs for NLI.

    Each item:
      premise:    "A man is playing guitar"
      hypothesis: "Someone is making music"
      label:      0 (entailment), 1 (neutral), 2 (contradiction)

    The tokenizer handles [CLS], [SEP], token_type_ids automatically
    when you pass two sentences.
    """

    def __init__(self, tokenizer, split: str = "train"):
        dataset = load_dataset("stanfordnlp/snli", split=split)
        self.data = dataset.filter(lambda x: x["label"] != -1)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        encoded = self.tokenizer(
            row["premise"],
            row["hypothesis"],
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "token_type_ids": encoded["token_type_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


def evaluate(bert, head, dataloader):
    """Run validation and return accuracy."""
    bert.eval()
    head.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            hidden = bert(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            cls_output = hidden.last_hidden_state[:, 0, :]  # CLS token
            logits = head(cls_output)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_dataset = SNLIDataset(tokenizer, split="train")
    val_dataset = SNLIDataset(tokenizer, split="validation")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Pretrained BERT — already understands language from MLM pre-training
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)

    # Classification head — Linear(768, 3) on CLS token
    # NOTE: this uses nn.Linear directly, not your ClassificationHead,
    # because HuggingFace BERT outputs a different format
    head = nn.Linear(768, NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(bert.parameters()) + list(head.parameters()), lr=LR
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- Training loop ---
    # TODO: wire it up!
    # for each epoch:
    #   bert.train(); head.train()
    #   for step, batch in enumerate(train_loader):
    #     1. move input_ids, token_type_ids, attention_mask, labels to DEVICE
    #     2. hidden = bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    #     3. cls_output = hidden.last_hidden_state[:, 0, :]   ← CLS token
    #     4. logits = head(cls_output)                        ← (B, 3)
    #     5. loss = loss_fn(logits, labels)
    #     6. optimizer.zero_grad() → loss.backward() → optimizer.step()
    #     7. print loss every 200 steps
    #
    #   val_acc = evaluate(bert, head, val_loader)
    #   print(f"epoch {epoch} | val accuracy: {val_acc:.4f}")
    #
    # Target: ~83-84% val accuracy after 3 epochs
    for epoch in range(EPOCHS):
        bert.train()
        head.train()
        for step, batch in tqdm(enumerate(train_loader)):
            input_ids = batch["input_ids"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            hidden = bert(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            cls_output = hidden.last_hidden_state[:, 0, :]
            logits = head(cls_output)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {loss.item()}"
                )

        val_acc = evaluate(bert, head, val_loader)
        print(f"Epoch {epoch} | Step {step} | Validation Accuracy {val_acc}")


if __name__ == "__main__":
    train()

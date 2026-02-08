import torch
from transformers import AutoTokenizer

from sft.dataset import AlpacaDataset
from sft.load_tinyllama import MODEL_DIR, load
from sft.lora_llama import apply_lora

from torch.nn import functional as F

if __name__ == "__main__":
    epochs = 3
    lr = 2e-4
    B = 4
    warmup_steps = 100

    model = load()
    model = apply_lora(model)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "models/tinyllama-1.1b")
    data_loader = AlpacaDataset(tokenizer)

    for epoch in range(epochs):
        for batch in data_loader:
            input_ids, labels = batch["input_ids"], batch["labels"]

            logits, *_ = model(input_ids)

            loss = F.cross_entropy(logits[:, :-1, :], labels[:, 1:])

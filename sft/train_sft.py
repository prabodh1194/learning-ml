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

    step = 0
    for epoch in range(epochs):
        for batch in data_loader:
            input_ids, labels = batch["input_ids"], batch["labels"]

            input_ids = torch.tensor(input_ids).unsqueeze(0)  # (T) -> (1, T)
            labels = torch.tensor(labels).unsqueeze(0)  # (T) -> (1, T)

            logits, *_ = model(input_ids)

            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, 32000), labels[:, 1:].reshape(-1), ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Live counter (overwrites itself)
            print(f"\rstep {step} | epoch {epoch} | loss: {loss.item():.4f}", end="", flush=True)

            # Sticky log every 500 steps (stays visible)
            if step % 50 == 0:
                print()  # newline to "stick" the current line
            step += 1

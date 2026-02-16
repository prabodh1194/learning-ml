import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer

from sft.dataset import AlpacaDataset
from sft.load_tinyllama import MODEL_DIR, load
from sft.lora_llama import apply_lora, save_lora_weights, load_lora_weights

from torch.nn import functional as F

if __name__ == "__main__":
    epochs = 3
    lr = 2e-4
    accum_steps = 4  # gradient accumulation
    max_steps = 1000
    device = "mps"

    model = load(freeze=True)
    model = apply_lora(model)

    # Resume from checkpoint if exists
    adapter_path = Path("adapters/alpaca.pt")
    if adapter_path.exists():
        print(f"Loading existing adapter from {adapter_path}")
        weights = torch.load(adapter_path)
        load_lora_weights(model, weights)
    else:
        print("Starting fresh training")

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "models/tinyllama-1.1b")
    data_loader = AlpacaDataset(tokenizer)

    # CSV logging
    log_file = open("training_log.csv", "w")
    log_file.write("step,timestamp,loss\n")

    step = 0
    accum_loss = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            input_ids, labels = batch["input_ids"], batch["labels"]

            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)  # (T) -> (1, T)
            labels = torch.tensor(labels).unsqueeze(0).to(device)  # (T) -> (1, T)

            logits, *_ = model(input_ids)

            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, 32000), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = loss / accum_steps  # scale loss for accumulation
            loss.backward()
            accum_loss += loss.item()

            # Update weights every accum_steps
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                now = datetime.now().strftime("%H:%M:%S")

                # CSV log
                log_file.write(f"{step},{now},{accum_loss:.4f}\n")
                log_file.flush()

                # Live counter
                print(f"\r[{now}] step {step} | epoch {epoch} | loss: {accum_loss:.4f}", end="", flush=True)

                # Sticky log every 50 steps
                if step % 50 == 0:
                    print()

                accum_loss = 0.0
                step += 1

                if step >= max_steps:
                    break
        if step >= max_steps:
            break

    log_file.close()
    print("\nTraining complete!")

    # Save adapter weights
    weights = save_lora_weights(model)
    torch.save(weights, "adapters/alpaca.pt")
    print("Saved adapter to adapters/alpaca.pt")

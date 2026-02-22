import torch
from datetime import datetime
from torch import optim
from transformers import AutoTokenizer

from distil.loss import distillation_loss
from distil.student import create_student
from distil.teacher import load_teacher
from sft.dataset import AlpacaDataset
from sft.load_tinyllama import MODEL_DIR
from torch.nn import functional as F


def train():
    epochs = 3
    lr = 2e-4
    temperature = 4
    device = "mps"

    teacher = load_teacher()
    teacher.to(device)

    student = create_student(vocab_size=32000)
    student.to(device)
    student.train()

    optimizer = optim.AdamW(student.parameters(), lr=lr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "models/tinyllama-1.1b")
    data_loader = AlpacaDataset(tokenizer)

    log_file = open("training_log.csv", "w")
    log_file.write("step,epoch,timestamp,loss\n")

    for epoch in range(epochs):
        step = 0
        for batch in data_loader:
            input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device)
            with torch.no_grad():
                teacher_logits, *_ = teacher(input_ids)
            student_logits, *_ = student(input_ids)

            kl_loss = distillation_loss(teacher_logits, student_logits, temperature)
            ce_loss = F.cross_entropy(
                student_logits[:, :-1, :].reshape(-1, 32000),
                torch.tensor(batch["labels"]).unsqueeze(0).to(device)[:, 1:].reshape(-1),
                ignore_index=-100,
            )

            loss = 0.7 * kl_loss + 0.3 * ce_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            now = datetime.now().strftime("%H:%M:%S")
            log_file.write(f"{step},{epoch},{now},{loss.item():.4f}\n")
            log_file.flush()

            print(
                f"\r[{now}] step {step} | epoch {epoch} | kl: {kl_loss.item():.4f} | ce: {ce_loss.item():.4f} | loss: {loss.item():.4f}",
                end="",
                flush=True,
            )
            if step % 50 == 0:
                print()

            step += 1

            if step >= 2000:
                break

    log_file.close()
    print("\nDistillation complete!")

    torch.save(student.state_dict(), "student.pt")
    print("Saved student weights to distil/student.pt")


if __name__ == "__main__":
    train()

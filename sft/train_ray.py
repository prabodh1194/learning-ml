import logging

import ray
import ray.data
import torch
from ray.train import RunConfig, ScalingConfig, get_context, get_dataset_shard, report
from ray.train.torch import TorchTrainer, prepare_model
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from sft.dataset import AlpacaDataset
from sft.load_tinyllama import load
from sft.lora_llama import apply_lora, save_lora_weights

logger = logging.getLogger("sft.train_ray")


def collate_fn(batch: dict[str, list[list]]) -> dict[str, torch.Tensor]:
    input_ids, labels = batch["input_ids"], batch["labels"]

    pt_input_ids = pad_sequence(
        list(map(torch.LongTensor, input_ids)), batch_first=True
    )
    pt_labels = pad_sequence(
        list(map(torch.LongTensor, labels)), batch_first=True, padding_value=-100.0
    )

    return {"input_ids": pt_input_ids, "labels": pt_labels}


def train_func(config):
    epochs = 3
    lr = 2e-4
    num_workers = config["num_workers"]
    total_rows = config["total_rows"]
    steps_per_epoch = total_rows // num_workers // 16  # batch_size=16
    total_steps = steps_per_epoch * epochs

    model = load(freeze=True)
    model = apply_lora(model)

    model = prepare_model(model)

    logger.info(f"~{steps_per_epoch} steps/epoch, ~{total_steps} total steps")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    train_shard = get_dataset_shard("train")

    for epoch in range(epochs):
        step = 0
        running_loss = 0.0
        running_steps = 0

        for batch in train_shard.iter_torch_batches(
            batch_size=16, collate_fn=collate_fn
        ):
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, *_ = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, 32000),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            running_steps += 1
            step += 1

            if step % 50 == 0:
                avg_loss = running_loss / running_steps
                report({"epoch": epoch, "step": step, "loss": avg_loss})
                logger.info(f"epoch {epoch} step {step}/{steps_per_epoch} loss {avg_loss:.4f}")
                running_loss = 0.0
                running_steps = 0

        # End-of-epoch report with whatever's left
        if running_steps > 0:
            avg_loss = running_loss / running_steps
            report({"epoch": epoch, "step": step, "loss": avg_loss, "epoch_end": True})
            logger.info(f"epoch {epoch} done — step {step} loss {avg_loss:.4f}")

    if get_context().get_world_rank() == 0:
        weights = save_lora_weights(model.module)
        torch.save(weights, "adapters/alpaca_ray.pt")
        logger.info("saved LoRA weights to adapters/alpaca_ray.pt")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    data_loader = AlpacaDataset(tokenizer)
    alpaca_ds = ray.data.from_items(data_loader.examples)

    num_workers = 16

    trainer = TorchTrainer(
        train_func,
        train_loop_config={"total_rows": len(data_loader.examples), "num_workers": num_workers},
        datasets={"train": alpaca_ds},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True),
        run_config=RunConfig(name="lora-tinyllama"),
    )

    trainer.fit()

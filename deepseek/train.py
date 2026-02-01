"""Train DeepSeek on TinyShakespeare."""

import torch

from training_config import DeepSeekConfig
from deepseek.model import Deepseek
from llama.train import CharDataset, train


if __name__ == "__main__":
    cfg = DeepSeekConfig()
    device = torch.device(cfg.device)

    with open(cfg.data_path, "r") as f:
        text = f.read()

    dataset = CharDataset(text, block_size=cfg.context_len)

    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Training on {len(dataset)} samples")
    print(f"Device: {device}")
    print()

    model = Deepseek(
        vocab_size=dataset.vocab_size,
        dim=cfg.dim,
        num_layers=cfg.num_layers,
        dim_latent=cfg.dim_latent,
        num_heads=cfg.num_heads,
        context_length=cfg.context_len,
        num_segments=cfg.num_segments,
        num_shared_experts=cfg.num_shared_experts,
        num_routed_experts=cfg.num_routed_experts,
    ).to(device)

    train(
        model,
        dataset,
        device,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        aux_weight=cfg.aux_weight,
    )

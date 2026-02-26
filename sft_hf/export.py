"""
Merge LoRA adapters into base TinyLlama and save as a single HF model.

What this does:
    1. Load TinyLlama 1.1B in HuggingFace format
    2. Load our LoRA A/B matrices from sft/adapters/alpaca.pt
    3. For each layer, merge: W += (B @ A) * scaling
    4. Save the merged model + tokenizer to sft-merged-hf/

After merging, LoRA is "baked in" — no adapter needed at inference time.

Run: uv run -m sft_hf.export
"""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = Path("models/tinyllama-1.1b")
ADAPTER_PATH = Path("sft/adapters/alpaca.pt")
OUTPUT_DIR = Path("sft-merged-hf")

RANK = 8
ALPHA = 16.0
SCALING = ALPHA / RANK  # 2.0

# LoRA key prefix -> HF attribute name
PROJ_MAP = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
}


def merge():
    logger.info(f"Loading base model from {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    logger.info(f"Loading LoRA weights from {ADAPTER_PATH}")
    lora_weights = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=True)

    n_layers = model.config.num_hidden_layers
    logger.info(f"Merging LoRA into {n_layers} layers (scaling={SCALING})")

    for i in range(n_layers):
        for lora_key, hf_attr in PROJ_MAP.items():
            A = lora_weights[f"layer.{i}.W_{lora_key}_A"]  # (rank, in_dim)
            B = lora_weights[f"layer.{i}.W_{lora_key}_B"]  # (out_dim, rank)

            proj = getattr(model.model.layers[i].self_attn, hf_attr)
            proj.weight.data += (B @ A) * SCALING

        logger.debug(f"  layer {i} merged")

    logger.info(f"Saving merged model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    merge()

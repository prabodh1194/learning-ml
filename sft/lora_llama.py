import itertools
import torch

from transformers import AutoTokenizer, TokenizersBackend
from pathlib import Path

from llama.block import LLaMABlock
from llama.model import LLaMA
from sft.dataset import format_example
from sft.load_tinyllama import MODEL_DIR, load
from sft.lora_linear import LoRALinear


def apply_lora(model: LLaMA, rank: int = 8, alpha: float = 16.0):
    """
    Replace attn projections with LoRA-wrapped versions.
    """
    for param in model.parameters():
        param.requires_grad = False

    layer: LLaMABlock
    for layer in model.layers:
        # wrap q, k, v, o projections
        layer.attn.W_q = LoRALinear(layer.attn.W_q, rank, alpha)
        layer.attn.W_k = LoRALinear(layer.attn.W_k, rank, alpha)
        layer.attn.W_v = LoRALinear(layer.attn.W_v, rank, alpha)
        layer.attn.W_o = LoRALinear(layer.attn.W_o, rank, alpha)

    return model


def save_lora_weights(model: LLaMA) -> dict:
    out = {}
    qkv = ["q", "k", "v", "o"]
    types = ["A", "B"]

    for i, layer in enumerate(model.layers):
        for _qkv, _types in itertools.product(qkv, types):
            out[f"layer.{i}.W_{_qkv}_{_types}"] = getattr(
                getattr(layer.attn, f"W_{_qkv}"), _types
            )

    return out


def load_lora_weights(model: LLaMA, lora_weights: dict) -> LLaMA:
    qkv = ["q", "k", "v", "o"]
    types = ["A", "B"]
    for i, layer in enumerate(model.layers):
        for _qkv, _types in itertools.product(qkv, types):
            lora_layer = getattr(layer.attn, f"W_{_qkv}")
            setattr(lora_layer, _types, lora_weights[f"layer.{i}.W_{_qkv}_{_types}"])

    return model


if __name__ == "__main__":
    print("=== Loading model with LoRA adapter ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "models/tinyllama-1.1b")

    model = load()
    model = apply_lora(model)

    # Load adapter weights
    adapter_path = Path("adapters/alpaca.pt")
    if adapter_path.exists():
        print(f"Loading adapter from {adapter_path}")
        weights = torch.load(adapter_path)
        model = load_lora_weights(model, weights)
    else:
        print("No adapter found, using base LoRA (untrained)")

    model.to("mps")
    model.eval()

    print("\n=== Generation Test ===")
    print("Type 'quit' to exit\n")

    while True:
        instruction = input("Instruction: ")
        if instruction.lower() == "quit":
            break

        # Format as instruction prompt
        prompt = format_example({"instruction": instruction, "input": "", "output": ""}, skip_response=True)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

        # Generate and decode all at once for proper spacing
        output_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
        )
        response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        print(f"Response: {response}")
        print()

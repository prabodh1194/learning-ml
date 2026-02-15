import itertools

from transformers import AutoTokenizer, TokenizersBackend

from llama.block import LLaMABlock
from llama.model import LLaMA
from sft.dataset import tokenize_with_mask
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
    # Test tokenize_with_mask
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "models/tinyllama-1.1b")

    example = {"instruction": "What is 2+2?", "input": "", "output": "4"}

    result = tokenize_with_mask(example, tokenizer)

    print("=== Masking Test ===")
    print(f"Input IDs length: {len(result['input_ids'])}")
    print(f"Labels length: {len(result['labels'])}")
    print(f"Masked tokens (label=-100): {result['labels'].count(-100)}")
    print(f"Unmasked tokens: {len(result['labels']) - result['labels'].count(-100)}")
    print()
    print("Decoded input:", tokenizer.decode(result["input_ids"]))
    print()
    print("Labels (showing mask):")
    for i, (tok, lab) in enumerate(zip(result["input_ids"], result["labels"])):
        marker = "MASK" if lab == -100 else "TRAIN"
        print(f"  {marker}: {tokenizer.decode([tok])!r}")

    print("\n=== LoRA Test ===")
    llama_model = load()
    lora_model = apply_lora(llama_model)

    total = sum(p.numel() for p in lora_model.parameters())
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable %: {100 * trainable / total:.2f}%")

from transformers import TokenizersBackend

from llama.block import LLaMABlock
from llama.model import LLaMA
from sft.dataset import format_example
from sft.load_tinyllama import load
from sft.lora_linear import LoRALinear
from transformers import AutoTokenizer
from sft.load_tinyllama import MODEL_DIR


def tokenize_with_mask(
    example: dict, tokenizer: TokenizersBackend, max_length: int = 512
):
    prompt = format_example(example, skip_response=True)
    response = f"{example['output']}</s>"

    # tokenise separately
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # combine
    input_ids = prompt_ids + response_ids

    labels = [-100] * len(prompt_ids) + response_ids

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


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
    model = load()
    model = apply_lora(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable %: {100 * trainable / total:.2f}%")

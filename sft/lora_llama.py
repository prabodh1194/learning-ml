from llama.block import LLaMABlock
from llama.model import LLaMA
from sft.load_tinyllama import load
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


if __name__ == "__main__":
    model = load()
    model = apply_lora(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable %: {100 * trainable / total:.2f}%")

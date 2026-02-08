import logging
from pathlib import Path

import safetensors
from transformers import AutoTokenizer

from llama.block import LLaMABlock
from llama.model import LLaMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Get the directory where this script lives
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR.parent


def load():
    logger.info("Creating LLaMA model...")
    # fetch from: https://arxiv.org/pdf/2401.02385
    model = LLaMA(
        n_layers=22,
        vocab_size=32000,
        dim=2048,
        hidden_dim=5632,
        context_length=2048,
        num_head=32,
        num_kv_head=4,
    )

    logger.info("Loading weights from safetensors...")
    with safetensors.safe_open(
        MODEL_DIR / "models/tinyllama-1.1b/model.safetensors", framework="pt"
    ) as f:
        model.embed.weight.data.copy_(f.get_tensor("model.embed_tokens.weight"))

        layer: LLaMABlock
        for i, layer in enumerate(model.layers):
            logger.debug(f"Loading layer {i}/21...")
            layer.attn_norm.gamma.data.copy_(
                f.get_tensor(f"model.layers.{i}.input_layernorm.weight")
            )
            layer.attn.W_q.weight.data.copy_(
                f.get_tensor(f"model.layers.{i}.self_attn.q_proj.weight")
            )
            layer.attn.W_k.weight.data.copy_(
                f.get_tensor(f"model.layers.{i}.self_attn.k_proj.weight")
            )
            layer.attn.W_v.weight.data.copy_(
                f.get_tensor(f"model.layers.{i}.self_attn.v_proj.weight")
            )
            layer.attn.W_o.weight.data.copy_(
                f.get_tensor(f"model.layers.{i}.self_attn.o_proj.weight")
            )

            layer.ffn.w_down.weight.data.copy_(
                f.get_tensor(f"model.layers.{i}.mlp.down_proj.weight")
            )
            layer.ffn.w_gate.weight.data.copy_(
                f.get_tensor(f"model.layers.{i}.mlp.gate_proj.weight")
            )
            layer.ffn.w_up.weight.data.copy_(
                f.get_tensor(f"model.layers.{i}.mlp.up_proj.weight")
            )
            layer.ffn_norm.gamma.data.copy_(
                f.get_tensor(
                    f"model.layers.{i}.post_attention_layernorm.weight",
                )
            )

        model.norm.gamma.data.copy_(f.get_tensor("model.norm.weight"))
        model.lm_head.weight.data.copy_(f.get_tensor("lm_head.weight"))

    logger.info("Model loaded successfully!")
    return model


if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "models/tinyllama-1.1b")

    # Test prompt
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt, return_tensors="pt")

    model = load()

    model.eval()
    output = model.generate(tokens, max_new_tokens=20, temperature=.7)

    print(tokenizer.decode(output[0]))

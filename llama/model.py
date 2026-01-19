import torch
from torch import nn

from llama.rmsnorm import RMSNorm
from llama.block import LLaMABlock


class LLaMA(nn.Module):
    """
    tokens ─► Embed ─► [Block 0] ─► [Block 1] ─► ... ─► [Block N] ─► RMSNorm ─► LM Head ─► logits
     (B,T)    (B,T,dim)     │           │                   │         (B,T,dim)           (B,T,vocab)
                            ▼           ▼                   ▼
                        kv_cache[0] kv_cache[1]        kv_cache[N]
    """

    def __init__(
        self,
        *,
        n_layers: int,
        vocab_size: int,
        dim: int,
        context_length: int,
        num_head: int,
        num_kv_head: int,
    ):
        super().__init__()
        self.context_length = context_length

        # project (B, T) -> (B, T, C)
        self.embed = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList(
            [
                LLaMABlock(
                    dim=dim,
                    context_length=context_length,
                    num_head=num_head,
                    num_kv_head=num_kv_head,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        kv_caches: list[tuple] | None = None,
    ) -> tuple[torch.Tensor, list[tuple]]:
        # tokens: (B, T)
        x = self.embed(tokens)  # (B, T, C); C = dim

        new_caches = []

        for layer, kv_cache in zip(self.layers, kv_caches or [None] * len(self.layers)):
            x, cache = layer(x, start_pos, kv_cache)
            new_caches.append(cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_caches

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # logits is (B, vocab_size)
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

        probs = (logits / temperature).softmax(dim=-1)

        result = torch.multinomial(probs, num_samples=1)  # (B, 1)

        return result  # (B, 1)

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, decode_fn=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length :]  # slide window
            logits, _ = self.forward(idx_cond)
            next_token = self._sample(logits[:, -1, :], temperature)
            idx = torch.cat([idx, next_token], dim=1)
            if decode_fn:
                print(decode_fn(next_token[0].item()), end="", flush=True)
        if decode_fn:
            print()
        return idx[:, -max_new_tokens:]

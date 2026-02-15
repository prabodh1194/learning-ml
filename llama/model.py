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
        hidden_dim: int,
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
                    hidden_dim=hidden_dim,
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
    ) -> tuple[torch.Tensor, float, list[tuple]]:
        # tokens: (B, T)
        x = self.embed(tokens)  # (B, T, C); C = dim

        new_caches = []

        for layer, kv_cache in zip(self.layers, kv_caches or [None] * len(self.layers)):
            x, cache = layer(x, start_pos, kv_cache)
            new_caches.append(cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, 0.0, new_caches

    """
    theory on temp:
    
LOGITS (raw model output):
─────────────────────────
Paris:     2.0
Lyon:      1.0
Berlin:    0.5

SOFTMAX FORMULA:
softmax(x) = e^x / sum(e^all)

Step by step with different temperatures:

TEMPERATURE = 1.0 (divide by 1, no change)
───────────────────────────────────────────
logits/T:   2.0/1 = 2.0    1.0/1 = 1.0    0.5/1 = 0.5

e^(logit/T):  e^2.0 = 7.39   e^1.0 = 2.72   e^0.5 = 1.65
                                            sum = 11.76

probs:      7.39/11.76     2.72/11.76     1.65/11.76
            = 0.63         = 0.23         = 0.14
            ██████         ██             █


TEMPERATURE = 0.5 (divide by 0.5 = multiply by 2)
───────────────────────────────────────────
logits/T:   2.0/0.5 = 4.0   1.0/0.5 = 2.0   0.5/0.5 = 1.0
                 ↑               ↑               ↑
            gaps get BIGGER (4 vs 2 vs 1)

e^(logit/T):  e^4.0 = 54.6   e^2.0 = 7.39   e^1.0 = 2.72
                                            sum = 64.71

probs:      54.6/64.71     7.39/64.71     2.72/64.71
            = 0.84         = 0.11         = 0.04
            ████████       █              ▪
                 ↑
            Paris dominates!


TEMPERATURE = 2.0 (divide by 2)
───────────────────────────────────────────
logits/T:   2.0/2 = 1.0    1.0/2 = 0.5    0.5/2 = 0.25
                 ↑              ↑              ↑
            gaps get SMALLER (1 vs 0.5 vs 0.25)

e^(logit/T):  e^1.0 = 2.72   e^0.5 = 1.65   e^0.25 = 1.28
                                            sum = 5.65

probs:      2.72/5.65      1.65/5.65      1.28/5.65
            = 0.48         = 0.29         = 0.23
            ████           ███            ██
                 ↑
            More even, anyone could win!

The key insight:

e^x is EXPONENTIAL - small differences become HUGE

Original logits:     2.0    1.0    0.5     (gaps: 1.0, 0.5)

Low T (÷0.5):        4.0    2.0    1.0     (gaps: 2.0, 1.0)  → BIGGER gaps
                      ↓
                   e^4 vs e^2 = 54 vs 7    → winner takes all

High T (÷2):         1.0    0.5    0.25    (gaps: 0.5, 0.25) → SMALLER gaps
                      ↓
                   e^1 vs e^0.5 = 2.7 vs 1.6 → more competitive

TL;DR: Dividing by T shrinks/expands the gaps between logits. Exponential (e^x) amplifies those gaps into probabilities.

now multinomial interplay is:
  
STEP 1: Temperature adjusts probabilities
─────────────────────────────────────────
logits = [2.0, 1.0, 0.5]

probs = softmax(logits / T)

T=0.5 → [0.84, 0.11, 0.04]   (Paris dominates)
T=2.0 → [0.48, 0.29, 0.23]   (more even)


STEP 2: Multinomial samples ONE token based on those probs
─────────────────────────────────────────
torch.multinomial(probs, num_samples=1)

Think of it as spinning a weighted wheel:

T=0.5 probs [0.84, 0.11, 0.04]:
┌────────────────────────────────┐
│ ████████████████████  Paris 84%│
│ ███  Lyon 11%                  │
│ █  Berlin 4%                   │
└────────────────────────────────┘
→ Almost always lands on Paris

T=2.0 probs [0.48, 0.29, 0.23]:
┌────────────────────────────────┐
│ ██████████  Paris 48%          │
│ ██████  Lyon 29%               │
│ █████  Berlin 23%              │
└────────────────────────────────┘
→ Paris likely, but Lyon/Berlin have real chances

The interplay:

Temperature = shapes the wheel
Multinomial = spins the wheel

Low T  → big slice for winner → spin almost always hits it
High T → similar slices      → spin is more unpredictable
T=0    → skip wheel entirely → just pick argmax (greedy)

In code:
probs = (logits / temperature).softmax(dim=-1)  # shape the wheel
result = torch.multinomial(probs, num_samples=1) # spin it once
    """

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # logits is (B, vocab_size)
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

        probs = (logits / temperature).softmax(dim=-1)

        result = torch.multinomial(probs, num_samples=1)  # (B, 1)

        return result  # (B, 1)

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, decode_fn=None, eos_token_id=2):
        original_len = idx.shape[1]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length :]  # slide window
            logits, _, _ = self.forward(idx_cond)
            next_token = self._sample(logits[:, -1, :], temperature)
            idx = torch.cat([idx, next_token], dim=1)

            token_id = next_token[0].item()
            if decode_fn:
                print(decode_fn(token_id), end="", flush=True)

            # Stop at EOS token
            if token_id == eos_token_id:
                break

        if decode_fn:
            print()

        # Return only the NEW tokens
        return idx[:, original_len:]

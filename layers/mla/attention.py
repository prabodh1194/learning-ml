"""
Multi-head Latent Attention (MLA) - DeepSeek's KV cache compression

MHA vs MLA:
───────────────────────────────────────────────────────────────────
                MHA                         MLA
───────────────────────────────────────────────────────────────────
K path:     X → W_K → K              X → WK_c → K_latent → WK_u → K
V path:     X → W_V → V              X → WV_c → V_latent → WV_u → V
Cache:      Full K, V (large)        Latents only (small)
Compression: None                     C → C_latent (e.g., 8x)
───────────────────────────────────────────────────────────────────

How Learning Works:
───────────────────────────────────────────────────────────────────
Fixed (not learned):
    scores = Q @ K.T / sqrt(d)    # dot product similarity
    attn = softmax(scores)        # normalize to weights
    out = attn @ V                # weighted sum

Learned (weight matrices):
    W_Q:  "What am I looking for?" - query projection
    W_K:  "What do I contain?" - key projection (via WK_c, WK_u)
    W_V:  "What do I give if selected?" - value projection (via WV_c, WV_u)
    W_O:  "How to combine heads" - output projection

Gradient Flow:
    Loss → W_O → attn weights → W_Q, W_K, W_V

    If attending to token X reduced loss → Q/K pushed to match stronger
    If attending to token Y hurt loss → Q/K pushed to match weaker

MLA Bottleneck:
    WK_c, WK_u force model to learn compressed K representation
    WV_c, WV_u force model to learn compressed V representation

    Similar to autoencoders - bottleneck forces meaningful compression
    rather than memorizing everything.

DeepSeek's bet: compression loss is worth the 8x cache reduction at scale.
"""

import math

import torch
from torch import nn

from llama.rope import RoPE


class Attention(nn.Module):
    def __init__(self, *, C: int, C_latent: int, num_heads: int, context_length: int):
        super().__init__()

        self.C = C
        self.C_latent = C_latent
        self.num_heads = num_heads

        self.d_head = self.C // self.num_heads
        self.d_head_latent = self.C_latent // self.num_heads

        self.W_Q = nn.Linear(self.C, self.C)
        self.W_O = nn.Linear(self.C, self.C)

        self.WK_c = nn.Linear(self.C, self.C_latent, bias=False)
        self.WK_u = nn.Linear(self.C_latent, self.C, bias=False)

        self.WV_c = nn.Linear(self.C, self.C_latent, bias=False)
        self.WV_u = nn.Linear(self.C_latent, self.C, bias=False)

        self.rope = RoPE(dim=self.d_head, context_length=context_length)

    def forward(
        self, X: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = X.shape

        assert C == self.C, f"didn't get the full dimension {self.C}. got {C} instead."

        K_latent = self.WK_c(X)
        V_latent = self.WV_c(X)

        if cache is not None:
            k_c, v_c = cache
            K_latent = torch.cat([k_c, K_latent], dim=-2)
            V_latent = torch.cat([v_c, V_latent], dim=-2)

        K = self.WK_u(K_latent)
        V = self.WV_u(V_latent)
        Q = self.W_Q(X)

        # split into heads
        # (B, T, C) -> (B, num_heads, T, d_head)
        Q_h = self.rope(
            Q.view(B, -1, self.num_heads, self.d_head), start_pos=K.shape[1] - T
        ).transpose(1, 2)
        K_h = self.rope(K.view(B, -1, self.num_heads, self.d_head)).transpose(1, 2)
        V_h = V.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)

        # attention
        scores = Q_h @ K_h.transpose(-2, -1) / math.sqrt(self.d_head)

        if T != 1:
            mask = torch.tril(torch.ones(T, T))
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = scores.softmax(dim=-1) @ V_h
        out = attn.transpose_(1, 2).contiguous().view(B, -1, C)

        return self.W_O(out), (K_latent, V_latent)

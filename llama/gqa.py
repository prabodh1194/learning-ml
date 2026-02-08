import torch
from torch import nn

from llama.rope import RoPE


class GQA(nn.Module):
    """
    RoPE is applied inside attention (not at input) because it rotates Q and K
    so their dot product encodes relative position. Must be applied in every layer
    after projection, otherwise layer weights would destroy the rotation.
    """

    def __init__(self, rope: RoPE, dim: int, num_head: int, num_kv_head: int):
        super().__init__()
        self.rope = rope

        self.dim = dim
        self.num_head = num_head
        self.num_kv_head = num_kv_head

        self.d_head = dim // num_head
        self.d_kv = self.d_head * num_kv_head

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, self.d_kv, bias=False)
        self.W_v = nn.Linear(dim, self.d_kv, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        X: torch.Tensor,
        start_pos: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = X.shape

        Q = self.rope(self.W_q(X).view(B, T, self.num_head, -1), start_pos).transpose(
            1, 2
        )  # B, num_head, T, d_head
        K_new = self.rope(
            self.W_k(X).view(B, T, self.num_kv_head, -1), start_pos
        ).transpose(1, 2)  # B, kv_head, T, d_head
        V_new = (
            self.W_v(X).view(B, T, self.num_kv_head, -1).transpose(1, 2)
        )  # B, kv_head, T, d_head

        if kv_cache:
            K_c, V_c = kv_cache
            K = torch.cat((K_c, K_new), dim=2)
            V = torch.cat((V_c, V_new), dim=2)
        else:
            K = K_new
            V = V_new

        K_i = torch.repeat_interleave(K, self.num_head // self.num_kv_head, dim=1)
        V_i = torch.repeat_interleave(V, self.num_head // self.num_kv_head, dim=1)

        scores = Q @ K_i.transpose(-2, -1) / (K.shape[-1] ** 0.5)

        if T != 1:
            mask = torch.tril(torch.ones(T, T)).to(scores.device)
            scores = scores.masked_fill(mask == 0, -torch.inf)

        attn = scores.softmax(dim=-1)
        out = attn @ V_i  # B, num_head, T, d_head
        return self.W_o(out.transpose_(1, 2).reshape(B, T, C)), (K, V)

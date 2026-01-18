import torch


class MultiHeadAttention:
    def __init__(self, C: int, num_heads: int):
        self.num_heads = num_heads

        self.W_q = torch.nn.Linear(C, C)
        self.W_k = torch.nn.Linear(C, C)
        self.W_v = torch.nn.Linear(C, C)
        self.W_o = torch.nn.Linear(C, C)

    def forward(self, X: torch.Tensor):
        B, T, C = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # split C-dim into num_heads
        q = Q.view(B, T, self.num_heads, -1)  # B, T, num_heads, C // num_heads

        # what's this for ??
        """
        assume batch-size is 1; 4 elements in seq; 4-dim embedding; 2 heads.
        
      E = h1=[e1 e2] h2=[e3 e4]
    T = e      .  .       .  .
        m      .  .       .  .
        m      .  .       .  .
        m      .  .       .  .
        
        Transpose is done so that scaled dot-product attention is applied correctly on a per-head basis.
        """
        Q = q.transpose(1, 2)  # B, num_heads, T, C // num_heads

        K = K.view(B, T, self.num_heads, -1).transpose(1, 2)
        V = V.view(B, T, self.num_heads, -1).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)

        mask = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mask == 0, -torch.inf)
        attn = scores.softmax(dim=-1)

        out = attn @ V

        # 1. transpose back to (B, T, num_heads, C // num_heads)
        # 2. concat the num_heads vectors in the last dim into a single vector.
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.W_o(out)

if __name__ == "__main__":
    mha = MultiHeadAttention(C=16, num_heads=2)
    X = torch.randn(2, 16, 16)

    assert mha.forward(X).shape == (2, 16, 16)

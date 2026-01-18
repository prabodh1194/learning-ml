import torch


class MultiHeadAttention:
    def __init__(self, C: int, num_heads: int):
        self.num_heads = num_heads

        self.C = C
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

    def inference(self, X: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        X - (B, 1, C) if X is single token
        else,
        X - (B, T, C)
        """
        B, T, C = X.shape

        Q = self.W_q(X).view(B, T, self.num_heads, -1).transpose(1, 2)
        K_new = self.W_k(X).view(B, T, self.num_heads, -1).transpose(1, 2)
        V_new = self.W_v(X).view(B, T, self.num_heads, -1).transpose(1, 2)

        if kv_cache:
            K_c, V_c = kv_cache
            # (B, num_heads, T, d_head) + (B, num_heads, 1, d_head)
            K = torch.cat([K_c, K_new], 2)
            V = torch.cat([V_c, V_new], 2)
        else:
            K = K_new
            V = V_new

        scores = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)

        if T != 1:
            mask = torch.tril(torch.ones(T, T))
            scores = scores.masked_fill(mask == 0, -torch.inf)

        attn = scores.softmax(dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.W_o(out), (K, V)


class GroupedQueryAttention:
    def __init__(self, C: int, num_head: int, num_kv_head: int):
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.C = C

        """
        for num_kv_head kv_heads; there are num_head q_heads.
        but the head size is same for both.
        """
        self.d_head = C // num_head

        self.W_q = torch.nn.Linear(C, C)
        self.W_k = torch.nn.Linear(C, num_kv_head * self.d_head)
        self.W_v = torch.nn.Linear(C, num_kv_head * self.d_head)
        self.W_o = torch.nn.Linear(C, C)

    def forward(self, X: torch.Tensor):
        B, T, C = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.view(B, T, self.num_head, -1).transpose(1, 2)
        K = K.view(B, T, self.num_kv_head, -1).transpose(1, 2)
        V = V.view(B, T, self.num_kv_head, -1).transpose(1, 2)

        # Q - (B, num_head, T, d_head)
        # K, V - (B, num_kv_head, T, d_head)
        # r * num_kv_head = num_head
        # repeat K, V; r times essentially to amplify K, V into num_head dimension

        K = torch.repeat_interleave(K, self.num_head // self.num_kv_head, dim=1)
        V = torch.repeat_interleave(V, self.num_head // self.num_kv_head, dim=1)

        score = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)
        mask = torch.tril(torch.ones(T, T))
        score = score.masked_fill(mask == 0, -torch.inf)
        attn = score.softmax(dim=-1)
        out = attn @ V

        out = out.transpose(1, 2).reshape(B, T, C)

        return self.W_o(out)

    def inference(self, X: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None=None):
        B, T, C = X.shape

        # T is 1 when kv_cache is being passed, else it is the full input

        Q = self.W_q(X).view(B, T, self.num_head, -1).transpose(1, 2)
        K_new = self.W_k(X).view(B, T, self.num_kv_head, -1).transpose(1, 2)
        V_new = self.W_v(X).view(B, T, self.num_kv_head, -1).transpose(1, 2)

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
            mask = torch.tril(torch.ones(T, T))
            scores = scores.masked_fill(mask == 0, -torch.inf)

        attn = scores.softmax(dim=-1)
        out = attn @ V_i
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.W_o(out), (K, V)


def generate_with_cache(model, prompt: torch.Tensor, num_tokens: int):
    """Shared driver to test any model with inference(X, kv_cache) method."""
    out, kv_cache = model.inference(prompt)
    last_token = out[:, -1:, :]
    generated = [prompt, last_token]

    for _ in range(num_tokens - 1):
        last_token, kv_cache = model.inference(last_token, kv_cache)
        generated.append(last_token)

    return torch.cat(generated, dim=1)


def test_forward(model, X, name: str):
    out = model.forward(X)
    assert out.shape == X.shape, f"{name} forward shape mismatch: {out.shape}"
    print(f"✓ {name} forward: {X.shape} -> {out.shape}")


def test_inference(model, X, expected_cache_shape: tuple, name: str):
    out, cache = model.inference(X)
    B, T, C = X.shape
    assert out.shape == (B, T, C), f"{name} inference output mismatch: {out.shape}"
    assert cache[0].shape == expected_cache_shape, f"{name} K cache mismatch: {cache[0].shape}"
    assert cache[1].shape == expected_cache_shape, f"{name} V cache mismatch: {cache[1].shape}"
    print(f"✓ {name} inference: output={out.shape}, K_cache={cache[0].shape}")
    return cache


def test_cached_inference(model, cache, B, C, expected_cache_shape: tuple, name: str):
    X_new = torch.randn(B, 1, C)
    out, cache2 = model.inference(X_new, cache)
    assert out.shape == (B, 1, C), f"{name} cached inference output mismatch: {out.shape}"
    assert cache2[0].shape == expected_cache_shape, f"{name} K cache after append mismatch: {cache2[0].shape}"
    assert cache2[1].shape == expected_cache_shape, f"{name} V cache after append mismatch: {cache2[1].shape}"
    print(f"✓ {name} cached inference: output={out.shape}, K_cache={cache2[0].shape}")


def test_generate(model, prompt_len: int, num_tokens: int, C: int, name: str):
    prompt = torch.randn(1, prompt_len, C)
    out = generate_with_cache(model, prompt, num_tokens)
    expected_len = prompt_len + num_tokens
    assert out.shape == (1, expected_len, C), f"{name} generate mismatch: {out.shape}"
    print(f"✓ {name} generate: prompt={prompt_len}, tokens={num_tokens} -> {out.shape}")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing MultiHeadAttention")
    print("=" * 50)

    mha = MultiHeadAttention(C=512, num_heads=8)
    X = torch.randn(2, 64, 512)

    test_forward(mha, X, "MHA")
    cache = test_inference(mha, X, expected_cache_shape=(2, 8, 64, 64), name="MHA")
    test_cached_inference(mha, cache, B=2, C=512, expected_cache_shape=(2, 8, 65, 64), name="MHA")

    mha = MultiHeadAttention(C=512, num_heads=8)
    test_generate(mha, prompt_len=100, num_tokens=50, C=512, name="MHA")

    print()
    print("=" * 50)
    print("Testing GroupedQueryAttention")
    print("=" * 50)

    gqa = GroupedQueryAttention(C=512, num_head=32, num_kv_head=8)
    X = torch.randn(2, 64, 512)

    test_forward(gqa, X, "GQA")

    # Verify KV weights are smaller (GQA-specific)
    assert gqa.W_q.weight.shape == (512, 512), f"W_q shape mismatch"
    assert gqa.W_k.weight.shape == (128, 512), f"W_k shape mismatch"
    assert gqa.W_v.weight.shape == (128, 512), f"W_v shape mismatch"
    print(f"✓ GQA weight sizes: W_q={gqa.W_q.weight.shape}, W_k={gqa.W_k.weight.shape}, W_v={gqa.W_v.weight.shape}")

    # GQA cache uses num_kv_head (8), d_head=16
    cache = test_inference(gqa, X, expected_cache_shape=(2, 8, 64, 16), name="GQA")
    test_cached_inference(gqa, cache, B=2, C=512, expected_cache_shape=(2, 8, 65, 16), name="GQA")

    gqa = GroupedQueryAttention(C=512, num_head=32, num_kv_head=8)
    test_generate(gqa, prompt_len=100, num_tokens=50, C=512, name="GQA")

    print()
    print("=" * 50)
    print("Cache Size Comparison (MHA vs GQA)")
    print("=" * 50)

    # For seq_len=1000, compare cache sizes
    seq_len = 1000
    d_head = 64
    mha_heads = 32
    gqa_kv_heads = 8

    mha_cache_size = 2 * mha_heads * seq_len * d_head  # K + V
    gqa_cache_size = 2 * gqa_kv_heads * seq_len * d_head  # K + V

    print(f"Sequence length: {seq_len}")
    print(f"MHA cache: {mha_heads} heads x {seq_len} x {d_head} x 2 = {mha_cache_size:,} values")
    print(f"GQA cache: {gqa_kv_heads} heads x {seq_len} x {d_head} x 2 = {gqa_cache_size:,} values")
    print(f"GQA saves: {mha_cache_size / gqa_cache_size:.0f}x smaller cache!")

    print()
    print("All tests passed!")

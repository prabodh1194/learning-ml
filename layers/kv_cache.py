import time

import torch


class SimpleAttention:
    def __init__(self, C: int):
        self.W_q = torch.nn.Linear(C, C)
        self.W_k = torch.nn.Linear(C, C)
        self.W_v = torch.nn.Linear(C, C)
        self.W_o = torch.nn.Linear(C, C)

    def forward(self, X: torch.Tensor):
        # X = B, T, C
        B, T, C = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        scores = Q @ K.transpose(-2, -1) / (C**0.5)

        mask = torch.tril(torch.ones((T, T)))
        scores = torch.masked_fill(scores, mask == 0, -torch.inf)

        attn = scores.softmax(dim=-1)

        out = attn @ V

        return self.W_o(out)

    def cache_forward(self, X: torch.Tensor, kv_cache: tuple | None = None):
        B, T, C = X.shape

        # new token will ask new queries
        Q = self.W_q(X)

        # but the previous tokens who need to attend to this query; their
        # response won't change, hence K & V values cam be reused for the
        # entire prompt.
        K_new = self.W_k(X)
        V_new = self.W_v(X)

        if kv_cache is not None:
            K_cached, V_cached = kv_cache
            K = torch.cat((K_cached, K_new), dim=1)
            V = torch.cat((V_cached, V_new), dim=1)
        else:
            K = K_new
            V = V_new

        scores = Q @ K.transpose(-2, -1) / (C**0.5)

        T_q, T_kv = Q.shape[1], K.shape[1]

        if T_q != 1:
            mask = torch.tril(torch.ones(T_q, T_kv), diagonal=T_kv - T_q)
            scores = torch.masked_fill(scores, mask == 0, -torch.inf)

        attn = scores.softmax(dim=-1)
        out = attn @ V

        return self.W_o(out), (K, V)


def generate_naive(model: SimpleAttention, prompt: torch.Tensor, num_tokens: int):
    generated = prompt.clone()

    for i in range(num_tokens):
        start = time.perf_counter()

        out = model.forward(generated)
        next_token = out[:, -1:, :]  # (B, 1, C)
        generated = torch.cat((generated, next_token), dim=1)

        elapsed = (time.perf_counter() - start) * 1000

        # Print every 20 tokens to see slowdown
        if i % 20 == 0:
            print(f"Token {i}: seq_len={generated.shape[1]}, time={elapsed:.2f}ms")

    return generated


def generate_with_cache(model: SimpleAttention, prompt: torch.Tensor, num_tokens: int):
    start = time.perf_counter()
    out, kv_cache = model.cache_forward(prompt, None)
    elapsed = (time.perf_counter() - start) * 1000
    last_token = out[:, -1:, :]
    print(f"Token 0: kv_cache_len={kv_cache[0].shape[1]}, time={elapsed:.2f}ms")
    generated = [prompt, last_token]

    for i in range(1, num_tokens):
        start = time.perf_counter()
        out, kv_cache = model.cache_forward(last_token, kv_cache)
        last_token = out

        generated.append(last_token)

        elapsed = (time.perf_counter() - start) * 1000

        if i % 20 == 0:
            print(
                f"Token {i}: kv_cache_len={kv_cache[0].shape[1]}, time={elapsed:.2f}ms"
            )

    return torch.cat(generated, dim=1)


if __name__ == "__main__":
    # Larger config to see the slowdown
    model = SimpleAttention(C=512)
    prompt = torch.randn(1, int(3e3), 512)  # batch=1, seq=100, dim=512

    print("Generating 200 tokens (watch time increase as seq grows)...")
    print("-" * 50)

    # out = generate_naive(model, prompt, num_tokens=200)
    out = generate_with_cache(model, prompt, num_tokens=200)

    print("-" * 50)
    print(f"Final shape: {out.shape}")  # should be (1, 300, 512)

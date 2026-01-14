"""
Multi-Head Attention

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

---
What is W_O ?

W_O is essential because:
- Enables head communication - Heads learn complementary patterns, W_O combines them
- Mixes information - Each output dimension uses all heads
- Learned combination - W_O learns optimal mixing strategy
- Recovers capacity - Maps from subspaces back to full space
- Architectural efficiency - Mixing happens once, not re-learned everywhere

Without W_O: Multi-head attention is just "multiple attention, stored side-by-side"
With W_O: Multi-head attention is "multiple perspectives, intelligently synthesized"

---
Parallelization Insights:

1. MULTI-HEAD PARALLELISM
   - Each head is completely independent during forward pass
   - No communication between heads until final concatenation
   - GPU can run all heads simultaneously: 8 heads = 8x speedup potential

   Sequential (NumPy):
     for head in range(num_heads):
         output[head] = Attention(Q[head], K[head], V[head])
     Total time = num_heads × single_head_time

   Parallel (GPU/MLX):
     output = vmap(attention)(Q_heads, K_heads, V_heads)
     Total time ≈ single_head_time

2. BATCH PARALLELISM
   - W matrices: (C, C) - no batch dimension! Shared across all samples
   - Input X: (B, T, C) - batch dimension preserved through matmul
   - Each batch item processes independently

   Benefits:
     - Fewer parameters (scales with C, not B)
     - Batch size can vary at inference
     - Massive parallelism: B independent computations

3. WHY TRANSFORMERS SCALE
   - 3 levels of parallelism: Batch (B) × Heads (H) × BLAS ops
   - Example: B=32, H=8, C=512 → 67M parallel ops
   - This is why transformers killed RNNs (which are sequential by nature)

   GPU execution:
     Head 0: [████████]
     Head 1: [████████]  ← All parallel!
     Head 2: [████████]
     ...

   vs NumPy (sequential):
     Head 0: [████████]
     Head 1:          [████████]
     Head 2:                   [████████]
     ...

4. MLX PARALLEL IMPLEMENTATION
   - Reshape (B, T, C) -> (B, H, T, C/H) to add head dimension
   - Single batched matmul across all heads simultaneously
   - No Python loop overhead
---
"""

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import torch

from layers.attention import ScaledDotProductAttention, AttentionCache
from layers.base import Layer


@dataclass
class MultiHeadAttentionCache:
    X: np.ndarray
    W: np.ndarray
    head_cache: list[AttentionCache]


class MultiHeadAttention(Layer):
    class np:
        @staticmethod
        def forward(
            Q: list[np.ndarray],
            K: list[np.ndarray],
            V: list[np.ndarray],
            W: np.ndarray,
            mask: np.ndarray = None,
        ) -> tuple[np.ndarray, MultiHeadAttentionCache]:
            heads: list[tuple[np.ndarray, AttentionCache]] = []

            for q_h, k_h, v_h in zip(Q, K, V):
                heads.append(ScaledDotProductAttention.np.forward(q_h, k_h, v_h, mask))

            X = np.concatenate(list(map(lambda x: x[0], heads)), axis=-1)

            return X @ W, MultiHeadAttentionCache(
                X, W, list(map(lambda x: x[1], heads))
            )

        @staticmethod
        def backward(
            dout: np.ndarray, cache: MultiHeadAttentionCache
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            dW = (cache.X.transpose(0, 2, 1) @ dout).sum(axis=0)
            dX = dout @ cache.W.T

            dX_h = np.split(dX, len(cache.head_cache), axis=-1)

            dQ, dK, dV = [], [], []

            for x, head_cache in zip(dX_h, cache.head_cache):
                dQ_h, dK_h, dV_h = ScaledDotProductAttention.np.backward(x, head_cache)
                dQ.append(dQ_h)
                dK.append(dK_h)
                dV.append(dV_h)

            return (
                np.concatenate(dQ, axis=-1),
                np.concatenate(dK, axis=-1),
                np.concatenate(dV, axis=-1),
                dW,
            )

    class torch:
        @staticmethod
        def forward(
            Q: tuple[torch.Tensor, ...],
            K: tuple[torch.Tensor, ...],
            V: tuple[torch.Tensor, ...],
            W: torch.Tensor,
            mask: torch.Tensor = None,
        ) -> torch.Tensor:
            heads: list[torch.Tensor] = []

            for q, k, v in zip(Q, K, V):
                heads.append(ScaledDotProductAttention.torch.forward(q, k, v, mask))

            return torch.concatenate(heads, dim=-1) @ W

    class mlx:
        @staticmethod
        def forward(
            Q: mx.array, K: mx.array, V: mx.array, W: mx.array, num_heads: int
        ) -> mx.array:
            """
            Parallel MHA: reshape to add head dim, run attention once across all heads.

            (B, T, C) -> (B, H, T, C/H) -> attention -> (B, T, C)
            """
            B, T, C = Q.shape
            head_dim = C // num_heads

            # Reshape: (B, T, C) -> (B, T, H, C/H) -> (B, H, T, C/H)
            Q = Q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
            K = K.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
            V = V.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

            # Attention in parallel: (B, H, T, C/H) @ (B, H, C/H, T) -> (B, H, T, T)
            scores = Q @ mx.transpose(K, axes=(0, 1, 3, 2)) / mx.sqrt(head_dim)
            weights = mx.softmax(scores, axis=-1)
            out = weights @ V  # (B, H, T, C/H)

            # Reshape back: (B, H, T, C/H) -> (B, T, H, C/H) -> (B, T, C)
            out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

            return out @ W


if __name__ == "__main__":
    np.random.seed(42)

    B, T, C = 4, 8, 512
    num_heads = 8

    Q = np.random.randn(B, T, C)
    K = np.random.randn(B, T, C)
    V = np.random.randn(B, T, C)
    W = np.random.randn(C, C)

    # numpy forward
    t0 = time.perf_counter()
    out, cache = MultiHeadAttention.np.forward(
        np.split(Q, num_heads, axis=-1),
        np.split(K, num_heads, axis=-1),
        np.split(V, num_heads, axis=-1),
        W,
    )
    t1 = time.perf_counter()
    print(f"numpy forward: {(t1-t0)*1000:.2f}ms")

    # numpy backward
    dout = np.random.randn(B, T, C)
    t0 = time.perf_counter()
    dQ_np, dK_np, dV_np, dW_np = MultiHeadAttention.np.backward(dout, cache)
    t1 = time.perf_counter()
    print(f"numpy backward: {(t1-t0)*1000:.2f}ms")

    # pt
    Q_pt = torch.tensor(Q, requires_grad=True)
    K_pt = torch.tensor(K, requires_grad=True)
    V_pt = torch.tensor(V, requires_grad=True)
    W_pt = torch.tensor(W, requires_grad=True)

    # pytorch forward
    t0 = time.perf_counter()
    out_pt = MultiHeadAttention.torch.forward(
        torch.chunk(Q_pt, num_heads, dim=-1),
        torch.chunk(K_pt, num_heads, dim=-1),
        torch.chunk(V_pt, num_heads, dim=-1),
        W_pt,
    )
    t1 = time.perf_counter()
    print(f"torch forward: {(t1-t0)*1000:.2f}ms")

    # pytorch backward
    t0 = time.perf_counter()
    out_pt.backward(torch.from_numpy(dout))
    t1 = time.perf_counter()
    print(f"torch backward: {(t1-t0)*1000:.2f}ms")

    # mlx
    Q_mx = mx.array(Q)
    K_mx = mx.array(K)
    V_mx = mx.array(V)
    W_mx = mx.array(W)

    # mlx forward (parallel - no loop!)
    t0 = time.perf_counter()
    out_mx = MultiHeadAttention.mlx.forward(Q_mx, K_mx, V_mx, W_mx, num_heads)
    mx.eval(out_mx)  # force evaluation
    t1 = time.perf_counter()
    print(f"mlx forward: {(t1-t0)*1000:.2f}ms")

    print()
    print("Q", np.allclose(dQ_np, Q_pt.grad.numpy()))
    print("K", np.allclose(dK_np, K_pt.grad.numpy()))
    print("V", np.allclose(dV_np, V_pt.grad.numpy()))
    print("W", np.allclose(dW_np, W_pt.grad.numpy()))
    print("MLX fwd", np.allclose(out, np.array(out_mx), atol=1e-5))

    # mlx backward using autograd
    # Note: mx.grad re-runs forward internally to trace the computation graph
    dout_mx = mx.array(dout)

    def mlx_mha_loss(Q, K, V, W):
        out = MultiHeadAttention.mlx.forward(Q, K, V, W, num_heads)
        return (out * dout_mx).sum()

    dQ_mx, dK_mx, dV_mx, dW_mx = mx.grad(mlx_mha_loss, argnums=(0, 1, 2, 3))(
        Q_mx, K_mx, V_mx, W_mx
    )

    print("MLX dQ", np.allclose(dQ_np, np.array(dQ_mx), atol=1e-4))
    print("MLX dK", np.allclose(dK_np, np.array(dK_mx), atol=1e-4))
    print("MLX dV", np.allclose(dV_np, np.array(dV_mx), atol=1e-4))
    print("MLX dW", np.allclose(dW_np, np.array(dW_mx), atol=1e-4))

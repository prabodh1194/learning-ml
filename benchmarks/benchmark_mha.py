"""
Benchmark: NumPy (sequential) vs MLX (parallel) vs PyTorch Multi-Head Attention

Demonstrates parallelization benefits at scale.
Small tensors: overhead dominates, NumPy wins
Large tensors: MLX/PyTorch parallelism wins (17-23x speedup)
"""

import time

import mlx.core as mx
import numpy as np
import torch

from layers.multihead_attention import MultiHeadAttention


def benchmark_scale():
    """Compare at various tensor sizes."""
    print("=" * 80)
    print("SCALE COMPARISON")
    print("=" * 80)

    # Warmup MLX
    _ = mx.array([1.0]) @ mx.array([1.0])
    mx.eval(_)

    for B, T, C in [(4, 8, 512), (32, 128, 512), (32, 256, 768), (64, 512, 768)]:
        num_heads = 8

        Q = np.random.randn(B, T, C)
        K = np.random.randn(B, T, C)
        V = np.random.randn(B, T, C)
        W = np.random.randn(C, C)

        # NumPy (sequential)
        t0 = time.perf_counter()
        out_np, _ = MultiHeadAttention.np.forward(
            np.split(Q, num_heads, axis=-1),
            np.split(K, num_heads, axis=-1),
            np.split(V, num_heads, axis=-1),
            W,
        )
        t1 = time.perf_counter()
        np_time = (t1 - t0) * 1000

        # PyTorch (MPS)
        device = torch.device("mps")
        Q_pt = torch.tensor(Q, device=device, dtype=torch.float32)
        K_pt = torch.tensor(K, device=device, dtype=torch.float32)
        V_pt = torch.tensor(V, device=device, dtype=torch.float32)
        W_pt = torch.tensor(W, device=device, dtype=torch.float32)

        # Warmup
        _ = MultiHeadAttention.torch.forward(
            torch.chunk(Q_pt, num_heads, dim=-1),
            torch.chunk(K_pt, num_heads, dim=-1),
            torch.chunk(V_pt, num_heads, dim=-1),
            W_pt,
        )

        t0 = time.perf_counter()
        out_pt = MultiHeadAttention.torch.forward(
            torch.chunk(Q_pt, num_heads, dim=-1),
            torch.chunk(K_pt, num_heads, dim=-1),
            torch.chunk(V_pt, num_heads, dim=-1),
            W_pt,
        )
        torch.mps.synchronize()
        t1 = time.perf_counter()
        pt_time = (t1 - t0) * 1000

        # MLX (parallel)
        Q_mx = mx.array(Q)
        K_mx = mx.array(K)
        V_mx = mx.array(V)
        W_mx = mx.array(W)

        # Warmup
        _ = MultiHeadAttention.mlx.forward(Q_mx, K_mx, V_mx, W_mx, num_heads)
        mx.eval(_)

        t0 = time.perf_counter()
        out_mx = MultiHeadAttention.mlx.forward(Q_mx, K_mx, V_mx, W_mx, num_heads)
        mx.eval(out_mx)
        t1 = time.perf_counter()
        mlx_time = (t1 - t0) * 1000

        print(
            f"B={B:2}, T={T:3}, C={C:3} | "
            f"NumPy: {np_time:7.2f}ms | PyTorch: {pt_time:7.2f}ms | MLX: {mlx_time:6.2f}ms"
        )


def benchmark_transformer(num_blocks: int = 96):
    """Simulate full transformer forward pass."""
    print()
    print("=" * 80)
    print(f"TRANSFORMER SIMULATION ({num_blocks} blocks)")
    print("=" * 80)

    B, T, C = 64, 512, 768
    num_heads = 8

    Q = np.random.randn(B, T, C)
    K = np.random.randn(B, T, C)
    V = np.random.randn(B, T, C)
    W = np.random.randn(C, C)

    # PyTorch setup (MPS)
    device = torch.device("mps")
    Q_pt = torch.tensor(Q, device=device, dtype=torch.float32)
    K_pt = torch.tensor(K, device=device, dtype=torch.float32)
    V_pt = torch.tensor(V, device=device, dtype=torch.float32)
    W_pt = torch.tensor(W, device=device, dtype=torch.float32)

    # MLX setup
    Q_mx = mx.array(Q)
    K_mx = mx.array(K)
    V_mx = mx.array(V)
    W_mx = mx.array(W)

    # Warmup
    for _ in range(3):
        _ = MultiHeadAttention.torch.forward(
            torch.chunk(Q_pt, num_heads, dim=-1),
            torch.chunk(K_pt, num_heads, dim=-1),
            torch.chunk(V_pt, num_heads, dim=-1),
            W_pt,
        )
        out = MultiHeadAttention.mlx.forward(Q_mx, K_mx, V_mx, W_mx, num_heads)
        mx.eval(out)

    print(f"Config: B={B}, T={T}, C={C}, heads={num_heads}")
    print()

    # NumPy: N blocks
    t0 = time.perf_counter()
    for _ in range(num_blocks):
        out_np, _ = MultiHeadAttention.np.forward(
            np.split(Q, num_heads, axis=-1),
            np.split(K, num_heads, axis=-1),
            np.split(V, num_heads, axis=-1),
            W,
        )
    t1 = time.perf_counter()
    np_time = t1 - t0
    print(f"NumPy (sequential loops): {np_time:.2f}s")

    # PyTorch: N blocks
    t0 = time.perf_counter()
    for _ in range(num_blocks):
        out_pt = MultiHeadAttention.torch.forward(
            torch.chunk(Q_pt, num_heads, dim=-1),
            torch.chunk(K_pt, num_heads, dim=-1),
            torch.chunk(V_pt, num_heads, dim=-1),
            W_pt,
        )
    torch.mps.synchronize()
    t1 = time.perf_counter()
    pt_time = t1 - t0
    print(f"PyTorch MPS (sequential): {pt_time:.2f}s")

    # MLX: N blocks
    t0 = time.perf_counter()
    for _ in range(num_blocks):
        out_mx = MultiHeadAttention.mlx.forward(Q_mx, K_mx, V_mx, W_mx, num_heads)
        mx.eval(out_mx)
    t1 = time.perf_counter()
    mlx_time = t1 - t0
    print(f"MLX (parallel reshape):   {mlx_time:.2f}s")

    print()
    print(f"MLX vs NumPy:   {np_time/mlx_time:.1f}x faster")
    print(f"MLX vs PyTorch: {pt_time/mlx_time:.1f}x faster")


if __name__ == "__main__":
    np.random.seed(42)
    benchmark_scale()
    benchmark_transformer(num_blocks=96)

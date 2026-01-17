"""
Visualize attention weights as a heatmap.
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_attention(
    weights: np.ndarray, title: str = "Attention Weights", save_path: str = None
):
    """
    Visualize attention weights as a heatmap.

    weights: (T, T) or (B, T, T) - if batched, uses first sample
    """
    if weights.ndim == 3:
        weights = weights[0]

    T = weights.shape[0]

    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap="Blues")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title(title)
    plt.xticks(range(T), [f"t{i}" for i in range(T)])
    plt.yticks(range(T), [f"t{i}" for i in range(T)])

    for i in range(T):
        for j in range(T):
            val = weights[i, j]
            if val > 0.01:
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    from layers.attention import ScaledDotProductAttention

    np.random.seed(42)

    B, T, C = 1, 8, 64
    Q = np.random.randn(B, T, C)
    K = np.random.randn(B, T, C)
    V = np.random.randn(B, T, C)

    # Causal mask
    mask = np.tril(np.ones((T, T)))

    out, cache = ScaledDotProductAttention.np.forward(Q, K, V, mask)

    visualize_attention(cache.weight, title="Causal Self-Attention Weights")

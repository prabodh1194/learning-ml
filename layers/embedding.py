from dataclasses import dataclass
import numpy as np
import torch

from layers.base import Layer, LayerGradients


@dataclass
class EmbeddingCache:
    indices: np.ndarray
    vocab_size: int


class Embedding(Layer):
    """
    Embedding layer - lookup table for token encoding.

    Forward: Y = W[indices]  (simple lookup)
    Backward: dW[indices] += dout  (accumulate at looked-up rows)

    No dX because indices are integers, not differentiable.
    """
    class np:
        @staticmethod
        def forward(
            indices: np.ndarray, W: np.ndarray
        ) -> tuple[np.ndarray, EmbeddingCache]:
            Y = W[indices]
            return Y, EmbeddingCache(indices, W.shape[0])

        @staticmethod
        def backward(dout: np.ndarray, cache: EmbeddingCache, W: np.ndarray) -> LayerGradients:
            dW = np.zeros_like(W)

            np.add.at(dW, cache.indices, dout)

            return LayerGradients(None, dW=dW)

    class torch:
        @staticmethod
        def forward(indices: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
            return W[indices]


if __name__ == "__main__":
    np.random.seed(42)

    vocab_size = 10
    embed_dim = 4
    batch_size = 3
    seq_length = 5

    # embedding matrix (numpy)
    W = np.random.randn(vocab_size, embed_dim)

    # indices
    indices = np.random.randint(vocab_size, size=(batch_size, seq_length))

    print("Testing Embedding...")
    print(f"W shape: {W.shape}")
    print(f"indices shape: {indices.shape}")
    print("=" * 40)

    # NumPy forward/backward
    Y_np, cache = Embedding.np.forward(indices, W)
    dout = np.ones_like(Y_np)
    grads_np = Embedding.np.backward(dout, cache, W)

    print(f"NumPy Y shape: {Y_np.shape}")
    print(f"NumPy dW shape: {grads_np.dW.shape}")
    print("=" * 40)

    # PyTorch forward/backward
    W_pt = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    indices_pt = torch.tensor(indices, dtype=torch.long)

    Y_pt = Embedding.torch.forward(indices_pt, W_pt)
    # pass dout = ones as upstream gradient (same as numpy test)
    Y_pt.backward(torch.ones_like(Y_pt))

    print(f"PyTorch Y shape: {Y_pt.shape}")
    print(f"PyTorch dW shape: {W_pt.grad.shape}")
    print("=" * 40)

    # Assertions
    assert np.allclose(Y_np, Y_pt.detach().numpy()), "Forward mismatch!"
    print("✓ Forward pass matches PyTorch!")

    assert np.allclose(grads_np.dW, W_pt.grad.numpy()), "dW mismatch!"
    print("✓ dW matches PyTorch!")

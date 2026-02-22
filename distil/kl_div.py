import torch
import numpy as np
import torch.nn.functional as F

r'''
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
           = Σ P(x) × (log(P(x)) - log(Q(x)))
           = Σ P(x) × log(P(x)) - Σ P(x) × log(Q(x))
             \_________________/  \_______________/
                    entropy         cross-entropy

- P = teacher (target distribution)
- Q = student (predicted distribution)

P, Q are already softmaxxed
'''


def kl_div_np(p: np.ndarray, q: np.ndarray) -> float:
    # Flatten to 2D (B*T, vocab) if needed, then mean over rows
    p = p.reshape(-1, p.shape[-1])
    q = q.reshape(-1, q.shape[-1])
    e = p * np.log(p)
    ce = p * np.log(q)
    return np.mean(np.sum(e - ce, axis=-1))  # sum over vocab, mean over tokens


def kl_div_pt(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Flatten (B, T, vocab) → (B*T, vocab) so batchmean divides by B*T
    p = p.view(-1, p.size(-1))
    q = q.view(-1, q.size(-1))
    return F.kl_div(q.log(), p, reduction="batchmean")


if __name__ == "__main__":
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.4, 0.3, 0.3])

    print(kl_div_np(p, q))
    print(kl_div_pt(torch.Tensor(p), torch.Tensor(q)))

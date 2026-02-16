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
    e = p * np.log(p)
    ce = p * np.log(q)
    return np.sum(e - ce)


def kl_div_pt(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return F.kl_div(q.log(), p, reduction="sum")


if __name__ == "__main__":
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.4, 0.3, 0.3])

    print(kl_div_np(p, q))
    print(kl_div_pt(torch.Tensor(p), torch.Tensor(q)))

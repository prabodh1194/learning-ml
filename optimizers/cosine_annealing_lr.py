"""
Cosine annealing oscillates between base_lr (high) and min_lr (low) following a smooth cosine curve.

LR
│
base_lr ─╮       ╭─       ╭─
         │╲     ╱│╲     ╱
         │ ╲   ╱ │ ╲   ╱
         │  ╲ ╱  │  ╲ ╱
min_lr ──│───╳───│───╳───
         └───────────────→ steps
         0  T_max  2*T_max

One full cosine cycle = T_max steps.

Why oscillate?
- Some research (warm restarts) shows that periodically "reheating" the LR helps escape local minima
- Each cycle can explore new regions of the loss landscape

For a single decay (no restart), you just run one cycle and stop at T_max.
"""

import math

import numpy as np
import torch


class CosineAnnealingLR:
    def __init__(self, base_lr: float, T_max: float, min_lr: float):
        self.base_lr = base_lr
        self.T_max = T_max
        self.min_lr = min_lr

    def get_lr(self, step: int):
        # project current step on the cosine curve.
        cos_factor = math.cos(math.pi * step / self.T_max)

        # phase shift by 1, so that min_factor becomes 0
        cos_factor += 1

        # however phase shift converts [-1, 1] to [0, 2].
        # convert to [0, 1] by division by 2
        cos_factor *= 0.5

        # interpolate b/w min & base LR
        return self.min_lr + cos_factor * (self.base_lr - self.min_lr)


if __name__ == "__main__":
    _base_lr, _T_max, _min_lr = 0.1, 10, 0.0

    scheduler = CosineAnnealingLR(_base_lr, _T_max, _min_lr)

    dummy_param = torch.tensor([1.0], requires_grad=True)
    opt = torch.optim.SGD([dummy_param], lr=_base_lr)
    pt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, _T_max)

    for step in range(_T_max * 2):
        yours = scheduler.get_lr(step)
        pt_lr = pt_scheduler.get_last_lr()[0]

        print(
            f"step: {step}, yours: {yours}, pt_lr: {pt_lr}", np.allclose(yours, pt_lr)
        )
        pt_scheduler.step()

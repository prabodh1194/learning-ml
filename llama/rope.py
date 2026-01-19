import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(self, *, dim: int, context_length: int):
        super().__init__()
        theta = torch.tensor([int(1e4) ** (-2 * i / dim) for i in range(dim // 2)])
        m = torch.arange(context_length)
        angles = torch.outer(m, theta)

        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)

        # these non-learnable params move to gpu on doing
        # register_buffer
        self.register_buffer("cos_theta", cos_theta)
        self.register_buffer("sin_theta", sin_theta)

    def forward(self, X: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        T = X.shape[1]

        x_even = X[..., ::2]
        x_odd = X[..., 1::2]

        sin_theta = self.sin_theta[start_pos : start_pos + T]
        cos_theta = self.cos_theta[start_pos : start_pos + T]

        if X.ndim == 4:
            # add the head-dim; (T, C) -> (T, 1, C) to align with (B, T, H, C)
            # broadcast happens at batch as well as head level.
            sin_theta = sin_theta.unsqueeze(1)
            cos_theta = cos_theta.unsqueeze(1)

        out_even = x_even * cos_theta - x_odd * sin_theta
        out_odd = x_even * sin_theta + x_odd * cos_theta

        out = torch.empty_like(X)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd

        return out

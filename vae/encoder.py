import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, *, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.activation(self.hidden_layer(x))

        mu = self.mu(x)

        # we interpret this output as log(var ^ 2) . no special calc is needed.
        # Linear layer outputs:  [-1.2, -0.8]     ← just numbers
        # We INTERPRET them as:  log σ² = [-1.2, -0.8]

        # When we need σ²:       σ² = exp([-1.2, -0.8]) = [0.30, 0.45]  ← always positive!
        # When we need σ:         σ  = exp([-1.2, -0.8] / 2) = [0.55, 0.67]

        log_var = self.log_var(x)
        return mu, log_var

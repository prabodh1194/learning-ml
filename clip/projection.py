import torch
from torch import nn
import torch.nn.functional as F

"""
Image encoder outputs: (B, 768)   ← lives in "vision land"                                                                                                                                                         
Text encoder outputs:  (B, 512)   ← lives in "language land"
                                                                                                                                                                                                                   
Can you compare them?  NO! Different sizes, different spaces.                                                                                                                                                      
                                                                                                                                                                                                                   
Projection heads fix this:                                                                                                                                                                                         
                                                                                                                                                                                                                   
  Image (B, 768) → Linear(768, 256) → L2 norm → (B, 256) ─┐                                                                                                                                                        
                                                             ├─ SAME space!
  Text  (B, 512) → Linear(512, 256) → L2 norm → (B, 256) ─┘

Now dot product = cosine similarity = "how similar are these?"

Why L2 normalize?

Without:  vec_a = [100, 200]    vec_b = [0.1, 0.2]
          dot product = 50      ← dominated by magnitude, not meaning

With L2 norm (length = 1):
          vec_a = [0.45, 0.89]  vec_b = [0.45, 0.89]
          dot product = 1.0     ← pure direction comparison

"""


class Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.proj_head = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_head(x)
        return F.normalize(x, dim=-1)

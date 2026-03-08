import torch

"""
The network needs to know which timestep it's denoising.
Same idea as positional encoding in transformers — turn an integer into a rich vector:

t=500 (integer) → sinusoidal embedding → (256,) vector → add into each block

**dim**:
  dim is just how long you want the fingerprint to be. It's a hyperparameter you choose.                                                                                                                             
                                                                                                
  dim = 8    →  embedding = [_, _, _, _, _, _, _, _]        short fingerprint                                                                                                                                        
  dim = 256  →  embedding = [_, _, _, _, ... 256 numbers]   rich fingerprint                                                                                                                                         
                                                                                                                                                                                                                     
  Bigger dim = more expressive fingerprint, but more parameters in the network.                 

  For our tiny U-Net, we'll use dim = 256. So every timestep integer gets turned into a 256-number vector:

  t=500 (one integer) → sinusoidal_embedding(t, dim=256) → (256,) vector
                                                  ^^^
                                              this is dim

  It's the same concept as d_model in your transformer — the "width" of the representation.
  
  Where this embedding goes

  timestep t=500
      │
      ▼
  sinusoidal_embedding  →  (B, 256) rich vector
      │
      ▼
  added into every block of the U-Net
  so the network knows "how noisy is this?"
  at every layer of processing

  Without this, the network would have no idea whether to remove a little noise (early t) or a lot of noise (late t).

"""


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    # t: (B,) integers
    # returns: (B, dim) embedding
    half = dim // 2
    freqs = 10000 ** (torch.arange(half, device=t.device) / half)
    args = t[:, None] / freqs[None, :]  # (B, 1) / (1, half) → (B, half)
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)

import torch

"""
  Imagine you have a crystal clear photo of a cat. Now imagine a process where you slowly pour static/snow over it:                                                                                                  
                                                                                                
  t=0          t=250        t=500        t=750        t=999
  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
  │  🐱    │  │ 🐱.·.  │  │ ·.·.·  │  │ ···.·· │  │ ······ │
  │ clear  │  │ a bit  │  │ half   │  │ barely │  │ pure   │
  │ photo  │  │ fuzzy  │  │ gone   │  │ visible│  │ static │
  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘

  What is t?

  t is just which step you're at in this corruption process.

  - t = 0 → "don't add any noise" → clean cat
  - t = 500 → "add medium noise" → half cat, half static
  - t = 999 → "add max noise" → pure static, cat is gone

  Each image in your batch can have a different t. That's why t has shape (B,) — one timestep per image.

  What is alpha_bar (ᾱ_t)?

  It's a lookup table that answers: "at step t, how much original image survives?"

  ᾱ_t:  a number between 1.0 and ~0.0

  t=0     ᾱ = 0.9999   →  99.99% cat,  0.01% noise
  t=250   ᾱ = 0.70     →  70% cat,     30% noise
  t=500   ᾱ = 0.30     →  30% cat,     70% noise
  t=999   ᾱ = 0.0001   →  0.01% cat,   99.99% noise

  alpha_bar has shape (T,) — one value per possible timestep.

  How they combine in the formula

  x_t  =  √(ᾱ_t) * x_0  +  √(1 - ᾱ_t) * ε
          ─────────────     ───────────────
          "how much cat     "how much noise
           survives"         gets added"

  At t=0: √(0.9999) * cat + √(0.0001) * noise → almost pure cat
  At t=999: √(0.0001) * cat + √(0.9999) * noise → almost pure noise

  Why t indexes into alpha_bar

  Each image in the batch has its own timestep:

  Batch of 4 images:
    image_0 got t=100  →  grab ᾱ[100] = 0.85
    image_1 got t=500  →  grab ᾱ[500] = 0.30
    image_2 got t=50   →  grab ᾱ[50]  = 0.95
    image_3 got t=900  →  grab ᾱ[900] = 0.01

  That's why you need alpha_bar[t] — you're picking the right "how much cat survives" value for each
  image's specific timestep.
  
  Now, each image in the batch gets one ᾱ value, and that single value scales every pixel across
  every channel of that image.                                                                                    
                                                                                                
  alpha_bar[t].view(-1,1,1,1)        x_0                                                                                                                                                                             
                                                                                                
   (B, 1, 1, 1)                  (B, C, H, W)                                                                                                                                                                        
  ┌─────────┐                   ┌─────────────┐                                                 
  │ 0.85    │  ──broadcasts──▶  │ all pixels  │  image 0
  │ 0.30    │  ──broadcasts──▶  │ all pixels  │  image 1
  │ 0.95    │  ──broadcasts──▶  │ all pixels  │  image 2
  │ 0.01    │  ──broadcasts──▶  │ all pixels  │  image 3
  └─────────┘                   └─────────────┘


"""


def forward_diffusion(
    *, x_0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x_0:       (B, C, H, W) clean images
    t:         (B,) timestep indices
    alpha_bar: (T,) precomputed cumulative product values

    returns: (x_t, noise)  — noisy image and the noise that was added
    """
    # 1. sample ε ~ N(0, 1) same shape as x_0
    # 2. grab ᾱ_t for each sample in the batch
    # 3. x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
    # 4. return x_t, ε   (we return ε because the model will learn to predict it!)

    eps = torch.randn_like(x_0)
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
    x_t = alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * eps

    return x_t, eps

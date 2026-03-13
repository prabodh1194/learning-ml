"""
Training taught the U-Net to predict noise. Now you undo the noise step-by-step.

Training: clean image → add noise → predict noise (learn)
Sampling: pure noise → predict noise → remove it → repeat 1000 times → image!

The reverse process:
  x_T ~ N(0,1)           start from pure noise
    │
    ▼  (repeat T times, t = T-1 down to 0)
  ε_pred = UNet(x_t, t)  predict the noise
  remove ε_pred from x_t  denoise one step
  add tiny fresh noise    (except at t=0)
    │
    ▼
  x_0 = generated image!

The one-step formula:

  x_{t-1} = (1/√α_t) * (x_t  -  β_t/√(1-ᾱ_t) * ε_pred)  +  σ_t * z

  where:
    α_t, β_t, ᾱ_t  = from noise schedule
    ε_pred          = UNet's prediction of the noise
    z               = fresh random noise (zero at t=0)
    σ_t             = √β_t (the noise scale)

"""

import torch

from diffusion.unet import UNet


@torch.no_grad()
def sample(
    model: UNet,
    T: int,
    *,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: str,
    n_images: int,
):
    model.eval()

    # 1. pure noise - 32 * 32 RGB image
    x = torch.randn(n_images, 3, 32, 32, device=device)

    # 2. start sampling noise
    for t in reversed(range(T)):
        # 3. predict noise at this t.
        eps_pred = model(
            x, torch.full((n_images,), t, device=device, dtype=torch.float32)
        )

        # "how much noise to remove" — scaled prediction
        noise_removal = beta[t] / (1 - alpha_bar[t]).sqrt() * eps_pred

        # "take the denoised step" — remove noise and scale
        x_denoised = (1 / alpha[t].sqrt()) * (x - noise_removal)

        # "add fresh randomness" — keeps sampling stochastic
        z = torch.randn_like(x) if t > 0 else 0

        # 4. denoise at t & add some new noise.
        x = x_denoised + beta[t].sqrt() * z

        if t % 10 == 0:
            from datetime import datetime

            print(f"[{datetime.now().strftime('%H:%M:%S')}] step {T - t}/{T}")

    # 5. clamp to [-1, 1] and rescale to [0, 1] for visualization
    return (x.clamp(-1, 1) + 1) / 2


if __name__ == "__main__":
    from torchvision.utils import save_image
    from diffusion.noise_schedule import linear_noise_schedule

    device = "mps"
    T = 1000

    beta, alpha, alpha_bar = linear_noise_schedule(T)
    beta, alpha, alpha_bar = beta.to(device), alpha.to(device), alpha_bar.to(device)

    model = UNet().to(device)
    ckpt = torch.load(
        "diffusion/checkpoints/ddpm_step_3600.pt",
        weights_only=True,
        map_location=device,
    )
    model.load_state_dict(ckpt["model"])

    # 3840x2160 with 32x32 tiles = 120x68 = 8160 tiles
    # generate 20%, then randomly duplicate to fill
    n_cols, n_rows = 120, 68
    n_total = n_cols * n_rows
    n_unique = int(n_total * 0.2)  # 3264

    images = sample(
        model,
        T,
        beta=beta,
        alpha=alpha,
        alpha_bar=alpha_bar,
        device=device,
        n_images=n_unique,
    )

    # randomly pick from generated images to fill the full grid
    indices = torch.randint(0, n_unique, (n_total,))
    grid = images[indices]

    save_image(grid, "diffusion/outputs/wallpaper_4k.png", nrow=n_cols, padding=0)
    print("saved → diffusion/outputs/wallpaper_4k.png")

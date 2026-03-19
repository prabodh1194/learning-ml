import torch
from torchvision.utils import save_image

from diffusion.noise_schedule import linear_noise_schedule
from latent_diffusion.conv_vae import ConvVAE
from latent_diffusion.unet import UNet

DEVICE = "mps"
T = 1000


@torch.no_grad()
def sample(unet, vae, *, beta, alpha, alpha_bar, device, n_images):
    unet.eval()
    vae.eval()

    # 1. pure noise in latent space (8, 8, 8)
    z = torch.randn(n_images, 8, 8, 8, device=device)

    # 2. reverse diffusion loop
    for t in reversed(range(T)):
        t_batch = torch.full((n_images,), t, device=device, dtype=torch.float32)
        eps_pred = unet(x=z, t=t_batch)

        noise_removal = beta[t] / (1 - alpha_bar[t]).sqrt() * eps_pred
        z_denoised = (1 / alpha[t].sqrt()) * (z - noise_removal)

        fresh_noise = torch.randn_like(z) if t > 0 else 0
        z = z_denoised + beta[t].sqrt() * fresh_noise

        if t % 100 == 0:
            print(f"  step {T - t}/{T}")

    # 3. decode latents to images
    images = vae.decoder(z)

    # 4. rescale from [-1, 1] to [0, 1]
    return (images.clamp(-1, 1) + 1) / 2


if __name__ == "__main__":
    beta, alpha, alpha_bar = linear_noise_schedule(T)
    beta, alpha, alpha_bar = beta.to(DEVICE), alpha.to(DEVICE), alpha_bar.to(DEVICE)

    # load VAE
    vae = ConvVAE().to(DEVICE)
    vae_ckpt = torch.load(
        "latent_diffusion/checkpoints/conv_vae_final.pt", map_location=DEVICE
    )
    vae.load_state_dict(vae_ckpt["model"])

    # load U-Net
    unet = UNet(in_ch=8).to(DEVICE)
    ldm_ckpt = torch.load(
        "latent_diffusion/checkpoints/ldm_final.pt", map_location=DEVICE
    )
    unet.load_state_dict(ldm_ckpt["model"])

    images = sample(
        unet,
        vae,
        beta=beta,
        alpha=alpha,
        alpha_bar=alpha_bar,
        device=DEVICE,
        n_images=16,
    )
    save_image(images, "latent_diffusion/outputs/ldm_samples.png", nrow=4)
    print("saved → latent_diffusion/outputs/ldm_samples.png")

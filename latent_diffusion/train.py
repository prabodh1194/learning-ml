import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusion.forward import forward_diffusion
from diffusion.noise_schedule import linear_noise_schedule
from latent_diffusion.conv_vae import ConvVAE
from latent_diffusion.unet import UNet

DEVICE = "mps"
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 50
T = 1000
CHECKPOINT_EVERY = 300


def get_cifar10():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root="./datasets_cache", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train():
    # 1. load frozen VAE
    vae = ConvVAE().to(DEVICE)
    checkpoint = torch.load(
        "latent_diffusion/checkpoints/conv_vae_final.pt", map_location=DEVICE
    )
    vae.load_state_dict(checkpoint["model"])
    vae.requires_grad_(False)
    vae.eval()

    # 2. noise schedule
    beta, alpha, alpha_bar = linear_noise_schedule(T)
    alpha_bar = alpha_bar.to(DEVICE)

    # 3. U-Net for latent diffusion
    unet = UNet(in_ch=4).to(DEVICE)
    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

    dataloader = get_cifar10()
    global_step = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for batch_images, _ in dataloader:
            batch_images = batch_images.to(DEVICE)

            # encode to latent space (use mu, skip reparameterization)
            with torch.no_grad():
                mu, _ = vae.encoder(batch_images)
            z_0 = mu

            # pick random timesteps
            t = torch.randint(0, T, (z_0.shape[0],), device=DEVICE)

            # forward diffusion on latents
            z_t, noise = forward_diffusion(x_0=z_0, t=t, alpha_bar=alpha_bar)

            # predict noise
            predicted_noise = unet(x=z_t, t=t)

            # loss
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % CHECKPOINT_EVERY == 0:
                torch.save(
                    {
                        "model": unet.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                    f"latent_diffusion/checkpoints/ldm_step_{global_step}.pt",
                )
                print(f"  checkpoint saved at step {global_step}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"epoch {epoch + 1}/{EPOCHS}  loss: {avg_loss:.4f}")

    torch.save(
        {
            "model": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": EPOCHS,
            "step": global_step,
        },
        "latent_diffusion/checkpoints/ldm_final.pt",
    )
    print("training done. saved ldm_final.pt")


if __name__ == "__main__":
    train()

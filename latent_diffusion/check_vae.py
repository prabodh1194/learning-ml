import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from latent_diffusion.conv_vae import ConvVAE

DEVICE = "mps"


def check_reconstructions():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root="./datasets_cache", train=True, download=True, transform=transform
    )

    vae = ConvVAE().to(DEVICE)
    ckpt = torch.load(
        "latent_diffusion/checkpoints/conv_vae_final.pt", map_location=DEVICE
    )
    vae.load_state_dict(ckpt["model"])
    vae.eval()

    # grab 8 images
    images = torch.stack([dataset[i][0] for i in range(8)]).to(DEVICE)

    with torch.no_grad():
        recons, mu, log_var = vae(images)

    # rescale [-1, 1] → [0, 1]
    originals = (images + 1) / 2
    reconstructed = (recons.clamp(-1, 1) + 1) / 2

    # top row: originals, bottom row: reconstructions
    grid = torch.cat([originals, reconstructed], dim=0)
    save_image(grid, "latent_diffusion/outputs/vae_recons.png", nrow=8)
    print("saved → latent_diffusion/outputs/vae_recons.png")


if __name__ == "__main__":
    check_reconstructions()

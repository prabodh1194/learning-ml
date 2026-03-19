import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from latent_diffusion.conv_vae import ConvVAE, conv_vae_loss

DEVICE = "mps"
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 25
CHECKPOINT_EVERY = 300


def get_cifar10():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
        ]
    )
    dataset = datasets.CIFAR10(
        root="./datasets_cache", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train():
    model = ConvVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dataloader = get_cifar10()

    global_step = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for batch_images, _ in dataloader:
            batch_images = batch_images.to(DEVICE)

            reconstructed, mu, log_var = model(batch_images)
            loss = conv_vae_loss(
                orig_image=batch_images,
                cons_image=reconstructed,
                mu=mu,
                log_var=log_var,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % CHECKPOINT_EVERY == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                    f"latent_diffusion/checkpoints/conv_vae_step_{global_step}.pt",
                )
                print(f"  checkpoint saved at step {global_step}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"epoch {epoch + 1}/{EPOCHS}  loss: {avg_loss:.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": EPOCHS,
            "step": global_step,
        },
        "latent_diffusion/checkpoints/conv_vae_final.pt",
    )
    print("training done. saved conv_vae_final.pt")


if __name__ == "__main__":
    train()

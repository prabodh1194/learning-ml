import torch


def extract_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, C, H, W = images.shape
    P = patch_size
    gh = H // P
    gw = W // P

    # 1. B, C, H, W
    # 2. B, C, gh, P, gw, P
    # 3. B, gh, gw, P, P, C
    # 4. B, N, P * P * C
    patches = images.reshape(B, C, gh, P, gw, P)
    return patches.permute(0, 2, 4, 3, 5, 1).reshape(B, gh * gw, P * P * C)


if __name__ == "__main__":
    _shape = 2, 3, 32, 32
    images = torch.randn(_shape)

    patches = extract_patches(images, 4)

    assert patches.shape == (2, 64, 48)

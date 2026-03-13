import numpy as np
import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d

from mnist.mnist_loader.loader import load_mnist


def naive_conv2d(
    *, image: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1
) -> np.ndarray:
    """
    A kernel is just a small grid of numbers you chose on purpose to detect a specific pattern.

    Think of it like a stencil — you slide it over the image and at each position it "asks a question" about the pixels
    underneath.

    Vertical edge kernel:        What it's asking:

     -1   0   1                  "Is the left side different
     -2   0   2                   from the right side?"
     -1   0   1

    Left pixels get multiplied by negative numbers.
    Right pixels get multiplied by positive numbers.

    If left ≈ right  →  negatives cancel positives  →  output ≈ 0  (no edge)
    If left ≠ right  →  they DON'T cancel           →  output is big (edge!)

    Another example:

    Blur kernel:                 What it's asking:

     1/9  1/9  1/9               "What's the average of
     1/9  1/9  1/9                all 9 neighbors?"
     1/9  1/9  1/9

    Every pixel gets equal weight → output = average → smooth/blurry

    So "creating a kernel" = picking numbers that encode the pattern you care about. The numbers ARE the detector. There's
    nothing more to it — just a small array of weights you designed by hand.

    Later in CNNs, the network learns these kernel numbers via backprop instead of you hand-picking them. That's the whole
    magic.

    """
    image = np.pad(image, padding)

    H, W = image.shape
    K, _ = kernel.shape
    output = np.zeros(((H - K) // stride + 1, (W - K) // stride + 1))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            patch = image[i * stride : i * stride + K, j * stride : j * stride + K]
            output[i, j] = (patch * kernel).sum()

    return output


def multichannel_conv2d(*, image: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    C_in, H, W = image.shape
    C_out, _, K, _ = kernels.shape  # (C_out, C_in, K, K)
    output = np.zeros((C_out, H - K + 1, W - K + 1))

    for o in range(C_out):
        for c in range(C_in):
            output[o] += naive_conv2d(image=image[c], kernel=kernels[o][c])

    return output


def conv_transpose(image: torch.Tensor) -> torch.Tensor:
    """
    What is ConvTranspose2d actually doing?

    In regular conv with stride=2, you read a patch and skip positions → output gets smaller:

    Input (6x6):
    [x x x] . . .     → output[0]
    . . [x x x] .     → output[1]
    . . . . [x x x]   → output[2]

    6 → 3  (halved)

    ConvTranspose2d with stride=2 does the reverse operation. Each input value gets multiplied by the kernel and placed into the output, spaced stride apart:

    Input (3 values):  [A]  [B]  [C]

    Step 1: Place A's kernel output starting at position 0
    Step 2: Place B's kernel output starting at position 2  (stride=2)
    Step 3: Place C's kernel output starting at position 4

    With kernel_size=2:
      A writes to [0, 1]
      B writes to [2, 3]
      C writes to [4, 5]

    Output: 6 values.  3 → 6 (doubled)

    Where patches overlap, values get summed.

    The formula

    output = (input - 1) * stride - 2*padding + kernel_size

    With input=14, stride=2, padding=0, kernel=2:
      (14 - 1) * 2 - 0 + 2 = 28

    What to write

    A function that:
    1. Takes a (1, 1, 14, 14) tensor (fake data, just torch.randn)
    2. Passes through ConvTranspose2d(1, 1, kernel_size=2, stride=2)
    3. Asserts output is (1, 1, 28, 28)
    4. Then does the round-trip: Conv2d(stride=2) down, ConvTranspose2d(stride=2) back up, assert same spatial size

    The (1, 1, 14, 14) means: batch=1, channels=1, height=14, width=14. PyTorch conv layers always expect this 4D shape.

    Go ahead and try again.
    """
    cn = ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
    val = cn(image)

    return val


if __name__ == "__main__":
    X_train, _, X_test, _ = load_mnist(path="mnist-dataset")
    vertical_edge_detector_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical_out = naive_conv2d(image=X_train[0], kernel=vertical_edge_detector_kernel)

    horizontal_edge_detector_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    horizontal_out = naive_conv2d(
        image=X_train[0], kernel=horizontal_edge_detector_kernel
    )

    blurry_kernel = np.array(
        [
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
        ]
    )
    blurry_out = naive_conv2d(image=X_train[0], kernel=blurry_kernel)

    # Verify output shapes match the formula:
    # output = floor((input + 2*padding - kernel) / stride + 1)
    image = X_train[0]
    kernel = vertical_edge_detector_kernel

    assert naive_conv2d(image=image, kernel=kernel, padding=0, stride=1).shape == (
        26,
        26,
    )
    assert naive_conv2d(image=image, kernel=kernel, padding=1, stride=1).shape == (
        28,
        28,
    )
    assert naive_conv2d(image=image, kernel=kernel, padding=0, stride=2).shape == (
        13,
        13,
    )
    assert naive_conv2d(image=image, kernel=kernel, padding=1, stride=2).shape == (
        14,
        14,
    )

    print("All shape checks passed!")

    # Verify multichannel_conv2d against torch.nn.Conv2d

    # MNIST image: (28, 28) → (1, 28, 28) to simulate 1 input channel
    mc_image = X_train[0].astype(np.float32)[np.newaxis, :, :]  # (1, 28, 28)

    conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, bias=False)
    weights = conv.weight.detach().numpy()  # (16, 1, 3, 3)

    our_output = multichannel_conv2d(image=mc_image, kernels=weights)
    torch_output = (
        conv(torch.tensor(mc_image[np.newaxis])).detach().numpy()[0]
    )  # remove batch dim

    assert our_output.shape == torch_output.shape, (
        f"{our_output.shape} != {torch_output.shape}"
    )
    assert np.allclose(our_output, torch_output, atol=1e-5), "Values don't match torch!"

    print(
        f"Multichannel check passed! Shape: {our_output.shape}, values match torch.nn.Conv2d"
    )

    assert conv_transpose(torch.randn((1, 1, 14, 14))).shape == (1, 1, 28, 28)

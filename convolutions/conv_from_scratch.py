import numpy as np

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

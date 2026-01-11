"""
Very simple MNIST dataset loader that reads the raw binary files.

MNIST files are stored in IDX format - a simple binary format for vectors and matrices.
Each file starts with a header describing the data, followed by the raw data bytes.
"""

import numpy as np
import numpy.typing as npt
import struct
import os


def read_idx_images(filename: str) -> npt.NDArray[np.uint8]:
    """
    Read MNIST image file in IDX3 format.

    File structure:
    - Bytes 0-3: Magic number (2051 for images)
    - Bytes 4-7: Number of images
    - Bytes 8-11: Number of rows (28)
    - Bytes 12-15: Number of columns (28)
    - Bytes 16+: Pixel values (0-255), one byte per pixel

    What's happening:
    1. Open the file in binary read mode
    2. Read the 4-byte header values as big-endian unsigned integers ('>I')
    3. Read all remaining bytes as unsigned 8-bit integers (pixel values)
    4. Reshape the flat array into (num_images, 28, 28)
    """
    with open(filename, "rb") as f:
        # Read 4 bytes, interpret as big-endian unsigned int
        # '>I' means: '>' = big-endian, 'I' = unsigned int (4 bytes)
        _magic = struct.unpack(">I", f.read(4))[0]
        num_images = struct.unpack(">I", f.read(4))[0]
        num_rows = struct.unpack(">I", f.read(4))[0]
        num_cols = struct.unpack(">I", f.read(4))[0]

        # Read all remaining bytes as unsigned 8-bit integers
        # Each pixel is one byte with value 0-255 (grayscale)
        images = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape from flat array to (num_images, rows, cols)
        images = images.reshape(num_images, num_rows, num_cols)

    return images


def read_idx_labels(filename: str) -> npt.NDArray[np.uint8]:
    """
    Read MNIST label file in IDX1 format.

    File structure:
    - Bytes 0-3: Magic number (2049 for labels)
    - Bytes 4-7: Number of labels
    - Bytes 8+: Label values (0-9), one byte per label

    What's happening:
    1. Open the file in binary read mode
    2. Read the 2 header values (magic number and count)
    3. Read all remaining bytes as the label values (0-9)
    """
    with open(filename, "rb") as f:
        _magic = struct.unpack(">I", f.read(4))[0]
        _num_labels = struct.unpack(">I", f.read(4))[0]

        # Each label is one byte (0-9 for the digit)
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def load_mnist(
    path: str = "mnist-dataset",
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
]:
    """
    Load the MNIST dataset from raw binary files.

    Args:
        path: Directory containing the MNIST .idx files

    Returns:
        X_train: Training images (60000, 28, 28) numpy array, dtype=uint8
        y_train: Training labels (60000,) numpy array, dtype=uint8
        X_test: Test images (10000, 28, 28) numpy array, dtype=uint8
        y_test: Test labels (10000,) numpy array, dtype=uint8

    What's happening:
    1. Load training images (60,000 images of 28x28 pixels)
    2. Load training labels (60,000 labels, values 0-9)
    3. Load test images (10,000 images)
    4. Load test labels (10,000 labels)
    5. Return all four arrays
    """
    # Training data: 60,000 examples
    X_train = read_idx_images(os.path.join(path, "train-images.idx3-ubyte"))
    y_train = read_idx_labels(os.path.join(path, "train-labels.idx1-ubyte"))

    # Test data: 10,000 examples
    X_test = read_idx_images(os.path.join(path, "t10k-images.idx3-ubyte"))
    y_test = read_idx_labels(os.path.join(path, "t10k-labels.idx1-ubyte"))

    return X_train, y_train, X_test, y_test

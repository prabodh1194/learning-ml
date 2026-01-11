import numpy as np
import numpy.typing as npt


def render_svg(image: npt.NDArray[np.uint8], scale: int = 10) -> str:
    """
    Convert a single MNIST image to SVG format.

    Args:
        image: A 28x28 numpy array with pixel values 0-255
        scale: How many pixels wide/tall each MNIST pixel should be

    Returns:
        SVG string that can be saved to a file or embedded in HTML

    What's happening:
    1. Create SVG header with proper dimensions (28 * scale)
    2. For each pixel in the 28x28 image:
       - Calculate its position (row, col)
       - Convert pixel value (0-255) to grayscale color
       - Create a rectangle at that position with that color
    3. Close the SVG tag

    Why scale? Each MNIST pixel is tiny (1x1). Scale=10 means each
    pixel becomes a 10x10 square, making a 280x280 SVG image.
    """
    width = height = 28 * scale
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{width}" height="{height}" fill="black"/>',  # Background
    ]

    # Loop through each pixel
    for row in range(28):
        for col in range(28):
            pixel_value = image[row, col]
            # Convert 0-255 to grayscale color
            # MNIST has white digits on black background
            color = f"rgb({pixel_value},{pixel_value},{pixel_value})"

            # Create a rectangle for this pixel
            x = col * scale
            y = row * scale
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{scale}" height="{scale}" fill="{color}"/>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def render_svg_grid(
    X: npt.NDArray[np.uint8],
    y: npt.NDArray[np.uint8],
    n_samples: int = 10,
    scale: int = 8,
) -> str:
    """
    Create an SVG grid showing multiple MNIST samples with labels.

    Args:
        X: Images array (N, 28, 28)
        y: Labels array (N,)
        n_samples: How many samples to show
        scale: Pixel scaling factor

    Returns:
        SVG string showing a grid of images with labels

    What's happening:
    1. Calculate grid layout (images in a row)
    2. For each sample:
       - Render the image as SVG
       - Add a text label above it
    3. Combine all images into one large SVG
    """
    img_size = 28 * scale
    padding = 20
    label_height = 20

    # Grid layout: all in one row
    total_width = n_samples * (img_size + padding)
    total_height = img_size + label_height + padding

    svg_parts = [
        f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{total_width}" height="{total_height}" fill="white"/>',
    ]

    for i in range(n_samples):
        x_offset = i * (img_size + padding)
        y_offset = label_height

        # Add label text
        svg_parts.append(
            f'<text x="{x_offset + img_size // 2}" y="{label_height - 5}" '
            f'text-anchor="middle" font-family="monospace" font-size="14" fill="black">'
            f"Label: {y[i]}</text>"
        )

        # Add image
        for row in range(28):
            for col in range(28):
                pixel_value = X[i, row, col]
                if pixel_value > 0:  # Only draw non-black pixels (optimization)
                    color = f"rgb({pixel_value},{pixel_value},{pixel_value})"
                    rect_x = x_offset + col * scale
                    rect_y = y_offset + row * scale
                    svg_parts.append(
                        f'<rect x="{rect_x}" y="{rect_y}" width="{scale}" height="{scale}" fill="{color}"/>'
                    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def save_svg(svg_content: str, filename: str) -> None:
    """
    Save SVG content to a file.

    Args:
        svg_content: SVG string from render_svg() or render_svg_grid()
        filename: Output filename (e.g., 'digits.svg')

    What's happening:
    1. Open file in write mode
    2. Write the SVG string
    3. File can now be opened in any web browser
    """
    with open(filename, "w") as f:
        f.write(svg_content)
    print(f"Saved to {filename}")

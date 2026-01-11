from mnist.mnist_loader.loader import load_mnist
from mnist.mnist_loader.renderer import render_svg, save_svg, render_svg_grid

if __name__ == "__main__":
    # Example usage: run this file directly to test the loader
    X_train, y_train, X_test, y_test = load_mnist()

    print(f"Training images: {X_train.shape}")  # Should be (60000, 28, 28)
    print(f"Training labels: {y_train.shape}")  # Should be (60000,)
    print(f"Test images: {X_test.shape}")  # Should be (10000, 28, 28)
    print(f"Test labels: {y_test.shape}")  # Should be (10000,)
    print(f"\nPixel value range: {X_train.min()} to {X_train.max()}")

    # Create SVG visualizations
    print("\n" + "=" * 50)
    print("Creating SVG files...")
    print("=" * 50)

    # Save a single digit
    svg_single = render_svg(X_train[0], scale=10)
    save_svg(svg_single, "mnist_single.svg")

    # Save a grid of 10 digits
    svg_grid = render_svg_grid(X_train, y_train, n_samples=10, scale=8)
    save_svg(svg_grid, "mnist_grid.svg")

    print("\nOpen these files in your browser to see the digits!")

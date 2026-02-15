"""
Simple loss plotter.

Usage: python -m sft.plot_loss
"""

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "training_log.csv"

df = pd.read_csv(CSV_PATH)

if len(df) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["loss"], "b-", linewidth=1)
    ax.scatter(df["step"], df["loss"], c="red", s=20)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss (latest: {df['loss'].iloc[-1]:.4f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

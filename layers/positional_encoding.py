"""

PE(pos, 2i)   = sin(pos / 10000^(2i/C))
PE(pos, 2i+1) = cos(pos / 10000^(2i/C))

Shapes:
    T = sequence length
    C = embedding dimension (must be even)
    output: (T, C)

Positional Encoding: represent every token in sequence with an embedding vector ot C-dim.
The vector is filled by alternating sin & cos terms. 0th position is a sin term; 1st position is cos; 2nd is sin and so on.

sin / cos are generated as follows:
for a token; for the C - dim vector, select an index i & generate 10000 ^ (2i/|C|). Then again multiple this term with the index of the token & do alternating sin/cos

---
ELI5: Why Sinusoidal (Sin/Cos) Positional Encodings?

The Core Idea: Position as a Unique "Barcode"
Think of positional encoding like giving each position in a sentence a unique barcode made of waves at different speeds.

Analogy: Clock Hands
Imagine a clock with many hands spinning at different speeds:

Position 0:  All hands at 12 o'clock
Position 1:  Fast hand moved a bit, slow hand barely moved
Position 2:  Fast hand moved more, slow hand still slow
Position 10: Fast hand did full cycle, slow hand moved a bit
Position 100: Fast hand did many circles, slow hand halfway

Each position has a unique combination of hand positions!

---
Different "Frequencies" for Different Dimensions

Dimension 0:  sin(pos / 10000^0) = sin(pos / 1)        <- FAST wave
Dimension 64: sin(pos / 10000^(64/512)) = sin(pos / 21.5) <- SLOWER
Dimension 510: sin(pos / 10000^(510/512)) = sin(pos / 9102) <- VERY SLOW

---
Why This Is Brilliant:

1. Every Position Gets Unique Pattern - No two positions have the same pattern!

2. Relative Position is Learnable - PE(pos + k) can be written as LINEAR FUNCTION of PE(pos)
   Because of sin/cos addition formulas: sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
   The model can learn this matrix through training!

3. Works for Any Length - Unlike learned embeddings (limited to max length),
   sinusoidal works for any position. You can use a model trained on 512 tokens for 1000 token sequences!

---
Why Sin AND Cos (Even/Odd Pairs)?

Sin and cos are 90 deg out of phase. Together they form a "2D rotation":
[sin(t), cos(t)] traces a circle as t increases

This gives more information than just sin alone:
- Sin alone: can't distinguish pos=0 from pos=2*pi
- Sin+cos: unique point on circle for each position!

---
Analogy: Binary Numbers (But Smooth)

Binary:       0: 0000, 1: 0001, 2: 0010, 3: 0011, 4: 0100
Sinusoidal:   Like binary but smooth and continuous!
              - Lower dims change fast (like rightmost bit)
              - Higher dims change slow (like leftmost bit)

---
TL;DR:

1. Unique pattern for each position
2. Encodes relative position (model can learn "k positions away")
3. Works for any length (not limited to training length)
4. Multiple frequencies (like binary but smooth)
5. Sin+Cos pairs (full 2D information per frequency)

It's like giving each word a unique "location code" made of waves at different speeds,
and the model can learn to read these codes to understand where words are relative to each other!
---

"""

import numpy as np


class SinusoidalPositionalEncoding:
    class np:
        @staticmethod
        def forward(seq_len: int, d_model: int) -> np.ndarray:
            """
            Args:
                seq_len: T (sequence length)
                d_model: C (embedding dimension, must be even)

            Returns:
                PE: (T, C) positional encodings
            """
            # Step 1: Create position indices [0, 1, 2, ..., T-1] as column vector (T, 1)
            position = np.arange(seq_len)[:, np.newaxis]  # shape: (T, 1)

            # Step 2: Create the division term for each dimension pair
            # Formula: 10000^(2i/C) but we compute it as exp(2i * -log(10000)/C)
            # i goes from 0 to C/2-1 (one value per sin/cos pair)
            div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(1e4) / d_model))

            # Step 3: Initialize output matrix
            pe = np.zeros((seq_len, d_model))

            # Step 4: Fill even indices with sin, odd indices with cos
            # pe[:, 0::2] means all rows, columns 0, 2, 4, ... (even)
            # pe[:, 1::2] means all rows, columns 1, 3, 5, ... (odd)
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)

            return pe


class LearnedPositionalEmbedding:
    """
    Learned positional embeddings (GPT-style).
    Just a lookup table: position index -> embedding vector.

    Forward: PE = W[positions]
    Backward: same as Embedding layer (accumulate gradients at looked-up rows)
    """

    class np:
        @staticmethod
        def forward(seq_len: int, W: np.ndarray) -> np.ndarray:
            """
            Args:
                seq_len: T (actual sequence length)
                W: (max_seq_len, C) learnable embedding matrix

            Returns:
                PE: (T, C) positional embeddings
            """
            positions = np.arange(seq_len)
            pe = W[positions]
            return pe


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = 100
    C = 128

    pe = SinusoidalPositionalEncoding.np.forward(T, C)

    # Shape check
    print(f"Shape: {pe.shape} (expected ({T}, {C}))")

    # Range check
    print(f"Range: [{pe.min():.2f}, {pe.max():.2f}] (expected [-1, 1])")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap: rows = positions, cols = dimensions
    im = ax1.imshow(pe, cmap="RdBu", aspect="auto")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Position")
    ax1.set_title("Positional Encoding Heatmap")
    plt.colorbar(im, ax=ax1)

    # Wave plots: show a few dimensions across all positions
    for dim in [0, 1, 20, 21, 60, 61]:
        label = f"dim {dim} ({'sin' if dim % 2 == 0 else 'cos'})"
        ax2.plot(pe[:, dim], label=label, alpha=0.7)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Value")
    ax2.set_title("Individual Dimensions")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("positional_encoding.png")
    plt.show()

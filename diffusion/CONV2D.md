# Conv2d — Pattern Matching on Images

## What is it?

A small sliding window (kernel) that scans across the image looking for patterns.

```
3×3 kernel (the "window")        slides across the image
┌───────┐
│ w w w │    →  →  →
│ w w w │    step by step
│ w w w │    across every position
└───────┘
    ↓
one output number = sum of (pixel × weight) in the window
```

## How it works

The kernel is a **template**. At every position, it does a dot product:

```
Image patch:        Kernel:           Output:
┌─────────┐        ┌─────────┐
│ 1  0  0 │        │ 1  0  0 │       1×1 + 0×0 + 0×0
│ 0  1  0 │   ×    │ 0  1  0 │     + 0×0 + 1×1 + 0×0  = 3  ← HIGH! pattern matches!
│ 0  0  1 │        │ 0  0  1 │     + 0×0 + 0×0 + 1×1
└─────────┘        └─────────┘

Image patch:        Kernel:           Output:
┌─────────┐        ┌─────────┐
│ 0  0  0 │        │ 1  0  0 │       0×1 + 0×0 + 0×0
│ 0  0  0 │   ×    │ 0  1  0 │     + 0×0 + 0×0 + 0×0  = 0  ← LOW! no match
│ 0  0  1 │        │ 0  0  1 │     + 0×0 + 0×0 + 1×1
└─────────┘        └─────────┘
```

**High output = "this patch looks like my template."**
**Low output = "nope."**

## PyTorch API

```python
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         ^^^^^^^^^^^  ^^^^^^^^^^^^
#         channels IN  channels OUT (= number of pattern detectors)
```

## What does padding do?

Without padding, the 3×3 kernel can't center on edge pixels, so the output shrinks by 2.
`padding=1` adds a border of zeros so the output stays the same size.

```
Original 4×4:              Padded (padding=1):
                            0  0  0  0  0  0
  1  0  0  0                0 [1  0  0  0] 0
  0  1  0  0       →        0 [0  1  0  0] 0
  0  0  1  0                0 [0  0  1  0] 0
  0  0  0  1                0 [0  0  0  1] 0
                            0  0  0  0  0  0
```

Corner outputs are lower because the kernel partially lands on padded zeros (sees 2 of 3 diagonal pixels instead of 3).

## Multiple output channels

Each output channel is a **different pattern detector** running in parallel:

```python
conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
# channel 0 might learn to detect "\"
# channel 1 might learn to detect "/"

output = conv(image)    # (B, 2, H, W) — two feature maps
output[0, 0]            # "\" detector response at every pixel
output[0, 1]            # "/" detector response at every pixel
```

## Shapes

```
Input:  (B, in_channels,  H, W)
Output: (B, out_channels, H, W)     ← with padding=1, H and W stay the same
Kernel: (out_channels, in_channels, kernel_size, kernel_size)
```

## In a real network

You never hand-pick kernels. The network starts with random kernels and backprop figures out which patterns are useful — edges, corners, textures, etc.

## MaxPool2d — shrink the grid

```python
nn.MaxPool2d(2)
# takes every 2×2 block → keeps the max → halves H and W

# (B, C, 32, 32)  →  (B, C, 16, 16)
```

## Upsample — grow the grid

```python
nn.Upsample(scale_factor=2)
# stretches each pixel into a 2×2 block → doubles H and W

# (B, C, 16, 16)  →  (B, C, 32, 32)
```

ConvTranspose2d — ELI5                                                                                                                                                                                             
                                                                                              
Regular Conv shrinks things. ConvTranspose grows things.                                                                                                                                                           
                                                                                                                                                                                                                   
Regular Conv first (what you know)                                                                                                                                                                                 
                                                                                                                                                                                                                 
Input (4x4):          Kernel (3x3):        Output (2x2):
┌───┬───┬───┬───┐     ┌───┬───┬───┐
│ 1 │ 2 │ 3 │ 0 │     │ 1 │ 0 │ 1 │       ┌───┬───┐
├───┼───┼───┼───┤     ├───┼───┼───┤       │ 8 │ 5 │
│ 0 │ 1 │ 2 │ 1 │     │ 0 │ 1 │ 0 │       ├───┼───┤
├───┼───┼───┼───┤     ├───┼───┼───┤       │ 4 │ 7 │
│ 1 │ 0 │ 1 │ 0 │     │ 1 │ 0 │ 1 │       └───┴───┘
├───┼───┼───┼───┤     └───┴───┘
│ 0 │ 1 │ 1 │ 1 │
└───┴───┴───┴───┘

4x4 → 2x2  (shrank!)

Slide kernel over input, multiply-and-sum at each position. Output is smaller.

ConvTranspose — the reverse direction

Think of it as: each input pixel "stamps" the kernel onto a bigger output grid.

Input (2x2):          Kernel (3x3):
┌───┬───┐             ┌───┬───┬───┐
│ 1 │ 2 │             │ 1 │ 0 │ 1 │
├───┼───┤             ├───┼───┼───┤
│ 0 │ 3 │             │ 0 │ 1 │ 0 │
└───┴───┘             ├───┼───┼───┤
                      │ 1 │ 0 │ 1 │
                      └───┴───┘

Step by step — each input pixel stamps the kernel:

Input[0,0] = 1, stamp kernel × 1 at position (0,0):
┌───┬───┬───┬───┐
│ 1 │ 0 │ 1 │ . │
├───┼───┼───┼───┤
│ 0 │ 1 │ 0 │ . │
├───┼───┼───┼───┤
│ 1 │ 0 │ 1 │ . │
├───┼───┼───┼───┤
│ . │ . │ . │ . │
└───┴───┴───┴───┘

Input[0,1] = 2, stamp kernel × 2 at position (0,1):
┌───┬───┬───┬───┐
│ . │ 2 │ 0 │ 2 │
├───┼───┼───┼───┤
│ . │ 0 │ 2 │ 0 │
├───┼───┼───┼───┤
│ . │ 2 │ 0 │ 2 │
├───┼───┼───┼───┤
│ . │ . │ . │ . │
└───┴───┴───┴───┘

Input[1,0] = 0, stamp kernel × 0 at position (1,0):
  (all zeros, skip)

Input[1,1] = 3, stamp kernel × 3 at position (1,1):
┌───┬───┬───┬───┐
│ . │ . │ . │ . │
├───┼───┼───┼───┤
│ . │ 3 │ 0 │ 3 │
├───┼───┼───┼───┤
│ . │ 0 │ 3 │ 0 │
├───┼───┼───┼───┤
│ . │ 3 │ 0 │ 3 │
└───┴───┴───┴───┘

SUM all stamps:
┌───┬───┬───┬───┐
│ 1 │ 2 │ 1 │ 2 │
├───┼───┼───┼───┤
│ 0 │ 4 │ 2 │ 3 │
├───┼───┼───┼───┤
│ 1 │ 2 │ 4 │ 2 │
├───┼───┼───┼───┤
│ 0 │ 3 │ 0 │ 3 │
└───┴───┴───┴───┘

2x2 → 4x4  (grew!)

The mental model

Regular Conv:    many input pixels → 1 output pixel     (gather)
ConvTranspose:   1 input pixel → many output pixels      (scatter)

Conv:           big → small     (compress/downsample)
ConvTranspose:  small → big     (expand/upsample)

With stride=2 (what your decoder uses)

Stride in ConvTranspose means space between stamps, not how far the kernel slides:

stride=1: stamps overlap a lot → output barely bigger than input
stride=2: stamps spaced apart  → output roughly 2x the input size

Input (2x2), kernel (3x3), stride=2:

  Stamp positions on a 5x5 grid:
  Input[0,0] stamps at (0,0)
  Input[0,1] stamps at (0,2)   ← 2 apart!
  Input[1,0] stamps at (2,0)   ← 2 apart!
  Input[1,1] stamps at (2,2)

  2x2 → 5x5

The output_padding=1 you added bumps that to 2x2 → 6x6, ensuring exact doubling. Without it, the size is
ambiguous (could be 5 or 6).

Why it's called "transpose"

Mathematically, if you unroll a regular Conv into a matrix multiply, ConvTranspose uses the transposed
version of that same matrix. Same weights, opposite direction. That's why it can "undo" a convolution's
spatial shrinking.

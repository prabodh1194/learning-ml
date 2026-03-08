### What does the U-Net do?                                                                                                                                                                                            
 
"Hey neural network, here's a noisy cat photo.                                                                                                                                                                     
 It's at timestep t=500. Please tell me what the noise looks like."                           

Input: noisy image + timestep
Output: predicted noise (same shape as image)

**The U shape**

```
Image (B, 3, 32, 32)
    │
    ▼
┌──────────┐
│ Down 1   │  3→64,   32×32  ───────────────┐ skip1
└──────────┘                                │
    │ downsample                            │
    ▼                                       │
┌──────────┐                                │
│ Down 2   │  64→128, 16×16  ──────────┐    │ skip2
└──────────┘                           │    │
    │ downsample                       │    │
    ▼                                  │    │
┌──────────┐                           │    │
│Bottleneck│  128→256, 8×8             │    │
└──────────┘  256→128                  │    │
    │ upsample                         │    │
    ▼                                  │    │
┌──────────┐                           │    │
│  Up 1    │  128+128→64, 16×16  ◄─────┘    │
└──────────┘  (cat skip2!)                  │
    │ upsample                              │
    ▼                                       │
┌──────────┐                                │
│  Up 2    │  64+64→3, 32×32  ◄─────────────┘
└──────────┘  (cat skip1!)
    │
    ▼
Predicted noise (B, 3, 32, 32)
```

**Skip connections** are the key insight — they let the decoder say "oh yeah, I remember what was here before downsampling."

**Timestep conditioning**

The network also needs to know __which__ timestep it's denoising. Same idea as positional encoding in transformers — turn an integer into a rich vector:

### What is upsample & downsample?

What is a pixel grid?                                                                                                                                                                                              
                                                                                              
A 32×32 image is just a grid of numbers:                                                                                                                                                                           
                                                                                              
32×32 image (1024 pixels)
┌─┬─┬─┬─┬─┬─┬─┬─┐
│ │ │ │ │ │ │ │ │  ... 32 columns
├─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │ │ │ │
├─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │ │ │ │
  ...  32 rows

Downsample = shrink the grid

Take every 2×2 block and squish it into 1 pixel (e.g. by taking the max or average):

32×32                          16×16
┌──┬──┬──┬──┐                 ┌────┬────┐
│.3│.5│.1│.9│                 │ .5 │ .9 │
├──┼──┼──┼──┤   ──────▶       ├────┼────┤
│.2│.4│.7│.8│   downsample    │ .4 │ .8 │
└──┴──┴──┴──┘                 └────┴────┘

 4 pixels become 1              fewer pixels
 (take the max)                 but "denser" info

Why? Smaller grid = each pixel now "sees" a bigger area of the original image. It loses fine details but captures the big picture.

32×32 → "this pixel = 1 fur strand"
16×16 → "this pixel = cat's ear"
 8×8  → "this pixel = cat's whole face"

Upsample = grow the grid back

Stretch 1 pixel back into a 2×2 block:

16×16                          32×32
┌────┬────┐                   ┌──┬──┬──┬──┐
│ .5 │ .9 │                   │.5│.5│.9│.9│
├────┼────┤   ──────▶         ├──┼──┼──┼──┤
│ .4 │ .8 │   upsample        │.5│.5│.9│.9│
└────┴────┘                   ├──┼──┼──┼──┤
                              │.4│.4│.8│.8│
 fewer pixels                 ├──┼──┼──┼──┤
                              │.4│.4│.8│.8│
                              └──┴──┴──┴──┘
                               back to big grid
                               (but blocky!)

Why the U shape needs both

Downsample path:          "WHAT is in the image?"
  32×32 → 16×16 → 8×8
  fur → ear → whole cat    zooming OUT to understand

Upsample path:            "WHERE exactly is the noise?"
  8×8 → 16×16 → 32×32
  whole cat → ear → fur    zooming back IN to be precise

Skip connections:          "remember the details I lost!"
  encoder 32×32 ──────▶ decoder 32×32
  "hey, here's what the fur looked like before I zoomed out"

Without skip connections, the upsample is blocky and blurry (it lost the details). Skip connections hand back the fine details from before downsampling.

Think of it like:
- Down = read the whole page, understand the story, forget exact words
- Up = rewrite the page from memory
- Skip = cheat sheet with the exact words you forgot

In conv nets, the "width" is the number of channels (also called filters/feature maps). Each channel detects a different pattern:                                                                                  
                                                                                                                                                                                                                   
Down path: fewer pixels, MORE channels                                                                                                                                                                             
─────────────────────────────────────                                                                                                                                                                              
32×32 × 3ch     "3 colors (RGB)"
16×16 × 64ch    "64 different patterns: edges, corners, curves..."
 8×8  × 128ch   "128 higher-level patterns: eyes, ears, fur..."

Pixels shrink:   32 → 16 → 8     (less spatial detail)
Channels grow:    3 → 64 → 128   (more "what I detected" detail)

It's a tradeoff:

               spatial detail    pattern detail
               (WHERE things     (WHAT things
                are)              are)
Down:          lose ↓             gain ↑
Up:            gain ↑             lose ↓

On the way back up, you reverse it:

Up path: more pixels, FEWER channels
─────────────────────────────────────
 8×8  × 128ch   "I know WHAT's here but not exactly WHERE"
16×16 × 64ch    "getting more spatial precision back"
32×32 × 3ch     "full resolution, predict noise per pixel"

So the U shape in terms of dimensions:

          channels    spatial
Down 1:    3 → 64     32×32 → 16×16
Down 2:   64 → 128    16×16 →  8×8
Bottleneck: 128→256→128   8×8
Up 1:    128+128→64   8×8  → 16×16    (+skip from Down 2)
Up 2:     64+64→3    16×16 → 32×32    (+skip from Down 1)

The 128+128 is the skip connection — you concatenate the encoder's channels with the decoder's channels, doubling the channel count temporarily, then a conv brings it back down.

Now go write the timestep embedding!

U-net is built in 4 parts:

1. Timestep embedding
2. Down block — Conv + Conv + MaxPool
3. Up block — Upsample + cat skip + Conv + Conv
4. U-Net — wire it all together

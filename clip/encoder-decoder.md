That's literally how image captioning works:

```
CLIP image encoder (frozen)
         ↓
   image embedding
         ↓
  cross-attention   ← decoder peeks at image embedding
         ↓
  GPT-style decoder
         ↓
  "a dog playing in the park"
```

The CLIP encoder becomes the "encoder" half, and you plug a decoder on top. The decoder uses **cross-attention** to look at the image embedding while generating words.

This is exactly how these models work:

```
CLIP encoder + GPT decoder  =  rough idea behind DALL-E 1
CLIP encoder + bigger LLM   =  LLaVA, GPT-4 Vision
CLIP encoder + diffusion     =  DALL-E 2, Stable Diffusion
```

CLIP's encoder is so useful precisely BECAUSE it's frozen and reusable — plug anything on top of it!

---

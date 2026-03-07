Great question to pause and consolidate. Let me build it from scratch.

---

The original transformer was built for **translation** — English → French. It has two jobs:

```
"The cat sat" (English)
       ↓
  ENCODER — reads & understands the input
       ↓
  [rich representation of meaning]
       ↓
  DECODER — generates the output word by word
       ↓
"Le chat s'est assis" (French)
```

The key difference is **what each side is allowed to look at:**

```
ENCODER:
  "The  cat  sat"
    ↕    ↕    ↕     ← every token sees every other token
  full bidirectional attention
  job: UNDERSTAND the whole input

DECODER:
  "Le  chat  s'est  assis"
   →    →     →      →     ← each token only sees tokens to its LEFT
  causal (masked) attention
  job: GENERATE one token at a time
```

Why does the decoder use causal masking? Because at generation time, future words don't exist yet:

```
Generating word by word:
  step 1: "Le" → predict "chat"
  step 2: "Le chat" → predict "s'est"
  step 3: "Le chat s'est" → predict "assis"

Can't peek at future words — they haven't been generated yet!
So we MASK them during training to simulate this.
```

---

Now the modern variants:

**Encoder-only (BERT, ViT, CLIP):**

```
Task: UNDERSTAND, not generate
  BERT: "what's the sentiment of this sentence?"
  ViT:  "what's in this image?"
  CLIP: "does this image match this text?"

All tokens see all other tokens (bidirectional)
No generation needed → no causal mask
Output: rich vector representations
```

**Decoder-only (GPT, LLaMA):**

```
Task: GENERATE text
  "The cat sat on the ___" → "mat"

Each token only sees tokens to its left
Output: next token prediction

Wait — where's the encoder?
  There isn't one! The decoder reads AND generates
  from the same sequence. The "input" is just
  the beginning of the sequence, generation
  continues it.
```

**Encoder-Decoder (original Transformer, T5):**

```
Task: TRANSFORM one sequence into another
  Translation, summarisation, Q&A

Encoder reads input fully (bidirectional)
Decoder generates output (causal)
Decoder also has CROSS-ATTENTION — it peeks
at encoder output while generating:

  Encoder output: [meaning of "The cat sat"]
                           ↓
  Decoder: "Le" → attends to encoder → predicts "chat"
           "Le chat" → attends to encoder → predicts "s'est"
```

---

The full picture:

```
                    Sees input?   Generates?   Attention
                    
Encoder-only        fully         no           bidirectional
(BERT, ViT, CLIP)

Decoder-only        causally      yes          causal
(GPT, LLaMA)

Encoder-Decoder     encoder=full  yes          encoder=bi
(T5, original       decoder=causal             decoder=causal
 transformer)                                  + cross-attention
```

---

One thing that trips people up about GPT/LLaMA:

```
"If it's decoder-only, where's the encoding?"

Answer: it doesn't separate them!
The same stack of layers both "understands" 
the prompt AND generates the response.

Prompt:     "What is 2+2?"
                  ↓
         same transformer layers
                  ↓
Response:   "2+2 = 4"

No separate encoder. The decoder does both jobs.
This is why decoder-only scales so well — 
one simple architecture, no cross-attention overhead.
```

---

Your intuition mapped out:

```
ViT          → encoder only   (understand images)
BERT         → encoder only   (understand text)
CLIP         → encoder only × 2 (understand image + text)
GPT/LLaMA    → decoder only   (generate text)
Original T   → encoder+decoder (translate)
T5, BART     → encoder+decoder (summarise, translate)
```

Does this click? The core of it is just: **bidirectional = understand, causal = generate.**
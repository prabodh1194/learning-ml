# DeepSeek Learning Plan

## MILESTONE 4: DeepSeek Architecture Complete

**Objective:** Master DeepSeek's architectural innovations: Mixture of Experts (MoE) and Multi-head Latent Attention (MLA).

**Total Time:** ~13-14 hours

### Why DeepSeek?

- **Massive scale:** 671B parameters (DeepSeek-V3)
- **Efficient:** Only 37B parameters active per token (18x capacity increase!)
- **Cost-effective:** $5.6M training cost (vs $100M+ for dense models)
- **Performance:** Matches GPT-4 on many benchmarks

---

## Task Overview

| Task | Focus | Due |
|------|-------|-----|
| D1 | MoE Foundation | Jan 28 |
| D2 | DeepSeek MoE Innovations | Jan 29 |
| D3 | Multi-head Latent Attention (MLA) | Jan 30 |
| D4 | Complete DeepSeek Assembly | Jan 31 |

---

## D1. MoE Foundation

**Goal:** Understand Mixture of Experts - the foundation enabling massive model scale through sparse computation.

**Time:** 3-4 hours

### Subtasks

- [ ] D1.1 Implement Top-K Router
- [ ] D1.2 Implement Expert Array
- [ ] D1.3 Implement Token Dispatch & Combination
- [ ] D1.4 Monitor Load Balancing Metrics
- [ ] D1.5 Complete MoE Layer Integration

### Core Concepts

**Why MoE Matters:**
- 671B total params, only 37B active per token
- Same compute budget, 10-20x more model capacity
- Production-proven: DeepSeek, Mixtral, GPT-4 (rumored), Switch Transformer

**Three Core Components:**

1. **Router (Gating Network)**
   - Takes token embedding as input
   - Outputs probability distribution over N experts
   - Selects top-K experts with highest probability
   - Normalizes to create gating weights

2. **Expert Networks**
   - Specialized FFN sub-networks (typically SwiGLU)
   - Different learned weights per expert
   - Learn to specialize on patterns through training

3. **Weighted Combination**
   - Each selected expert produces output vector
   - Weight by router's gating probabilities
   - Sum weighted outputs

### Implementation Sequence

**1. Simple Top-K Router (Start here!)**
```python
# Input: token embedding (d_model)
# Output: K expert indices + K gate weights per token
logits = token @ W_router  # Linear projection
probs = softmax(logits)    # Softmax over experts
top_k_indices = topk(probs, k)  # Top-K selection
gate_weights = normalize(probs[top_k_indices])  # Renormalize
```
**Test:** Gate weights sum to 1.0 per token

**2. Single Expert FFN**
- Reuse SwiGLU implementation from LLaMA
- Just one expert to start

**3. Array of Experts**
- Array/list of N identical FFN modules
- Each with independent parameters

**4. Token Dispatch & Combine**
- Route each token to its top-K experts
- Gather outputs from selected experts
- Weight by gate probabilities
- Sum weighted outputs

### Tiny Config to Start
```python
d_model = 256
num_experts = 4
top_k = 2
expert_dim = 512  # hidden dim for SwiGLU
```

### Success Criteria
- Different tokens choose different experts
- Gradient flow to router and experts
- Some experts get more traffic (load imbalance - expected!)
- Loss decreases (model is learning)

---

## D2. DeepSeek MoE Innovations

**Goal:** Understand and implement DeepSeek's three key MoE innovations.

**Time:** 2-3 hours

**Prerequisites:** D1 (Basic MoE understanding)

### Subtasks

- [ ] D2.1 Implement Shared Experts
- [ ] D2.2 Implement Fine-Grained Segment Routing
- [ ] D2.3 Implement Bias-Based Load Balancing

### The Three Innovations

**1. Shared + Routed Expert Architecture**

```
Token -> [Shared Expert 1] -> always on
      -> [Shared Expert 2] -> always on
      -> [Route to top-K of 256 routed experts] -> sparse
      -> Combine all outputs
```

- **Shared Experts (1-2):** Process ALL tokens, learn common patterns
- **Routed Experts (64-256):** Top-K routing, learn specialized patterns

**2. Fine-Grained Expert Segmentation**

- Split token's hidden dimension into m segments (e.g., 8)
- Each segment routes independently to top-K experts
- Different segments can choose different experts

```
Token embedding (1024 dims)
-> Split into 8 segments of 128 dims each
-> Segment 1 routes to experts [3, 7]
-> Segment 2 routes to experts [1, 5]
-> Segment 3 routes to experts [2, 7]
... (different experts for different segments!)
-> Recombine outputs
```

**3. Bias-Based Load Balancing**

- Add learnable bias term to router logits
- Track expert utilization during training
- Increase bias for underused experts
- No auxiliary loss needed!

```python
# Standard: logits = token @ W_router
# DeepSeek: logits = token @ W_router + expert_bias
```

### Success Criteria
- Load balance CV < 0.1 (very balanced)
- Different segments choose different experts
- Bias values converge during training

---

## D3. Multi-head Latent Attention (MLA)

**Goal:** Understand MLA - DeepSeek's innovation for 5-10x KV cache compression.

**Time:** 3-4 hours

**Prerequisites:** Standard multi-head attention, GQA helpful

### Subtasks

- [ ] D3.1 Implement Latent Projections (Down/Up)
- [ ] D3.2 Implement Latent KV Cache Management
- [ ] D3.3 Full MLA Integration with Attention

### The KV Cache Problem

**Standard MHA Cache (LLaMA 70B example):**
- 80 layers x 64 heads x 4096 seq x 128 dim
- = 10.5 GB just for KV cache!
- Memory bandwidth is the bottleneck

### MLA Core Innovation

**Standard attention:**
```
Q, K, V projections: d_model -> num_heads x d_head
For 32 heads, d_head=128:
  K: d_model -> 4096 dimensions
  V: d_model -> 4096 dimensions
```

**MLA approach:**
```
1. Down-project to low-dim latent:
   K_latent: d_model -> d_latent (e.g., 512)
   V_latent: d_model -> d_latent (e.g., 512)

2. Cache the latents (compressed!)

3. At inference, up-project latents to full K, V:
   K_full: d_latent -> num_heads x d_head
   V_full: d_latent -> num_heads x d_head

Compression: 4096 / 512 = 8x smaller cache!
```

### MLA Flow
```
Input x (d_model)
  |
Q = x @ W_Q -> (num_heads, d_head)  [Full Q, no compression]
  |
K_latent = x @ W_Kc -> (d_latent)  [Compress!]
V_latent = x @ W_Vc -> (d_latent)  [Compress!]
  |
[CACHE K_latent, V_latent]  <- Much smaller!
  |
K = K_latent @ W_Ku -> (num_heads, d_head)  [Expand at inference]
V = V_latent @ W_Vu -> (num_heads, d_head)  [Expand at inference]
  |
Attention(Q, K, V)
```

### Success Criteria
- KV cache 5-10x smaller than standard MHA
- Attention output cosine similarity > 0.95 vs standard
- Faster inference (less memory bandwidth)

---

## D4. Complete DeepSeek Assembly

**Goal:** Assemble complete DeepSeek decoder block, integrating MoE and MLA.

**Time:** 3 hours

**Prerequisites:** D1-D3, LLaMA components (RoPE, RMSNorm)

### Subtasks

- [ ] D4.1 Assemble Single DeepSeek Block
- [ ] D4.2 Stack Blocks into Complete Decoder
- [ ] D4.3 Training & Validation

### DeepSeek Block Architecture

```
Input
  |
RMSNorm -> Multi-head Latent Attention (MLA) -> Residual
  |
RMSNorm -> DeepSeek MoE (Shared + Routed) -> Residual
  |
Output
```

### Single Block Pseudocode
```python
def forward(x):
    # Attention sublayer
    attn_out = MLA(RMSNorm(x))
    x = x + attn_out

    # MoE FFN sublayer
    moe_out = DeepSeek_MoE(RMSNorm(x))
    x = x + moe_out

    return x
```

### DeepSeek-V3 Configuration
- 61 layers
- d_model = 7168
- 128 attention heads
- d_latent = 1024 (MLA)
- 256 routed + 2 shared experts per layer
- Top-8 routing per segment
- 8 fine-grained segments

### Success Criteria
- Forward pass: input (B, T, d_model) -> output (B, T, d_model)
- Gradients flow to all components
- Memory usage lower than standard transformer
- Can generate coherent text

---

## Key Papers

1. **MoE Background:**
   - "Switch Transformers: Scaling to Trillion Parameter Models" (Google, 2021)
   - "Mixtral of Experts" (Mistral AI, 2023)

2. **DeepSeek:**
   - "DeepSeek-V2: A Strong, Economical, and Efficient MoE Model" (2024)
   - "DeepSeek-V3 Technical Report" (2024)

3. **Attention Optimization:**
   - "Multi-Query Attention" (Shazeer, 2019)
   - "GQA: Training Generalized Multi-Query Transformers" (2023)

---

## Capstone Achievement

After completing this milestone:

- Mastered sparse neural network architectures
- Understood production-scale LLM design
- Implemented state-of-the-art innovations
- Gained expertise few ML engineers possess

You'll be able to read DeepSeek papers and say: "I've built this. I understand how it works. I can explain the trade-offs."

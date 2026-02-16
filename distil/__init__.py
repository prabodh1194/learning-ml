"""
The Big Picture

GOAL: Make small model behave like big model

┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE DISTILLATION                      │
│                                                                 │
│   ┌───────────────┐              ┌───────────────┐              │
│   │    TEACHER    │              │    STUDENT    │              │
│   │  (TinyLLaMA)  │              │  (Mini LLaMA) │              │
│   │    1.1B       │              │     20M       │              │
│   │    FROZEN     │              │   TRAINABLE   │              │
│   └───────┬───────┘              └───────┬───────┘              │
│           │                              │                      │
│           ▼                              ▼                      │
│   "The capital of France is ___"    Same input                  │
│           │                              │                      │
│           ▼                              ▼                      │
│   ┌───────────────┐              ┌───────────────┐              │
│   │ Paris:  70%   │              │ Paris:  40%   │              │
│   │ Lyon:   15%   │   ───────►   │ Lyon:   20%   │              │
│   │ Berlin: 10%   │    MATCH     │ Berlin: 25%   │              │
│   │ Rome:    5%   │    THESE!    │ Rome:   15%   │              │
│   └───────────────┘              └───────────────┘              │
│         ▲                              ▲                        │
│         │                              │                        │
│         └──────────┬───────────────────┘                        │
│                    │                                            │
│                    ▼                                            │
│            ┌──────────────┐                                     │
│            │ KL DIVERGENCE│  ◄── "How different are these?"     │
│            │    LOSS      │                                     │
│            └──────────────┘                                     │
│                    │                                            │
│                    ▼                                            │
│              Backprop to student                                │
│              (make student match teacher)                       │
└─────────────────────────────────────────────────────────────────┘

---
What is KL Divergence?

ELI5: It measures "how surprised would I be if I expected distribution P but got distribution Q?"

TEACHER says:          STUDENT says:
Paris:  70%            Paris:  40%
Lyon:   15%            Lyon:   20%
Berlin: 10%            Berlin: 25%
Rome:    5%            Rome:   15%

KL Divergence = "How wrong is student compared to teacher?"

If student matches teacher perfectly → KL = 0
If student is totally different      → KL = large number
"""

import torch
from torch import nn

from deepseek.block import DeepseekBlock
from llama.rmsnorm import RMSNorm

"""                                                                                                                                                                                                
DeepSeek Model - Complete Decoder Stack                                                                                                                                                            
                                                                                                                                                                                                   
Token IDs                                                                                                                                                                                          
    │                                                                                                                                                                                              
    ▼                                                                                                                                                                                              
┌─────────────┐                                                                                                                                                                                    
│  Embedding  │                                                                                                                                                                                    
└──────┬──────┘                                                                                                                                                                                    
       │                                                                                                                                                                                           
       ▼                                                                                                                                                                                           
┌─────────────┐                                                                                                                                                                                    
│ DeepSeek    │ ×N layers                                                                                                                                                                          
│   Block     │                                                                                                                                                                                    
└──────┬──────┘                                                                                                                                                                                    
       │                                                                                                                                                                                           
       ▼                                                                                                                                                                                           
┌─────────────┐                                                                                                                                                                                    
│  RMSNorm    │                                                                                                                                                                                    
└──────┬──────┘                                                                                                                                                                                    
       │                                                                                                                                                                                           
       ▼                                                                                                                                                                                           
┌─────────────┐                                                                                                                                                                                    
│  LM Head    │                                                                                                                                                                                    
└──────┬──────┘                                                                                                                                                                                    
       │                                                                                                                                                                                           
       ▼                                                                                                                                                                                           
   Logits                                                                                                                                                                                          
"""


class Deepseek(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        num_layers: int,
        dim_latent: int,
        num_heads: int,
        context_length: int,
        num_segments: int,
        num_shared_experts: int,
        num_routed_experts: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList(
            [
                DeepseekBlock(
                    dim=dim,
                    dim_latent=dim_latent,
                    num_heads=num_heads,
                    context_length=context_length,
                    num_segments=num_segments,
                    num_shared_experts=num_shared_experts,
                    num_routed_experts=num_routed_experts,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(dim=dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        caches: list,
    ):
        X = self.embedding(tokens)
        new_caches = []
        total_aux_loss = 0

        for idx, block in enumerate(self.blocks):
            X, aux_loss, cache = block(X, caches[idx])
            new_caches.append(cache)
            total_aux_loss += aux_loss

        X = self.final_norm(X)
        logits = self.lm_head(X)

        return logits, total_aux_loss, new_caches

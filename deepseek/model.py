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


if __name__ == "__main__":
    num_layers = 8

    deepseek_model = Deepseek(
        vocab_size=256,
        dim=32,
        dim_latent=8,
        num_layers=num_layers,
        num_heads=4,
        context_length=10,
        num_segments=8,
        num_shared_experts=2,
        num_routed_experts=8,
    )

    # forward pass (B = 2, T = 5)
    tokens = torch.randint(0, 256, (2, 5))
    logits, aux_loss, caches = deepseek_model(tokens, [None] * num_layers)

    print("logits shape: ", logits.shape)
    print("auxiliary loss: ", aux_loss)

    # grad test
    loss = logits.sum() + aux_loss * 0.01
    print("loss: ", loss)
    loss.backward()

    print(f"Gradients flow: {deepseek_model.embedding.weight.grad is not None}")

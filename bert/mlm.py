import torch
from torch import nn


def mlm_mask(
    tokens: torch.Tensor, vocab_size: int, mask_token_id: int, mask_prob: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply BERT's 80/10/10 masking strategy.

    tokens:         (B, seq_len) — original token ids
    vocab_size:     size of tokenizer vocabulary
    mask_token_id:  id of the [MASK] token
    mask_prob:      probability of selecting a token for masking (0.15)

    Returns:
        masked_tokens: (B, seq_len) — tokens with masking applied
        labels:        (B, seq_len) — original token ids where masked, -100 elsewhere

    Steps:
        1. Clone tokens
        2. labels = full of -100
        3. selected = random < mask_prob  (which 15% to target)
        4. labels[selected] = original tokens  (what we want the model to predict)
        5. roll = another random for 80/10/10 split:
             80% of selected → replace with mask_token_id
             10% of selected → replace with randint(0, vocab_size)
             10% of selected → leave unchanged
    """
    masked_tokens = tokens.clone()
    labels = torch.full_like(
        tokens, -100
    )  # these are the tokens that our model will learn to predict

    selected = torch.rand(tokens.shape) < mask_prob
    labels[selected] = tokens[selected]

    roll = torch.rand(tokens.shape)

    # 80% -> mask
    mask_replace = selected & (roll < 0.8)
    masked_tokens[mask_replace] = mask_token_id

    # 10% -> random token
    random_replace = selected & (roll < 0.9) & (roll >= 0.8)

    # random_replace.sum() is a count of the items to be replaced.
    # we can generate only enough random tokens that are required to be filled.
    masked_tokens[random_replace] = torch.randint(0, vocab_size, (random_replace.sum()))

    return masked_tokens, labels


class MLMHead(nn.Module):
    """
    Projects BERT hidden states to vocabulary logits.

    (B, seq_len, d_model) → LayerNorm → Linear → (B, seq_len, vocab_size)
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.lin = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, seq_len, d_model)
        Returns: (B, seq_len, vocab_size)
        """
        return self.lin(self.ln(hidden_states))

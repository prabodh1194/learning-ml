import torch
from torch import nn


class ClassificationHead(nn.Module):
    """
    Takes the [CLS] token output and classifies it.

    BERT output: (B, seq_len, d_model)
                      │
                      ▼
              output[:, 0, :]     ← grab CLS token (position 0)
                      │
                      ▼
              Linear(d_model, num_classes)
                      │
                      ▼
                   logits (B, num_classes)

    Note: no softmax here — PyTorch's CrossEntropyLoss includes it.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        # TODO: Linear(d_model, num_classes)
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, seq_len, d_model) — full BERT output

        Returns: (B, num_classes) — classification logits
        """
        # TODO: extract CLS token (index 0), pass through linear
        pass

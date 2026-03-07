import torch
import torch.nn.functional as F
from torch import nn

"""
By the time inputs reach the loss function, the encoders have already collapsed each sequence into a single vector.

Full pipeline:

  Image: (B, C, H, W) → ImageEncoder → (B, 768) → Projection → (B, 256)
                                              ↑
                                         CLS token extracted
                                         T is gone!

  Text:  (B, 77) → TextEncoder → (B, 512) → Projection → (B, 256)
                                       ↑
                                  EOS token extracted
                                  T is gone!

What the loss sees:
  image_embeds: (B, 256)   ← one vector per image
  text_embeds:  (B, 256)   ← one vector per text

The whole point of grabbing the CLS/EOS token was to go from (B, T, C) → (B, C) — squishing the sequence into one
summary vector. The projection then maps it to the shared space.

So loss receives (B, d) tensors, not (B, T, C).

Loss calculation the core idea:

  You have a batch of N image-text PAIRS:
    pair 0: (dog_image,  "a photo of a dog")
    pair 1: (cat_image,  "a photo of a cat")
    pair 2: (car_image,  "a photo of a car")

  Step 1: Compute ALL pairwise similarities (N×N matrix)

             text_0   text_1   text_2
    img_0  [  0.9      0.1      0.0  ]    ← img_0 vs ALL texts
    img_1  [  0.1      0.9      0.0  ]    ← img_1 vs ALL texts
    img_2  [  0.0      0.0      0.9  ]    ← img_2 vs ALL texts
              ↑
              col = text_0 vs ALL images

    Diagonal = correct pairs   → push HIGH
    Off-diag = wrong pairs     → push LOW

  Step 2: This is just classification!

    Row 0: "which text matches img_0?" → answer: index 0
    Row 1: "which text matches img_1?" → answer: index 1
    Row 2: "which text matches img_2?" → answer: index 2

    Labels = [0, 1, 2]  ← always the diagonal!
    Loss   = CrossEntropy(similarity_matrix, labels)

  Step 3: Do it BOTH directions

    image→text:  each ROW is a classification problem
    text→image:  each COLUMN is a classification problem (just transpose!)

    final_loss = (row_loss + col_loss) / 2

  Step 4: Temperature τ

    similarity / τ    before passing to CrossEntropy

    τ is LEARNABLE — stored as nn.Parameter
    Initialized to something like ln(1/0.07) ≈ 2.66


"""


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.temperature = nn.Parameter((torch.ones(1) / 0.07).log())

    def forward(self, images: torch.Tensor, texts: torch.Tensor):
        logits = images @ texts.T * self.temperature.exp()
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        return loss

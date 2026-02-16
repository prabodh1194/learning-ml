import torch


def soft_targets(logits: torch.Tensor, temperature: float = 1.0):
    """
    convert logits to soft probability distribution
    """

    logits = logits / temperature

    return logits.softmax(dim=-1)


if __name__ == "__main__":
    logits = torch.tensor([2.0, 1.0, 0.5])

    # T=1: [0.63, 0.23, 0.14]  ← Peaky (Paris dominates)
    print(soft_targets(logits, temperature=1))

    # T=4: [0.41, 0.32, 0.28]  ← Smoother (dark knowledge visible)
    print(soft_targets(logits, temperature=4))

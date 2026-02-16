import torch

from distil.kl_div import kl_div_pt


def soft_targets(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    convert logits to soft probability distribution
    """

    logits = logits / temperature

    return logits.softmax(dim=-1)


def distillation_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    p = soft_targets(teacher_logits, temperature)
    q = soft_targets(student_logits, temperature)

    return kl_div_pt(p, q) * (temperature ** 2)  # temperature scaling for stability


if __name__ == "__main__":
    logits = torch.tensor([2.0, 1.0, 0.5])

    # assume Q is what's the capital of France?

    # T=1: [0.63, 0.23, 0.14]  ← Peaky (e.g.: Paris dominates)
    print(soft_targets(logits, temperature=1))

    # T=4: [0.41, 0.32, 0.28]  ← Smoother (dark knowledge visible, e.g. other French cities)
    print(soft_targets(logits, temperature=4))

    print(distillation_loss(logits, logits))

    teacher_logits = torch.tensor([5.0, 1.0, 0.1])
    student_logits = torch.tensor([1.0, 1.0, 1.0])

    print(distillation_loss(teacher_logits, student_logits))
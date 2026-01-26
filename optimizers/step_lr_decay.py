import torch


class StepLR:
    def __init__(self, base_lr: float, step_size: int, gamma: float):
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step: int) -> float:
        return self.base_lr * self.gamma ** (step // self.step_size)


if __name__ == "__main__":
    base_lr, step_size, gamma = 0.1, 10, 0.5

    scheduler = StepLR(base_lr, step_size, gamma)

    dummy_param = torch.tensor([1.0], requires_grad=True)
    opt = torch.optim.SGD([dummy_param], lr=base_lr)
    pt_scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=step_size, gamma=gamma
    )

    for step in range(35):
        np = scheduler.get_lr(step)
        pt_lr = pt_scheduler.get_last_lr()[0]
        print(f"Step {step}: yours={np:.4f}, torch={pt_lr:.4f}")

        pt_scheduler.step()

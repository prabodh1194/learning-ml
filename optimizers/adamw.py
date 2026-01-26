import numpy as np
import torch


class AdamW:
    class np:
        def __init__(self, lr: float, beta1: float, beta2: float, weight_decay: float):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.weight_decay = weight_decay

            self.m = 0
            self.v = 0
            self.t = 0

        def optimise(
            self, *, param: np.ndarray, grad: np.ndarray, eps: float = 1e-8
        ) -> np.ndarray:
            self.t += 1

            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)

            param = param - self.lr * (
                m_hat / (np.sqrt(v_hat) + eps) + self.weight_decay * param
            )

            return param


if __name__ == "__main__":
    np.random.seed(42)

    init_param = np.random.randn(5).astype(np.float32)
    init_grad = np.random.randn(5).astype(np.float32)

    param_np = init_param.copy()
    grad_np = init_grad.copy()
    opt_np = AdamW.np(lr=0.1, beta1=0.9, beta2=0.999, weight_decay=0.01)

    param_pt = torch.tensor(init_param, requires_grad=True)
    opt_pt = torch.optim.AdamW([param_pt], lr=0.1)

    for i in range(5):
        param_np = opt_np.optimise(param=param_np, grad=grad_np)

        opt_pt.zero_grad()
        param_pt.grad = torch.tensor(init_grad.copy())
        opt_pt.step()

    print(f"Yours:   {param_np}")
    print(f"PyTorch: {param_pt.detach().numpy()}")
    print(f"Match: {np.allclose(param_np, param_pt.detach().numpy())}")

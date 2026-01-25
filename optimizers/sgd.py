import numpy as np
import torch


class SGD:
    class np:
        def __init__(self, *, lr: float, momentum: float, weight_decay: float = 0):
            self.lr = lr
            self.momentum = momentum
            self.v = 0
            self.weight_decay = weight_decay

        def optimise(self, *, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
            grad = grad + self.weight_decay * param
            self.v = self.momentum * self.v + grad
            param = param - self.lr * self.v

            return param


if __name__ == "__main__":
    lr, momentum = 0.001, 0.9

    np.random.seed(42)

    init_param = np.random.randn(5).astype(np.float64)
    init_grad = np.random.randn(5).astype(np.float64)

    param_np = init_param.copy()
    opt_np = SGD.np(lr=lr, momentum=momentum)

    param_pt = torch.tensor(init_param.copy(), requires_grad=True)
    opt_pt = torch.optim.SGD([param_pt], lr=lr, momentum=momentum)

    for i in range(5):
        param_np = opt_np.optimise(param=param_np, grad=init_grad)
        opt_pt.zero_grad()
        param_pt.grad = torch.tensor(init_grad.copy())
        opt_pt.step()

    print(f"Yours:   {param_np}")
    print(f"PyTorch: {param_pt.detach().numpy()}")
    print(f"Match: {np.allclose(param_np, param_pt.detach().numpy())}")

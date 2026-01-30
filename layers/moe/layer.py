from torch import nn

from layers.moe.expert import ExpertArray
from layers.moe.router import MOERouter


class MOELayer(nn.Module):
    def __init__(self, num_experts: int, dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim

        self.router = MOERouter.torch.forward
        self.experts = ExpertArray(num_experts, dim)

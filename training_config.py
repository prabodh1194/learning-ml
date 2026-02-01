from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 256  # B
    context_len: int = 64  # T
    dim: int = 128  # C
    epochs: int = 3
    lr: float = 3e-4
    device: str = "mps"
    data_path: str = "data/tinyshakespeare/input.txt"


@dataclass
class LLaMAConfig(TrainConfig):
    n_layers: int = 6
    num_head: int = 4
    num_kv_head: int = 2


@dataclass
class DeepSeekConfig(TrainConfig):
    num_layers: int = 4
    dim_latent: int = 32
    num_heads: int = 4
    num_segments: int = 4
    num_shared_experts: int = 1
    num_routed_experts: int = 4
    aux_weight: float = 0.01

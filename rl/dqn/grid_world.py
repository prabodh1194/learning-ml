import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self, *, state: int, action: int, reward: int, next_state: int, done: bool
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple[int, int, int, int, bool]]:
        batch = random.sample(list(self.buffer), batch_size)
        return tuple(map(np.array, zip(*batch)))

    def __len__(self) -> int:
        return len(self.buffer)

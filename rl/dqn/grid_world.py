import random
from collections import deque

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from rl.micro_rl import GridWorld


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self, *, state: int, action: int, reward: int, next_state: int, done: bool
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        batch = random.sample(list(self.buffer), batch_size)
        return tuple(map(np.array, zip(*batch)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        n_states: int = 12,
        n_actions: int = 4,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps: float = 1.0,
    ):
        self.online_net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

        self.target_net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.eps = eps
        self.gamma = gamma

        self.n_actions = n_actions
        self.n_states = n_states

    def choose_action(self, state: int) -> int:
        choice = random.random()

        if choice < self.eps:
            return random.randint(0, self.n_actions - 1)
        else:
            x = torch.eye(self.n_states)[state]
            self.online_net.eval()
            with torch.no_grad():
                return self.online_net(x.unsqueeze(0)).argmax().item()

    def learn(self, batch: tuple[np.ndarray, ...]):
        self.online_net.train()
        states, _actions, _rewards, next_states, _dones = batch

        actions = torch.LongTensor(_actions).unsqueeze(1)
        rewards = torch.FloatTensor(_rewards)
        dones = torch.FloatTensor(_dones)

        states_ohe = torch.eye(self.n_states)[states]
        next_states_ohe = torch.eye(self.n_states)[next_states]

        predicted_q = self.online_net(states_ohe).gather(1, actions).squeeze()

        with torch.no_grad():
            target_q = rewards + self.gamma * (
                self.target_net(next_states_ohe).max(dim=1).values * (1 - dones)
            )

        loss = F.mse_loss(predicted_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self) -> None:
        self.eps *= 0.995
        self.eps = max(self.eps, 0.01)


def train():
    env = GridWorld(r=3, c=4)
    agent = DQNAgent()
    buffer = ReplayBuffer(capacity=10_000)
    step = 0
    wins = 0

    for episode in range(1_000):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            if len(buffer) >= 64:
                agent.learn(buffer.sample(32))

            step += 1

            if step % 100 == 0:
                agent.sync_target()

            state = next_state

            if done and reward == 1:
                wins += 1

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1:>5} | wins last 100: {wins:>3} | eps: {agent.eps:.3f}"
            )
            wins = 0

    # print learned policy
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    specials = {3: " G", 5: "XX", 7: " P"}
    print("\nLearned policy:")
    for s in range(12):
        if s % 4 == 0:
            print("  ", end="")
        if s in specials:
            print(f" {specials[s]} ", end="")
        else:
            old_eps, agent.eps = agent.eps, 0
            a = agent.choose_action(s)
            agent.eps = old_eps
            print(f"  {arrows[a]} ", end="")
        if s % 4 == 3:
            print()


if __name__ == "__main__":
    train()

import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from rl.dqn.grid_world import ReplayBuffer


class _DQNAgent:
    def __init__(
        self,
        *,
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

    def choose_action(self, state: np.ndarray) -> int:
        choice = random.random()

        if choice < self.eps:
            return random.randint(0, self.n_actions - 1)
        else:
            x = torch.FloatTensor(state).unsqueeze(0)
            self.online_net.eval()
            with torch.no_grad():
                return self.online_net(x).argmax().item()

    def learn(self, batch: tuple[np.ndarray, ...]):
        self.online_net.train()
        states, _actions, _rewards, next_states, _dones = batch

        actions = torch.LongTensor(_actions).unsqueeze(1)
        rewards = torch.FloatTensor(_rewards)
        dones = torch.FloatTensor(_dones)

        s = torch.FloatTensor(states)
        ns = torch.FloatTensor(next_states)

        predicted_q = self.online_net(s).gather(1, actions).squeeze()

        with torch.no_grad():
            target_q = rewards + self.gamma * (
                self.target_net(ns).max(dim=1).values * (1 - dones)
            )

        loss = F.mse_loss(predicted_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10)
        self.optimizer.step()

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self) -> None:
        self.eps *= 0.995
        self.eps = max(self.eps, 0.01)


def train():
    env = gym.make("CartPole-v1")
    agent = _DQNAgent(
        n_states=4,
        n_actions=2,
    )
    buffer = ReplayBuffer(capacity=10_000)
    step = 0
    rewards_history = []

    for episode in range(3_000):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

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

            if step % 1000 == 0:
                agent.sync_target()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(
                f"Episode {episode + 1:>5} | avg reward last 100: {avg:>6.1f} | eps: {agent.eps:.3f}"
            )


if __name__ == "__main__":
    train()

"""
Grid world - environment:

    ┌───┬───┬───┬───┐
    │ S │   │   │ G │   5 lines of state logic
    ├───┼───┼───┼───┤   step() returns (new_state, reward, done)
    │   │ X │   │ P │   reset() puts agent back at start
    ├───┼───┼───┼───┤   S = start,         G = goal
    │   │   │   │   │   X = wall no entry, P = pit
    └───┴───┴───┴───┘

    - Actions: 0=up, 1=right, 2=down, 3=left
    - If you hit a wall or edge, you stay in place
    - Episode ends when you reach goal or pit

    State numbering:
    ┌────┬────┬────┬────┐
    │  0 │  1 │  2 │  3 │  ← goal (3)
    ├────┼────┼────┼────┤
    │  4 │  5 │  6 │  7 │  ← wall (5), pit (7)
    ├────┼────┼────┼────┤
    │  8 │  9 │ 10 │ 11 │
    └────┴────┴────┴────┘
"""

import random
from pathlib import Path

import numpy as np


class GridWorld:
    ACTION = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }

    def __init__(self, *, r: int, c: int) -> None:
        self.state = 0
        self.goal = 3
        self.wall = 5
        self.pit = 7

        self.r = r
        self.c = c

    def step(self, action: int) -> tuple:
        state = self.state
        _r, _c = state // self.c, state % self.c
        _dr, _dc = self.ACTION[action]
        _r += _dr
        _c += _dc

        state = _r * self.c + _c
        done = False
        reward = 0

        if _r < 0 or _r >= self.r or _c < 0 or _c >= self.c:
            state = self.state
        if state == self.wall:
            state = self.state
        if state == self.goal or state == self.pit:
            done = True
        if state == self.goal:
            reward = 1
        if state == self.pit:
            reward = -1

        self.state = state
        return state, reward, done

    def reset(self) -> int:
        self.state = 0
        return self.state


class QAgent:
    def __init__(
        self,
        n_states: int = 12,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        eps: float = 1.0,
    ) -> None:
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def choose_action(self, state: int) -> int:
        choice = random.random()

        if choice < self.eps:
            return random.randint(0, 3)
        return int(np.argmax(self.q[state]))

    def learn(self, state: int, action: int, reward: float, next_state: int) -> None:
        self.q[state][action] += self.alpha * (
            reward + self.gamma * self.q[next_state].max() - self.q[state][action]
        )

    ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    SPECIALS = {3: " G", 5: "XX", 7: " P"}

    def decay_epsilon(self) -> None:
        self.eps *= 0.995
        self.eps = max(self.eps, 0.01)

    def policy_grid(self) -> str:
        rows = []
        for s in range(self.q.shape[0]):
            if s % 4 == 0:
                rows.append("  ")
            if s in self.SPECIALS:
                rows.append(f" {self.SPECIALS[s]} ")
            else:
                best = int(np.argmax(self.q[s]))
                rows.append(f"  {self.ARROWS[best]} ")
            if s % 4 == 3:
                rows.append("\n")
        return "".join(rows)

    def value_grid(self) -> str:
        rows = []
        for s in range(self.q.shape[0]):
            if s % 4 == 0:
                rows.append("  ")
            if s in self.SPECIALS:
                rows.append(f" {self.SPECIALS[s]:>5}")
            else:
                rows.append(f" {self.q[s].max():5.2f}")
            if s % 4 == 3:
                rows.append("\n")
        return "".join(rows)

    def __repr__(self) -> str:
        return f"Policy:\n{self.policy_grid()}Q-Values:\n{self.value_grid()}"


def train():
    env = GridWorld(r=3, c=4)
    agent = QAgent()

    wr = 0
    snapshots = []

    for episode in range(500):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

            if done and reward == 1:
                wr += 1

        if episode % 100 == 0:
            print(f"Episode {episode:>5} | win rate: {wr:>3}% | eps: {agent.eps:.3f}")
            wr = 0

        snapshots.append((episode, agent.q.copy()))

        agent.decay_epsilon()

    print()
    print(agent)

    animate(snapshots)


def animate(snapshots: list[tuple[int, np.ndarray]]):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import TwoSlopeNorm

    specials = {3: "G", 5: "X", 7: "P"}
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    fig, (ax_val, ax_pol) = plt.subplots(1, 2, figsize=(10, 4))
    title = fig.suptitle("Episode 0", fontsize=14)

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    def draw(frame):
        episode, q = snapshots[frame]
        title.set_text(f"Episode {episode}")

        # --- Q-value heatmap ---
        ax_val.clear()
        grid = q.max(axis=1).reshape(3, 4)
        ax_val.imshow(grid, cmap="RdYlGn", norm=norm)
        ax_val.set_title("Q-Values (max)")
        ax_val.set_xticks([])
        ax_val.set_yticks([])
        for s in range(12):
            r, c = s // 4, s % 4
            if s in specials:
                ax_val.text(
                    c,
                    r,
                    specials[s],
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                )
            else:
                ax_val.text(
                    c, r, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=11
                )

        # --- policy arrows ---
        ax_pol.clear()
        ax_pol.imshow(grid, cmap="RdYlGn", norm=norm, alpha=0.3)
        ax_pol.set_title("Policy (best action)")
        ax_pol.set_xticks([])
        ax_pol.set_yticks([])
        for s in range(12):
            r, c = s // 4, s % 4
            if s in specials:
                ax_pol.text(
                    c,
                    r,
                    specials[s],
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                )
            else:
                best = int(np.argmax(q[s]))
                ax_pol.text(c, r, arrows[best], ha="center", va="center", fontsize=20)

    anim = FuncAnimation(fig, draw, frames=len(snapshots), interval=150, repeat=True)
    out = Path(__file__).parent / "q_learning_progress.gif"
    anim.save(out, writer="pillow", fps=6)
    print(f"Animation saved to {out}")
    plt.show()


if __name__ == "__main__":
    train()

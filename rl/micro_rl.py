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

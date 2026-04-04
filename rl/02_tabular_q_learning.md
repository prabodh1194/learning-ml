# RL 2 — Tabular Q-Learning

## The Q-Table

A numpy array of shape `(n_states, n_actions)`. Start with all zeros. Each cell `Q[s, a]` stores "how good is it to do action `a` in state `s`?"

## The Q-Learning Update (THE one-liner)

```
Q[s,a] += alpha * (r + gamma * max(Q[s']) - Q[s,a])
          ─────    ─────────────────────   ────────
          learning     target (what it      current estimate
          rate         SHOULD be)           (what I think)

The bracket = TD error = "how wrong was I?"
```

Like adjusting a restaurant rating: you thought 7/10, experience says 9/10, so you nudge to 7.2. Over many visits, you converge.

## Epsilon-Greedy Exploration

```
With probability (1-eps): pick BEST action (exploit)
With probability eps:     pick RANDOM action (explore)

Start eps=1.0 (all random), decay to eps=0.01 (mostly exploit)
Early: you know nothing → explore!
Later: you've learned   → exploit!
```

## SARSA vs Q-Learning

```
Q-learning (off-policy):  uses max(Q[s']) — "assume I'll be smart next time"
SARSA (on-policy):        uses Q[s',a']  — "use the action I actually picked"

Q-learning = optimistic, learns faster, riskier near cliffs
SARSA      = realistic, safer, slower
```

## What We Built: micro_rl.py

- `GridWorld`: 3x4 grid, goal(+1), pit(-1), wall. row/col movement logic.
- `QAgent`: Q-table, epsilon-greedy, Bellman update, epsilon decay.
- Training: 10k episodes. Win rate 0% → 100% by episode 300.

## Key Observations from Training

```
Learned Policy:           Q-Values:
  →  →  →  G               0.98  0.99  1.00   G
  ↑ XX  ↑  P               0.97    XX  0.99   P
  ↑  ←  ↑  ←               0.87  0.37  0.70  0.09
```

- Q-values decay away from goal (Bellman ripple effect)
- Agent avoids pit column — state 11 (next to pit) has lowest Q
- State 9 goes LEFT (to 0.87) instead of RIGHT (toward pit) — safety learned automatically
- Knowledge flows backward from goal, one Bellman update at a time

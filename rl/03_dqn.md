# DQN — Deep Q-Network

Tabular Q-learning stores one number per (state, action) pair.
DQN replaces that table with a neural net.

## 1. Neural net as Q-function approximator

```
TABULAR (what we had):
  Q-table with 12 rows x 4 columns = 48 numbers
  Q[state=5, action=up] = 0.87    ← just a lookup

DQN:
  state ──→ [ neural net ] ──→ [Q(up)=0.87, Q(right)=0.12, Q(down)=-0.3, Q(left)=0.45]
                                 pick the highest ──→ action = "up"
```

The net takes a state vector as input and spits out one Q-value per action.

For GridWorld: state = one-hot vector of length 12 (state 5 → [0,0,0,0,0,1,0,0,0,0,0,0]).
For CartPole: state = [cart_pos, cart_vel, pole_angle, pole_vel] — 4 continuous floats.

Training = minimize how wrong our Q-predictions are:

```
loss = (predicted_Q - target_Q)²

where target_Q = reward + gamma * max(Q_target(next_state))
                 ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 what we    best possible future value
                 got now    (from the target net)
```

## 2. Experience replay buffer

```
WITHOUT replay:                    WITH replay:
  train on step 1                    store step 1 in buffer
  train on step 2                    store step 2 in buffer
  train on step 3                    store step 3 in buffer
  ...                                ...
  consecutive steps are              sample RANDOM batch of 32
  correlated! NN overfits            from buffer
  to recent experience               → breaks correlation
                                     → stable training
```

It's a list of (state, action, reward, next_state, done) tuples.
When full, oldest entry gets kicked out (ring buffer / deque).
Each training step, sample a random batch.

Implementation: `collections.deque(maxlen=capacity)` handles the ring buffer for free.

## 3. Target network

```
WITHOUT target net:                WITH target net:
  Q_predicted = net(s)               Q_predicted = online_net(s)
  Q_target = r + γ·max(net(s'))      Q_target = r + γ·max(target_net(s'))
            ↑ same net on both sides!           ↑ target_net is a FROZEN copy
            = chasing your own tail             updated every N steps
            = unstable, oscillates              = stable target to aim at
```

Like a student grading their own test vs having a teacher grade it.
The target net is the "teacher" — it changes slowly so the student has a stable goal.

Two ways to update the target net:
- **Hard sync**: copy all weights every N steps (simple but causes jumps)
- **Soft sync** (Polyak): blend every step: `target = 0.5%·online + 99.5%·target` (smoother)

## Tabular vs DQN on GridWorld

Same environment (3x4 grid, goal + pit + wall). Same result.

```
Tabular Q-learning:              DQN:
  stores 48 numbers                stores ~5000 neural net weights
  learns in ~200 episodes          learns in ~200 episodes
  100% win rate                    100% win rate
  exact same optimal policy        exact same optimal policy

  → DQN is overkill here. 12 states fit easily in a table.
```

So why bother? Because...

## CartPole — where DQN earns its keep

```
GridWorld state: integer 0-11     CartPole state: [0.0234, -0.412, 0.087, 0.551]
  → 12 possible states              → infinite possible states (continuous floats)
  → table works fine                 → CAN'T build a table for this
```

CartPole = balance a pole on a cart. Push left or right. +1 reward per step alive.
"Solved" = surviving 500 steps on average.

Our results:
- DQN learns to balance (avg reward 300-450 range)
- But it's UNSTABLE — learns, forgets, relearns, forgets again
- Never cleanly "solves" at 475+

This instability is not a bug. It's a fundamental property of DQN.

## The deadly triad

Three things that, when combined, cause instability:

```
  1. Function approximation    ← neural net (can't represent Q perfectly)
  +
  2. Bootstrapping             ← target depends on our own predictions
  +                               (reward + gamma * max Q_target)
  3. Off-policy learning       ← learning from old buffer data,
                                  not current policy
  ═══════════════════════════
  = unstable training

  Any TWO of these are fine. All THREE together = trouble.
```

```
Tabular Q-learning:
  1. No function approx (exact table)  ← missing one leg
  2. Bootstrapping ✓
  3. Off-policy ✓
  → stable! (converges guaranteed)

DQN:
  1. Function approx ✓    ← all three present
  2. Bootstrapping ✓
  3. Off-policy ✓
  → unstable (no convergence guarantee)
```

This is why we saw the yo-yo pattern on CartPole:

```
Episode  500: avg 375  ← learned well!
Episode  900: avg 195  ← forgot everything
Episode 2000: avg 436  ← relearned!
Episode 2400: avg 148  ← forgot again
```

The replay buffer and target net are PATCHES for this instability — they help, but don't fully solve it. Further improvements (Double DQN, Dueling DQN, Prioritized Replay) exist to reduce it further.

## Hyperparameter observations

What we tried and what mattered:

| Knob | Too low | Too high | Sweet spot |
|------|---------|----------|------------|
| Learning rate | 1e-4: learns too slow, eps decays before agent learns anything | - | 1e-3 worked best |
| Epsilon decay | 0.999: too much random exploration pollutes buffer | 0.995 in 1000 eps: hits 0.01 by ep 1000, stops exploring | 0.995 was OK for 1000 eps |
| Hidden size | 32: too few params for continuous states | 256: overkill, slower | 64 worked for CartPole |
| Buffer size | 1K: forgets good experiences | 100K: too much stale data | 10K was fine |
| Target sync | every step: target too jittery | every 10K: target too stale | every 100-1000 steps |

Key lesson: vanilla DQN is VERY sensitive to hyperparameters. Small changes cause big swings.
That's another reason improvements like Double DQN exist — they're more robust to tuning.

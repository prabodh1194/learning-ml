# RL 1 — What is Reinforcement Learning?

## The Big Idea

Nobody gives you a manual. You try stuff, get rewards (or punishments), and figure it out.

## The RL Loop

```
         ┌──────────────────────────────┐
         │                              │
         v                              │
    ┌─────────┐   action    ┌───────────┴───┐
    │  AGENT  │────────────>│  ENVIRONMENT  │
    │ (brain) │             │   (world)     │
    │         │<────────────│               │
    └─────────┘  state +    └───────────────┘
                 reward

  1. Agent SEES the state     (where am I?)
  2. Agent PICKS an action    (go left? go right?)
  3. Environment RESPONDS     (new state + reward)
  4. Repeat forever
```

## Supervised Learning vs RL

```
SUPERVISED LEARNING:
  "Here's a cat photo, the label is CAT"
  -> Teacher gives you the RIGHT answer every time

RL:
  "Here's the game screen. You moved right. Nothing happened.
   You moved up. YOU DIED. Reward: -1"
  -> No teacher. Just consequences.
```

Key difference: in RL you don't know WHICH action was the mistake. This is the **credit assignment problem**.

## Markov Decision Process (MDP)

An MDP is a world where decisions matter. It has 4 things:

```
MDP = (S, A, P, R)

S = States        -> all possible positions
A = Actions       -> {up, down, left, right}
P = Transitions   -> P(s'|s,a) = "if I'm at s and do a, where do I end up?"
R = Rewards       -> R(s,a) = "what reward do I get for doing a in state s?"
```

The **Markov property**: only the CURRENT state matters, not how you got there. Like chess — the board position right now tells you everything.

## Grid World Example

```
┌─────┬─────┬─────┬─────┐
│  S  │     │     │ +1  │   S = start
├─────┼─────┼─────┼─────┤   +1 = goal (reward +1)
│     │ XXX │     │ -1  │   -1 = pit (reward -1)
├─────┼─────┼─────┼─────┤   XXX = wall
│     │     │     │     │
└─────┴─────┴─────┴─────┘
```

## R, G, V — The Three Levels of Reward

```
R(s,a) = ONE reward, ONE step
         "I moved right and got +1"

G      = SUM of ALL rewards in an episode (discounted)
         "I played the whole game and collected 0.729 total"

V(s)   = AVERAGE of G across MANY episodes from state s
         "If I play 1000 games from here, G averages to 0.85"
```

Analogy:

```
R = one paycheck
G = your total earnings for the year
V = your EXPECTED annual earnings (what you'd tell the bank)
```

G is random (depends on what happens). V is the expectation (the average).

## Discount Factor (gamma)

A reward NOW is better than a reward LATER.

```
G = r1 + y*r2 + y^2*r3 + y^3*r4 + ...

y (gamma) = discount factor, between 0 and 1

  y = 0    -> greedy, only sees NOW
  y = 0.9  -> balanced, plans ahead but prefers sooner
  y = 0.99 -> very far-sighted, patient
  y = 1    -> math breaks (sum can be infinite), no urgency
```

## The Bellman Equation

THE equation of RL. Everything builds on this.

```
V(s) = max [ R(s,a) + y * V(s') ]
        a

"The value of a state =
  the best immediate reward I can get
  + the discounted value of where I end up"
```

It's recursive — the value of HERE depends on the value of THERE.

```
Example (y = 0.9):

  V(goal) = +1.0
  V(one-left-of-goal)  = 0 + 0.9 * 1.0  = 0.90
  V(two-left-of-goal)  = 0 + 0.9 * 0.9  = 0.81
```

## V(s) vs Q(s,a)

```
V(s)   = "how good is this STATE?"
Q(s,a) = "how good is this STATE + ACTION combo?"

V(s) = max Q(s,a)
        a
```

Why Q is more useful:

```
V(s)   = "this place is nice"       (okay but WHAT DO I DO?)
Q(s,a) = "going RIGHT here is nice" (NOW I know what to do!)
```

V tells you the weather forecast. Q tells you whether to bring an umbrella.

## Bellman Equation for Q

```
Q(s,a) = R(s,a) + y * max Q(s', a')
                        a'

"The value of doing action a in state s =
  immediate reward +
  discounted value of the BEST action in the next state"
```

This is what Q-learning uses. You build a table of Q(s,a) for every state-action pair and update it with this equation until it converges.

## Key Takeaways

- RL = learn from consequences, not labels
- MDP = (States, Actions, Transitions, Rewards) + Markov property
- Discount factor y controls how far ahead the agent plans
- Bellman equation = recursive definition of value
- Q(s,a) > V(s) because it's actionable — tells you WHAT to do

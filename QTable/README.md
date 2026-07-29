# QTable — two tabular Q-learning bots

A second pair of bots for the obstacle course, learning by Q-table instead of by
neural network. Separate from `QN/` in every respect except the two that have to
match for a comparison to mean anything: identical action set, identical reward
constants.

Pure Python. No torch, no numpy — a `dict` of small float lists is genuinely the
right data structure, and keeping it plain makes the update rule readable
against the textbook version.

```
QT_paths.py     puts Architecture/ on sys.path (both modules below need it)
QT_encoder.py   room -> hashable table key. The hard part.
QT_env.py       environment adapter. Sibling of QN/env_bridge.py.
QT_model.py     QTable + TabularAgent. Mirrors QN_train.Agent's surface.
QT_train.py     Config, eps_at, build_agents, train, evaluate, plot.
```

## Running it

```bash
cd QTable
python QT_train.py fixed                  # one room, exact encoder  -> 100%
python QT_train.py procedural tutorial    # new room every episode   -> ~52%
python QT_env.py                          # smoke test, no training
```

Both write `.pkl` tables and a reward plot next to themselves.

## The problem this module had to solve

`QN/env_bridge.py` hands its network a **191-float vector** (5×5×7 local view +
16 globals). A network interpolates over that happily. A table cannot index it —
tabular methods need a small, discrete, hashable state space. Choosing that
space *is* the design of a tabular agent, so `QT_encoder.py` is where the real
work went. Two encoders ship, at opposite ends of the trade-off:

| | `PositionEncoder` | `FeatureEncoder` |
|---|---|---|
| key | `(x, y, keys, switches, checkpoints, exit_open)` | local terrain, goal bearing, goal distance, underfoot, partner, hold-duty, last move |
| Markov | yes | no — aliased |
| converges to optimal | yes, provably | no |
| transfers to a new room | no, not at all | yes |
| use with | `room_seeds=[n]` | procedural generation |

`PositionEncoder` is indexed by literal coordinates, so it transfers to a new
seed about as well as a street map of one city transfers to another. That is not
a flaw to fix; it is the defining limitation of tabular methods, and showing it
cleanly is worth more than hiding it.

## Measured results

| setting | encoder | solve rate |
|---|---|---|
| fixed room (seed 7, "easy") | `PositionEncoder` | **100%** (200/200), 30 steps of a 200 budget |
| procedural, "tutorial" | `FeatureEncoder` | **52%** at ε=0.05, 27.5% greedy |
| procedural, "easy" | `FeatureEncoder` | near 0% — see below |

3000–4000 episodes, under two minutes each on CPU.

## Three findings that apply to the DQN too

These came out of building this and are not tabular-specific. Worth acting on
regardless of which learner you keep.

**1. Reachability-aware targeting is worth more than the learning algorithm.**
A scripted *perfect* BFS policy — one that always walks the shortest path to the
current objective — solved only **22/60** tutorial rooms. It beelines at
objectives sitting behind locked doors and never works out that the key is the
thing to fetch. Teaching the target pointer to check "can I actually reach
this?" first, and to fall back to whatever unlocks the way, took the same
policy to **60/60**. `Reachability` in `QT_encoder.py` does this, caching the
flood fill and recomputing only when a door opens or a crate moves.

That 22/60 is a **ceiling on any learner** using that targeting, including your
DQN. It is not a tabular limitation.

**2. Your generator makes genuinely cooperative puzzles, and "easy" is not easy.**
Every "easy" room sampled had 1–2 HOLD switches — one bot must stand on a switch
while the other walks through the door it controls. Step off and it re-locks.
Even with reachability-aware targeting, a perfect pathfinder solves 10/60 on
"easy" and 0/60 on "standard", because pathfinding alone cannot express "wait
here while my partner goes". `FeatureEncoder` exposes a `duty` feature and a
"stay put" target so the table *can* represent it, but tutorial is the honest
starting point for the curriculum. Ladder: fixed room → small `room_seeds` pool
→ tutorial → easy.

**3. Greedy evaluation badly understates the learned policy.**
Under an aliased encoder, a deterministic greedy policy can walk into a
two-state cycle it cannot perceive, and burn the rest of the episode there.
Same tables, same rooms:

```
eval(eps=0.00): 27.5%      eval(eps=0.05): 52.0%      eval(eps=0.10): 51.5%
```

`evaluate()` therefore defaults to `eps=0.05`, which looks like cheating and is
not — the gap between the two numbers is a direct readout of how much aliasing
the encoder carries. `PositionEncoder` has none and scores the same either way.
Adding a last-move feature to break oscillation roughly doubled both numbers
(7%→26% greedy, 35%→60% noisy) for one extra dimension.

## One trap worth knowing about

Eligibility traces multiply the effective step size by about `1/(1 - γλ)`. At
the defaults (γ=0.99, λ=0.8) that is **4.8×**, so α=0.3 becomes an effective
1.44 — above 1.0, which overshoots every update and oscillates instead of
converging.

This produced a genuinely alarming result during development: on the *same fixed
room*, `alpha_decay=0.9995` scored **100%** and `alpha_decay=0.9998` scored
**0%**. Neither number is wrong; the faster decay simply fell under the
stability threshold sooner, by luck.

`TabularAgent` now caps α at `0.9 × (1 - γλ)` automatically and reports
`(capped)` in `describe()`. It also rejects `alpha_decay >= 1.0` when traces are
on, because a constant step size does not converge under Robbins–Monro — it
hovers, and with traces that hovering is large enough to destroy the policy.
Failing loudly beats failing mysteriously.

## How it differs from `QN/`

| | `QN/` (DQN) | `QTable/` |
|---|---|---|
| replay buffer | yes | **no** — a table has no cross-state interference to average out |
| target network | yes | **no** — no shared function to destabilise |
| optimiser | Adam | one scalar α per entry, decayed on a schedule |
| credit assignment | replay | eligibility traces (`λ=0.8`) |
| update rule | Q-learning | selectable: `q_learning`, `sarsa`, `expected_sarsa` |
| memory | fixed | grows with states visited (~19k states/agent after 4k episodes) |
| reward shaping | none | optional potential-based, on by default |

Set `Config.shaping = 0.0` for a run strictly comparable to the DQN. It is on by
default because on procedural rooms the exit bonus alone is too sparse to find.

`train(env, cfg)` returns `(agents, history)` exactly as `QN_train.train` does,
so the same plotting and comparison code drives either family.

## Things I would look at next

- Fold `Reachability` into `QN/env_bridge.py` — the DQN is paying the same
  22/60 targeting ceiling, and its `_extras()` "nearest uncollected key" vector
  has the identical blind spot.
- `DQN/DQN_train.py` has no training loop, and `DQN_model.py` expects a 96-wide
  input against the bridge's 191. Either wire it up or delete it; right now it
  reads as the real DQN and `QN/` reads as something smaller.
- Try `shared_table=True` (`build_agents`) — the roles are near-symmetric, so
  sharing roughly halves the time to fill the table. The cost is that the bots
  can no longer specialise, which matters exactly when one needs to hold a
  switch while the other walks.

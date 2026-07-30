# QTable — tabular Q-learning bots for the obstacle course

A second pair of bots that learn by Q-table instead of by neural network,
sharing the DQN's environment exactly so the two can be compared.

Drop the folder next to `Architecture/` and `DQN/`:

```
Ai-Obstacle-Course-/
├── Architecture/
├── DQN/
└── QTable/        ← here
```

```bash
cd QTable
python QT_train.py                        # single-stage, 'easy' preset
python QT_curriculum.py --list            # show the 31 stages
python QT_curriculum.py                   # staged training
python QT_curriculum.py --stages 14 --episodes 400 --rounds 1
```

Requires Python 3.10+, torch (via the bridge), and matplotlib for plots.

## Files

```
QT_paths.py       path bootstrap; matches the DQN's mixed import convention
QT_env.py         TabularBridge(CoopEnvBridge) -- 60 lines, overrides _obs and reset
QT_encoder.py     the bridge's 51 globals -> a 13-field discrete key
QT_model.py       QTable + TabularAgent, mirroring DQN_train.Agent
QT_train.py       Config, eps_at, train, evaluate, plot
QT_curriculum.py  runs the DQN's own default_stages()
```

## The one design decision that matters

`QT_env.py` **inherits** `CoopEnvBridge` rather than reimplementing it.

An earlier version of this package hand-copied the bridge's dynamics so the
tabular stack could stay torch-free. Within a week the copy had drifted — still
nine actions when the bridge had moved to six, still `R_HAZARD = -1.0` when the
bridge said `-0.5`. A drifted copy does not produce a comparison; it produces two
unrelated numbers.

So the subclass overrides exactly two methods:

```python
def _obs(self, i):          # 1325 floats -> a hashable tuple
def reset(self, seed=None):  # redraw until the stage's accepts() passes
```

Every reward, every mechanic, every room comes from the parent untouched. The
expensive observation vector is never built. Torch comes along as a dependency,
which is the price.

## The state key

The DQN reads 1325 floats: a 7×7×26 view plus 51 globals. A table has to index
that, so it gets bucketed to 13 fields:

| # | field | values | source |
|---|---|---|---|
| 0 | route | 5 | `route_dx`, `route_dy` |
| 1 | route_tile | 3 | free / blocked / hazard |
| 2 | hazards_adj | 3 | 0 / 1 / 2+ |
| 3 | route_wait | 2 | `route_wait` |
| 4 | goal_kind | 6 | `goal_key`/`switch`/`checkpoint`/`exit`/`crate` |
| 5 | reachable | 2 | `goal_reachable` |
| 6 | dist | 5 | banded `goal_distance` |
| 7 | can_interact | 2 | `can_interact` |
| 8 | can_push | 2 | `can_push` |
| 9 | switch_mode | 4 | toggle / hold / oneshot |
| 10 | mech | 5 | timed door + bridge status, collapsed |
| 11 | teammate | 3 | far / near / same tile |
| 12 | ball | 2 | wipeout ball threatens an adjacent tile |

`_extras()` had already done the hard part — `route_dx`, `goal_reachable`,
`can_push` and the rest are a strictly better feature set than anything
hand-rolled, and they come free.

**Be clear about the route hint.** `route_dx`/`route_dy` is a BFS-derived next
step toward the goal, and it is in the key because the DQN gets it too
(`ROUTE_Q_BIAS = 0.5` in `DQN_model`). With it, the table is not learning to
navigate from raw terrain — it is learning *when to follow the route, when to
deviate for a hazard, and what to do on arrival*. That is a real learning
problem and it is the same one the DQN faces, which is the point. But a high
score here is not evidence of navigation learned from scratch.

## What matches the DQN, and what doesn't

| | DQN | QTable |
|---|---|---|
| action masking | `action_mask` | same, imported |
| route bias | `ROUTE_Q_BIAS` | same constant, imported |
| Agent surface | act/remember/learn_batch/sync/save/load | identical |
| rewards, rooms, dynamics | `CoopEnvBridge` | inherited unchanged |
| replay buffer | grouped, weighted | **none** — no cross-state interference to average out |
| target network | yes | **none** — no shared weights to destabilise |
| dueling heads | value + advantage | **none** — a table shares nothing between entries |
| optimiser | Adam | one scalar α, decayed per episode |
| credit assignment | replay | eligibility traces, λ=0.8 |
| update rule | Q-learning | selectable: `q_learning`, `sarsa`, `expected_sarsa` |

`remember()` performs the update immediately, since a table learns online.
`learn_batch()` and `sync()` are no-ops kept only so `TabularAgent` is drop-in
for `DQN_train.Agent`.

## Curriculum

`QT_curriculum.py` imports `default_stages()` from `DQN/curriculum.py` rather
than defining its own, so both families face the same 31 lessons in the same
order. Add a stage for the DQN and the table gets it too.

**Every episode is a freshly generated room.** A stage is a `GenerationConfig`
plus an `accepts` predicate; `TabularBridge.reset()` redraws until a generated
room satisfies it. The tables never see a layout twice, so promotion is measured
on unseen rooms by construction. Two bridges on different master seeds give the
train and validation numbers — they should come out close, and a large gap is an
alarm rather than a result.

Tables carry forward between stages, which is tabular's structural advantage
here: "the route says east and east is free" transfers intact into harder rooms.
This runner deliberately omits `CurriculumRunner`'s regression detection,
retention scoring and rollback — that is ~2200 lines protecting a function
approximator from catastrophic forgetting, and a table cannot catastrophically
forget, since writing entry X never touches entry Y.

## Measured results

Single stage, procedural `easy`, 2000 episodes (~207s):

```
eval(eps=0.05): solved 150/200 (75.0%)  return 10.31  steps 92.3
eval(eps=0.00): solved 128/200 (64.0%)  return  8.68  steps 103.0
```

Curriculum, stages 1–6, fresh room every episode, one round each:

| stage | train | validation | states |
|---|---|---|---|
| open_navigation | 100% | 100% | 88 |
| layout_variation | 100% | 100% | 102 |
| obstacle_navigation | 100% | 100% | 102 |
| hazard_avoidance | 100% | 100% | 102 |
| checkpoint_exit | 100% | 100% | 180 |
| key_door | 100% | 100% | 239 |

Stages 1–14 all promoted on the first round in a separate run.

Two things to notice. The tables are *small* — 239 states after six stages. And
state counts barely move across the navigation stages (102 → 102 → 102), which is
the curriculum working: obstacle and hazard rooms reuse entries already learned
in open navigation, so only genuinely new mechanics add rows.

## Known gaps

- **Stages 15–31 are untrained.** Room generation is verified for all of them
  (8s worst case for `full_course_mix`), but hold switches, paired levers,
  crate-on-switch, timed doors, bridges and wipeout balls have not been trained
  through. Expect the first real failures there.
- **`can_push` is a boolean.** The key says a crate is pushable but not *where*
  it would go. `crate_hold_switch` needs a crate parked on a switch, and the
  table may be unable to represent the difference between a useful push and a
  useless one. If that stage stalls, this is the first thing to widen.
- **Greedy evaluation understates the policy.** 75% at ε=0.05 against 64% at
  ε=0.00. Aliased keys let a deterministic policy enter a cycle it cannot
  perceive; a trace of noise breaks it. `evaluate()` defaults to `eps=0.05` for
  that reason, and the gap between the two is a readout of how much aliasing the
  key carries.
- **No held-out test harness yet.** An earlier version had `QT_test.py` with
  baselines (random, and a scripted pathfinder as a ceiling). Worth rebuilding
  against this bridge — a solve count means little without knowing what the room
  set permits.

## A trap worth knowing about

Eligibility traces multiply the effective step size by about `1/(1 - γλ)` — at
γ=0.99, λ=0.8 that is 4.8×, so α=0.3 becomes an effective 1.44 and oscillates
instead of converging. In an earlier version, on the same fixed room,
`alpha_decay=0.9995` scored 100% and `0.9998` scored 0%. Neither was wrong; the
faster decay dropped under the stability threshold sooner, by luck.

`TabularAgent` now caps α at `0.9 × (1 - γλ)` and reports `(capped)` in
`describe()`. It also rejects `alpha_decay >= 1.0` when traces are on, because a
constant step size does not converge under Robbins–Monro. Failing loudly beats
failing mysteriously.

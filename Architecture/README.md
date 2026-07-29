# coop_env

A procedurally generated 2D grid environment for **future** multi-agent
reinforcement learning experiments. It generates an unlimited number of
randomized cooperative rooms from a seed, proves each one is completable before
handing it over, renders it for inspection, and resets cleanly between episodes.

Pure Python 3.11+, **standard library only**. No install step.

## What this is not

By design, this project contains **no agents**. There are no policies, reward
functions, neural networks, training loops, action handling, agent pathfinding,
or human controls anywhere in it. Rooms provide two *spawn tiles*; nothing ever
stands on them.

`coop_env/interfaces.py` holds unimplemented placeholders describing where a
Gymnasium / PettingZoo / ML-Agents layer would attach later. Every method there
raises `NotImplementedError`. `tests/test_scope.py` enforces this boundary: it
fails if an agent-shaped class, an ML dependency, or a stored solution appears
in the package.

**One clarification.** The brief rules out pathfinding, and also requires proof
that keys are reachable and the exit can be unlocked. Those are different
things, and both are honoured. `coop_env/validation/` runs a static reachability
analysis **once, at generation time**, over an abstract region graph. It answers
"could this room be completed?" and returns a boolean. It never produces a
route, never runs during an episode, and nothing consumes its output but the
accept/reject decision. No navigation exists for anything to use.

## Quick start

```bash
python3 run_tests.py
```

```bash
python3 tools/inspect_room.py --seed 42 --preset standard
```

```bash
python3 tools/generate_gallery.py --sweep --count 30 --out output/sweep.html
```

```python
from coop_env import EnvironmentSession, GenerationConfig

session = EnvironmentSession(GenerationConfig.preset("standard"), master_seed=7)

session.reset()                    # brand new room, fresh seed
session.reset(seed=1234)           # rebuild that exact room
session.reset(same_room=True)      # same layout, mechanisms back to defaults
session.reset(reroll=True)         # regenerate from the current seed
session.reset_state()              # reset objects without a new episode

print(session.room.summary())
print(session.report.summary())    # validation verdict for the loaded room
```

Straight to the generator, no session:

```python
from coop_env import RoomGenerator, GenerationConfig

generator = RoomGenerator(GenerationConfig.from_complexity(0.75))
room = generator.generate(seed=99)
rooms = generator.generate_many(100, start_seed=0)
```

## Reproducibility

The same seed always produces the same room — across processes, and regardless
of what else the generator did first. Each subsystem draws from its own named
sub-stream (`rng.derive("hazards")`), so adding a random draw to one stage
cannot shift the numbers another stage sees. Named streams are hashed with
BLAKE2b rather than Python's `hash()`, which is salted per process.

## The world

Terrain is static and lives in an integer grid (`Tile`): void, floor, wall,
static obstacle, and four hazard surfaces (lava, spikes, water, pit).

Everything with state is an entity, immutable in the blueprint and mutable only
in `EpisodeState`:

| Entity | Behaviour |
| --- | --- |
| `AgentSpawn` | A reserved start tile. Two per room. Marks a location only. |
| `ExitDoor` | The objective. Opens when its requirement is met. |
| `Key` | Portable token. May open more than one door (shared keys). |
| `LockedDoor` | Blocks a doorway. Latching, hold-open, or timed. |
| `Switch` | Lever. `TOGGLE`, `ONESHOT`, or `HOLD` (active only while weighed down by an agent slot or a crate). |
| `MovingPlatform` | Shuttles a fixed track; position is a pure function of the tick. |
| `PushableBlock` | Crate. Can weigh down a hold-lever, freeing an agent. |
| `Checkpoint` | Progress marker; can feed a door requirement. |
| `ResetZone` | Area that returns whatever enters it to a spawn or checkpoint. |
| `TemporaryBridge` | Hazard tiles that phase solid on a fixed cycle. |

Unlock conditions are declarative `Requirement` objects (`KeyRequirement`,
`SwitchRequirement`, `CheckpointRequirement`, `CompositeRequirement`), combined
with `ALL`, `ANY`, or `SIMULTANEOUS`.

## How cooperation arises

The generator never scripts a solution. It places mechanisms whose *structure*
can make two agents useful, and lets the layout speak:

- **Paired levers** — one door, two `HOLD` levers, `SIMULTANEOUS`. Both must be
  weighed down at the same instant, then the door latches open for good. Toggle
  switches would not work here: one agent could just flip them in sequence.
- **Hold-levers** — a non-latching door that stays open only while its lever is
  weighed down. One slot holds, the other passes; the passage is one-way.
- **Shared keys** — a single key gating two different doors.
- **Split routes** — the two spawns may start in different regions of the
  initially-open zone, on branches that reconnect later.
- **Hazard crossings** — platform-bridged gaps that must be timed.

The validator reports which doors *no single agent slot could have opened
alone* (`cooperative_clusters`) and whether the exit is reachable by both slots
at once (`exit_jointly_reachable`). These are measured from the finished room,
not asserted by the generator.

## Difficulty

One scalar drives everything:

```python
GenerationConfig.from_complexity(0.0)   # tutorial
GenerationConfig.from_complexity(1.0)   # brutal
GenerationConfig.preset("hard")         # named point on the same axis
GenerationConfig.preset("hard", hazard_density=0.02, num_keys=(4, 4))  # override
```

Every parameter can also be set directly: `width`, `height`, `shape_weights`,
`region_count`, `branching_factor`, `corridor_width`, `obstacle_density`,
`hazard_density`, `hazard_weights`, `hazard_blob_size`, `num_keys`,
`num_locked_doors`, `num_switches`, `num_moving_platforms`, `num_pushable_blocks`, `num_checkpoints`,
`num_reset_zones`, `num_temporary_bridges`, `puzzle_chain_length`,
`exit_objective_count`, `required_cooperative_actions`,
`timed_door_probability`, `platform_bridge_probability`,
`separate_spawns_probability`, `exit_requires_both_agents`, `max_attempts`,
`raise_on_failure`.

`config.validate()` returns readable problems; `RoomGenerator` calls
`require_valid()` and raises on a bad config rather than generating nonsense.

Measured across 40 seeds per preset (`python3 tools/benchmark.py`):

| preset | ms/room | attempts | fallbacks | unique layouts | rooms with a co-op door | mean regions |
| --- | --- | --- | --- | --- | --- | --- |
| tutorial | 2.5 | 1.00 | 0 | 40/40 | 0/40 (budget is 0) | 2.1 |
| easy | 6.1 | 1.02 | 0 | 40/40 | 38/40 | 3.7 |
| standard | 17.1 | 1.00 | 0 | 40/40 | 40/40 | 6.5 |
| hard | 51.4 | 1.10 | 0 | 40/40 | 40/40 | 9.6 |
| brutal | 131.1 | 1.12 | 0 | 40/40 | 40/40 | 14.0 |

Lock-chain depth (validator-measured) rises with difficulty: tutorial rooms are
mostly 0–1 gates deep, brutal rooms mostly 2–4.

## Validation

No room is returned until it passes. Structural rules check placement, terrain,
stacking, and dangling references; the solvability analysis then proves the room
is completable. A failure regenerates from a fresh sub-seed, up to
`max_attempts`. If the budget runs out, a small correct-by-construction fallback
room is returned and tagged `metadata["fallback"]` — or an exception is raised
if you set `raise_on_failure=True`.

```python
from coop_env import validate_room
report = validate_room(room)
report.ok            # bool
report.summary()     # 'valid' / 'invalid: <first error>'
report.issues        # errors and warnings, each with a code
report.solvability   # reachable regions, co-op doors, chain length
```

The analysis is deliberately pessimistic: hazards are never crossable without a
platform, crates never move, and a trigger only counts if some slot can reach
it. A room it accepts is completable; a room it rejects might have been fine and
is simply regenerated. That trade keeps the guarantee meaningful.

## Rendering

Rendering is strictly downstream — no generation or validation module imports
it, and rooms carry no visual data. Swapping in a different renderer means
writing one module against `Room` and `EpisodeState`.

```python
from coop_env.rendering import render_ascii, render_svg, save_gallery

print(render_ascii(room))                   # terminal / logs / diffs
svg = render_svg(room, state)               # standalone, no external assets
save_gallery(rooms, "output/gallery.html")  # many rooms on one page
```

Colours and glyphs live in `coop_env/rendering/palette.py`.

## Layout

```
coop_env/
  config.py         GenerationConfig, presets, difficulty scaling
  rng.py            seeded randomness with independent named sub-streams
  tiles.py          terrain vocabulary
  entities.py       interactive object blueprints
  requirements.py   declarative unlock conditions
  room.py           immutable generated room + region topology
  state.py          mutable per-episode state, mechanism behaviour, reset
  episode.py        EnvironmentSession: lifecycle and reset behaviours
  interfaces.py     placeholders for a future RL layer (all NotImplementedError)
  generation/       shapes -> partition -> layout -> terrain -> topology -> mechanisms
  validation/       connectivity model, solvability analysis, rule registry
  rendering/        palette, ASCII, SVG, HTML gallery
  utils/            geometry, grid, graph
tools/              inspect_room.py, generate_gallery.py, benchmark.py
tests/              110 tests, stdlib unittest
```

See `ARCHITECTURE.md` for the generation pipeline, the solvability model and its
assumptions, and how to add a new mechanic.

## Extending

Adding a mechanic touches three places and nothing else:

1. a dataclass in `entities.py`
2. a placement rule in `generation/mechanisms.py`
3. a glyph in the renderers

New validation invariants are a function appended to `STRUCTURAL_RULES`. New
room silhouettes are a builder registered in `SHAPE_BUILDERS`.

## Handoff to a training stack

Already in place for whoever builds that layer:

- `Room.terrain.to_list()` — flat row-major terrain, ready for an array backend
- `Tile` / `EntityKind` — stable integer ids for encoding
- `Room.spawns` — the two start tiles
- `EpisodeState.is_walkable(pos)` — the movement rule a physics layer enforces
- `EpisodeState.collect_key/set_switch/place_block` — mechanism seams
- `EpisodeState.snapshot()/restore()` — serialisable episode state
- `EnvironmentSession.on_reset` — observer hook for a wrapper
- `session.advance_time(n)` — clock only; platforms and timers, nobody moving

Not provided, and out of scope: observations, actions, rewards, termination.

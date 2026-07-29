# Architecture

How a seed becomes a validated room, why the solvability proof holds, and where
to attach new work.

## Layering

Dependencies run strictly downward. Nothing below imports anything above it.

```
tools/            CLI entry points
  rendering/      ASCII, SVG, HTML gallery          <- consumes Room + EpisodeState
    episode.py    EnvironmentSession, reset lifecycle
      state.py    mutable episode state, mechanism behaviour
      validation/ connectivity model, solvability, rules
      generation/ the pipeline
        room.py entities.py requirements.py tiles.py config.py rng.py
          utils/ geometry, grid, graph
```

Two consequences worth naming:

- **Rendering is a leaf.** No generation or validation module imports it, and a
  `Room` carries no colours, glyphs, or layout hints. Replacing the renderer
  cannot affect what gets generated.
- **Validation does not import the generator.** It rebuilds its own model of the
  room from terrain and entities. A check that read the generator's notes would
  only restate them.

## The generation pipeline

```
seed
 └─> shapes.py      silhouette: rectangle, L, T, plus, donut, diamond, cavern, terrace
     partition.py   BSP split into sub-areas, each split reserving a wall line
     layout.py      paint dividers as wall, carve doorways (spanning tree + branches)
     terrain.py     obstacles and hazard pools
     topology.py    region graph derived from the finished terrain
     mechanisms.py  spawns, gates, triggers, exit, optional extras
     validator.py   accept, or throw the room away and retry
```

Every stage takes plain data and returns plain data. Any one can be replaced
without the others noticing.

### Silhouette

Eight builders produce a floor mask, each reduced to its largest 4-connected
component so a shape can never hand a disconnected blob downstream. A
silhouette that fills less than 28% of its bounding box falls back to a plain
rectangle, so generation cannot stall on an unlucky cellular-automata draw.

### Partition and doorways

BSP splits the largest splittable leaf each round, which keeps sub-areas similar
in size instead of producing one hall and a row of slivers. Each split reserves
a one-tile divider that becomes wall.

Doorway candidates are wall tiles with floor on opposite sides belonging to
different leaves. A randomised spanning tree over those candidates guarantees
every sub-area is reachable; `branching_factor` then decides how many redundant
links stay open — the difference between one forced route and alternative paths
that reconnect later. Floor left stranded by a thin silhouette neck is walled
off rather than treated as a failure.

### Terrain

Obstacles and hazards are grown as blobs. Every blob is written through a
connectivity guard: if the open floor stops being one piece, the blob is rolled
back. The floor set is tracked incrementally rather than rescanned — that one
change took brutal-preset generation from ~2.7 s to ~130 ms per room.

Doorway tiles and their approaches are protected from scatter, so a hazard pool
can never quietly seal a link.

Because nothing is allowed to sever the floor, the walkable area stays a
single connected piece from this stage onward.

### Topology

A **region** is a patch of floor you cannot leave without passing through a
doorway. Regions are computed from the finished terrain, not carried over from
the BSP leaves, so obstacles and hazard pools that reshaped an area are
reflected honestly. Touching doorway tiles are grouped into one link, so a
widened corridor counts once.

Edges are doorways. If the graph comes out
disconnected, the attempt is rejected before mechanisms are placed.

## How solvability is constructed

This is the core of the design.

The generator roots the region graph at the spawn region and processes candidate
gates **in order of increasing depth**. When it processes gate `e`, it computes
which regions are still reachable from the spawn with `e` *and every
not-yet-processed gate* treated as closed. That set is the gate's
**prerequisite zone**, and the gate's trigger — key, lever, lever pair — is
placed only inside it.

Because a gate's trigger always lives in territory that opens before the gate
does, the room is solvable by construction. Redundant links do not break the
argument: the prerequisite zone is computed by an actual traversal of the graph,
so a cycle that bypasses `e` simply makes the zone larger.

This is a structural argument, not a solution. No ordering of actions is
recorded anywhere, and `tests/test_scope.py` asserts that no walkthrough leaks
into room metadata.

Two rules keep cooperative gates sane:

- A **hold-lever** gate is refused if the path already crosses one. With two
  agents, a second would leave nobody free to move on.
- Hold-levers are refused entirely when `exit_requires_both_agents` is set,
  since any of them strands one agent behind.

## How solvability is verified

`validation/` re-derives everything independently and runs a monotone fixpoint:

- Each agent slot starts with the regions its spawn can reach over open ground.
- A door opens when its requirement is satisfiable by the slots that can
  currently reach the relevant triggers.
- Opening a door grows the reachable sets, which may satisfy further doors.
  Repeat until nothing changes.

The two door flavours differ, and that difference is what makes cooperative
layouts *verifiable* rather than merely intended:

| flavour | condition to pass | effect |
| --- | --- | --- |
| latching | requirement satisfiable by the slots between them | opens permanently, for both |
| non-latching (hold-lever) | the *other* slot can reach the lever | slot passes one-way; door shuts behind |

`SIMULTANEOUS` requirements need one distinct slot per trigger, checked as a
matching over slot permutations — the structural form of "two buttons, two
agents, same instant".

Derived measurements, all computed from the finished room:

- `cooperative_clusters` — doors no single slot could have opened alone
- `one_way_clusters` — hold-lever passages
- `joint_reachable` / `exit_jointly_reachable` — where both slots can stand at
  the *same time*, using permanently-open doors only
- `unlock_round` / `chain_length` — which pass of the fixpoint each door fell on.
  A door opening on round 2 needed something found past a round-1 door, so the
  highest round is the room's longest lock-and-key chain.

### Deliberate pessimism

The model refuses to assume anything clever:

- hazards are never crossable without a bridge
- crates never move
- a trigger counts only if some slot can reach it
- an unrecognised requirement type is treated as unsatisfiable

So the analysis is **sound but incomplete**: every room it accepts is
completable, and a room it rejects might have been fine. Rejections cost one
regeneration, which measures at 1.0–1.12 attempts per room across all presets —
cheap insurance for a guarantee that actually means something.

### Known limits

- Both spawns are placed in the initially-open zone. Asymmetric starts, where
  each agent begins sealed in a separate pocket and must free the other, would
  need a richer per-slot model than the current fixpoint.
- Crates are analysed as permanently solid, so a puzzle that *requires* pushing
  a crate to progress would be rejected rather than accepted. Crates are
  therefore placed as optional tools, in open ground, never as gates.

## Reproducibility

`SeededRandom` wraps `random.Random` and can spawn named sub-streams:

```python
rng.derive("layout")   # stable child stream
rng.fork("attempt:3")  # fresh stream, used per retry
```

Two properties this buys:

1. **Stage independence.** Adding a draw inside the hazard pass cannot shift the
   numbers the layout pass sees.
2. **Cross-process stability.** Labels are hashed with BLAKE2b, not Python's
   salted `hash()`.

Anywhere a random choice touches a `set`, it is sorted first. Iteration order of
a set is not a reproducible thing to draw from.

## Blueprint and state

`Room` is immutable: terrain, entities, topology, metadata. `EpisodeState` holds
everything that can change and is **rebuilt** from the room on reset rather than
mutated back. That is why reset is exact by construction — there is no
accumulated drift to undo, and `state.snapshot()` before and after round-trips
identically.

Time-driven mechanics are pure functions of the tick:

```python
bridge.is_solid_at(tick)     # total, deterministic, no simulation loop
bridge.is_solid_at(tick)
```

`advance(n)` moves the clock and counts down timed doors. A timed door that
expires stays shut until its trigger is released and thrown again — otherwise a
still-active lever would re-open it on the same tick.

## Extension points

| To add | Where |
| --- | --- |
| a mechanic | dataclass in `entities.py`, placement in `generation/mechanisms.py`, glyph in the renderers |
| an unlock condition | `Requirement` subclass in `requirements.py`, plus a branch in `solvability._satisfiable` |
| a room silhouette | builder registered in `shapes.SHAPE_BUILDERS`, member in `config.RoomShape` |
| a validation rule | function appended to `validator.STRUCTURAL_RULES` |
| a gate style | member of `mechanisms.GateKind`, branch in `_install_gate` |
| a renderer | new module consuming `Room` + `EpisodeState` |

If a new `Requirement` type is added without teaching `_satisfiable` about it,
the analysis treats it as unsatisfiable and rooms using it are regenerated. That
fails loudly and safely rather than silently accepting an unsolvable room.

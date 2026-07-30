"""Stage 6: install mechanisms onto the region graph.

This is where a floor plan becomes a puzzle. The method is a dependency
ordering rather than a script:

1. Root the region graph at the spawn region and walk its edges outward.
2. Process candidate gates in order of increasing depth. When gate `e` is
   processed, compute which regions are still reachable with `e` **and every
   not-yet-processed gate** treated as closed. That set is the gate's
   *prerequisite zone*.
3. Place whatever `e` needs -- a key, a lever, a pair of levers -- somewhere
   inside that prerequisite zone, and only there.

Because a gate's trigger always lives in territory that is open before the gate
is, the room is solvable by construction. That is a structural argument, not a
solution: nothing here records an order of operations, and the validator
re-derives solvability from scratch rather than trusting this stage.

Cooperative pressure comes from the *kind* of gate chosen -- a pair of levers
that must be held at the same instant, or a single hold-lever that keeps a door
open only while it is weighed down. The layout makes two agents useful; it never says how
they should coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from ..config import GenerationConfig
from ..entities import (
    AgentSpawn,
    Checkpoint,
    Entity,
    ExitDoor,
    Key,
    LockedDoor,
    PushableBlock,
    ResetZone,
    Switch,
    SwitchMode,
    TemporaryBridge,
    WipeoutBall,
    WipeoutBallSize,
)
from ..requirements import (
    AlwaysOpen,
    CheckpointRequirement,
    KeyRequirement,
    Requirement,
    SwitchRequirement,
    TriggerMode,
    combine,
)
from ..rng import SeededRandom
from ..room import PortalKey
from ..tiles import HAZARD_TILES, Tile, is_hazard
from ..utils.geometry import DIRECTIONS4, Rect, Vec2
from ..utils.grid import Grid, distance_field
from .layout import Layout
from .terrain import Decoration
from .topology import Topology

KEY_COLORS = ("gold", "azure", "crimson", "emerald", "violet", "amber")


class GateKind(str, Enum):
    """The mechanism styles a locked edge can use."""

    KEY = "key"
    SHARED_KEY = "shared_key"          # reuses a key an earlier door already needs
    SWITCH = "switch"
    TIMED_SWITCH = "timed_switch"
    HOLD_SWITCH = "hold_switch"        # cooperative: someone must stay behind
    PAIRED_LEVERS = "paired_levers"    # cooperative: two hold-levers, same instant


COOPERATIVE_GATES = frozenset({GateKind.HOLD_SWITCH, GateKind.PAIRED_LEVERS})


@dataclass(slots=True)
class GateRecord:
    """Bookkeeping for one locked edge -- reporting and rendering only."""

    edge: PortalKey
    kind: GateKind
    door_ids: tuple[str, ...]
    trigger_ids: tuple[str, ...]
    prerequisite_regions: tuple[int, ...]
    depth: int


@dataclass(slots=True)
class MechanismResult:
    entities: list[Entity] = field(default_factory=list)
    spawn_regions: tuple[int, ...] = ()
    exit_region: int = 0
    depths: dict[int, int] = field(default_factory=dict)
    gates: list[GateRecord] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class _Placer:
    """Hands out floor tiles, never the same one twice."""

    def __init__(self, terrain: Grid, topology: Topology, rng: SeededRandom) -> None:
        self.terrain = terrain
        self.topology = topology
        self.rng = rng
        self.reserved: set[Vec2] = set()
        for tiles in topology.portals.values():
            self.reserved.update(tiles)
            for tile in tiles:
                self.reserved.update(tile.neighbors4())

    def free_tiles(self, region_ids: Iterable[int]) -> list[Vec2]:
        out: list[Vec2] = []
        for rid in sorted(set(region_ids)):
            region = self.topology.regions.get(rid)
            if region is None:
                continue
            for tile in region.sorted_tiles():
                if tile in self.reserved:
                    continue
                if self.terrain[tile] != Tile.FLOOR:
                    continue
                out.append(tile)
        return out

    def take(self, region_ids: Iterable[int]) -> Vec2 | None:
        options = self.free_tiles(region_ids)
        if not options:
            return None
        pick = self.rng.choice(options)
        self.reserved.add(pick)
        return pick

    def take_spread(self, region_ids: Sequence[int], count: int) -> list[Vec2]:
        """Take `count` tiles, preferring one per region so triggers sit apart."""
        picks: list[Vec2] = []
        regions = [r for r in region_ids if self.free_tiles([r])]
        regions = self.rng.shuffled(regions)
        for rid in regions:
            if len(picks) >= count:
                break
            tile = self.take([rid])
            if tile is not None:
                picks.append(tile)
        while len(picks) < count:
            tile = self.take(region_ids)
            if tile is None:
                break
            picks.append(tile)
        return picks

    def take_far_from(self, region_ids: Iterable[int], anchors: Sequence[Vec2]) -> Vec2 | None:
        """Take the free tile furthest (by step count) from `anchors`.

        This is a static spread measurement over the finished map, used to keep
        the exit away from the spawn. It moves nothing.
        """
        options = self.free_tiles(region_ids)
        if not options:
            return None
        if not anchors:
            return self.take(region_ids)
        walkable = lambda p: self.terrain.in_bounds(p) and self.terrain[p] in (
            Tile.FLOOR,
        )
        field_map = distance_field(anchors, walkable, self.terrain.bounds)
        scored = sorted(
            options, key=lambda p: (-field_map.get(p, -1), p[1], p[0])
        )
        top = scored[: max(1, len(scored) // 6)]
        pick = self.rng.choice(top)
        self.reserved.add(pick)
        return pick


def populate_mechanisms(
    layout: Layout,
    topology: Topology,
    decoration: Decoration,
    config: GenerationConfig,
    rng: SeededRandom,
) -> MechanismResult:
    """Place spawns, gates, triggers, the exit, and the optional extras."""
    result = MechanismResult()
    if not topology.regions:
        return result

    placer = _Placer(layout.terrain, topology, rng.derive("placement"))
    counter = _IdCounter()

    spawn_region = _choose_spawn_region(topology, rng.derive("spawn"))
    depths = topology.graph.depths(spawn_region)
    result.depths = depths

    gate_edges = _choose_gate_edges(topology, spawn_region, depths, config, rng.derive("gates"))

    budgets = _Budgets(
        keys=rng.derive("budget").in_range(config.num_keys),
        switches=rng.derive("budget").in_range(config.num_switches),
        cooperative=config.required_cooperative_actions,
        blocks=rng.derive("block_budget").in_range(config.num_pushable_blocks),
    )
    result.stats["requested_pushable_blocks"] = budgets.blocks
    if config.require_key_for_each_agent:
        budgets.keys = max(2, budgets.keys)
    if budgets.cooperative > 0 and config.exit_requires_both_agents:
        # Single hold-levers are unavailable in this mode, so paired levers are
        # the only cooperative gate left. Guarantee enough levers to build one
        # rather than silently under-delivering on required_cooperative_actions.
        budgets.switches = max(budgets.switches, 2)

    entities: list[Entity] = []
    resolved: set[PortalKey] = set()
    unresolved = set(gate_edges)
    hold_gated_regions: set[int] = set()

    for edge in gate_edges:
        available = topology.graph.reachable_from(spawn_region, blocked_edges=unresolved)
        if not available:
            continue
        # A key already sitting in unlocked territory can gate a second door --
        # the "shared resource" case, where one token matters in two places.
        reusable_keys = [
            entity
            for entity in entities
            if isinstance(entity, Key)
            and topology.tile_region.get(entity.pos) in available
        ] if config.allow_shared_keys else []
        owned = {
            entity.agent_index
            for entity in entities
            if isinstance(entity, Key) and entity.agent_index is not None
        }
        if config.require_key_for_each_agent and len(owned) < 2 and budgets.keys > 0:
            kind = GateKind.KEY
        else:
            kind = _pick_gate_kind(
                budgets,
                config,
                edge,
                depths,
                hold_gated_regions,
                bool(reusable_keys),
                rng.derive("gate_kind"),
            )
        if kind is None:
            continue
        record = _install_gate(
            kind=kind,
            edge=edge,
            available=sorted(available),
            reusable_keys=reusable_keys,
            topology=topology,
            layout=layout,
            placer=placer,
            counter=counter,
            budgets=budgets,
            config=config,
            rng=rng.derive(f"gate_{edge[0]}_{edge[1]}"),
            entities=entities,
            depth=max(depths.get(edge[0], 0), depths.get(edge[1], 0)),
        )
        if record is None:
            continue
        result.gates.append(record)
        resolved.add(edge)
        unresolved.discard(edge)
        if kind is GateKind.HOLD_SWITCH:
            # everything past a hold-gate is one-way for a single agent
            beyond = set(topology.graph.nodes) - set(available)
            hold_gated_regions.update(beyond)

    # Regions we could not gate stay open; drop them from the blocked set.
    unresolved.clear()

    exit_region = _choose_exit_region(topology, spawn_region, depths, resolved, placer)
    result.exit_region = exit_region

    spawn_zone = topology.graph.reachable_from(spawn_region, blocked_edges=resolved)
    result.spawn_regions = _place_spawns(
        spawn_zone, spawn_region, placer, config, rng.derive("spawn_tiles"), entities
    )

    _place_exit(
        exit_region=exit_region,
        all_regions=sorted(topology.regions),
        spawn_tiles=[e.pos for e in entities if isinstance(e, AgentSpawn)],
        placer=placer,
        counter=counter,
        budgets=budgets,
        config=config,
        rng=rng.derive("exit"),
        entities=entities,
        result=result,
    )

    _place_extras(
        layout,
        topology,
        decoration,
        config,
        budgets,
        placer,
        counter,
        rng.derive("extras"),
        entities,
        result,
    )

    result.entities = entities
    result.stats.update(
        {
            "spawn_region": spawn_region,
            "gate_count": len(result.gates),
            "cooperative_gates": sum(
                1 for g in result.gates if g.kind in COOPERATIVE_GATES
            ),
            "chain_depth": max((g.depth for g in result.gates), default=0),
        }
    )
    return result


# ---------------------------------------------------------------------------
# region selection
# ---------------------------------------------------------------------------


def _choose_spawn_region(topology: Topology, rng: SeededRandom) -> int:
    """Prefer a dead-end region so the room unfolds away from the start."""
    regions = sorted(topology.regions)
    if not regions:
        return 0
    leaves = [r for r in regions if topology.graph.degree(r) == 1 and len(topology.regions[r].tiles) >= 4]
    pool = leaves or [r for r in regions if len(topology.regions[r].tiles) >= 4] or regions
    return rng.choice(pool)


def _choose_exit_region(
    topology: Topology,
    spawn_region: int,
    depths: dict[int, int],
    gated: set[PortalKey],
    placer: _Placer,
) -> int:
    """Deepest region that still has somewhere to put the exit door."""
    candidates = [
        rid
        for rid in sorted(topology.regions)
        if placer.free_tiles([rid])
    ]
    if not candidates:
        return spawn_region
    behind_gate = {rid for edge in gated for rid in edge}
    candidates.sort(
        key=lambda rid: (
            -depths.get(rid, 0),
            -(1 if rid in behind_gate else 0),
            -len(topology.regions[rid].tiles),
            rid,
        )
    )
    return candidates[0]


def _choose_gate_edges(
    topology: Topology,
    spawn_region: int,
    depths: dict[int, int],
    config: GenerationConfig,
    rng: SeededRandom,
) -> list[PortalKey]:
    """Pick which region links become locked, shallowest first.

    Edges on the spawn-to-deepest-region route are favoured, because gating
    them in sequence is what produces a multi-step chain rather than several
    independent one-step locks.
    """
    tree = topology.graph.bfs_tree(spawn_region)
    tree_edges: list[PortalKey] = []
    for child, parent in sorted(tree.items()):
        if parent is None:
            continue
        key = (child, parent) if child <= parent else (parent, child)
        if key in topology.portals:
            tree_edges.append(key)

    if not tree_edges:
        return []

    deepest = max(depths, key=lambda r: (depths[r], r))
    critical = set()
    for node in topology.graph.path_to_root(deepest, tree):
        parent = tree.get(node)
        if parent is not None:
            key = (node, parent) if node <= parent else (parent, node)
            critical.add(key)

    def sort_key(edge: PortalKey) -> tuple[int, int, int, int]:
        depth = max(depths.get(edge[0], 0), depths.get(edge[1], 0))
        return (0 if edge in critical else 1, depth, edge[0], edge[1])

    ordered = sorted(set(tree_edges), key=sort_key)
    wanted = rng.in_range(config.num_locked_doors)
    wanted = min(wanted, len(ordered))
    if config.puzzle_chain_length > 0:
        wanted = max(wanted, min(config.puzzle_chain_length, len(ordered)))
        wanted = min(wanted, config.num_locked_doors[1], len(ordered))
    chosen = ordered[:wanted]
    return sorted(chosen, key=lambda e: (max(depths.get(e[0], 0), depths.get(e[1], 0)), e))


# ---------------------------------------------------------------------------
# gate installation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Budgets:
    keys: int
    switches: int
    cooperative: int
    blocks: int


class _IdCounter:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def next(self, prefix: str) -> str:
        index = self._counts.get(prefix, 0)
        self._counts[prefix] = index + 1
        return f"{prefix}_{index}"


def _pick_gate_kind(
    budgets: _Budgets,
    config: GenerationConfig,
    edge: PortalKey,
    depths: dict[int, int],
    hold_gated_regions: set[int],
    reusable_keys: bool,
    rng: SeededRandom,
) -> GateKind | None:
    """Choose a mechanism style this gate can afford.

    Cooperative styles are spent first while the budget lasts. A hold-lever is
    refused past another hold-lever: with only two agents, a second one would
    leave nobody free to move on.
    """
    coop_possible: list[GateKind] = []
    if budgets.cooperative > 0:
        if budgets.switches >= 2:
            coop_possible.append(GateKind.PAIRED_LEVERS)
        already_split = any(r in hold_gated_regions for r in edge)
        # A hold-lever always strands one agent behind it, so it is off the
        # table when the room is meant to end with both of them at the exit.
        if (
            budgets.switches >= 1
            and not already_split
            and not config.exit_requires_both_agents
        ):
            coop_possible.append(GateKind.HOLD_SWITCH)
    if coop_possible:
        return rng.choice(coop_possible)

    plain: list[GateKind] = []
    if budgets.keys >= 1:
        plain.append(GateKind.KEY)
    if budgets.switches >= 1:
        plain.append(
            GateKind.TIMED_SWITCH
            if rng.chance(config.timed_door_probability)
            else GateKind.SWITCH
        )
    if reusable_keys and config.allow_shared_keys:
        # Costs no budget, which is also what makes it the fallback when the
        # mechanism budget is spent but there are still links worth locking.
        plain.append(GateKind.SHARED_KEY)
    if not plain:
        return None
    return rng.choice(plain)


def _crate_switch_lines(
    placer: _Placer,
    region_ids: Iterable[int],
) -> list[tuple[Vec2, Vec2, Vec2]]:
    free = set(placer.free_tiles(region_ids))
    lines: list[tuple[Vec2, Vec2, Vec2]] = []
    for switch_tile in sorted(free, key=lambda tile: (tile.y, tile.x)):
        for direction in DIRECTIONS4:
            block_tile = switch_tile - direction
            standing_tile = block_tile - direction
            if block_tile in free and standing_tile in free:
                lines.append((switch_tile, block_tile, standing_tile))
    return lines


def _take_crate_switch_line(
    placer: _Placer,
    region_ids: Iterable[int],
    rng: SeededRandom,
) -> tuple[Vec2, Vec2, Vec2] | None:
    lines = _crate_switch_lines(placer, region_ids)
    if not lines:
        return None
    line = rng.choice(lines)
    placer.reserved.update(line)
    return line


def _take_paired_switch_plan(
    placer: _Placer,
    region_ids: Sequence[int],
    topology: Topology,
    rng: SeededRandom,
) -> tuple[tuple[Vec2, Vec2], tuple[Vec2, Vec2, Vec2]] | None:
    free = set(placer.free_tiles(region_ids))
    for line in rng.shuffled(_crate_switch_lines(placer, region_ids)):
        remaining = free - set(line)
        if not remaining:
            continue
        first_region = topology.tile_region.get(line[0])
        spread = [
            tile
            for tile in remaining
            if topology.tile_region.get(tile) != first_region
        ]
        options = spread or list(remaining)
        second = rng.choice(sorted(options, key=lambda tile: (tile.y, tile.x)))
        placer.reserved.update((*line, second))
        return (line[0], second), line
    return None


def _install_gate(
    kind: GateKind,
    edge: PortalKey,
    available: list[int],
    reusable_keys: list[Key],
    topology: Topology,
    layout: Layout,
    placer: _Placer,
    counter: _IdCounter,
    budgets: _Budgets,
    config: GenerationConfig,
    rng: SeededRandom,
    entities: list[Entity],
    depth: int,
) -> GateRecord | None:
    """Create the door(s) for `edge` and the trigger(s) inside `available`."""
    portal_tiles = topology.portals.get(edge, ())
    if not portal_tiles:
        return None

    door_ids = [counter.next("door") for _ in portal_tiles]
    trigger_ids: list[str] = []
    requirement: Requirement
    latching = True
    timer: int | None = None

    if kind is GateKind.KEY:
        tile = placer.take(available)
        if tile is None:
            return None
        key_id = counter.next("key")
        owner = _next_key_owner(config, entities, rng)
        color = _key_color(owner, entities)
        entities.append(
            Key(
                id=key_id,
                pos=tile,
                color=color,
                opens=tuple(door_ids),
                agent_index=owner,
            )
        )
        trigger_ids.append(key_id)
        requirement = KeyRequirement((key_id,))
        budgets.keys -= 1

    elif kind is GateKind.SHARED_KEY:
        if not reusable_keys:
            return None
        existing = rng.choice(sorted(reusable_keys, key=lambda k: k.id))
        # record the extra doors on the key itself, so the blueprint stays honest
        entities[entities.index(existing)] = replace(
            existing, opens=existing.opens + tuple(door_ids)
        )
        trigger_ids.append(existing.id)
        requirement = KeyRequirement((existing.id,))

    elif kind in (GateKind.SWITCH, GateKind.TIMED_SWITCH):
        tile = placer.take(available)
        if tile is None:
            return None
        switch_id = counter.next("switch")
        entities.append(
            Switch(
                id=switch_id,
                pos=tile,
                mode=SwitchMode.TOGGLE,
                controls=tuple(door_ids),
            )
        )
        trigger_ids.append(switch_id)
        requirement = SwitchRequirement((switch_id,))
        budgets.switches -= 1
        if kind is GateKind.TIMED_SWITCH:
            timer = rng.randint(18, 44)

    elif kind is GateKind.HOLD_SWITCH:
        crate_line = None
        if budgets.blocks > 0:
            crate_line = _take_crate_switch_line(
                placer, available, rng.derive("crate_line")
            )
            if crate_line is None:
                return None
            tile = crate_line[0]
        else:
            tile = placer.take(available)
            if tile is None:
                return None
        switch_id = counter.next("switch")
        entities.append(
            Switch(
                id=switch_id,
                pos=tile,
                mode=SwitchMode.HOLD,
                group="hold",
                controls=tuple(door_ids),
            )
        )
        if crate_line is not None:
            entities.append(
                PushableBlock(
                    id=counter.next("block"),
                    pos=crate_line[1],
                    target_switch_id=switch_id,
                    push_from=crate_line[2],
                )
            )
            budgets.blocks -= 1
        trigger_ids.append(switch_id)
        requirement = SwitchRequirement((switch_id,))
        latching = False  # closes the moment the lever is released
        budgets.switches -= 1
        budgets.cooperative -= 1

    elif kind is GateKind.PAIRED_LEVERS:
        crate_line = None
        if budgets.blocks > 0:
            plan = _take_paired_switch_plan(
                placer,
                available,
                topology,
                rng.derive("crate_line"),
            )
            if plan is None:
                return None
            tiles, crate_line = plan
        else:
            tiles = tuple(placer.take_spread(available, 2))
            if len(tiles) < 2:
                return None
        lever_ids = [counter.next("switch") for _ in tiles]
        group = f"pair_{edge[0]}_{edge[1]}"
        for lever_id, tile in zip(lever_ids, tiles):
            entities.append(
                Switch(
                    id=lever_id,
                    pos=tile,
                    mode=SwitchMode.HOLD,
                    group=group,
                    controls=tuple(door_ids),
                )
            )
        if crate_line is not None:
            entities.append(
                PushableBlock(
                    id=counter.next("block"),
                    pos=crate_line[1],
                    target_switch_id=lever_ids[0],
                    push_from=crate_line[2],
                )
            )
            budgets.blocks -= 1
        trigger_ids.extend(lever_ids)
        requirement = SwitchRequirement(tuple(lever_ids), TriggerMode.SIMULTANEOUS)
        budgets.switches -= 2
        budgets.cooperative -= 1

    else:  # pragma: no cover - exhaustive above
        return None

    for door_id, tile in zip(door_ids, portal_tiles):
        candidate = layout.doorways.get(tile)
        entities.append(
            LockedDoor(
                id=door_id,
                pos=tile,
                requirement=requirement,
                latching=latching,
                timer=timer,
                horizontal=bool(candidate and not candidate.vertical),
                region_a=edge[0],
                region_b=edge[1],
            )
        )

    return GateRecord(
        edge=edge,
        kind=kind,
        door_ids=tuple(door_ids),
        trigger_ids=tuple(trigger_ids),
        prerequisite_regions=tuple(available),
        depth=depth,
    )


# ---------------------------------------------------------------------------
# spawns, exit, extras
# ---------------------------------------------------------------------------


def _place_spawns(
    spawn_zone: set[int],
    spawn_region: int,
    placer: _Placer,
    config: GenerationConfig,
    rng: SeededRandom,
    entities: list[Entity],
) -> tuple[int, ...]:
    """Reserve two start tiles inside the initially-open zone.

    Both spawns sit in territory that is open before any gate, so neither agent
    starts sealed away from the other. They may still be in different regions
    of that zone -- separate routes that reconnect later.
    """
    zone = sorted(spawn_zone) or [spawn_region]
    regions_with_space = [r for r in zone if placer.free_tiles([r])]
    if not regions_with_space:
        regions_with_space = zone

    chosen_regions: list[int] = []
    if len(regions_with_space) >= 2 and rng.chance(config.separate_spawns_probability):
        chosen_regions = rng.sample(regions_with_space, 2)
    else:
        home = spawn_region if spawn_region in regions_with_space else regions_with_space[0]
        chosen_regions = [home, home]

    placed: list[int] = []
    for index, rid in enumerate(chosen_regions):
        tile = placer.take([rid])
        if tile is None:
            tile = placer.take(regions_with_space)
        if tile is None:
            continue
        entities.append(AgentSpawn(id=f"spawn_{index}", pos=tile, index=index))
        placed.append(rid)
    return tuple(placed)


def _place_exit(
    exit_region: int,
    all_regions: list[int],
    spawn_tiles: list[Vec2],
    placer: _Placer,
    counter: _IdCounter,
    budgets: _Budgets,
    config: GenerationConfig,
    rng: SeededRandom,
    entities: list[Entity],
    result: MechanismResult,
) -> None:
    """Install the exit and the objectives that unlock it."""
    tile = placer.take_far_from([exit_region], spawn_tiles)
    if tile is None:
        tile = placer.take(all_regions)
    if tile is None:
        return

    parts: list[Requirement] = []
    objective_ids: list[str] = []
    wanted = max(0, config.exit_objective_count)

    for _ in range(wanted):
        style = _pick_objective_style(budgets, rng)
        target = placer.take(all_regions)
        if target is None:
            break
        if style == "key":
            key_id = counter.next("key")
            owner = _next_key_owner(config, entities, rng)
            color = _key_color(owner, entities)
            entities.append(
                Key(
                    id=key_id,
                    pos=target,
                    color=color,
                    opens=("exit",),
                    agent_index=owner,
                )
            )
            parts.append(KeyRequirement((key_id,)))
            objective_ids.append(key_id)
            budgets.keys -= 1
        elif style == "switch":
            switch_id = counter.next("switch")
            entities.append(
                Switch(id=switch_id, pos=target, mode=SwitchMode.ONESHOT, controls=("exit",))
            )
            parts.append(SwitchRequirement((switch_id,)))
            objective_ids.append(switch_id)
            budgets.switches -= 1
        else:
            checkpoint_id = counter.next("checkpoint")
            entities.append(
                Checkpoint(id=checkpoint_id, pos=target, order=len(objective_ids), group="exit")
            )
            parts.append(CheckpointRequirement((checkpoint_id,)))
            objective_ids.append(checkpoint_id)

    requirement = combine(parts, TriggerMode.ALL) if parts else AlwaysOpen()
    entities.append(ExitDoor(id="exit", pos=tile, requirement=requirement))
    result.stats["exit_objectives"] = tuple(objective_ids)


def _pick_objective_style(budgets: _Budgets, rng: SeededRandom) -> str:
    """Exit objectives may overdraw the scatter budget -- the exit comes first."""
    options: list[str] = []
    if budgets.keys > 0:
        options.append("key")
    if budgets.switches > 0:
        options.append("switch")
    options.append("checkpoint")
    if not options:
        return "key"
    return rng.choice(options)


def _place_extras(
    layout: Layout,
    topology: Topology,
    decoration: Decoration,
    config: GenerationConfig,
    budgets: _Budgets,
    placer: _Placer,
    counter: _IdCounter,
    rng: SeededRandom,
    entities: list[Entity],
    result: MechanismResult,
) -> None:
    """Place extra mechanics after the gate skeleton."""
    regions = sorted(topology.regions)

    _place_wipeout_balls(
        layout.terrain,
        config,
        placer,
        counter,
        rng.derive("wipeout"),
        entities,
        result,
    )
    if config.require_bridge_crossing:
        _place_bridges(
            layout.terrain,
            decoration,
            config,
            placer,
            counter,
            rng.derive("bridges"),
            entities,
            result,
        )
    if config.require_reset_detour:
        _place_required_reset_detour(
            layout.terrain,
            decoration,
            placer,
            counter,
            rng.derive("reset_detour"),
            entities,
            result,
        )

    for _ in range(budgets.blocks):
        tile = _take_open_tile(layout.terrain, placer, regions, rng)
        if tile is None:
            break
        entities.append(PushableBlock(id=counter.next("block"), pos=tile))
    budgets.blocks = 0

    blocks = [entity for entity in entities if isinstance(entity, PushableBlock)]
    result.stats["placed_pushable_blocks"] = len(blocks)
    result.stats["crate_switch_pairs"] = tuple(
        {
            "block": block.id,
            "switch": block.target_switch_id,
            "push_from": list(block.push_from) if block.push_from is not None else None,
        }
        for block in blocks
        if block.target_switch_id is not None
    )

    for index in range(rng.in_range(config.num_checkpoints)):
        tile = placer.take(regions)
        if tile is None:
            break
        entities.append(
            Checkpoint(id=counter.next("checkpoint"), pos=tile, order=index, group="route")
        )

    if not config.require_reset_detour:
        wanted_resets = rng.in_range(config.num_reset_zones)
        placed_resets = 0
        for _ in range(wanted_resets):
            zone = _find_reset_zone(layout.terrain, placer, rng)
            if zone is None:
                break
            entities.append(
                ResetZone(id=counter.next("reset"), pos=Vec2(zone.x, zone.y), rect=zone)
            )
            placed_resets += 1
        result.stats["requested_reset_zones"] = wanted_resets
        result.stats["placed_reset_zones"] = placed_resets
        result.stats["required_reset_zone_id"] = None

    if not config.require_bridge_crossing:
        wanted = rng.in_range(config.num_temporary_bridges)
        placed = 0
        used_bridge_tiles: set[Vec2] = set()
        for _ in range(wanted):
            run = _find_bridge_run(layout.terrain, used_bridge_tiles, rng)
            if not run:
                break
            used_bridge_tiles.update(run)
            period = rng.randint(10, 20)
            entities.append(
                TemporaryBridge(
                    id=counter.next("bridge"),
                    pos=run[0],
                    tiles=tuple(run),
                    period=period,
                    on_ticks=max(3, period // 2),
                    phase=rng.randint(0, period - 1),
                )
            )
            placed += 1
        result.stats["requested_temporary_bridges"] = wanted
        result.stats["placed_temporary_bridges"] = placed
        result.stats["required_bridge_id"] = None


def _place_bridges(
    terrain: Grid,
    decoration: Decoration,
    config: GenerationConfig,
    placer: _Placer,
    counter: _IdCounter,
    rng: SeededRandom,
    entities: list[Entity],
    result: MechanismResult,
) -> None:
    wanted = rng.derive("count").in_range(config.num_temporary_bridges)
    placed = 0
    used: set[Vec2] = set()
    result.stats["required_bridge_id"] = None

    if config.require_bridge_crossing:
        run = _find_required_bridge_run(
            terrain,
            placer,
            entities,
            rng.derive("required"),
        )
        if run:
            choices = {
                tile: weight
                for tile, weight in config.hazard_weights.items()
                if tile in HAZARD_TILES and weight > 0
            }
            hazard = rng.derive("required_hazard").weighted_choice(choices)
            for tile in run:
                terrain[tile] = hazard
            decoration.hazard_tiles.update(run)
            bridge = _add_bridge(
                run,
                placer,
                counter,
                rng.derive("required_timing"),
                entities,
            )
            result.stats["required_bridge_id"] = bridge.id
            used.update(run)
            placed += 1

    for index in range(placed, wanted):
        run = _find_bridge_run(terrain, used, rng.derive(f"extra_{index}"))
        if not run:
            break
        _add_bridge(
            run,
            placer,
            counter,
            rng.derive(f"timing_{index}"),
            entities,
        )
        used.update(run)
        placed += 1

    result.stats["requested_temporary_bridges"] = wanted
    result.stats["placed_temporary_bridges"] = placed


def _add_bridge(
    run: list[Vec2],
    placer: _Placer,
    counter: _IdCounter,
    rng: SeededRandom,
    entities: list[Entity],
) -> TemporaryBridge:
    period = rng.randint(10, 20)
    bridge = TemporaryBridge(
        id=counter.next("bridge"),
        pos=run[0],
        tiles=tuple(run),
        period=period,
        on_ticks=max(3, period // 2),
        phase=rng.randint(0, period - 1),
    )
    placer.reserved.update(run)
    entities.append(bridge)
    return bridge


def _find_required_bridge_run(
    terrain: Grid,
    placer: _Placer,
    entities: Sequence[Entity],
    rng: SeededRandom,
) -> list[Vec2]:
    separates = _required_crossing_predicate(terrain, entities)
    candidates: list[list[Vec2]] = []
    for run in _bridge_course_candidates(terrain):
        tiles = set(run)
        if tiles & placer.reserved:
            continue
        if separates(run, tiles):
            candidates.append(run)
    return rng.choice(candidates) if candidates else []


def _bridge_course_candidates(terrain: Grid) -> list[list[Vec2]]:
    runs: list[list[Vec2]] = []
    for y in range(terrain.height):
        row = [Vec2(x, y) for x in range(terrain.width)]
        runs.extend(_bounded_floor_runs(terrain, row, Vec2(0, 1)))
    for x in range(terrain.width):
        column = [Vec2(x, y) for y in range(terrain.height)]
        runs.extend(_bounded_floor_runs(terrain, column, Vec2(1, 0)))
    return runs


def _bounded_floor_runs(
    terrain: Grid,
    line: list[Vec2],
    side: Vec2,
) -> list[list[Vec2]]:
    found: list[list[Vec2]] = []
    current: list[Vec2] = []
    for tile in (*line, None):
        if tile is not None and terrain.get(tile, Tile.VOID) == Tile.FLOOR:
            current.append(tile)
            continue
        if 2 <= len(current) <= 7:
            found.append(current)
        elif len(current) > 7 and all(
            terrain.get(point + side, Tile.VOID) != Tile.FLOOR
            and terrain.get(point - side, Tile.VOID) != Tile.FLOOR
            for point in current
        ):
            found.extend(current[index : index + 2] for index in range(len(current) - 1))
        current = []
    return found


@dataclass(frozen=True, slots=True)
class _ResetDetourPlan:
    barrier: tuple[Vec2, ...]
    reset_tile: Vec2
    safe_tile: Vec2
    shortest_steps: tuple[int, ...]
    safe_steps: tuple[int, ...]


def _place_required_reset_detour(
    terrain: Grid,
    decoration: Decoration,
    placer: _Placer,
    counter: _IdCounter,
    rng: SeededRandom,
    entities: list[Entity],
    result: MechanismResult,
) -> None:
    result.stats["requested_reset_zones"] = 1
    result.stats["placed_reset_zones"] = 0
    result.stats["required_reset_zone_id"] = None
    plan = _find_reset_detour_plan(terrain, placer, entities, rng)
    if plan is None:
        return

    blocked = set(plan.barrier) - {plan.reset_tile, plan.safe_tile}
    for tile in blocked:
        terrain[tile] = Tile.OBSTACLE
    decoration.obstacle_tiles.update(blocked)
    placer.reserved.update((plan.reset_tile, plan.safe_tile))
    reset_id = counter.next("reset")
    entities.append(
        ResetZone(
            id=reset_id,
            pos=plan.reset_tile,
            rect=Rect(plan.reset_tile.x, plan.reset_tile.y, 1, 1),
        )
    )
    result.stats["placed_reset_zones"] = 1
    result.stats["required_reset_zone_id"] = reset_id
    result.stats["reset_shortest_steps"] = plan.shortest_steps
    result.stats["reset_safe_steps"] = plan.safe_steps


def _find_reset_detour_plan(
    terrain: Grid,
    placer: _Placer,
    entities: Sequence[Entity],
    rng: SeededRandom,
) -> _ResetDetourPlan | None:
    spawns = [entity.pos for entity in entities if isinstance(entity, AgentSpawn)]
    exits = [entity.pos for entity in entities if isinstance(entity, ExitDoor)]
    if len(spawns) != 2 or len(exits) != 1:
        return None
    exit_pos = exits[0]
    floor = {pos for pos in terrain.positions() if terrain[pos] == Tile.FLOOR}
    plans: list[_ResetDetourPlan] = []

    for barrier in _reset_barrier_candidates(terrain):
        if set(barrier) & placer.reserved:
            continue
        horizontal = len({tile.y for tile in barrier}) == 1
        boundary = barrier[0].y if horizontal else barrier[0].x
        spawn_sides = [
            (spawn.y - boundary) if horizontal else (spawn.x - boundary)
            for spawn in spawns
        ]
        exit_side = (
            exit_pos.y - boundary if horizontal else exit_pos.x - boundary
        )
        if exit_side == 0 or any(side == 0 or side * exit_side >= 0 for side in spawn_sides):
            continue

        shortcut_options = sorted(
            barrier,
            key=lambda tile: (
                sum(spawn.manhattan(tile) for spawn in spawns)
                + tile.manhattan(exit_pos),
                tile.y,
                tile.x,
            ),
        )[:4]
        safe_options = sorted(
            barrier,
            key=lambda tile: (
                -sum(spawn.manhattan(tile) for spawn in spawns)
                - tile.manhattan(exit_pos),
                tile.y,
                tile.x,
            ),
        )[:4]
        for reset_tile in shortcut_options:
            for safe_tile in safe_options:
                if reset_tile == safe_tile:
                    continue
                blocked = set(barrier) - {reset_tile, safe_tile}
                course = floor - blocked
                shortest: list[int] = []
                detours: list[int] = []
                valid = True
                for spawn in spawns:
                    direct = _distance_between(terrain, course, spawn, exit_pos)
                    to_reset = _distance_between(
                        terrain, course, spawn, reset_tile
                    )
                    from_reset = _distance_between(
                        terrain, course, reset_tile, exit_pos
                    )
                    safe = _distance_between(
                        terrain, course - {reset_tile}, spawn, exit_pos
                    )
                    if (
                        direct is None
                        or to_reset is None
                        or from_reset is None
                        or safe is None
                        or to_reset + from_reset != direct
                        or safe <= direct
                    ):
                        valid = False
                        break
                    shortest.append(direct)
                    detours.append(safe)
                if valid:
                    plans.append(
                        _ResetDetourPlan(
                            barrier=tuple(barrier),
                            reset_tile=reset_tile,
                            safe_tile=safe_tile,
                            shortest_steps=tuple(shortest),
                            safe_steps=tuple(detours),
                        )
                    )
    return rng.choice(plans) if plans else None


def _reset_barrier_candidates(terrain: Grid) -> list[list[Vec2]]:
    barriers: list[list[Vec2]] = []
    for y in range(terrain.height):
        barriers.extend(
            run
            for run in _maximal_floor_runs(
                terrain,
                [Vec2(x, y) for x in range(terrain.width)],
            )
            if len(run) >= 6
        )
    for x in range(terrain.width):
        barriers.extend(
            run
            for run in _maximal_floor_runs(
                terrain,
                [Vec2(x, y) for y in range(terrain.height)],
            )
            if len(run) >= 6
        )
    return barriers


def _maximal_floor_runs(terrain: Grid, line: list[Vec2]) -> list[list[Vec2]]:
    found: list[list[Vec2]] = []
    current: list[Vec2] = []
    for tile in (*line, None):
        if tile is not None and terrain.get(tile, Tile.VOID) == Tile.FLOOR:
            current.append(tile)
            continue
        if current:
            found.append(current)
        current = []
    return found


def _distance_between(
    terrain: Grid,
    allowed: set[Vec2],
    start: Vec2,
    goal: Vec2,
) -> int | None:
    if start not in allowed or goal not in allowed:
        return None
    return distance_field(
        (start,),
        lambda tile: tile in allowed,
        terrain.bounds,
    ).get(goal)


def _next_key_owner(
    config: GenerationConfig,
    entities: Sequence[Entity],
    rng: SeededRandom,
) -> int | None:
    if not config.agent_specific_keys:
        return None
    counts = [
        sum(
            isinstance(entity, Key) and entity.agent_index == index
            for entity in entities
        )
        for index in range(2)
    ]
    if counts[0] == counts[1]:
        return rng.choice([0, 1])
    return 0 if counts[0] < counts[1] else 1


def _key_color(owner: int | None, entities: Sequence[Entity]) -> str:
    if owner == 0:
        return "azure"
    if owner == 1:
        return "crimson"
    count = sum(isinstance(entity, Key) for entity in entities)
    return KEY_COLORS[count % len(KEY_COLORS)]


def _place_wipeout_balls(
    terrain: Grid,
    config: GenerationConfig,
    placer: _Placer,
    counter: _IdCounter,
    rng: SeededRandom,
    entities: list[Entity],
    result: MechanismResult,
) -> None:
    wanted = {
        WipeoutBallSize.NORMAL: rng.derive("normal_count").in_range(
            config.num_normal_wipeout_balls
        ),
        WipeoutBallSize.BIG: rng.derive("big_count").in_range(
            config.num_big_wipeout_balls
        ),
    }
    placed = {size: 0 for size in wanted}
    result.stats["required_wipeout_ball_id"] = None

    if config.require_wipeout_crossing:
        crossing_predicate = _required_crossing_predicate(terrain, entities)
        for size in (WipeoutBallSize.BIG, WipeoutBallSize.NORMAL):
            if wanted[size] < 1:
                continue
            track = _find_wipeout_track(
                terrain,
                placer,
                7 if size is WipeoutBallSize.NORMAL else 11,
                0 if size is WipeoutBallSize.NORMAL else 1,
                rng.derive(f"required_{size.value}"),
                predicate=crossing_predicate,
            )
            if not track:
                continue
            ball = _add_wipeout_ball(size, track, placer, counter, entities)
            placed[size] += 1
            result.stats["required_wipeout_ball_id"] = ball.id
            break

    for size in (WipeoutBallSize.BIG, WipeoutBallSize.NORMAL):
        for index in range(placed[size], wanted[size]):
            track = _find_wipeout_track(
                terrain,
                placer,
                7 if size is WipeoutBallSize.NORMAL else 11,
                0 if size is WipeoutBallSize.NORMAL else 1,
                rng.derive(f"{size.value}_{index}"),
            )
            if not track:
                break
            _add_wipeout_ball(size, track, placer, counter, entities)
            placed[size] += 1

    for size in (WipeoutBallSize.NORMAL, WipeoutBallSize.BIG):
        result.stats[f"requested_{size.value}_wipeout_balls"] = wanted[size]
        result.stats[f"placed_{size.value}_wipeout_balls"] = placed[size]


def _add_wipeout_ball(
    size: WipeoutBallSize,
    track: list[Vec2],
    placer: _Placer,
    counter: _IdCounter,
    entities: list[Entity],
) -> WipeoutBall:
    ball = WipeoutBall(
        id=counter.next(f"wipeout_{size.value}"),
        pos=track[0],
        track=tuple(track),
        size=size,
    )
    placer.reserved.update(ball.footprint())
    entities.append(ball)
    return ball


def _find_wipeout_track(
    terrain: Grid,
    placer: _Placer,
    length: int,
    radius: int,
    rng: SeededRandom,
    predicate: Callable[[list[Vec2], set[Vec2]], bool] | None = None,
) -> list[Vec2]:
    candidates: list[list[Vec2]] = []
    for y in range(terrain.height):
        for x in range(terrain.width - length + 1):
            track = [Vec2(x + offset, y) for offset in range(length)]
            swept = {
                Vec2(center.x + dx, center.y + dy)
                for center in track
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
            }
            if swept & placer.reserved:
                continue
            if not all(terrain.get(tile, Tile.VOID) == Tile.FLOOR for tile in swept):
                continue
            if predicate is not None and not predicate(track, swept):
                continue
            candidates.append(track)
    return rng.choice(candidates) if candidates else []


def _required_crossing_predicate(
    terrain: Grid,
    entities: Sequence[Entity],
) -> Callable[[list[Vec2], set[Vec2]], bool]:
    spawns = [entity.pos for entity in entities if isinstance(entity, AgentSpawn)]
    exits = [entity.pos for entity in entities if isinstance(entity, ExitDoor)]
    floor = {pos for pos in terrain.positions() if terrain[pos] == Tile.FLOOR}
    if len(spawns) != 2 or len(exits) != 1:
        return lambda _track, _swept: False

    exit_pos = exits[0]
    if any(not _tiles_connect(spawn, exit_pos, floor) for spawn in spawns):
        return lambda _track, _swept: False

    def separates(_track: list[Vec2], swept: set[Vec2]) -> bool:
        return all(not _tiles_connect(spawn, exit_pos, floor, swept) for spawn in spawns)

    return separates


def _tiles_connect(
    start: Vec2,
    goal: Vec2,
    floor: set[Vec2],
    blocked: set[Vec2] | None = None,
) -> bool:
    blocked = blocked or set()
    if start not in floor or goal not in floor or start in blocked or goal in blocked:
        return False
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        if current == goal:
            return True
        for neighbour in current.neighbors4():
            if (
                neighbour in floor
                and neighbour not in blocked
                and neighbour not in seen
            ):
                seen.add(neighbour)
                stack.append(neighbour)
    return False


def _take_open_tile(
    terrain: Grid, placer: _Placer, regions: list[int], rng: SeededRandom
) -> Vec2 | None:
    """Claim a free tile that has open floor on at least three sides."""
    options = [
        tile
        for tile in placer.free_tiles(regions)
        if sum(1 for n in tile.neighbors4() if terrain.get(n, Tile.VOID) == Tile.FLOOR) >= 3
    ]
    if not options:
        return placer.take(regions)
    pick = rng.choice(options)
    placer.reserved.add(pick)
    return pick


def _find_reset_zone(terrain: Grid, placer: _Placer, rng: SeededRandom) -> Rect | None:
    """A small patch of floor next to a hazard, used as a 'sent back' area."""
    options = [
        p
        for p in terrain.positions()
        if terrain[p] == Tile.FLOOR
        and p not in placer.reserved
        and any(is_hazard(terrain.get(n, Tile.VOID)) for n in terrain.neighbors4(p))
    ]
    if not options:
        options = [
            p for p in terrain.positions() if terrain[p] == Tile.FLOOR and p not in placer.reserved
        ]
    if not options:
        return None
    origin = rng.choice(options)
    for size in ((2, 2), (2, 1), (1, 2), (1, 1)):
        rect = Rect(origin[0], origin[1], size[0], size[1])
        tiles = list(rect.positions())
        if all(
            terrain.get(t, Tile.VOID) == Tile.FLOOR and t not in placer.reserved
            for t in tiles
        ):
            placer.reserved.update(tiles)
            return rect
    return None


def _find_bridge_run(
    terrain: Grid, used: set[Vec2], rng: SeededRandom
) -> list[Vec2]:
    """A straight hazard run that a temporary bridge can phase across."""
    hazards = [
        p
        for p in terrain.positions()
        if is_hazard(terrain[p]) and p not in used
    ]
    if not hazards:
        return []
    for seed in rng.shuffled(hazards)[:24]:
        for direction in (Vec2(1, 0), Vec2(0, 1)):
            run = [seed]
            cursor = seed + direction
            while (
                terrain.in_bounds(cursor)
                and is_hazard(terrain[cursor])
                and cursor not in used
                and len(run) < 5
            ):
                run.append(cursor)
                cursor = cursor + direction
            if len(run) >= 2:
                return run
    return []

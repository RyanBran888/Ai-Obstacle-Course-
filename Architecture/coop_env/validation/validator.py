"""The acceptance gate a room must pass before the generator returns it.

Two layers of checking:

* **Structural rules** -- cheap sanity checks on placement and references.
  Anything with severity `ERROR` rejects the room.
* **Solvability analysis** -- the fixpoint in `solvability.py`.

Rules live in the `STRUCTURAL_RULES` list and take `(room, model)`, so adding a
new invariant is a matter of writing one function and appending it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..entities import (
    AgentSpawn,
    Checkpoint,
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
from ..room import Room
from ..requirements import (
    CheckpointRequirement,
    CompositeRequirement,
    KeyRequirement,
    SwitchRequirement,
)
from ..tiles import Tile, is_hazard, is_walkable
from ..utils.geometry import Vec2
from .bridge import BridgeReport, analyse_bridge
from .combined import CombinedCourseReport, analyse_combined_course
from .connectivity import ConnectivityModel, build_connectivity
from .reset_detour import ResetDetourReport, analyse_reset_detour
from .solvability import SolvabilityReport, analyse
from .wipeout import WipeoutReport, analyse_wipeout


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Issue:
    severity: Severity
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code}: {self.message}"


@dataclass(slots=True)
class ValidationReport:
    ok: bool = False
    issues: list[Issue] = field(default_factory=list)
    solvability: SolvabilityReport | None = None
    wipeout: WipeoutReport | None = None
    bridge: BridgeReport | None = None
    reset_detour: ResetDetourReport | None = None
    combined: CombinedCourseReport | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def summary(self) -> str:
        if self.ok:
            extra = f" ({len(self.warnings)} warnings)" if self.warnings else ""
            return f"valid{extra}"
        first = self.errors[0].message if self.errors else "unknown failure"
        return f"invalid: {first}"

    def report_lines(self) -> list[str]:
        return [str(issue) for issue in self.issues]


Rule = Callable[[Room, ConnectivityModel], list[Issue]]


def _error(code: str, message: str) -> Issue:
    return Issue(Severity.ERROR, code, message)


def _warning(code: str, message: str) -> Issue:
    return Issue(Severity.WARNING, code, message)


# ---------------------------------------------------------------------------
# structural rules
# ---------------------------------------------------------------------------


def rule_spawns(room: Room, model: ConnectivityModel) -> list[Issue]:
    """Two distinct, standable start tiles."""
    issues: list[Issue] = []
    spawns = room.spawns
    if len(spawns) != 2:
        issues.append(
            _error("spawn_count", f"expected exactly 2 agent spawns, found {len(spawns)}")
        )
    positions = [s.pos for s in spawns]
    if len(set(positions)) != len(positions):
        issues.append(_error("spawn_overlap", "the two spawn points share a tile"))
    for spawn in spawns:
        tile = room.terrain_at(spawn.pos)
        if not is_walkable(tile):
            issues.append(
                _error("spawn_terrain", f"spawn {spawn.id} sits on non-walkable {tile.name}")
            )
        if is_hazard(tile):
            issues.append(_error("spawn_hazard", f"spawn {spawn.id} sits on a hazard"))
        if model.region_of_tile(spawn.pos) is None:
            issues.append(_error("spawn_region", f"spawn {spawn.id} is not inside any region"))
    return issues


def rule_exit(room: Room, model: ConnectivityModel) -> list[Issue]:
    """Exactly one exit, on floor."""
    issues: list[Issue] = []
    exits = room.of_type(ExitDoor)
    if len(exits) != 1:
        issues.append(_error("exit_count", f"expected exactly 1 exit, found {len(exits)}"))
        return issues
    exit_door = exits[0]
    if not is_walkable(room.terrain_at(exit_door.pos)):
        issues.append(_error("exit_terrain", "exit door is not on walkable terrain"))
    if any(s.pos == exit_door.pos for s in room.spawns):
        issues.append(_error("exit_on_spawn", "exit door shares a tile with a spawn"))
    return issues


def rule_entity_placement(room: Room, model: ConnectivityModel) -> list[Issue]:
    """Every object sits somewhere it makes sense."""
    issues: list[Issue] = []
    floor_bound = (Key, Switch, Checkpoint, PushableBlock, AgentSpawn, ExitDoor)
    for entity in room.entities:
        for tile in entity.footprint():
            if not room.terrain.in_bounds(tile):
                issues.append(
                    _error("out_of_bounds", f"{entity.id} occupies {tuple(tile)} outside the grid")
                )
        if isinstance(entity, floor_bound):
            tile = room.terrain_at(entity.pos)
            if tile != Tile.FLOOR:
                issues.append(
                    _error(
                        "bad_terrain",
                        f"{entity.id} ({entity.kind.name.lower()}) sits on {tile.name}",
                    )
                )
        if isinstance(entity, LockedDoor):
            if not is_walkable(room.terrain_at(entity.pos)):
                issues.append(_error("door_terrain", f"door {entity.id} is embedded in solid terrain"))
        if isinstance(entity, TemporaryBridge):
            if entity.on_ticks <= 0 or entity.on_ticks > entity.period:
                issues.append(
                    _error("bridge_timing", f"bridge {entity.id} has an impossible duty cycle")
                )
        if isinstance(entity, ResetZone):
            if entity.rect.area <= 0:
                issues.append(_error("reset_zone", f"reset zone {entity.id} has no area"))
    return issues


def rule_no_stacking(room: Room, model: ConnectivityModel) -> list[Issue]:
    """Interactive objects should not be piled onto the same tile."""
    issues: list[Issue] = []
    exclusive = (Key, Switch, LockedDoor, ExitDoor, AgentSpawn, PushableBlock)
    seen: dict[Any, str] = {}
    for entity in room.entities:
        if not isinstance(entity, exclusive):
            continue
        previous = seen.get(entity.pos)
        if previous is not None:
            issues.append(
                _error("stacked", f"{entity.id} and {previous} both occupy {tuple(entity.pos)}")
            )
        else:
            seen[entity.pos] = entity.id
    return issues


def rule_requirement_references(room: Room, model: ConnectivityModel) -> list[Issue]:
    """Every requirement points at an object that actually exists."""
    issues: list[Issue] = []
    holders: list[tuple[str, Any]] = [(d.id, d.requirement) for d in room.doors]
    for exit_door in room.of_type(ExitDoor):
        holders.append((exit_door.id, exit_door.requirement))
    for holder_id, requirement in holders:
        for entity_id, expected, label in _typed_references(requirement):
            entity = room.find(entity_id)
            if entity is None:
                issues.append(
                    _error(
                        "dangling_requirement",
                        f"{holder_id} requires {entity_id!r}, which is not in the room",
                    )
                )
            elif expected is not None and not isinstance(entity, expected):
                issues.append(
                    _error(
                        "wrong_requirement_type",
                        f"{holder_id} expects {entity_id!r} to be a {label}",
                    )
                )
    return issues


def _typed_references(requirement):
    if isinstance(requirement, CompositeRequirement):
        for part in requirement.parts:
            yield from _typed_references(part)
        return
    if isinstance(requirement, KeyRequirement):
        for entity_id in sorted(requirement.key_ids):
            yield entity_id, Key, "key"
        return
    if isinstance(requirement, SwitchRequirement):
        for entity_id in sorted(requirement.switch_ids):
            yield entity_id, Switch, "switch"
        return
    if isinstance(requirement, CheckpointRequirement):
        for entity_id in sorted(requirement.checkpoint_ids):
            yield entity_id, Checkpoint, "checkpoint"
        return
    for entity_id in sorted(requirement.referenced_ids()):
        yield entity_id, None, "entity"


def rule_agent_keys(room: Room, model: ConnectivityModel) -> list[Issue]:
    issues: list[Issue] = []
    required_by: dict[str, set[str]] = {key.id: set() for key in room.keys}
    holders = [*room.doors, room.exit]
    for holder in holders:
        for key_id in holder.requirement.referenced_ids():
            if key_id in required_by:
                required_by[key_id].add(holder.id)

    owners: set[int] = set()
    for key in room.keys:
        if key.agent_index not in (None, 0, 1):
            issues.append(
                _error(
                    "key_owner",
                    f"key {key.id} has invalid agent index {key.agent_index}",
                )
            )
        if key.agent_index is not None:
            owners.add(key.agent_index)
        if room.config.agent_specific_keys and key.agent_index is None:
            issues.append(
                _error("key_owner", f"key {key.id} is not assigned to an agent")
            )
        strict_links = room.config.agent_specific_keys or not room.config.allow_shared_keys
        if strict_links and set(key.opens) != required_by[key.id]:
            issues.append(
                _error(
                    "key_door_mismatch",
                    f"key {key.id} opens {sorted(key.opens)}, but requirements use "
                    f"{sorted(required_by[key.id])}",
                )
            )
        if not room.config.allow_shared_keys:
            logical_targets = set()
            for target_id in key.opens:
                target = room.find(target_id)
                if isinstance(target, LockedDoor):
                    logical_targets.add(
                        ("door", min(target.region_a, target.region_b), max(target.region_a, target.region_b))
                    )
                else:
                    logical_targets.add(("exit", target_id))
            if len(logical_targets) > 1:
                issues.append(
                    _error(
                        "shared_key",
                        f"key {key.id} is connected to multiple logical doors",
                    )
                )

    if room.config.require_key_for_each_agent and owners != {0, 1}:
        issues.append(
            _error("missing_agent_key", "the room needs at least one key for each agent")
        )
    return issues


def rule_crate_switch_pairs(room: Room, model: ConnectivityModel) -> list[Issue]:
    issues: list[Issue] = []
    paired: list[PushableBlock] = []

    for block in room.blocks:
        if block.target_switch_id is None and block.push_from is None:
            continue
        if block.target_switch_id is None or block.push_from is None:
            issues.append(
                _error(
                    "crate_switch_pair",
                    f"crate {block.id} has an incomplete switch plan",
                )
            )
            continue

        target = room.find(block.target_switch_id)
        if not isinstance(target, Switch) or target.mode is not SwitchMode.HOLD:
            issues.append(
                _error(
                    "crate_switch_target",
                    f"crate {block.id} targets a missing or non-HOLD switch",
                )
            )
            continue
        paired.append(block)

        if block.pos.manhattan(target.pos) != 1:
            issues.append(
                _error(
                    "crate_switch_distance",
                    f"crate {block.id} is not one push from {target.id}",
                )
            )
            continue

        direction = target.pos - block.pos
        expected_standing = block.pos - direction
        if block.push_from != expected_standing:
            issues.append(
                _error(
                    "crate_switch_direction",
                    f"crate {block.id} needs standing tile {tuple(expected_standing)}",
                )
            )
            continue

        if room.terrain_at(block.push_from) != Tile.FLOOR:
            issues.append(
                _error(
                    "crate_switch_standing",
                    f"crate {block.id} standing tile is not clear floor",
                )
            )
        elif room.entities_at(block.push_from):
            issues.append(
                _error(
                    "crate_switch_standing",
                    f"crate {block.id} standing tile is occupied",
                )
            )
        elif model.region_of_tile(block.push_from) is None:
            issues.append(
                _error(
                    "crate_switch_standing",
                    f"crate {block.id} standing tile is unreachable",
                )
            )

        target_occupants = [
            entity.id
            for entity in room.entities_at(target.pos)
            if entity.id != target.id
        ]
        if room.terrain_at(target.pos) != Tile.FLOOR or target_occupants:
            issues.append(
                _error(
                    "crate_switch_destination",
                    f"crate {block.id} cannot be pushed onto {target.id}",
                )
            )

    requested = room.metadata.get("requested_pushable_blocks")
    if requested is not None and len(room.blocks) != requested:
        issues.append(
            _error(
                "crate_count",
                f"expected {requested} pushable blocks, found {len(room.blocks)}",
            )
        )

    hold_switches = [
        switch for switch in room.switches if switch.mode is SwitchMode.HOLD
    ]
    if requested and hold_switches and not paired:
        issues.append(
            _error(
                "crate_switch_missing",
                "a requested crate was not paired with a HOLD switch",
            )
        )

    saved_pairs = room.metadata.get("crate_switch_pairs")
    if saved_pairs is not None:
        actual_pairs = tuple(
            {
                "block": block.id,
                "switch": block.target_switch_id,
                "push_from": (
                    list(block.push_from) if block.push_from is not None else None
                ),
            }
            for block in room.blocks
            if block.target_switch_id is not None
        )
        if tuple(saved_pairs) != actual_pairs:
            issues.append(
                _error(
                    "crate_switch_metadata",
                    "crate-switch metadata does not match the room blueprint",
                )
            )
    return issues


def rule_wipeout_balls(room: Room, model: ConnectivityModel) -> list[Issue]:
    issues: list[Issue] = []
    swept: set[Any] = set()
    static_tiles = {
        tile
        for entity in room.entities
        if not isinstance(entity, WipeoutBall)
        for tile in entity.footprint()
    }

    for ball in room.wipeout_balls:
        expected = 7 if ball.size is WipeoutBallSize.NORMAL else 11
        if len(ball.track) != expected:
            issues.append(
                _error(
                    "wipeout_length",
                    f"{ball.id} needs a {expected}x1 track, found {len(ball.track)} tiles",
                )
            )
        if not ball.track or ball.pos != ball.track[0]:
            issues.append(
                _error("wipeout_origin", f"{ball.id} must start at the left track endpoint")
            )
        if len(set(ball.track)) != len(ball.track):
            issues.append(_error("wipeout_track", f"{ball.id} repeats a track tile"))
        for previous, current in zip(ball.track, ball.track[1:]):
            if current != Vec2(previous.x + 1, previous.y):
                issues.append(
                    _error(
                        "wipeout_track",
                        f"{ball.id} track must be a contiguous horizontal run",
                    )
                )
                break

        footprint = set(ball.footprint())
        for tile in footprint:
            if not room.terrain.in_bounds(tile):
                issues.append(
                    _error(
                        "wipeout_bounds",
                        f"{ball.id} collision area leaves the room at {tuple(tile)}",
                    )
                )
            elif room.terrain_at(tile) != Tile.FLOOR:
                issues.append(
                    _error(
                        "wipeout_terrain",
                        f"{ball.id} collision area crosses {room.terrain_at(tile).name}",
                    )
                )
        overlap = footprint & static_tiles
        if overlap:
            issues.append(
                _error(
                    "wipeout_overlap",
                    f"{ball.id} overlaps a static entity at {tuple(sorted(overlap)[0])}",
                )
            )
        ball_overlap = footprint & swept
        if ball_overlap:
            issues.append(
                _error(
                    "wipeout_overlap",
                    f"{ball.id} overlaps another wipeout track",
                )
            )
        swept.update(footprint)

    for size in ("normal", "big"):
        requested = room.metadata.get(f"requested_{size}_wipeout_balls")
        if requested is None:
            continue
        actual = sum(ball.size.value == size for ball in room.wipeout_balls)
        if actual != requested:
            issues.append(
                _error(
                    "wipeout_count",
                    f"expected {requested} {size} wipeout balls, found {actual}",
                )
            )
    return issues


def rule_temporary_bridges(room: Room, model: ConnectivityModel) -> list[Issue]:
    issues: list[Issue] = []
    occupied: set[Vec2] = set()
    for bridge in room.bridges:
        tiles = bridge.footprint()
        if not tiles or bridge.pos != tiles[0]:
            issues.append(
                _error(
                    "bridge_origin",
                    f"{bridge.id} must start at its first bridge tile",
                )
            )
            continue
        if len(tiles) < 2:
            issues.append(
                _error("bridge_length", f"{bridge.id} needs at least 2 tiles")
            )
        if len(set(tiles)) != len(tiles):
            issues.append(_error("bridge_path", f"{bridge.id} repeats a bridge tile"))
        xs = {tile.x for tile in tiles}
        ys = {tile.y for tile in tiles}
        if len(xs) > 1 and len(ys) > 1:
            issues.append(_error("bridge_path", f"{bridge.id} must be a straight run"))
        ordered = sorted(tiles, key=lambda tile: (tile.y, tile.x))
        if any(a.manhattan(b) != 1 for a, b in zip(ordered, ordered[1:])):
            issues.append(_error("bridge_path", f"{bridge.id} has a gap in its run"))
        for tile in tiles:
            if not room.terrain.in_bounds(tile):
                issues.append(
                    _error(
                        "bridge_bounds",
                        f"{bridge.id} leaves the room at {tuple(tile)}",
                    )
                )
            elif not is_hazard(room.terrain_at(tile)):
                issues.append(
                    _error(
                        "bridge_terrain",
                        f"{bridge.id} must cover hazard terrain at {tuple(tile)}",
                    )
                )
        if set(tiles) & occupied:
            issues.append(_error("bridge_overlap", f"{bridge.id} overlaps another bridge"))
        occupied.update(tiles)

    if room.config.require_bridge_crossing or room.config.require_combined_course:
        requested = room.metadata.get("requested_temporary_bridges")
        placed = room.metadata.get("placed_temporary_bridges")
        actual = len(room.bridges)
        if requested != 1 or placed != 1 or actual != 1:
            issues.append(
                _error(
                    "bridge_count",
                    f"required bridge course expected 1 bridge, found {actual}",
                )
            )
    return issues


def rule_reset_detours(room: Room, model: ConnectivityModel) -> list[Issue]:
    if not (
        room.config.require_reset_detour
        or room.config.require_combined_course
    ):
        return []
    requested = room.metadata.get("requested_reset_zones")
    placed = room.metadata.get("placed_reset_zones")
    if requested == 1 and placed == 1 and len(room.reset_zones) == 1:
        return []
    return [
        _error(
            "reset_count",
            f"required reset detour expected 1 reset zone, found {len(room.reset_zones)}",
        )
    ]


def rule_reachable_mechanisms(room: Room, model: ConnectivityModel) -> list[Issue]:
    """No object may be sealed inside solid terrain."""
    issues: list[Issue] = []
    for entity_id in model.unreachable_entities:
        issues.append(
            _error("orphan_entity", f"{entity_id} is not on any walkable region")
        )
    return issues


def rule_room_scale(room: Room, model: ConnectivityModel) -> list[Issue]:
    """A room needs enough open floor to be worth generating."""
    issues: list[Issue] = []
    walkable = sum(1 for p in room.terrain.positions() if is_walkable(room.terrain[p]))
    if walkable < 24:
        issues.append(_error("too_small", f"only {walkable} walkable tiles"))
    if not model.regions:
        issues.append(_error("no_regions", "no walkable regions were found"))
    return issues


def rule_hazard_sanity(room: Room, model: ConnectivityModel) -> list[Issue]:
    """Warn when hazards swallow most of the room."""
    issues: list[Issue] = []
    total = sum(1 for p in room.terrain.positions() if room.terrain[p] != Tile.VOID)
    hazards = sum(1 for p in room.terrain.positions() if is_hazard(room.terrain[p]))
    if total and hazards / total > 0.4:
        issues.append(
            _warning("hazard_heavy", f"{hazards}/{total} tiles are hazardous")
        )
    return issues


STRUCTURAL_RULES: list[Rule] = [
    rule_room_scale,
    rule_spawns,
    rule_exit,
    rule_entity_placement,
    rule_no_stacking,
    rule_requirement_references,
    rule_agent_keys,
    rule_crate_switch_pairs,
    rule_wipeout_balls,
    rule_temporary_bridges,
    rule_reset_detours,
    rule_reachable_mechanisms,
    rule_hazard_sanity,
]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def validate_room(room: Room, rules: list[Rule] | None = None) -> ValidationReport:
    """Check a room and report whether it may be handed out."""
    model = build_connectivity(room)
    report = ValidationReport()

    for rule in rules if rules is not None else STRUCTURAL_RULES:
        report.issues.extend(rule(room, model))

    if report.errors:
        report.ok = False
        report.stats["stage"] = "structural"
        return report

    if room.wipeout_balls or room.config.require_wipeout_crossing:
        wipeout = analyse_wipeout(room)
        report.wipeout = wipeout
        report.stats.update(
            {
                "wipeout_period": wipeout.period,
                "wipeout_reachable_by": wipeout.reachable_by,
                "wipeout_min_steps": wipeout.min_steps_by_agent,
                "required_wipeout_crossing": wipeout.structural_crossing,
            }
        )
        for reason in wipeout.reasons:
            report.issues.append(_error("wipeout_unsolvable", reason))
        if report.errors:
            report.ok = False
            report.stats["stage"] = "wipeout"
            return report

    if room.config.require_bridge_crossing or room.config.require_combined_course:
        bridge = analyse_bridge(room)
        report.bridge = bridge
        report.stats.update(
            {
                "bridge_period": bridge.period,
                "bridge_reachable_by": bridge.reachable_by,
                "bridge_min_steps": bridge.min_steps_by_agent,
                "required_bridge_crossing": bridge.structural_crossing,
            }
        )
        for reason in bridge.reasons:
            report.issues.append(_error("bridge_unsolvable", reason))
        if report.errors:
            report.ok = False
            report.stats["stage"] = "bridge"
            return report

    if room.config.require_reset_detour or room.config.require_combined_course:
        reset_detour = analyse_reset_detour(room)
        report.reset_detour = reset_detour
        report.stats.update(
            {
                "reset_shortest_steps": reset_detour.shortest_steps_by_agent,
                "reset_safe_steps": reset_detour.safe_steps_by_agent,
                "required_reset_shortcut": reset_detour.shortcut_valid,
                "required_reset_detour": reset_detour.safe_detour,
            }
        )
        for reason in reset_detour.reasons:
            report.issues.append(_error("reset_detour_invalid", reason))
        if report.errors:
            report.ok = False
            report.stats["stage"] = "reset_detour"
            return report

    if room.config.require_combined_course:
        combined = analyse_combined_course(
            room,
            report.wipeout,
            report.bridge,
            report.reset_detour,
        )
        report.combined = combined
        report.stats.update(
            {
                "combined_sections": combined.section_ids,
                "combined_gate_cuts": combined.gate_cut_ids,
                "combined_ball_cuts": combined.ball_cut_ids,
                "combined_timed_steps": combined.timed_max_steps,
                "combined_route_budget": combined.route_budget,
            }
        )
        for reason in combined.reasons:
            report.issues.append(_error("combined_course_invalid", reason))
        if report.errors:
            report.ok = False
            report.stats["stage"] = "combined_course"
            return report

    solvability = analyse(room, model)
    report.solvability = solvability
    for reason in solvability.reasons:
        report.issues.append(_error("unsolvable", reason))

    reachable = solvability.reachable_union
    report.stats.update(
        {
            "stage": "solvability",
            "regions": len(model.regions),
            "regions_reachable": len(reachable),
            "door_clusters": len(model.clusters),
            "doors_openable": len(solvability.opened_clusters),
            "doors_blocked": len(solvability.blocked_clusters),
            "cooperative_doors": len(solvability.cooperative_clusters),
            "chain_length": solvability.chain_length,
            "one_way_doors": len(solvability.one_way_clusters),
            "exit_reachable_by": solvability.exit_reachable_by,
            "exit_jointly_reachable": solvability.exit_jointly_reachable,
        }
    )

    if model.regions and len(reachable) < len(model.regions):
        stranded = len(model.regions) - len(reachable)
        report.issues.append(
            _warning("unreachable_regions", f"{stranded} region(s) can never be entered")
        )

    report.ok = not report.errors
    return report

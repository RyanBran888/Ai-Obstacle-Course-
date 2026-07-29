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
    MovingPlatform,
    PushableBlock,
    ResetZone,
    Switch,
    TemporaryBridge,
)
from ..room import Room
from ..tiles import Tile, is_hazard, is_walkable
from .connectivity import ConnectivityModel, build_connectivity
from .solvability import SolvabilityReport, analyse


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
        if isinstance(entity, MovingPlatform):
            if not entity.path:
                issues.append(_error("platform_path", f"platform {entity.id} has an empty track"))
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
        for entity_id in sorted(requirement.referenced_ids()):
            if room.find(entity_id) is None:
                issues.append(
                    _error(
                        "dangling_requirement",
                        f"{holder_id} requires {entity_id!r}, which is not in the room",
                    )
                )
    return issues


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

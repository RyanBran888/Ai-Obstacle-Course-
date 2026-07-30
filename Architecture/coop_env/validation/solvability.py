"""Prove that a generated room can, in principle, be completed.

The analysis is a monotone fixpoint over the region model:

* Each of the two agent slots starts with the set of regions its spawn can
  reach through open ground.
* A door opens when its requirement is satisfiable by the agents that can
  currently reach the relevant triggers.
* Opening a door grows the reachable sets, which may make further doors
  satisfiable. Repeat until nothing changes.

Two door flavours behave differently, and the distinction is what makes
cooperative layouts verifiable:

* **Latching** doors stay open once triggered, so satisfying them once opens
  the passage for both agent slots permanently.
* **Non-latching** doors are one-way unless an aligned crate can keep their
  HOLD switch pressed.

Deliberately conservative choices: hazards are never crossable without a
bridge, crates never move, and a trigger counts only if some slot can reach
it. A room this analysis accepts is completable; a room it rejects might still
be completable in some cleverer way, and is simply regenerated. Erring that
direction keeps the guarantee meaningful.

Nothing here is an agent. There is no state machine for a player, no ordering
of actions, and no output describing what anyone should do -- only a yes/no on
whether the structure permits completion at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

from ..entities import AgentSpawn
from ..requirements import (
    AlwaysOpen,
    CheckpointRequirement,
    CompositeRequirement,
    KeyRequirement,
    NeverOpen,
    Requirement,
    SwitchRequirement,
    TriggerMode,
)
from ..room import Room
from .connectivity import ConnectivityModel, DoorCluster

AGENT_SLOTS = 2


@dataclass(slots=True)
class SolvabilityReport:
    solvable: bool = False
    reasons: list[str] = field(default_factory=list)
    reachable: list[set[int]] = field(default_factory=list)
    joint_reachable: set[int] = field(default_factory=set)
    """Regions both slots can occupy at the same time.

    Reaching a region past a hold-lever costs the other slot its freedom, so
    those regions are reachable individually but never jointly. This is the
    distinction `exit_requires_both_agents` turns on.
    """
    opened_clusters: set[str] = field(default_factory=set)
    blocked_clusters: set[str] = field(default_factory=set)
    one_way_clusters: set[str] = field(default_factory=set)
    unlock_round: dict[str, int] = field(default_factory=dict)
    """Which pass of the fixpoint each door became passable on.

    Round 0 doors are openable from the starting area. A door that only opens
    on round 2 needed something found past a round-1 door, so the highest round
    is the length of the room's longest lock-and-key chain.
    """
    cooperative_clusters: set[str] = field(default_factory=set)
    """Doors that no single agent slot could have opened alone."""
    exit_reachable_by: tuple[int, ...] = ()
    exit_jointly_reachable: bool = False
    exit_requirement_met: bool = False
    unreachable_entities: tuple[str, ...] = ()

    @property
    def reachable_union(self) -> set[int]:
        out: set[int] = set()
        for group in self.reachable:
            out |= group
        return out

    @property
    def chain_length(self) -> int:
        """Longest dependency chain, in gates. 0 means nothing is locked."""
        if not self.unlock_round:
            return 0
        return max(self.unlock_round.values()) + 1


def analyse(room: Room, model: ConnectivityModel) -> SolvabilityReport:
    """Run the fixpoint and report whether the room is completable."""
    report = SolvabilityReport()
    spawns: tuple[AgentSpawn, ...] = room.spawns
    if len(spawns) < AGENT_SLOTS:
        report.reasons.append(f"expected {AGENT_SLOTS} spawn points, found {len(spawns)}")
        return report

    home: list[int | None] = [model.region_of_tile(s.pos) for s in spawns[:AGENT_SLOTS]]
    for index, region in enumerate(home):
        if region is None:
            report.reasons.append(f"spawn {index} is not on walkable floor")
    if any(r is None for r in home):
        return report

    reach: list[set[int]] = [set() for _ in range(AGENT_SLOTS)]
    open_clusters: set[str] = set()
    one_way: dict[str, set[int]] = {}

    unlock_round: dict[str, int] = {}
    for round_index in range(len(model.clusters) * AGENT_SLOTS + 4):
        for slot in range(AGENT_SLOTS):
            allowed = set(open_clusters) | {
                cid for cid, slots in one_way.items() if slot in slots
            }
            reach[slot] = _expand(model, home[slot], allowed)

        changed = False
        for cluster_id, cluster in sorted(model.clusters.items()):
            if cluster_id in open_clusters:
                continue
            if cluster.latching or _crate_sustains(
                cluster.requirement, model
            ):
                if _satisfiable(cluster.requirement, range(AGENT_SLOTS), reach, model):
                    open_clusters.add(cluster_id)
                    unlock_round[cluster_id] = round_index
                    changed = True
                continue
            # hold-lever: someone else keeps it open while one slot slips through
            for slot in range(AGENT_SLOTS):
                if slot in one_way.get(cluster_id, set()):
                    continue
                others = [s for s in range(AGENT_SLOTS) if s != slot]
                if not _satisfiable(cluster.requirement, others, reach, model):
                    continue
                if not any(r in reach[slot] for r in cluster.regions):
                    continue
                one_way.setdefault(cluster_id, set()).add(slot)
                unlock_round.setdefault(cluster_id, round_index)
                changed = True
        if not changed:
            break

    report.reachable = reach
    # Permanently-open doors only: nobody is tied to a lever, so both slots are
    # free to stand anywhere in this set at once.
    joint_sets = [_expand(model, home[slot], set(open_clusters)) for slot in range(AGENT_SLOTS)]
    report.joint_reachable = set.intersection(*joint_sets) if joint_sets else set()
    report.opened_clusters = open_clusters
    report.unlock_round = unlock_round
    report.one_way_clusters = {cid for cid, slots in one_way.items() if slots}
    report.blocked_clusters = {
        cid
        for cid in model.clusters
        if cid not in open_clusters and cid not in report.one_way_clusters
    }
    report.cooperative_clusters = _find_cooperative(model, open_clusters, one_way, reach)
    report.unreachable_entities = model.unreachable_entities

    exit_door = room.exit
    exit_region = model.region_of_entity(exit_door.id)
    if exit_region is None:
        report.reasons.append("exit door is not on walkable floor")
        return report

    report.exit_reachable_by = tuple(
        slot for slot in range(AGENT_SLOTS) if exit_region in reach[slot]
    )
    report.exit_jointly_reachable = exit_region in report.joint_reachable
    report.exit_requirement_met = _satisfiable(
        exit_door.requirement, range(AGENT_SLOTS), reach, model
    )

    if not report.exit_reachable_by:
        report.reasons.append("no agent slot can reach the exit door")
    if not report.exit_requirement_met:
        report.reasons.append(
            f"exit requirement cannot be satisfied: {exit_door.requirement.describe()}"
        )
    if room.config.exit_requires_both_agents and not report.exit_jointly_reachable:
        report.reasons.append(
            "exit_requires_both_agents is set, but the exit is only reachable one "
            "slot at a time (a hold-lever passage splits the pair)"
        )

    report.solvable = not report.reasons
    return report


def _expand(model: ConnectivityModel, start: int, allowed_clusters: set[str]) -> set[int]:
    """Regions reachable from `start` using free links and `allowed_clusters`."""
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbour in model.neighbours_of(current):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
        for cluster in model.clusters_touching(current):
            if cluster.id not in allowed_clusters:
                continue
            for region in cluster.regions:
                if region not in seen:
                    seen.add(region)
                    stack.append(region)
    return seen


def _satisfiable(
    requirement: Requirement,
    slots,
    reach: list[set[int]],
    model: ConnectivityModel,
) -> bool:
    """Can the given agent slots, between them, meet this requirement?"""
    slots = list(slots)
    if not slots:
        return False

    if isinstance(requirement, NeverOpen):
        return False
    if isinstance(requirement, AlwaysOpen) or type(requirement) is Requirement:
        return True

    if isinstance(requirement, CompositeRequirement):
        results = [_satisfiable(p, slots, reach, model) for p in requirement.parts]
        return any(results) if requirement.mode is TriggerMode.ANY else all(results)

    if isinstance(
        requirement, (KeyRequirement, SwitchRequirement, CheckpointRequirement)
    ):
        ids = sorted(requirement.referenced_ids())
        mode = getattr(requirement, "mode", TriggerMode.ALL)
        if isinstance(requirement, SwitchRequirement) and requirement.needs_simultaneity():
            return _has_simultaneous_cover(tuple(ids), slots, reach, model)
        checks = [_any_slot_reaches(entity_id, slots, reach, model) for entity_id in ids]
        return any(checks) if mode is TriggerMode.ANY else all(checks)

    # Unknown requirement type: refuse rather than assume it is satisfiable.
    return False


def _crate_sustains(
    requirement: Requirement,
    model: ConnectivityModel,
) -> bool:
    if not isinstance(requirement, SwitchRequirement):
        return False
    checks = [
        switch_id in model.crate_held_switches
        for switch_id in requirement.switch_ids
    ]
    if requirement.mode is TriggerMode.ANY:
        return any(checks)
    return bool(checks) and all(checks)


def _any_slot_reaches(
    entity_id: str, slots, reach: list[set[int]], model: ConnectivityModel
) -> bool:
    region = model.region_of_entity(entity_id)
    if region is None:
        return False
    owner = model.key_owners.get(entity_id)
    if owner is not None:
        return owner in slots and region in reach[owner]
    return any(region in reach[s] for s in slots)


def _has_simultaneous_cover(
    entity_ids: tuple[str, ...], slots, reach: list[set[int]], model: ConnectivityModel
) -> bool:
    """One distinct agent slot per trigger, all reachable at once.

    This is the structural form of "two buttons, two agents, same instant".
    """
    slots = list(slots)
    ids = list(entity_ids)
    if len(ids) > len(slots):
        return False
    regions = [model.region_of_entity(i) for i in ids]
    if any(r is None for r in regions):
        return False
    for assignment in permutations(slots, len(ids)):
        if all(regions[k] in reach[assignment[k]] for k in range(len(ids))):
            return True
    return False


def _find_cooperative(
    model: ConnectivityModel,
    open_clusters: set[str],
    one_way: dict[str, set[int]],
    reach: list[set[int]],
) -> set[str]:
    """Doors that a single agent slot could not have opened by itself."""
    cooperative: set[str] = set()
    for cluster_id in open_clusters | set(one_way):
        cluster: DoorCluster = model.clusters[cluster_id]
        if not cluster.latching:
            cooperative.add(cluster_id)  # hold-levers always need a partner
            continue
        solo = any(
            _satisfiable(cluster.requirement, [slot], reach, model)
            for slot in range(AGENT_SLOTS)
        )
        if not solo:
            cooperative.add(cluster_id)
    return cooperative

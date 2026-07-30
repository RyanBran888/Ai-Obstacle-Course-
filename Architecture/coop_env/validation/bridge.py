"""Spatial and timed checks for required temporary bridges."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import lcm

from ..entities import PushableBlock, TemporaryBridge
from ..room import Room
from ..tiles import is_hazard, is_walkable
from ..utils.geometry import Vec2
from ..utils.grid import flood_fill


@dataclass(frozen=True, slots=True)
class BridgeReport:
    required_bridge_id: str | None
    structural_crossing: bool
    time_solvable: bool
    reachable_by: tuple[int, ...]
    min_steps_by_agent: tuple[int | None, ...]
    period: int
    reasons: tuple[str, ...]


def analyse_bridge(room: Room) -> BridgeReport:
    """Check the required cut and find safe timed routes for both agents."""
    required_id = room.metadata.get("required_bridge_id")
    required = room.find(required_id) if isinstance(required_id, str) else None
    reasons: list[str] = []
    base = _base_walkable(room)

    structural = True
    if not isinstance(required, TemporaryBridge):
        structural = False
        reasons.append("required bridge crossing has no selected bridge")
    elif not _is_spawn_exit_cut(room, base, required):
        structural = False
        reasons.append(
            f"{required.id} does not lie on every spawn-to-exit route"
        )

    periods = [max(1, bridge.period) for bridge in room.bridges]
    periods.extend(
        2 * (len(ball.track) - 1)
        for ball in room.wipeout_balls
        if len(ball.track) > 1
    )
    period = lcm(*periods) if periods else 1
    support = tuple(
        {
            tile
            for bridge in room.bridges
            if bridge.is_solid_at(tick)
            for tile in bridge.footprint()
        }
        for tick in range(period)
    )
    danger = tuple(
        {
            tile
            for ball in room.wipeout_balls
            for tile in ball.collision_tiles_at(tick)
        }
        for tick in range(period)
    )
    distances = tuple(
        _timed_distance(room, base, spawn.pos, support, danger)
        for spawn in room.spawns
    )
    reachable = tuple(
        index for index, distance in enumerate(distances) if distance is not None
    )
    timed = len(reachable) == len(room.spawns)
    if not timed:
        reasons.append("no phase-safe route lets both agents cross the bridge")

    return BridgeReport(
        required_bridge_id=required_id if isinstance(required_id, str) else None,
        structural_crossing=structural,
        time_solvable=timed,
        reachable_by=reachable,
        min_steps_by_agent=distances,
        period=period,
        reasons=tuple(reasons),
    )


def _base_walkable(room: Room) -> set[Vec2]:
    tiles = {
        pos
        for pos in room.terrain.positions()
        if is_walkable(room.terrain[pos])
    }
    for bridge in room.bridges:
        tiles.update(bridge.footprint())
    for block in room.of_type(PushableBlock):
        tiles.difference_update(block.footprint())
    return tiles


def _is_spawn_exit_cut(
    room: Room,
    base: set[Vec2],
    bridge: TemporaryBridge,
) -> bool:
    exit_pos = room.exit.pos
    for spawn in room.spawns:
        reachable = flood_fill(spawn.pos, lambda pos: pos in base, room.bounds)
        if exit_pos not in reachable:
            return False

    without_bridge = base - set(bridge.footprint())
    return all(
        exit_pos
        not in flood_fill(spawn.pos, lambda pos: pos in without_bridge, room.bounds)
        for spawn in room.spawns
    )


def _timed_distance(
    room: Room,
    base: set[Vec2],
    start: Vec2,
    support: tuple[set[Vec2], ...],
    danger: tuple[set[Vec2], ...],
) -> int | None:
    period = len(support)

    def safe(pos: Vec2, phase: int) -> bool:
        if pos not in base or pos in danger[phase]:
            return False
        if is_walkable(room.terrain_at(pos)):
            return True
        return is_hazard(room.terrain_at(pos)) and pos in support[phase]

    if not safe(start, 0):
        return None

    queue = deque([(start, 0, 0)])
    seen = {(start, 0)}
    while queue:
        current, phase, distance = queue.popleft()
        if current == room.exit.pos:
            return distance
        next_phase = (phase + 1) % period
        for destination in (current, *current.neighbors4()):
            state = (destination, next_phase)
            if state in seen:
                continue
            if not safe(destination, phase) or not safe(destination, next_phase):
                continue
            seen.add(state)
            queue.append((destination, next_phase, distance + 1))
    return None

"""Spatial and timed checks for moving wipeout balls."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import lcm

from ..entities import PushableBlock, WipeoutBall
from ..room import Room
from ..tiles import is_walkable
from ..utils.geometry import Vec2
from ..utils.grid import flood_fill


@dataclass(frozen=True, slots=True)
class WipeoutReport:
    required_ball_id: str | None
    structural_crossing: bool
    time_solvable: bool
    reachable_by: tuple[int, ...]
    min_steps_by_agent: tuple[int | None, ...]
    period: int
    reasons: tuple[str, ...]


def analyse_wipeout(room: Room) -> WipeoutReport:
    """Check the required cut and find safe timed routes."""
    balls = room.wipeout_balls
    required_id = room.metadata.get("required_wipeout_ball_id")
    reasons: list[str] = []
    base = _base_walkable(room)

    structural = True
    requires_crossing = (
        room.config.require_wipeout_crossing
        or room.config.require_combined_course
    )
    if requires_crossing:
        required = room.find(required_id) if isinstance(required_id, str) else None
        if not isinstance(required, WipeoutBall):
            structural = False
            reasons.append("required wipeout crossing has no selected ball")
        elif not _is_spawn_exit_cut(room, base, required):
            structural = False
            reasons.append(
                f"{required.id} does not lie on every spawn-to-exit route"
            )

    period = lcm(
        *(2 * (len(ball.track) - 1) for ball in balls if len(ball.track) > 1)
    ) if balls else 1
    danger = tuple(
        {
            tile
            for ball in balls
            for tile in ball.collision_tiles_at(tick)
        }
        for tick in range(period)
    )
    distances = tuple(
        _timed_distance(room, base, spawn.pos, danger)
        for spawn in room.spawns
    )
    reachable = tuple(index for index, distance in enumerate(distances) if distance is not None)
    needs_both = (
        room.config.exit_requires_both_agents
        or requires_crossing
    )
    timed = len(reachable) == len(room.spawns) if needs_both else bool(reachable)
    if balls and not timed:
        needed = "both agents" if needs_both else "either agent"
        reasons.append(f"no collision-free timed route lets {needed} reach the exit")

    return WipeoutReport(
        required_ball_id=required_id if isinstance(required_id, str) else None,
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
    ball: WipeoutBall,
) -> bool:
    exit_pos = room.exit.pos
    for spawn in room.spawns:
        reachable = flood_fill(spawn.pos, lambda pos: pos in base, room.bounds)
        if exit_pos not in reachable:
            return False

    without_track = base - set(ball.footprint())
    return all(
        exit_pos
        not in flood_fill(spawn.pos, lambda pos: pos in without_track, room.bounds)
        for spawn in room.spawns
    )


def _timed_distance(
    room: Room,
    base: set[Vec2],
    start: Vec2,
    danger: tuple[set[Vec2], ...],
) -> int | None:
    period = len(danger)
    if start not in base or start in danger[0]:
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
            if state in seen or destination not in base:
                continue
            if destination in danger[phase] or destination in danger[next_phase]:
                continue
            seen.add(state)
            queue.append((destination, next_phase, distance + 1))
    return None

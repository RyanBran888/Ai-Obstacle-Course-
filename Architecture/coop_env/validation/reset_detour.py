"""Static checks for an intentional reset-zone shortcut and safe detour."""

from __future__ import annotations

from dataclasses import dataclass

from ..entities import PushableBlock, ResetZone
from ..room import Room
from ..tiles import is_walkable
from ..utils.geometry import Vec2
from ..utils.grid import distance_field


@dataclass(frozen=True, slots=True)
class ResetDetourReport:
    required_reset_zone_id: str | None
    shortcut_valid: bool
    safe_detour: bool
    shortest_steps_by_agent: tuple[int | None, ...]
    safe_steps_by_agent: tuple[int | None, ...]
    reasons: tuple[str, ...]


def analyse_reset_detour(room: Room) -> ResetDetourReport:
    required_id = room.metadata.get("required_reset_zone_id")
    reset = room.find(required_id) if isinstance(required_id, str) else None
    reasons: list[str] = []
    shortest: list[int | None] = []
    safe_steps: list[int | None] = []
    shortcut_valid = isinstance(reset, ResetZone) and reset.rect.area == 1
    safe_detour = shortcut_valid

    if not isinstance(reset, ResetZone) or reset.rect.area != 1:
        reasons.append("required reset detour has no selected 1x1 reset zone")
    else:
        base = _base_walkable(room)
        reset_tile = reset.pos
        for spawn in room.spawns:
            direct = _distance(room, base, spawn.pos, room.exit.pos)
            to_reset = _distance(room, base, spawn.pos, reset_tile)
            from_reset = _distance(room, base, reset_tile, room.exit.pos)
            safe = _distance(room, base - {reset_tile}, spawn.pos, room.exit.pos)
            shortest.append(direct)
            safe_steps.append(safe)

            on_shortest = (
                direct is not None
                and to_reset is not None
                and from_reset is not None
                and to_reset + from_reset == direct
            )
            if not on_shortest:
                shortcut_valid = False
            if direct is None or safe is None or safe <= direct:
                safe_detour = False

        if not shortcut_valid:
            reasons.append(
                f"{reset.id} is not on a shortest route for both agents"
            )
        if not safe_detour:
            reasons.append(
                f"avoiding {reset.id} does not leave a strictly longer safe route"
            )

    while len(shortest) < len(room.spawns):
        shortest.append(None)
        safe_steps.append(None)
    return ResetDetourReport(
        required_reset_zone_id=required_id if isinstance(required_id, str) else None,
        shortcut_valid=shortcut_valid,
        safe_detour=safe_detour,
        shortest_steps_by_agent=tuple(shortest),
        safe_steps_by_agent=tuple(safe_steps),
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


def _distance(
    room: Room,
    allowed: set[Vec2],
    start: Vec2,
    goal: Vec2,
) -> int | None:
    if start not in allowed or goal not in allowed:
        return None
    return distance_field(
        (start,),
        lambda tile: tile in allowed,
        room.bounds,
    ).get(goal)

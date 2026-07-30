"""Checks for the serial combined course."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import lcm
from typing import Any

from ..entities import (
    Checkpoint,
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
from ..requirements import CheckpointRequirement, KeyRequirement, SwitchRequirement
from ..room import Room
from ..tiles import Tile, is_hazard, is_walkable
from ..utils.geometry import Rect, Vec2
from ..utils.grid import distance_field, flood_fill
from .bridge import BridgeReport
from .reset_detour import ResetDetourReport
from .wipeout import WipeoutReport


COMBINED_SECTION_ENTITIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("owned_key_0", ("key_agent_0", "door_key_0")),
    ("owned_key_1", ("key_agent_1", "door_key_1")),
    ("crate_hold", ("crate_0", "switch_crate", "door_crate")),
    ("wipeout_cut", ("wipeout_normal_0", "wipeout_big_0")),
    ("bridge_cut", ("bridge_0",)),
    ("reset_detour", ("reset_0",)),
    ("checkpoint_exit", ("checkpoint_exit", "exit")),
)


@dataclass(frozen=True, slots=True)
class CombinedCourseReport:
    section_ids: tuple[str, ...]
    gate_cut_ids: tuple[str, ...]
    ball_cut_ids: tuple[str, ...]
    timed_max_steps: tuple[tuple[str, int], ...]
    route_budget: int | None
    reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.reasons


def combined_route_budget(room: Room) -> dict[str, int]:
    """Return the fixed-course upper budget."""
    allowed = {
        pos
        for pos in room.terrain.positions()
        if is_walkable(room.terrain[pos])
    }
    for bridge in room.bridges:
        allowed.update(bridge.footprint())
    for reset in room.reset_zones:
        allowed.difference_update(reset.footprint())

    def gap(first: Vec2, second: Vec2) -> int:
        distance = distance_field(
            (first,),
            lambda pos: pos in allowed,
            room.bounds,
        ).get(second)
        return distance if distance is not None else 10_000

    key_0 = room.find("key_agent_0")
    key_1 = room.find("key_agent_1")
    door_0 = room.find("door_key_0")
    door_1 = room.find("door_key_1")
    block = room.find("crate_0")
    hold_door = room.find("door_crate")
    checkpoint = room.find("checkpoint_exit")
    if not (
        isinstance(key_0, Key)
        and isinstance(key_1, Key)
        and isinstance(door_0, LockedDoor)
        and isinstance(door_1, LockedDoor)
        and isinstance(block, PushableBlock)
        and block.push_from is not None
        and isinstance(hold_door, LockedDoor)
        and isinstance(checkpoint, Checkpoint)
    ):
        planned = 10_000
    else:
        phase_0 = max(
            gap(room.spawns[0].pos, key_0.pos) + 1 + gap(key_0.pos, door_0.pos),
            gap(room.spawns[1].pos, door_0.pos),
        )
        phase_1 = max(
            gap(door_0.pos, key_1.pos) + 1 + gap(key_1.pos, door_1.pos),
            gap(door_0.pos, door_1.pos),
        )
        phase_2 = (
            gap(door_1.pos, block.push_from)
            + 1
            + gap(block.pos, hold_door.pos)
        )
        phase_3 = (
            gap(hold_door.pos, checkpoint.pos)
            + 1
            + gap(checkpoint.pos, room.exit.pos)
        )
        planned = phase_0 + phase_1 + phase_2 + phase_3

    periods = [
        2 * (len(ball.track) - 1)
        for ball in room.wipeout_balls
        if len(ball.track) > 1
    ]
    periods.extend(max(1, bridge.period) for bridge in room.bridges)
    timing = lcm(*periods) if periods else 1
    movement = 2 * (room.width - 2) + room.height
    margin = 10
    return {
        "horizon": 200,
        "planned_action_steps": planned,
        "movement_budget": movement,
        "timing_budget": timing,
        "interaction_margin": margin,
        "total": movement + timing + margin,
    }


def analyse_combined_course(
    room: Room,
    wipeout: WipeoutReport | None,
    bridge: BridgeReport | None,
    reset_detour: ResetDetourReport | None,
) -> CombinedCourseReport:
    reasons: list[str] = []
    course = room.metadata.get("combined_course")
    if not isinstance(course, dict):
        return CombinedCourseReport((), (), (), (), None, ("combined metadata is missing",))
    if course.get("version") != 1:
        reasons.append("combined metadata version is not supported")

    sections = course.get("sections")
    section_ids: tuple[str, ...] = ()
    section_map: dict[str, dict[str, Any]] = {}
    if isinstance(sections, (tuple, list)):
        parsed: list[str] = []
        for section in sections:
            if not isinstance(section, dict) or not isinstance(section.get("id"), str):
                continue
            section_id = section["id"]
            parsed.append(section_id)
            section_map[section_id] = section
        section_ids = tuple(parsed)
    expected_ids = tuple(section_id for section_id, _ in COMBINED_SECTION_ENTITIES)
    if section_ids != expected_ids:
        reasons.append("combined sections are missing or out of order")
    for section_id, entity_ids in COMBINED_SECTION_ENTITIES:
        section = section_map.get(section_id)
        saved = tuple(section.get("entities", ())) if section else ()
        if saved != entity_ids:
            reasons.append(f"{section_id} entity metadata does not match")

    keys = (room.find("key_agent_0"), room.find("key_agent_1"))
    key_doors = (room.find("door_key_0"), room.find("door_key_1"))
    for owner, (key, door) in enumerate(zip(keys, key_doors)):
        if not (
            isinstance(key, Key)
            and key.agent_index == owner
            and isinstance(door, LockedDoor)
            and key.opens == (door.id,)
            and isinstance(door.requirement, KeyRequirement)
            and door.requirement.key_ids == (key.id,)
        ):
            reasons.append(f"owned key gate {owner} is not exact")

    block = room.find("crate_0")
    switch = room.find("switch_crate")
    crate_door = room.find("door_crate")
    crate_exact = (
        isinstance(block, PushableBlock)
        and isinstance(switch, Switch)
        and switch.mode is SwitchMode.HOLD
        and block.target_switch_id == switch.id
        and block.push_from is not None
        and isinstance(crate_door, LockedDoor)
        and not crate_door.latching
        and isinstance(crate_door.requirement, SwitchRequirement)
        and crate_door.requirement.switch_ids == (switch.id,)
        and switch.controls == (crate_door.id,)
    )
    if not crate_exact:
        reasons.append("crate HOLD gate is not exact")

    checkpoint = room.find("checkpoint_exit")
    if not (
        isinstance(checkpoint, Checkpoint)
        and isinstance(room.exit.requirement, CheckpointRequirement)
        and room.exit.requirement.checkpoint_ids == (checkpoint.id,)
        and room.config.exit_requires_both_agents
    ):
        reasons.append("checkpoint exit contract is not exact")

    expected_counts = {
        "keys": 2,
        "doors": 3,
        "switches": 1,
        "blocks": 1,
        "checkpoints": 1,
        "reset_zones": 1,
        "bridges": 1,
        "wipeout_balls": 2,
    }
    for name, count in expected_counts.items():
        if len(getattr(room, name)) != count:
            reasons.append(f"combined course needs exactly {count} {name}")

    base = _base_walkable(room)
    gate_cut_ids = tuple(
        door.id
        for door in room.doors
        if _is_cut(room, base, set(door.footprint()))
    )
    expected_gate_ids = ("door_key_0", "door_key_1", "door_crate")
    if set(gate_cut_ids) != set(expected_gate_ids):
        reasons.append("every combined gate must be a spawn-to-exit cut")

    if crate_exact:
        assert isinstance(block, PushableBlock)
        assert isinstance(switch, Switch)
        assert isinstance(crate_door, LockedDoor)
        near = flood_fill(
            room.spawns[0].pos,
            lambda pos: pos in base - set(crate_door.footprint()),
            room.bounds,
        )
        if (
            block.push_from not in near
            or switch.pos not in near
            or room.exit.pos in near
        ):
            reasons.append("crate cannot permanently hold the required near-side gate")

    balls = tuple(room.wipeout_balls)
    ball_cut_ids = tuple(
        ball.id
        for ball in balls
        if _is_cut(room, base, set(ball.footprint()))
    )
    required_ball_id = room.metadata.get("required_wipeout_ball_id")
    if ball_cut_ids != (required_ball_id,):
        reasons.append("exactly the selected wipeout ball must be a required cut")
    required_ball = room.find(required_ball_id) if isinstance(required_ball_id, str) else None
    if not isinstance(required_ball, WipeoutBall):
        reasons.append("required wipeout metadata is missing")
    else:
        if course.get("required_ball_size") != required_ball.size.value:
            reasons.append("required wipeout size metadata does not match")
        saved_barrier = _saved_positions(course.get("ball_barrier"))
        if (
            saved_barrier is None
            or not set(required_ball.footprint()).issubset(saved_barrier)
        ):
            reasons.append("wipeout barrier metadata does not match")
    sizes = {ball.size for ball in balls}
    if sizes != {WipeoutBallSize.NORMAL, WipeoutBallSize.BIG}:
        reasons.append("combined course needs one normal and one big wipeout ball")
    if wipeout is None or not wipeout.structural_crossing or not wipeout.time_solvable:
        reasons.append("required wipeout crossing is not time-solvable")

    required_bridge = room.find(room.metadata.get("required_bridge_id", ""))
    if not isinstance(required_bridge, TemporaryBridge):
        reasons.append("required bridge metadata is missing")
    elif not _is_cut(room, base, set(required_bridge.footprint())):
        reasons.append("required bridge has a bypass")
    elif course.get("bridge_hazard") != int(
        room.terrain_at(required_bridge.pos)
    ):
        reasons.append("bridge hazard metadata does not match")
    if bridge is None or not bridge.structural_crossing or not bridge.time_solvable:
        reasons.append("required bridge crossing is not time-solvable")

    reset = room.find(room.metadata.get("required_reset_zone_id", ""))
    if not isinstance(reset, ResetZone):
        reasons.append("required reset metadata is missing")
    if (
        reset_detour is None
        or not reset_detour.shortcut_valid
        or not reset_detour.safe_detour
    ):
        reasons.append("required reset safe detour is invalid")

    bridge_tiles = {
        tile for item in room.bridges for tile in item.footprint()
    }
    non_bridge_hazards = {
        pos
        for pos in room.terrain.positions()
        if is_hazard(room.terrain[pos]) and pos not in bridge_tiles
    }
    if not non_bridge_hazards:
        reasons.append("combined course needs non-bridge hazards")
    saved_hazards = _saved_positions(course.get("non_bridge_hazards"))
    if saved_hazards != non_bridge_hazards:
        reasons.append("non-bridge hazard metadata does not match")
    if not any(
        room.terrain_at(pos) is Tile.OBSTACLE
        for pos in room.terrain.positions()
    ):
        reasons.append("combined course needs obstacles")

    timed: list[tuple[str, int]] = []
    period = _course_period(room)
    for section_id in ("wipeout_cut", "bridge_cut"):
        section = section_map.get(section_id)
        start = _saved_vec(section, "entry")
        goal = _saved_vec(section, "exit")
        bounds = _saved_rect(section)
        if start is None or goal is None or bounds is None:
            reasons.append(f"{section_id} timing metadata is missing")
            continue
        distances = [
            _timed_distance(room, start, goal, bounds, phase, period)
            for phase in range(period)
        ]
        if any(distance is None for distance in distances):
            reasons.append(f"{section_id} is not safe from every arrival phase")
            continue
        timed.append((section_id, max(distance for distance in distances if distance is not None)))

    saved_budget = course.get("route_budget")
    expected_budget = combined_route_budget(room)
    route_budget = saved_budget.get("total") if isinstance(saved_budget, dict) else None
    if saved_budget != expected_budget:
        reasons.append("combined route budget metadata does not match")
    if (
        expected_budget["planned_action_steps"] > expected_budget["movement_budget"]
        or expected_budget["total"] >= expected_budget["horizon"]
    ):
        reasons.append("combined route budget exceeds the 200-step horizon")
    if timed and max(distance for _, distance in timed) > expected_budget["timing_budget"]:
        reasons.append("a timed section exceeds its conservative wait budget")

    return CombinedCourseReport(
        section_ids=section_ids,
        gate_cut_ids=gate_cut_ids,
        ball_cut_ids=ball_cut_ids,
        timed_max_steps=tuple(timed),
        route_budget=route_budget if isinstance(route_budget, int) else None,
        reasons=tuple(reasons),
    )


def _base_walkable(room: Room) -> set[Vec2]:
    allowed = {
        pos
        for pos in room.terrain.positions()
        if is_walkable(room.terrain[pos])
    }
    for bridge in room.bridges:
        allowed.update(bridge.footprint())
    for block in room.blocks:
        allowed.discard(block.pos)
    return allowed


def _is_cut(room: Room, base: set[Vec2], removed: set[Vec2]) -> bool:
    if not all(
        room.exit.pos
        in flood_fill(spawn.pos, lambda pos: pos in base, room.bounds)
        for spawn in room.spawns
    ):
        return False
    allowed = base - removed
    return all(
        room.exit.pos
        not in flood_fill(spawn.pos, lambda pos: pos in allowed, room.bounds)
        for spawn in room.spawns
    )


def _course_period(room: Room) -> int:
    periods = [
        2 * (len(ball.track) - 1)
        for ball in room.wipeout_balls
        if len(ball.track) > 1
    ]
    periods.extend(max(1, bridge.period) for bridge in room.bridges)
    return lcm(*periods) if periods else 1


def _saved_vec(section: dict[str, Any] | None, name: str) -> Vec2 | None:
    value = section.get(name) if section else None
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return Vec2(value[0], value[1])
    return None


def _saved_positions(value: Any) -> set[Vec2] | None:
    if not isinstance(value, (tuple, list)):
        return None
    positions: set[Vec2] = set()
    for item in value:
        if (
            not isinstance(item, (tuple, list))
            or len(item) != 2
            or not all(isinstance(part, int) for part in item)
        ):
            return None
        positions.add(Vec2(item[0], item[1]))
    return positions


def _saved_rect(section: dict[str, Any] | None) -> Rect | None:
    value = section.get("bounds") if section else None
    if (
        isinstance(value, (tuple, list))
        and len(value) == 4
        and all(isinstance(item, int) for item in value)
    ):
        return Rect(value[0], value[1], value[2], value[3])
    return None


def _timed_distance(
    room: Room,
    start: Vec2,
    goal: Vec2,
    bounds: Rect,
    start_phase: int,
    period: int,
) -> int | None:
    bridges = {
        tile: bridge
        for bridge in room.bridges
        for tile in bridge.footprint()
    }

    def safe(pos: Vec2, phase: int) -> bool:
        if not bounds.contains(pos):
            return False
        if any(pos in ball.collision_tiles_at(phase) for ball in room.wipeout_balls):
            return False
        tile = room.terrain_at(pos)
        if is_walkable(tile):
            return True
        bridge = bridges.get(pos)
        return (
            bridge is not None
            and is_hazard(tile)
            and bridge.is_solid_at(phase)
        )

    phase = start_phase % period
    if not safe(start, phase):
        return None
    queue = deque([(start, phase, 0)])
    seen = {(start, phase)}
    limit = period + bounds.w * bounds.h
    while queue:
        current, at, steps = queue.popleft()
        if current == goal:
            return steps
        if steps >= limit:
            continue
        after = (at + 1) % period
        for destination in (current, *current.neighbors4()):
            state = (destination, after)
            if state in seen:
                continue
            if not safe(destination, at) or not safe(destination, after):
                continue
            seen.add(state)
            queue.append((destination, after, steps + 1))
    return None

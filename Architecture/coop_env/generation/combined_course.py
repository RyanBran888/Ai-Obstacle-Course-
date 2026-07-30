"""By-construction serial cooperative course."""

from __future__ import annotations

from ..config import GenerationConfig, RoomShape
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
from ..requirements import CheckpointRequirement, KeyRequirement, SwitchRequirement
from ..rng import SeededRandom
from ..room import Region, Room, RoomTopology, portal_key
from ..tiles import HAZARD_TILES, Tile, is_walkable
from ..utils.geometry import Rect, Vec2
from ..utils.graph import Graph
from ..utils.grid import Grid
from ..validation.combined import (
    COMBINED_SECTION_ENTITIES,
    combined_route_budget,
)


def build_combined_course(seed: int, config: GenerationConfig) -> Room:
    rng = SeededRandom(seed).derive("combined_course")
    width = rng.derive("width").randint(max(37, config.width[0]), min(38, config.width[1]))
    height = rng.derive("height").randint(max(25, config.height[0]), min(32, config.height[1]))
    top_y = 5
    bottom_y = height - 6
    bottom_top = bottom_y - 3
    ball_y = (10 + bottom_top - 1) // 2
    course_right = 35
    safe_below = rng.derive("reset_side").chance(0.5)
    safe_y = bottom_y + 2 if safe_below else bottom_y - 2
    required_size = rng.derive("required_ball").choice(
        (WipeoutBallSize.NORMAL, WipeoutBallSize.BIG)
    )

    terrain = Grid(width, height, Tile.WALL)
    for pos in Rect(1, 1, course_right, 9).positions():
        terrain[pos] = Tile.FLOOR
    for pos in Rect(23, 10, course_right - 22, bottom_top - 10).positions():
        terrain[pos] = Tile.FLOOR
    for pos in Rect(1, bottom_top, course_right, 7).positions():
        terrain[pos] = Tile.FLOOR

    door_positions = (
        Vec2(4, top_y),
        Vec2(8, top_y),
        Vec2(14, top_y),
    )
    for x in (4, 8, 14):
        for y in range(1, 10):
            terrain[Vec2(x, y)] = Tile.WALL
        terrain[Vec2(x, top_y)] = Tile.FLOOR

    normal_track = tuple(Vec2(27 + offset, ball_y) for offset in range(7))
    big_track = tuple(Vec2(24 + offset, ball_y) for offset in range(11))
    if required_size is WipeoutBallSize.NORMAL:
        big_track = tuple(Vec2(23 + offset, 8) for offset in range(11))
    else:
        normal_track = tuple(Vec2(27 + offset, 8) for offset in range(7))
    normal_ball = WipeoutBall(
        id="wipeout_normal_0",
        pos=normal_track[0],
        track=normal_track,
        size=WipeoutBallSize.NORMAL,
    )
    big_ball = WipeoutBall(
        id="wipeout_big_0",
        pos=big_track[0],
        track=big_track,
        size=WipeoutBallSize.BIG,
    )
    required_ball = normal_ball if required_size is WipeoutBallSize.NORMAL else big_ball
    barrier_rows = (
        (ball_y,)
        if required_size is WipeoutBallSize.NORMAL
        else (ball_y - 1, ball_y, ball_y + 1)
    )
    for y in barrier_rows:
        for x in range(23, course_right + 1):
            terrain[Vec2(x, y)] = Tile.WALL
    for pos in required_ball.footprint():
        terrain[pos] = Tile.FLOOR

    hazards = {
        tile: weight
        for tile, weight in config.hazard_weights.items()
        if tile in HAZARD_TILES and weight > 0
    }
    bridge_hazard = rng.derive("bridge_hazard").weighted_choice(hazards)
    side_hazard = rng.derive("side_hazard").weighted_choice(hazards)
    bridge_tiles = tuple(Vec2(20, bottom_y + offset) for offset in (-1, 0, 1))
    for y in range(bottom_top, bottom_y + 4):
        terrain[Vec2(20, y)] = Tile.WALL
    for pos in bridge_tiles:
        terrain[pos] = bridge_hazard

    for y in range(bottom_top, bottom_y + 4):
        terrain[Vec2(10, y)] = Tile.OBSTACLE
    reset_pos = Vec2(10, bottom_y)
    safe_pos = Vec2(10, safe_y)
    terrain[reset_pos] = Tile.FLOOR
    terrain[safe_pos] = Tile.FLOOR

    decorative_obstacles = (
        Vec2(2, 2),
        Vec2(6, 8),
        Vec2(10, 2),
        Vec2(25, bottom_y + 2),
    )
    for pos in decorative_obstacles:
        terrain[pos] = Tile.OBSTACLE
    hazard_y = 2 if safe_below else 7
    non_bridge_hazards = (
        Vec2(17, hazard_y),
        Vec2(18, hazard_y),
        Vec2(17, hazard_y + 1),
        Vec2(18, hazard_y + 1),
    )
    for pos in non_bridge_hazards:
        terrain[pos] = side_hazard

    key_sign = -1 if rng.derive("key_side").chance(0.5) else 1
    spawns = (
        AgentSpawn(id="spawn_0", pos=Vec2(2, top_y - 1), index=0),
        AgentSpawn(id="spawn_1", pos=Vec2(2, top_y + 1), index=1),
    )
    keys = (
        Key(
            id="key_agent_0",
            pos=Vec2(2, top_y + key_sign * 4),
            color="azure",
            opens=("door_key_0",),
            agent_index=0,
        ),
        Key(
            id="key_agent_1",
            pos=Vec2(6, top_y - key_sign * 4),
            color="crimson",
            opens=("door_key_1",),
            agent_index=1,
        ),
    )
    switch = Switch(
        id="switch_crate",
        pos=Vec2(12, top_y),
        mode=SwitchMode.HOLD,
        group="crate_hold",
        controls=("door_crate",),
    )
    block = PushableBlock(
        id="crate_0",
        pos=Vec2(11, top_y),
        target_switch_id=switch.id,
        push_from=Vec2(10, top_y),
    )
    doors = (
        LockedDoor(
            id="door_key_0",
            pos=door_positions[0],
            requirement=KeyRequirement((keys[0].id,)),
            latching=True,
            horizontal=False,
            region_a=0,
            region_b=1,
        ),
        LockedDoor(
            id="door_key_1",
            pos=door_positions[1],
            requirement=KeyRequirement((keys[1].id,)),
            latching=True,
            horizontal=False,
            region_a=1,
            region_b=2,
        ),
        LockedDoor(
            id="door_crate",
            pos=door_positions[2],
            requirement=SwitchRequirement((switch.id,)),
            latching=False,
            horizontal=False,
            region_a=2,
            region_b=3,
        ),
    )
    bridge = TemporaryBridge(
        id="bridge_0",
        pos=bridge_tiles[0],
        tiles=bridge_tiles,
        period=12,
        on_ticks=6,
        phase=rng.derive("bridge_phase").randint(0, 11),
    )
    reset = ResetZone(
        id="reset_0",
        pos=reset_pos,
        rect=Rect(reset_pos.x, reset_pos.y, 1, 1),
    )
    checkpoint = Checkpoint(
        id="checkpoint_exit",
        pos=Vec2(5, bottom_y),
        order=0,
        group="exit",
    )
    exit_door = ExitDoor(
        id="exit",
        pos=Vec2(2, bottom_y),
        requirement=CheckpointRequirement((checkpoint.id,)),
    )
    entities: tuple[Entity, ...] = (
        *spawns,
        *keys,
        *doors,
        switch,
        block,
        normal_ball,
        big_ball,
        bridge,
        reset,
        checkpoint,
        exit_door,
    )

    topology = _topology(
        terrain,
        bridge_tiles,
        door_positions,
        bottom_top,
    )
    section_bounds = {
        "owned_key_0": Rect(1, 1, 4, 9),
        "owned_key_1": Rect(4, 1, 5, 9),
        "crate_hold": Rect(8, 1, 7, 9),
        "wipeout_cut": Rect(
            23,
            ball_y - 2,
            course_right - 22,
            5,
        ),
        "bridge_cut": Rect(19, bottom_top, 3, 7),
        "reset_detour": Rect(9, bottom_top, 11, 7),
        "checkpoint_exit": Rect(1, bottom_top, 9, 7),
    }
    ball_entry_y = (
        ball_y - 1
        if required_size is WipeoutBallSize.NORMAL
        else ball_y - 2
    )
    ball_exit_y = (
        ball_y + 1
        if required_size is WipeoutBallSize.NORMAL
        else ball_y + 2
    )
    section_edges = {
        "owned_key_0": (spawns[0].pos, door_positions[0]),
        "owned_key_1": (Vec2(5, top_y), door_positions[1]),
        "crate_hold": (Vec2(9, top_y), door_positions[2]),
        "wipeout_cut": (
            Vec2(29, ball_entry_y),
            Vec2(29, ball_exit_y),
        ),
        "bridge_cut": (
            Vec2(21, bottom_y),
            Vec2(19, bottom_y),
        ),
        "reset_detour": (
            Vec2(19, bottom_y),
            Vec2(9, bottom_y),
        ),
        "checkpoint_exit": (
            Vec2(9, bottom_y),
            exit_door.pos,
        ),
    }
    sections = []
    for section_id, entity_ids in COMBINED_SECTION_ENTITIES:
        entry, leave = section_edges[section_id]
        bounds = section_bounds[section_id]
        sections.append(
            {
                "id": section_id,
                "entities": list(entity_ids),
                "entry": list(entry),
                "exit": list(leave),
                "bounds": list(bounds),
            }
        )

    assert block.push_from is not None
    metadata = {
        "attempt": 0,
        "shape": RoomShape.RECTANGLE.value,
        "complexity": config.complexity,
        "gates": [
            {
                "edge": [0, 1],
                "kind": "key",
                "doors": ["door_key_0"],
                "triggers": ["key_agent_0"],
                "depth": 0,
            },
            {
                "edge": [1, 2],
                "kind": "key",
                "doors": ["door_key_1"],
                "triggers": ["key_agent_1"],
                "depth": 1,
            },
            {
                "edge": [2, 3],
                "kind": "hold_switch",
                "doors": ["door_crate"],
                "triggers": ["switch_crate"],
                "depth": 2,
            },
        ],
        "cooperative_gates": 1,
        "hazard_tiles": sum(
            is_hazard_tile(terrain[pos]) for pos in terrain.positions()
        ),
        "obstacle_tiles": terrain.count(Tile.OBSTACLE),
        "requested_normal_wipeout_balls": 1,
        "placed_normal_wipeout_balls": 1,
        "requested_big_wipeout_balls": 1,
        "placed_big_wipeout_balls": 1,
        "required_wipeout_ball_id": required_ball.id,
        "requested_temporary_bridges": 1,
        "placed_temporary_bridges": 1,
        "required_bridge_id": bridge.id,
        "requested_reset_zones": 1,
        "placed_reset_zones": 1,
        "required_reset_zone_id": reset.id,
        "requested_pushable_blocks": 1,
        "placed_pushable_blocks": 1,
        "crate_switch_pairs": (
            {
                "block": block.id,
                "switch": switch.id,
                "push_from": list(block.push_from),
            },
        ),
        "combined_course": {
            "version": 1,
            "sections": sections,
            "required_ball_size": required_size.value,
            "ball_barrier": [
                list(Vec2(x, y))
                for y in barrier_rows
                for x in range(23, course_right + 1)
            ],
            "bridge_hazard": int(bridge_hazard),
            "non_bridge_hazards": [list(pos) for pos in non_bridge_hazards],
        },
    }
    room = Room(
        seed=seed,
        config=config,
        terrain=terrain,
        entities=entities,
        topology=topology,
        shape=RoomShape.RECTANGLE,
        metadata=metadata,
    )
    room.metadata["combined_course"]["route_budget"] = combined_route_budget(room)
    return room


def _topology(
    terrain: Grid,
    bridge_tiles: tuple[Vec2, ...],
    door_positions: tuple[Vec2, ...],
    bottom_top: int,
) -> RoomTopology:
    groups = (
        {
            Vec2(x, y)
            for x in range(1, 4)
            for y in range(1, 10)
        },
        {
            Vec2(x, y)
            for x in range(5, 8)
            for y in range(1, 10)
        },
        {
            Vec2(x, y)
            for x in range(9, 14)
            for y in range(1, 10)
        },
        {
            *(
                Vec2(x, y)
                for x in range(15, 36)
                for y in range(1, 10)
            ),
            *(
                Vec2(x, y)
                for x in range(23, 36)
                for y in range(10, bottom_top)
            ),
            *(
                Vec2(x, y)
                for x in range(21, 36)
                for y in range(bottom_top, bottom_top + 7)
            ),
        },
        {
            Vec2(x, y)
            for x in range(11, 20)
            for y in range(bottom_top, bottom_top + 7)
        },
        {
            Vec2(x, y)
            for x in range(1, 10)
            for y in range(bottom_top, bottom_top + 7)
        },
    )
    regions = {
        index: Region(
            index,
            frozenset(
                pos
                for pos in candidates
                if terrain.in_bounds(pos) and is_walkable(terrain[pos])
            ),
        )
        for index, candidates in enumerate(groups)
    }
    graph: Graph[int] = Graph()
    for index in range(len(regions) - 1):
        graph.add_edge(index, index + 1)
    return RoomTopology(
        regions=regions,
        graph=graph,
        portals={
            portal_key(0, 1): (door_positions[0],),
            portal_key(1, 2): (door_positions[1],),
            portal_key(2, 3): (door_positions[2],),
            portal_key(3, 4): bridge_tiles,
            portal_key(4, 5): tuple(
                Vec2(10, y)
                for y in range(bottom_top, bottom_top + 7)
                if is_walkable(terrain[Vec2(10, y)])
            ),
        },
        spawn_regions=(0, 0),
        exit_region=5,
        depths={index: index for index in regions},
    )


def is_hazard_tile(value: int) -> bool:
    return Tile(value) in HAZARD_TILES

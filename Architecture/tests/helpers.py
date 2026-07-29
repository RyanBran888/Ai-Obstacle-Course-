"""Hand-built rooms for tests that need an exact, known structure.

The generator is designed never to emit a broken room, so the negative cases
(key sealed behind the door it opens, spawn inside a wall, ...) have to be
constructed by hand to prove the validator actually catches them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coop_env import GenerationConfig  # noqa: E402
from coop_env.entities import (  # noqa: E402
    AgentSpawn,
    Entity,
    ExitDoor,
    Key,
    LockedDoor,
    Switch,
    SwitchMode,
)
from coop_env.requirements import (  # noqa: E402
    AlwaysOpen,
    KeyRequirement,
    Requirement,
    SwitchRequirement,
    TriggerMode,
)
from coop_env.room import Region, Room, RoomTopology, portal_key  # noqa: E402
from coop_env.tiles import Tile  # noqa: E402
from coop_env.utils.geometry import Rect, Vec2  # noqa: E402
from coop_env.utils.graph import Graph  # noqa: E402
from coop_env.utils.grid import Grid  # noqa: E402

WIDTH, HEIGHT = 17, 11
DIVIDER_X = 8
DOORWAY = Vec2(DIVIDER_X, 5)

LEFT_TILE = Vec2(3, 3)
LEFT_TILE_B = Vec2(3, 7)
RIGHT_TILE = Vec2(12, 3)
RIGHT_TILE_B = Vec2(12, 7)
EXIT_TILE = Vec2(14, 5)


def two_region_room(
    *,
    door_requirement: Requirement | None = None,
    exit_requirement: Requirement | None = None,
    latching: bool = True,
    extra_entities: tuple[Entity, ...] = (),
    spawns: tuple[Vec2, Vec2] = (Vec2(2, 2), Vec2(2, 8)),
    config: GenerationConfig | None = None,
) -> Room:
    """A 17x11 box split by a wall, joined by one doorway holding a door.

    Left region is region 0, right region is region 1, the doorway sits between
    them. Callers decide what the door and the exit require, and where the
    triggers live.
    """
    terrain = Grid(WIDTH, HEIGHT, Tile.WALL)
    for pos in Rect(1, 1, WIDTH - 2, HEIGHT - 2).positions():
        terrain[pos] = Tile.FLOOR
    for y in range(1, HEIGHT - 1):
        terrain[Vec2(DIVIDER_X, y)] = Tile.WALL
    terrain[DOORWAY] = Tile.FLOOR

    left = frozenset(
        Vec2(x, y) for y in range(1, HEIGHT - 1) for x in range(1, DIVIDER_X)
    )
    right = frozenset(
        Vec2(x, y) for y in range(1, HEIGHT - 1) for x in range(DIVIDER_X + 1, WIDTH - 1)
    )

    entities: list[Entity] = [
        AgentSpawn(id="spawn_0", pos=spawns[0], index=0),
        AgentSpawn(id="spawn_1", pos=spawns[1], index=1),
        LockedDoor(
            id="door_0",
            pos=DOORWAY,
            requirement=door_requirement or AlwaysOpen(),
            latching=latching,
            horizontal=False,
            region_a=0,
            region_b=1,
        ),
        ExitDoor(id="exit", pos=EXIT_TILE, requirement=exit_requirement or AlwaysOpen()),
        *extra_entities,
    ]

    graph: Graph[int] = Graph()
    graph.add_edge(0, 1)
    topology = RoomTopology(
        regions={0: Region(0, left), 1: Region(1, right)},
        graph=graph,
        portals={portal_key(0, 1): (DOORWAY,)},
        spawn_regions=(0, 0),
        exit_region=1,
        depths={0: 0, 1: 1},
    )
    return Room(
        seed=1,
        config=config or GenerationConfig(),
        terrain=terrain,
        entities=tuple(entities),
        topology=topology,
    )


def solvable_key_room() -> Room:
    """Key on the near side of the door it opens -- completable."""
    return two_region_room(
        door_requirement=KeyRequirement(("key_0",)),
        extra_entities=(Key(id="key_0", pos=LEFT_TILE, opens=("door_0",)),),
    )


def sealed_key_room() -> Room:
    """Key on the far side of the door it opens -- not completable."""
    return two_region_room(
        door_requirement=KeyRequirement(("key_0",)),
        extra_entities=(Key(id="key_0", pos=RIGHT_TILE, opens=("door_0",)),),
    )


def paired_lever_room() -> Room:
    """Door needs two hold-levers held at the same instant: needs both slots.

    The door latches, so once the pair triggers it, it stays open and neither
    agent ends up stranded.
    """
    return two_region_room(
        door_requirement=SwitchRequirement(
            ("switch_0", "switch_1"), TriggerMode.SIMULTANEOUS
        ),
        extra_entities=(
            Switch(id="switch_0", pos=LEFT_TILE, mode=SwitchMode.HOLD,
                   group="pair", controls=("door_0",)),
            Switch(id="switch_1", pos=LEFT_TILE_B, mode=SwitchMode.HOLD,
                   group="pair", controls=("door_0",)),
        ),
    )


def hold_switch_room() -> Room:
    """Non-latching door held open by a lever on the near side: one-way."""
    return two_region_room(
        door_requirement=SwitchRequirement(("switch_0",)),
        latching=False,
        extra_entities=(
            Switch(id="switch_0", pos=LEFT_TILE, mode=SwitchMode.HOLD, controls=("door_0",)),
        ),
    )

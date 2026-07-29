"""Static terrain vocabulary.

Terrain is everything that never changes during an episode: void, floor, walls,
static obstacles, and hazard surfaces. Anything with mutable state (doors,
switches, keys, blocks) is an *entity* and lives in `entities.py`.
Keeping that split means the terrain grid can be handed to a future observation
encoder as a plain integer array with no special cases.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


class Tile(IntEnum):
    """Terrain tile ids. Values are stable and safe to serialise."""

    VOID = 0          # outside the room silhouette; never enterable
    FLOOR = 1
    WALL = 2
    OBSTACLE = 3      # static blocker inside the play area (rubble, pillar, crate)
    HAZARD_LAVA = 4
    HAZARD_SPIKES = 5
    HAZARD_WATER = 6
    HAZARD_PIT = 7


class TileProperties(NamedTuple):
    """Declarative description of how a terrain tile behaves.

    These flags describe the *world*, not any policy for dealing with it. The
    validator reads `walkable` to decide whether a route can exist; a future
    training stack can read `lethal`/`resets_progress` to build its own
    termination rules.
    """

    name: str
    walkable: bool
    blocks_sight: bool
    is_hazard: bool
    lethal: bool
    resets_progress: bool
    glyph: str


_PROPERTIES: dict[Tile, TileProperties] = {
    Tile.VOID: TileProperties("void", False, True, False, False, False, " "),
    Tile.FLOOR: TileProperties("floor", True, False, False, False, False, "."),
    Tile.WALL: TileProperties("wall", False, True, False, False, False, "#"),
    Tile.OBSTACLE: TileProperties("obstacle", False, True, False, False, False, "@"),
    Tile.HAZARD_LAVA: TileProperties("lava", False, False, True, True, False, "~"),
    Tile.HAZARD_SPIKES: TileProperties("spikes", False, False, True, True, False, "^"),
    Tile.HAZARD_WATER: TileProperties("water", False, False, True, False, True, "="),
    Tile.HAZARD_PIT: TileProperties("pit", False, False, True, False, True, "v"),
}

HAZARD_TILES: tuple[Tile, ...] = (
    Tile.HAZARD_LAVA,
    Tile.HAZARD_SPIKES,
    Tile.HAZARD_WATER,
    Tile.HAZARD_PIT,
)


def properties(tile: Tile | int) -> TileProperties:
    return _PROPERTIES[Tile(tile)]


def is_walkable(tile: Tile | int) -> bool:
    """True when terrain alone permits occupying the tile.

    Hazards are *not* walkable. That is deliberately conservative: the validator
    proves solvability using only hazard-free routes, so a room is never
    accepted on the assumption that something can survive crossing lava.
    """
    return _PROPERTIES[Tile(tile)].walkable


def is_hazard(tile: Tile | int) -> bool:
    return _PROPERTIES[Tile(tile)].is_hazard


def is_lethal(tile: Tile | int) -> bool:
    return _PROPERTIES[Tile(tile)].lethal


def glyph(tile: Tile | int) -> str:
    return _PROPERTIES[Tile(tile)].glyph


def tile_name(tile: Tile | int) -> str:
    return _PROPERTIES[Tile(tile)].name

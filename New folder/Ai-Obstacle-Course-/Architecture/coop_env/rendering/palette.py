"""Colours and glyphs for the inspection renderers.

Rendering is kept strictly downstream of generation: nothing in this package is
imported by the generator or the validator, and a room carries no visual
information of its own. Swapping in a different renderer means writing one
module against `Room` and `EpisodeState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..entities import EntityKind
from ..tiles import Tile


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    background: str
    panel: str
    text: str
    muted: str
    grid: str
    tiles: dict[Tile, str]
    entities: dict[EntityKind, str]
    accents: dict[str, str] = field(default_factory=dict)

    def tile_color(self, tile: Tile | int) -> str:
        return self.tiles.get(Tile(tile), "#ff00ff")

    def entity_color(self, kind: EntityKind) -> str:
        return self.entities.get(kind, "#ff00ff")


DARK = Theme(
    name="dark",
    background="#0d1017",
    panel="#151a24",
    text="#e6ebf5",
    muted="#8a94a8",
    grid="#2a3244",
    tiles={
        Tile.VOID: "#0a0c11",
        # Floor sits well below wall in luminance so the layout reads at a
        # glance; pit is kept distinct from void so holes are not mistaken for
        # the space outside the room.
        Tile.FLOOR: "#1c2230",
        Tile.WALL: "#737e96",
        Tile.OBSTACLE: "#7d6a52",
        Tile.HAZARD_LAVA: "#d94f2b",
        Tile.HAZARD_SPIKES: "#a8415c",
        Tile.HAZARD_WATER: "#2b6c9e",
        Tile.HAZARD_PIT: "#12161f",
    },
    entities={
        EntityKind.AGENT_SPAWN: "#7fb2ff",
        EntityKind.EXIT_DOOR: "#57e39b",
        EntityKind.KEY: "#ffcc4d",
        EntityKind.LOCKED_DOOR: "#c9a227",
        EntityKind.SWITCH: "#4fd6e8",
        EntityKind.MOVING_PLATFORM: "#9ed35a",
        EntityKind.PUSHABLE_BLOCK: "#b98a5e",
        EntityKind.CHECKPOINT: "#dfe4ee",
        EntityKind.RESET_ZONE: "#6b5ce0",
        EntityKind.TEMPORARY_BRIDGE: "#77c6a8",
    },
    accents={
        "lock_key": "#ffcc4d",
        "lock_switch": "#4fd6e8",
        "lock_paired": "#e368c8",
        "lock_hold": "#ff9a3c",
        "lock_timed": "#ff7a7a",
        "open": "#3d8a63",
    },
)

LIGHT = Theme(
    name="light",
    background="#f4f6fa",
    panel="#ffffff",
    text="#1a1f2b",
    muted="#5c667a",
    grid="#dfe4ee",
    tiles={
        Tile.VOID: "#f4f6fa",
        Tile.FLOOR: "#e5e9f2",
        Tile.WALL: "#8e97ab",
        Tile.OBSTACLE: "#b39a76",
        Tile.HAZARD_LAVA: "#e8623c",
        Tile.HAZARD_SPIKES: "#c05374",
        Tile.HAZARD_WATER: "#4d92c4",
        Tile.HAZARD_PIT: "#3c4451",
    },
    entities=DARK.entities,
    accents=DARK.accents,
)

THEMES: dict[str, Theme] = {"dark": DARK, "light": LIGHT}


def get_theme(name: str | Theme = "dark") -> Theme:
    if isinstance(name, Theme):
        return name
    return THEMES.get(name.lower(), DARK)


#: Single characters used by the ASCII renderer for entities. Terrain glyphs
#: live with the tile definitions in `coop_env.tiles`.
ENTITY_GLYPHS: dict[EntityKind, str] = {
    EntityKind.AGENT_SPAWN: "1",
    EntityKind.EXIT_DOOR: "E",
    EntityKind.KEY: "k",
    EntityKind.LOCKED_DOOR: "D",
    EntityKind.SWITCH: "S",
    EntityKind.MOVING_PLATFORM: "P",
    EntityKind.PUSHABLE_BLOCK: "B",
    EntityKind.CHECKPOINT: "C",
    EntityKind.RESET_ZONE: ",",
    EntityKind.TEMPORARY_BRIDGE: "-",
}

#: Draw order -- later kinds win the tile when several share one.
ENTITY_DRAW_ORDER: tuple[EntityKind, ...] = (
    EntityKind.RESET_ZONE,
    EntityKind.TEMPORARY_BRIDGE,
    EntityKind.MOVING_PLATFORM,
    EntityKind.CHECKPOINT,
    EntityKind.KEY,
    EntityKind.SWITCH,
    EntityKind.PUSHABLE_BLOCK,
    EntityKind.LOCKED_DOOR,
    EntityKind.EXIT_DOOR,
    EntityKind.AGENT_SPAWN,
)

"""The immutable room blueprint produced by the generator.

A `Room` is a finished, validated level: terrain, entities, and the region
topology that was used to build it. It never changes. Everything that varies
during an episode lives in `EpisodeState`, which is constructed from a room and
can be thrown away and rebuilt at zero cost.

The topology is kept alongside the room for rendering, analytics, and
debugging. The validator deliberately does *not* trust it -- it rebuilds the
connectivity graph from terrain and entities so that its verdict is an
independent check on the generator rather than a restatement of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

from .config import GenerationConfig, RoomShape
from .entities import (
    AgentSpawn,
    Checkpoint,
    Entity,
    EntityKind,
    ExitDoor,
    Key,
    LockedDoor,
    MovingPlatform,
    PushableBlock,
    ResetZone,
    Switch,
    TemporaryBridge,
)
from .tiles import Tile, is_walkable
from .utils.geometry import Rect, Vec2
from .utils.graph import Graph
from .utils.grid import Grid

E = TypeVar("E", bound=Entity)

#: Portal keys are sorted region-id pairs so lookups are order independent.
PortalKey = tuple[int, int]


def portal_key(a: int, b: int) -> PortalKey:
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True, slots=True)
class Region:
    """One connected area of walkable floor."""

    id: int
    tiles: frozenset[Vec2]

    @property
    def size(self) -> int:
        return len(self.tiles)

    @property
    def bounds(self) -> Rect:
        xs = [p[0] for p in self.tiles]
        ys = [p[1] for p in self.tiles]
        return Rect(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)

    @property
    def centroid(self) -> Vec2:
        n = len(self.tiles)
        return Vec2(
            sum(p[0] for p in self.tiles) // n,
            sum(p[1] for p in self.tiles) // n,
        )

    def sorted_tiles(self) -> list[Vec2]:
        """Deterministic tile ordering, for reproducible placement draws."""
        return sorted(self.tiles, key=lambda p: (p[1], p[0]))


@dataclass(frozen=True, slots=True)
class RoomTopology:
    """How the room's regions connect, as understood by the generator."""

    regions: Mapping[int, Region]
    graph: Graph[int]
    portals: Mapping[PortalKey, tuple[Vec2, ...]]
    spawn_regions: tuple[int, ...]
    exit_region: int
    depths: Mapping[int, int]
    platform_links: tuple[PortalKey, ...] = ()

    def region_of(self, pos: Vec2) -> int | None:
        for rid, region in self.regions.items():
            if pos in region.tiles:
                return rid
        return None

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def max_depth(self) -> int:
        return max(self.depths.values(), default=0)


@dataclass(frozen=True, slots=True)
class Room:
    """A generated, validated level blueprint."""

    seed: int
    config: GenerationConfig
    terrain: Grid
    entities: tuple[Entity, ...]
    topology: RoomTopology
    shape: RoomShape = RoomShape.RECTANGLE
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- cached lookups ----------------------------------------------------

    def __post_init__(self) -> None:
        by_id: dict[str, Entity] = {}
        by_pos: dict[Vec2, list[Entity]] = {}
        for entity in self.entities:
            if entity.id in by_id:
                raise ValueError(f"duplicate entity id {entity.id!r}")
            by_id[entity.id] = entity
            for tile in entity.footprint():
                by_pos.setdefault(tile, []).append(entity)
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(self, "_by_pos", by_pos)

    # dataclass(slots=True) needs the cache attributes declared
    _by_id: dict[str, Entity] = field(default_factory=dict, init=False, repr=False, compare=False)
    _by_pos: dict[Vec2, list[Entity]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # -- dimensions --------------------------------------------------------

    @property
    def width(self) -> int:
        return self.terrain.width

    @property
    def height(self) -> int:
        return self.terrain.height

    @property
    def bounds(self) -> Rect:
        return self.terrain.bounds

    def terrain_at(self, pos: Vec2) -> Tile:
        return Tile(self.terrain.get(pos, Tile.VOID))

    def is_terrain_walkable(self, pos: Vec2) -> bool:
        return self.terrain.in_bounds(pos) and is_walkable(self.terrain[pos])

    def walkable_tiles(self) -> list[Vec2]:
        return [p for p in self.terrain.positions() if is_walkable(self.terrain[p])]

    # -- entity access -----------------------------------------------------

    def entity(self, entity_id: str) -> Entity:
        return self._by_id[entity_id]

    def find(self, entity_id: str) -> Entity | None:
        return self._by_id.get(entity_id)

    def entities_at(self, pos: Vec2) -> tuple[Entity, ...]:
        return tuple(self._by_pos.get(pos, ()))

    def of_type(self, entity_type: type[E]) -> tuple[E, ...]:
        return tuple(e for e in self.entities if isinstance(e, entity_type))

    def of_kind(self, kind: EntityKind) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.kind is kind)

    @property
    def spawns(self) -> tuple[AgentSpawn, ...]:
        return tuple(sorted(self.of_type(AgentSpawn), key=lambda s: s.index))

    @property
    def exit(self) -> ExitDoor:
        exits = self.of_type(ExitDoor)
        if not exits:
            raise ValueError("room has no exit door")
        return exits[0]

    @property
    def keys(self) -> tuple[Key, ...]:
        return self.of_type(Key)

    @property
    def doors(self) -> tuple[LockedDoor, ...]:
        return self.of_type(LockedDoor)

    @property
    def switches(self) -> tuple[Switch, ...]:
        return self.of_type(Switch)

    @property
    def platforms(self) -> tuple[MovingPlatform, ...]:
        return self.of_type(MovingPlatform)

    @property
    def blocks(self) -> tuple[PushableBlock, ...]:
        return self.of_type(PushableBlock)

    @property
    def checkpoints(self) -> tuple[Checkpoint, ...]:
        return self.of_type(Checkpoint)

    @property
    def reset_zones(self) -> tuple[ResetZone, ...]:
        return self.of_type(ResetZone)

    @property
    def bridges(self) -> tuple[TemporaryBridge, ...]:
        return self.of_type(TemporaryBridge)

    # -- reporting ---------------------------------------------------------

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for entity in self.entities:
            name = entity.kind.name.lower()
            tally[name] = tally.get(name, 0) + 1
        return dict(sorted(tally.items()))

    def tile_histogram(self) -> dict[str, int]:
        from .tiles import tile_name

        tally: dict[str, int] = {}
        for tile in Tile:
            count = self.terrain.count(tile)
            if count:
                tally[tile_name(tile)] = count
        return tally

    def summary(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in self.counts().items())
        return (
            f"Room(seed={self.seed}, {self.width}x{self.height}, shape={self.shape.value}, "
            f"regions={self.topology.region_count}, depth={self.topology.max_depth}, {counts})"
        )

    def __repr__(self) -> str:
        return self.summary()


def iter_positions(entities: Iterable[Entity]) -> Iterator[Vec2]:
    for entity in entities:
        yield from entity.footprint()


def occupied_positions(entities: Sequence[Entity]) -> set[Vec2]:
    return set(iter_positions(entities))

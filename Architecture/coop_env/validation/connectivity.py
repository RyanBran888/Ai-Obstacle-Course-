"""Rebuild a room's connectivity model from terrain and entities alone.

The generator already knows how it wired the room together, but a check that
reads the generator's own notes proves nothing. This module reconstructs the
picture from scratch -- floor tiles, door positions, platform tracks -- so the
verdict in `solvability.py` is an independent second opinion.

Model: regions are patches of floor separated by doors. Doors become their own
nodes, so passing from one region to another means entering a door node, which
is only possible when that door's requirement is met.

This is static structural analysis of the map. It computes what *could* be
connected under what conditions; it never produces a route, a plan, or a
sequence of moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..entities import LockedDoor, MovingPlatform, PushableBlock, TemporaryBridge
from ..requirements import Requirement, combine
from ..room import Room
from ..tiles import is_walkable
from ..utils.geometry import Vec2
from ..utils.grid import connected_components


@dataclass(frozen=True, slots=True)
class DoorCluster:
    """One or more touching door tiles, treated as a single passage.

    Touching doors must all be open to pass, so their requirements combine.
    """

    id: str
    door_ids: tuple[str, ...]
    tiles: tuple[Vec2, ...]
    regions: tuple[int, ...]
    requirement: Requirement
    latching: bool
    """False when any door in the cluster closes as soon as its trigger is released."""


@dataclass(slots=True)
class ConnectivityModel:
    """Regions, the doors between them, and where every entity lives."""

    regions: dict[int, frozenset[Vec2]] = field(default_factory=dict)
    tile_region: dict[Vec2, int] = field(default_factory=dict)
    free_links: set[tuple[int, int]] = field(default_factory=set)
    clusters: dict[str, DoorCluster] = field(default_factory=dict)
    entity_region: dict[str, int] = field(default_factory=dict)
    unreachable_entities: tuple[str, ...] = ()

    def region_of_tile(self, pos: Vec2) -> int | None:
        return self.tile_region.get(pos)

    def region_of_entity(self, entity_id: str) -> int | None:
        return self.entity_region.get(entity_id)

    def neighbours_of(self, region: int) -> set[int]:
        out: set[int] = set()
        for a, b in self.free_links:
            if a == region:
                out.add(b)
            elif b == region:
                out.add(a)
        return out

    def clusters_touching(self, region: int) -> list[DoorCluster]:
        return [c for c in self.clusters.values() if region in c.regions]


def build_connectivity(room: Room) -> ConnectivityModel:
    """Derive the region/door model from the room blueprint."""
    terrain = room.terrain
    door_tiles: dict[Vec2, LockedDoor] = {}
    for door in room.doors:
        door_tiles[door.pos] = door

    # Crates are treated as solid. A crate can in principle be pushed aside,
    # but assuming so would let an unsolvable room slip through, so the
    # pessimistic reading is the safe one here.
    blocked = {b.pos for b in room.of_type(PushableBlock)}

    base = {
        p
        for p in terrain.positions()
        if is_walkable(terrain[p]) and p not in door_tiles and p not in blocked
    }
    components = connected_components(base, lambda p: p in base, terrain.bounds)

    regions: dict[int, frozenset[Vec2]] = {}
    tile_region: dict[Vec2, int] = {}
    for index, component in enumerate(components):
        regions[index] = frozenset(component)
        for pos in component:
            tile_region[pos] = index

    clusters = _build_door_clusters(door_tiles, tile_region)
    free_links = _platform_links(room, tile_region)

    entity_region: dict[str, int] = {}
    unreachable: list[str] = []
    for entity in room.entities:
        if isinstance(entity, LockedDoor):
            continue
        rid = tile_region.get(entity.pos)
        if rid is None and isinstance(entity, PushableBlock):
            # A crate blocks its own tile, so it is never in `base`. It still
            # belongs to whatever region surrounds it.
            for neighbour in entity.pos.neighbors4():
                rid = tile_region.get(neighbour)
                if rid is not None:
                    break
        if rid is None:
            # platforms and bridges sit on hazard tiles by design
            if isinstance(entity, (MovingPlatform, TemporaryBridge)):
                continue
            unreachable.append(entity.id)
            continue
        entity_region[entity.id] = rid

    return ConnectivityModel(
        regions=regions,
        tile_region=tile_region,
        free_links=free_links,
        clusters=clusters,
        entity_region=entity_region,
        unreachable_entities=tuple(sorted(unreachable)),
    )


def _build_door_clusters(
    door_tiles: dict[Vec2, LockedDoor], tile_region: dict[Vec2, int]
) -> dict[str, DoorCluster]:
    remaining = set(door_tiles)
    clusters: dict[str, DoorCluster] = {}
    index = 0
    while remaining:
        seed = min(remaining, key=lambda p: (p[1], p[0]))
        stack = [seed]
        remaining.discard(seed)
        group = [seed]
        while stack:
            current = stack.pop()
            for n in current.neighbors4():
                if n in remaining:
                    remaining.discard(n)
                    group.append(n)
                    stack.append(n)
        group.sort(key=lambda p: (p[1], p[0]))
        doors = [door_tiles[t] for t in group]
        touching: set[int] = set()
        for tile in group:
            for n in tile.neighbors4():
                rid = tile_region.get(n)
                if rid is not None:
                    touching.add(rid)
        cluster_id = f"cluster_{index}"
        index += 1
        clusters[cluster_id] = DoorCluster(
            id=cluster_id,
            door_ids=tuple(d.id for d in doors),
            tiles=tuple(group),
            regions=tuple(sorted(touching)),
            requirement=combine([d.requirement for d in doors]),
            latching=all(d.latching for d in doors),
        )
    return clusters


def _platform_links(room: Room, tile_region: dict[Vec2, int]) -> set[tuple[int, int]]:
    """Regions joined by a platform track or a temporary bridge.

    A platform on a fixed cycle always returns, and a temporary bridge always
    comes back, so both count as unconditional links between whatever they
    touch. What they do *not* do is open a locked door.
    """
    links: set[tuple[int, int]] = set()
    spans: list[tuple[Vec2, ...]] = []
    for platform in room.platforms:
        spans.append(platform.footprint())
    for bridge in room.bridges:
        if bridge.on_ticks > 0:
            spans.append(bridge.footprint())

    for span in spans:
        touching: set[int] = set()
        for tile in span:
            for n in tile.neighbors4():
                rid = tile_region.get(n)
                if rid is not None:
                    touching.add(rid)
        ordered = sorted(touching)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                links.add((a, b))
    return links

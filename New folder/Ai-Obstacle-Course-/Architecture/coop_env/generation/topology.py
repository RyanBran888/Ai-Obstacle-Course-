"""Stage 5: derive the region graph from finished terrain.

A *region* is a connected patch of floor that you cannot leave without passing
through a doorway. Regions are computed from the terrain itself rather than
carried over from the BSP leaves, so obstacles and hazard pools that reshaped
an area are reflected honestly.

The resulting graph is what the puzzle layer gates: nodes are regions, edges are
doorways (or platform crossings), and locking an edge is what makes part of the
room conditional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..room import PortalKey, Region, portal_key
from ..tiles import Tile, is_walkable
from ..utils.geometry import Vec2
from ..utils.graph import Graph
from ..utils.grid import connected_components
from .layout import Layout
from .terrain import Decoration, PlatformTrack


@dataclass(slots=True)
class Topology:
    regions: dict[int, Region]
    graph: Graph[int]
    portals: dict[PortalKey, tuple[Vec2, ...]]
    """Doorway tiles per region pair -- the only places a door may be installed."""
    platform_links: dict[PortalKey, PlatformTrack] = field(default_factory=dict)
    tile_region: dict[Vec2, int] = field(default_factory=dict)

    def region_of(self, pos: Vec2) -> int | None:
        return self.tile_region.get(pos)

    def is_connected(self) -> bool:
        return self.graph.is_connected()

    def gateable_edges(self) -> list[PortalKey]:
        """Region pairs joined by a real doorway (platform links can't hold a door)."""
        return sorted(self.portals)


def build_topology(layout: Layout, decoration: Decoration) -> Topology:
    terrain = layout.terrain
    doorway_tiles = {
        p for p in layout.doorways if terrain.get(p, Tile.VOID) == Tile.FLOOR
    }
    walkable = {p for p in terrain.positions() if is_walkable(terrain[p])}
    body = walkable - doorway_tiles

    components = connected_components(body, lambda p: p in body, terrain.bounds)
    tile_region: dict[Vec2, int] = {}
    region_tiles: dict[int, set[Vec2]] = {}
    for index, component in enumerate(components):
        region_tiles[index] = set(component)
        for pos in component:
            tile_region[pos] = index

    clusters = _doorway_clusters(doorway_tiles, terrain.bounds)
    portals: dict[PortalKey, list[Vec2]] = {}

    for cluster in clusters:
        touching: dict[int, list[Vec2]] = {}
        for tile in cluster:
            for neighbour in tile.neighbors4():
                rid = tile_region.get(neighbour)
                if rid is not None:
                    touching.setdefault(rid, []).append(tile)
        region_ids = sorted(touching)
        if len(region_ids) < 2:
            # a stub doorway that leads nowhere -- fold it into its own region
            if region_ids:
                rid = region_ids[0]
                region_tiles[rid].update(cluster)
                for tile in cluster:
                    tile_region[tile] = rid
            continue
        for i, a in enumerate(region_ids):
            for b in region_ids[i + 1 :]:
                key = portal_key(a, b)
                shared = sorted(
                    set(touching[a]) & set(touching[b]) or set(cluster),
                    key=lambda p: (p[1], p[0]),
                )
                portals.setdefault(key, []).extend(shared)

    graph: Graph[int] = Graph()
    for rid in region_tiles:
        graph.add_node(rid)
    for (a, b) in portals:
        graph.add_edge(a, b)

    platform_links: dict[PortalKey, PlatformTrack] = {}
    for track in decoration.tracks:
        if not track.bridges:
            continue
        ra = tile_region.get(track.dock_a)
        rb = tile_region.get(track.dock_b)
        if ra is None or rb is None or ra == rb:
            continue
        key = portal_key(ra, rb)
        platform_links[key] = track
        graph.add_edge(ra, rb)

    regions = {
        rid: Region(id=rid, tiles=frozenset(tiles))
        for rid, tiles in sorted(region_tiles.items())
    }
    deduped = {
        key: tuple(sorted(set(tiles), key=lambda p: (p[1], p[0])))
        for key, tiles in sorted(portals.items())
    }
    return Topology(
        regions=regions,
        graph=graph,
        portals=deduped,
        platform_links=platform_links,
        tile_region=tile_region,
    )


def _doorway_clusters(tiles: set[Vec2], bounds) -> list[list[Vec2]]:
    """Group touching doorway tiles so a widened corridor counts as one link."""
    remaining = set(tiles)
    clusters: list[list[Vec2]] = []
    while remaining:
        seed = min(remaining, key=lambda p: (p[1], p[0]))
        stack = [seed]
        remaining.discard(seed)
        cluster = [seed]
        while stack:
            current = stack.pop()
            for n in current.neighbors4():
                if n in remaining:
                    remaining.discard(n)
                    cluster.append(n)
                    stack.append(n)
        clusters.append(sorted(cluster, key=lambda p: (p[1], p[0])))
    return clusters

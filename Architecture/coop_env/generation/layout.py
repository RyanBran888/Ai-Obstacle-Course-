"""Stage 3: turn a silhouette and a partition into a walled floor plan.

Output of this stage is a terrain grid of VOID / FLOOR / WALL plus the set of
doorway tiles that join sub-areas. Those doorways are the only places a locked
door can later be installed, so this is where the room's *possible* puzzle
topology is decided -- though no mechanic exists yet at this point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import GenerationConfig, RoomShape
from ..rng import SeededRandom
from ..tiles import Tile
from ..utils.geometry import Rect, Vec2
from ..utils.grid import Grid, largest_component
from .partition import Divider, partition_area
from .shapes import build_silhouette

#: A doorway candidate: the wall tile, and the two floor tiles it joins.
@dataclass(frozen=True, slots=True)
class DoorwayCandidate:
    tile: Vec2
    side_a: Vec2
    side_b: Vec2
    leaf_a: int
    leaf_b: int
    vertical: bool
    """True when the wall line runs vertically (so the doorway is a gap in a column)."""


@dataclass(slots=True)
class Layout:
    terrain: Grid
    shape: RoomShape
    area: Rect
    leaf_labels: Grid
    doorways: dict[Vec2, DoorwayCandidate] = field(default_factory=dict)
    """Carved-open doorway tiles, keyed by position."""
    sealed: tuple[DoorwayCandidate, ...] = ()
    """Candidates that were left walled -- spare links for later mechanics."""

    def floor_tiles(self) -> set[Vec2]:
        return {p for p in self.terrain.positions() if self.terrain[p] == Tile.FLOOR}


class _UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {i: i for i in items}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def build_layout(config: GenerationConfig, rng: SeededRandom) -> Layout:
    """Run silhouette -> partition -> doorway carving and return the floor plan."""
    shape = rng.derive("shape").weighted_choice(config.shape_weights)
    width = rng.derive("size").in_range(config.width)
    height = rng.derive("size").in_range(config.height)

    terrain = Grid(width, height, Tile.VOID)
    area = Rect(1, 1, width - 2, height - 2)

    silhouette = build_silhouette(shape, area, rng.derive("silhouette"))
    for tile in silhouette:
        terrain[tile] = Tile.FLOOR

    target_leaves = rng.derive("regions").in_range(config.region_count)
    partition = partition_area(
        area, rng.derive("partition"), target_leaves, config.min_region_span
    )

    leaf_labels = _label_leaves(terrain, partition.leaves)
    divider_tiles = _paint_dividers(terrain, partition.dividers, leaf_labels)

    candidates = _collect_doorways(terrain, leaf_labels, divider_tiles, partition.dividers)
    opened, sealed = _carve_doorways(terrain, candidates, config, rng.derive("doorways"))

    _prune_islands(terrain, set(opened))
    _wrap_with_walls(terrain)

    # doorways may have been pruned along with their island
    opened = {p: c for p, c in opened.items() if terrain[p] == Tile.FLOOR}

    return Layout(
        terrain=terrain,
        shape=shape,
        area=area,
        leaf_labels=leaf_labels,
        doorways=opened,
        sealed=tuple(sealed),
    )


def _label_leaves(terrain: Grid, leaves: tuple[Rect, ...]) -> Grid:
    """Tag every floor tile with the index of the BSP leaf that contains it."""
    labels = Grid(terrain.width, terrain.height, -1)
    for index, rect in enumerate(leaves):
        for pos in rect.positions():
            if terrain.in_bounds(pos) and terrain[pos] == Tile.FLOOR:
                labels[pos] = index
    return labels


def _paint_dividers(
    terrain: Grid, dividers: tuple[Divider, ...], labels: Grid
) -> set[Vec2]:
    """Turn every divider line into wall, and return those tiles."""
    painted: set[Vec2] = set()
    for divider in dividers:
        for pos in divider.rect.positions():
            if terrain.in_bounds(pos) and terrain[pos] == Tile.FLOOR:
                terrain[pos] = Tile.WALL
                labels[pos] = -1
                painted.add(pos)
    return painted


def _collect_doorways(
    terrain: Grid,
    labels: Grid,
    divider_tiles: set[Vec2],
    dividers: tuple[Divider, ...],
) -> dict[tuple[int, int], list[DoorwayCandidate]]:
    """Find every wall tile that could become a doorway between two leaves."""
    vertical_lines = {
        pos for d in dividers if d.vertical for pos in d.rect.positions()
    }
    grouped: dict[tuple[int, int], list[DoorwayCandidate]] = {}
    for tile in sorted(divider_tiles, key=lambda p: (p[1], p[0])):
        vertical = tile in vertical_lines
        if vertical:
            a, b = Vec2(tile[0] - 1, tile[1]), Vec2(tile[0] + 1, tile[1])
        else:
            a, b = Vec2(tile[0], tile[1] - 1), Vec2(tile[0], tile[1] + 1)
        if terrain.get(a, Tile.VOID) != Tile.FLOOR or terrain.get(b, Tile.VOID) != Tile.FLOOR:
            continue
        label_a, label_b = labels[a], labels[b]
        if label_a < 0 or label_b < 0 or label_a == label_b:
            continue
        key = (label_a, label_b) if label_a < label_b else (label_b, label_a)
        grouped.setdefault(key, []).append(
            DoorwayCandidate(tile, a, b, label_a, label_b, vertical)
        )
    return grouped


def _carve_doorways(
    terrain: Grid,
    candidates: dict[tuple[int, int], list[DoorwayCandidate]],
    config: GenerationConfig,
    rng: SeededRandom,
) -> tuple[dict[Vec2, DoorwayCandidate], list[DoorwayCandidate]]:
    """Open a spanning set of doorways, plus extra links for branching routes.

    A spanning tree guarantees every sub-area is reachable; `branching_factor`
    then decides how many redundant links stay open, which is the difference
    between a single forced route and a layout with alternative paths that
    reconnect later.
    """
    if not candidates:
        return {}, []

    leaves = sorted({leaf for key in candidates for leaf in key})
    union = _UnionFind(leaves)
    edges = rng.shuffled(sorted(candidates))

    tree_edges: list[tuple[int, int]] = []
    extra_edges: list[tuple[int, int]] = []
    for edge in edges:
        if union.union(*edge):
            tree_edges.append(edge)
        else:
            extra_edges.append(edge)

    chosen = list(tree_edges)
    for edge in extra_edges:
        if rng.chance(config.branching_factor):
            chosen.append(edge)

    opened: dict[Vec2, DoorwayCandidate] = {}
    for edge in sorted(chosen):
        options = candidates[edge]
        pick = rng.choice(options)
        _open_tile(terrain, pick, opened)
        # a wider corridor reads better
        if rng.randint(*config.corridor_width) > 1:
            neighbours = [
                c
                for c in options
                if c.tile != pick.tile and c.tile.manhattan(pick.tile) == 1
            ]
            if neighbours:
                _open_tile(terrain, rng.choice(neighbours), opened)

    sealed = [
        c
        for edge, options in sorted(candidates.items())
        for c in options
        if c.tile not in opened
    ]
    return opened, sealed


def _open_tile(
    terrain: Grid, candidate: DoorwayCandidate, opened: dict[Vec2, DoorwayCandidate]
) -> None:
    terrain[candidate.tile] = Tile.FLOOR
    opened[candidate.tile] = candidate


def _prune_islands(terrain: Grid, doorway_tiles: set[Vec2]) -> None:
    """Wall off any floor that ended up cut away from the main body.

    Silhouettes with thin necks can leave a leaf touching the rest of the room
    only diagonally. Rather than fail, the stray floor becomes wall.
    """
    floors = {p for p in terrain.positions() if terrain[p] == Tile.FLOOR}
    if not floors:
        return
    main = largest_component(floors, lambda p: p in floors, terrain.bounds)
    for pos in floors - main:
        terrain[pos] = Tile.WALL


def _wrap_with_walls(terrain: Grid) -> None:
    """Give every floor tile a solid edge, so the room is sealed."""
    for pos in terrain.positions():
        if terrain[pos] != Tile.VOID:
            continue
        if any(
            terrain.get(n, Tile.VOID) == Tile.FLOOR for n in terrain.neighbors8(pos)
        ):
            terrain[pos] = Tile.WALL

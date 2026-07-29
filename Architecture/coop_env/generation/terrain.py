"""Stage 4: scatter obstacles and hazards.

Everything here is guarded: after each blob is written, the floor is re-checked
for connectivity and the blob is rolled back if it cut the room in two. That
keeps the invariant "the walkable floor is one piece" true right up until the
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..config import GenerationConfig
from ..rng import SeededRandom
from ..tiles import Tile
from ..utils.geometry import Vec2
from ..utils.grid import Grid
from .layout import Layout


@dataclass(slots=True)
class Decoration:
    hazard_tiles: set[Vec2] = field(default_factory=set)
    obstacle_tiles: set[Vec2] = field(default_factory=set)


def decorate_terrain(layout: Layout, config: GenerationConfig, rng: SeededRandom) -> Decoration:
    """Add static obstacles and hazard pools in place."""
    decoration = Decoration()
    protected = _protected_tiles(layout)
    tracker = _FloorTracker(layout.terrain)

    _scatter_obstacles(tracker, config, rng.derive("obstacles"), protected, decoration)
    _scatter_hazards(tracker, config, rng.derive("hazards"), protected, decoration)
    return decoration


def _protected_tiles(layout: Layout) -> set[Vec2]:
    """Doorway tiles and their approaches must stay clear."""
    protected: set[Vec2] = set()
    for tile, candidate in layout.doorways.items():
        protected.add(tile)
        protected.add(candidate.side_a)
        protected.add(candidate.side_b)
        for n in tile.neighbors4():
            protected.add(n)
    return protected


class _FloorTracker:
    """Keeps the set of open floor tiles in step with the grid.

    The connectivity guard runs after every blob, so rebuilding the floor set
    from the grid each time was the single most expensive thing in generation.
    Tracking it incrementally turns each check into one flood fill over a set.
    """

    __slots__ = ("terrain", "floor")

    def __init__(self, terrain: Grid) -> None:
        self.terrain = terrain
        self.floor: set[Vec2] = {
            p for p in terrain.positions() if terrain[p] == Tile.FLOOR
        }

    def is_connected(self) -> bool:
        if not self.floor:
            return False
        floor = self.floor
        start = next(iter(floor))
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for n in current.neighbors4():
                if n in floor and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen) == len(floor)

    def apply_guarded(self, blob: list[Vec2], value: Tile) -> bool:
        """Write a blob, rolling it back if the floor stops being one piece."""
        if not blob:
            return False
        previous = [(p, Tile(self.terrain[p])) for p in blob]
        for pos in blob:
            self.terrain[pos] = value
            self.floor.discard(pos)
        if self.is_connected():
            return True
        for pos, old in previous:
            self.terrain[pos] = old
            if old == Tile.FLOOR:
                self.floor.add(pos)
        return False

    def carve(self, tiles: Iterable[Vec2], value: Tile) -> list[tuple[Vec2, Tile]]:
        """Unguarded write, bypassing the connectivity guard.

        Returns the previous tile values so the caller can roll back.
        """
        previous: list[tuple[Vec2, Tile]] = []
        for pos in tiles:
            previous.append((pos, Tile(self.terrain[pos])))
            self.terrain[pos] = value
            self.floor.discard(pos)
        return previous

    def restore(self, previous: list[tuple[Vec2, Tile]]) -> None:
        for pos, old in previous:
            self.terrain[pos] = old
            if old == Tile.FLOOR:
                self.floor.add(pos)

    def components(self) -> list[set[Vec2]]:
        """4-connected components of the open floor."""
        remaining = set(self.floor)
        found: list[set[Vec2]] = []
        while remaining:
            start = next(iter(remaining))
            remaining.discard(start)
            group = {start}
            stack = [start]
            while stack:
                current = stack.pop()
                for n in current.neighbors4():
                    if n in remaining:
                        remaining.discard(n)
                        group.add(n)
                        stack.append(n)
            found.append(group)
        return found


def _grow_blob(
    terrain: Grid, seed: Vec2, size: int, rng: SeededRandom, protected: set[Vec2]
) -> list[Vec2]:
    """Random-walk outward from `seed` collecting up to `size` floor tiles."""
    blob = [seed]
    frontier = [seed]
    claimed = {seed}
    while len(blob) < size and frontier:
        current = rng.choice(frontier)
        options = [
            n
            for n in current.neighbors4()
            if n not in claimed
            and terrain.get(n, Tile.VOID) == Tile.FLOOR
            and n not in protected
        ]
        if not options:
            frontier.remove(current)
            continue
        pick = rng.choice(options)
        claimed.add(pick)
        blob.append(pick)
        frontier.append(pick)
    return blob


def _scatter(
    tracker: _FloorTracker,
    rng: SeededRandom,
    protected: set[Vec2],
    density: float,
    size_range: tuple[int, int],
    pick_tile: Callable[[], Tile],
    record: set[Vec2],
) -> None:
    """Grow blobs over open floor until `density` of it is consumed.

    Candidates are drawn from one pre-shuffled pass rather than re-scanning the
    grid, and each blob is applied through the connectivity guard.
    """
    terrain = tracker.terrain
    eligible = [p for p in tracker.floor if p not in protected]
    budget = int(len(eligible) * density)
    if budget <= 0 or not eligible:
        return

    placed = 0
    for seed in rng.shuffled(eligible):
        if placed >= budget:
            break
        if seed not in tracker.floor or seed in protected:
            continue
        size = min(rng.in_range(size_range), budget - placed)
        blob = _grow_blob(terrain, seed, max(1, size), rng, protected)
        if tracker.apply_guarded(blob, pick_tile()):
            record.update(blob)
            placed += len(blob)


def _scatter_obstacles(
    tracker: _FloorTracker,
    config: GenerationConfig,
    rng: SeededRandom,
    protected: set[Vec2],
    decoration: Decoration,
) -> None:
    _scatter(
        tracker,
        rng,
        protected,
        config.obstacle_density,
        (1, 3),
        lambda: Tile.OBSTACLE,
        decoration.obstacle_tiles,
    )


def _scatter_hazards(
    tracker: _FloorTracker,
    config: GenerationConfig,
    rng: SeededRandom,
    protected: set[Vec2],
    decoration: Decoration,
) -> None:
    _scatter(
        tracker,
        rng,
        protected,
        config.hazard_density,
        config.hazard_blob_size,
        lambda: rng.weighted_choice(config.hazard_weights),
        decoration.hazard_tiles,
    )

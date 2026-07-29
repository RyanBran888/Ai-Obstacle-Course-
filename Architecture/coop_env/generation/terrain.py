"""Stage 4: scatter obstacles, hazards, and platform crossings.

Everything here is guarded: after each blob is written, the floor is re-checked
for connectivity and the blob is rolled back if it cut the room in two. That
keeps the invariant "the walkable floor is one piece" true right up until the
platform-bridge pass, which is the one place a gap is opened *deliberately* --
and it always installs a platform track across it in the same step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..config import GenerationConfig
from ..rng import SeededRandom
from ..tiles import Tile
from ..utils.geometry import Vec2
from ..utils.grid import Grid
from .layout import DoorwayCandidate, Layout


@dataclass(frozen=True, slots=True)
class PlatformTrack:
    """A prepared route for a moving platform.

    `bridges` is True when the track is the only way across the gap it spans,
    which the topology stage turns into a graph edge.
    """

    path: tuple[Vec2, ...]
    dock_a: Vec2
    dock_b: Vec2
    bridges: bool


@dataclass(slots=True)
class Decoration:
    tracks: list[PlatformTrack] = field(default_factory=list)
    hazard_tiles: set[Vec2] = field(default_factory=set)
    obstacle_tiles: set[Vec2] = field(default_factory=set)
    removed_doorways: set[Vec2] = field(default_factory=set)


def decorate_terrain(layout: Layout, config: GenerationConfig, rng: SeededRandom) -> Decoration:
    """Add static obstacles, hazard pools, and platform crossings in place."""
    decoration = Decoration()
    protected = _protected_tiles(layout)
    tracker = _FloorTracker(layout.terrain)

    _scatter_obstacles(tracker, config, rng.derive("obstacles"), protected, decoration)
    _scatter_hazards(tracker, config, rng.derive("hazards"), protected, decoration)
    _open_platform_gaps(layout, tracker, config, rng.derive("bridges"), decoration)
    _add_hazard_platforms(layout, config, rng.derive("platforms"), decoration)
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
        """Unguarded write -- only for the deliberate platform gaps.

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


def _open_platform_gaps(
    layout: Layout,
    tracker: _FloorTracker,
    config: GenerationConfig,
    rng: SeededRandom,
    decoration: Decoration,
) -> None:
    """Sever selected doorways into hazard gaps spanned by a platform track.

    This is the only pass allowed to disconnect the floor, and it repays the
    debt immediately by recording a `PlatformTrack` that re-links the two
    sides. The topology stage treats that track as a graph edge.
    """
    if config.platform_bridge_probability <= 0 or not layout.doorways:
        return
    candidates = [layout.doorways[p] for p in sorted(layout.doorways)]
    for candidate in rng.shuffled(candidates):
        if not rng.chance(config.platform_bridge_probability):
            continue
        track = _try_open_gap(layout, tracker, candidate, rng)
        if track is not None:
            decoration.tracks.append(track)
            decoration.removed_doorways.add(candidate.tile)
            decoration.hazard_tiles.update(track.path)
            layout.doorways.pop(candidate.tile, None)


def _try_open_gap(
    layout: Layout,
    tracker: _FloorTracker,
    candidate: DoorwayCandidate,
    rng: SeededRandom,
) -> PlatformTrack | None:
    """Widen a doorway into a three-tile hazard gap with a dock on each side."""
    terrain = layout.terrain
    step = Vec2(1, 0) if candidate.vertical else Vec2(0, 1)
    gap = (candidate.tile - step, candidate.tile, candidate.tile + step)
    dock_a = candidate.tile - step.scaled(2)
    dock_b = candidate.tile + step.scaled(2)

    if any(terrain.get(p, Tile.VOID) != Tile.FLOOR for p in gap):
        return None
    if terrain.get(dock_a, Tile.VOID) != Tile.FLOOR:
        return None
    if terrain.get(dock_b, Tile.VOID) != Tile.FLOOR:
        return None
    # docks must not themselves be doorways, or the gap swallows two links
    if dock_a in layout.doorways or dock_b in layout.doorways:
        return None

    hazard = rng.choice((Tile.HAZARD_PIT, Tile.HAZARD_LAVA, Tile.HAZARD_WATER))
    before = len(tracker.components())
    previous = tracker.carve(gap, hazard)

    # The platform re-links exactly the two docks. If the gap severed anything
    # else -- a side passage that happened to run through these tiles -- the
    # room would end up with a region nothing can enter, so roll it back.
    after = tracker.components()
    home = {tile: index for index, group in enumerate(after) for tile in group}
    side_a, side_b = home.get(dock_a), home.get(dock_b)
    if side_a is None or side_b is None:
        tracker.restore(previous)
        return None
    effective = len(after) - (1 if side_a != side_b else 0)
    if effective != before:
        tracker.restore(previous)
        return None

    return PlatformTrack(path=tuple(gap), dock_a=dock_a, dock_b=dock_b, bridges=True)


def _add_hazard_platforms(
    layout: Layout,
    config: GenerationConfig,
    rng: SeededRandom,
    decoration: Decoration,
) -> None:
    """Lay decorative platform tracks along straight runs inside hazard pools.

    These do not change connectivity -- they are traversal flavour, and give a
    future policy something to time its movement against.
    """
    wanted = rng.in_range(config.num_moving_platforms) - len(decoration.tracks)
    if wanted <= 0:
        return
    terrain = layout.terrain
    hazards = sorted(
        (p for p in decoration.hazard_tiles if _is_hazard_tile(terrain, p)),
        key=lambda p: (p[1], p[0]),
    )
    used: set[Vec2] = set()
    for track in decoration.tracks:
        used.update(track.path)

    for seed in rng.shuffled(hazards):
        if wanted <= 0:
            break
        if seed in used:
            continue
        for direction in rng.shuffled([Vec2(1, 0), Vec2(0, 1)]):
            run = _straight_hazard_run(terrain, seed, direction, used)
            if len(run) < 2:
                continue
            dock_a = run[0] - direction
            dock_b = run[-1] + direction
            if terrain.get(dock_a, Tile.VOID) != Tile.FLOOR:
                dock_a = run[0]
            if terrain.get(dock_b, Tile.VOID) != Tile.FLOOR:
                dock_b = run[-1]
            decoration.tracks.append(
                PlatformTrack(tuple(run), dock_a, dock_b, bridges=False)
            )
            used.update(run)
            wanted -= 1
            break


def _is_hazard_tile(terrain: Grid, pos: Vec2) -> bool:
    from ..tiles import is_hazard

    return terrain.in_bounds(pos) and is_hazard(terrain[pos])


def _straight_hazard_run(
    terrain: Grid, start: Vec2, direction: Vec2, used: set[Vec2]
) -> list[Vec2]:
    run = [start]
    cursor = start + direction
    while _is_hazard_tile(terrain, cursor) and cursor not in used and len(run) < 6:
        run.append(cursor)
        cursor = cursor + direction
    return run

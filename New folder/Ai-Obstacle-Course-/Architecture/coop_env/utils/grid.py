"""A dense 2D integer grid plus the flood-fill primitives built on top of it.

The grid stores small integers (tile ids, region labels, distances). It is
deliberately dependency-free and knows nothing about tiles, entities, or
rendering -- those meanings live in the layers above.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Iterator

from .geometry import Rect, Vec2


class Grid:
    """A width x height grid of integers, stored flat in row-major order."""

    __slots__ = ("width", "height", "_data", "_positions")

    def __init__(self, width: int, height: int, fill: int = 0) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"grid dimensions must be positive, got {width}x{height}")
        self.width = width
        self.height = height
        self._data = [fill] * (width * height)
        # Position objects are immutable and get walked constantly, so they are
        # built once and shared rather than reallocated on every sweep.
        self._positions: tuple[Vec2, ...] | None = None

    # -- basic access ----------------------------------------------------

    def __getitem__(self, pos: Vec2) -> int:
        return self._data[pos[1] * self.width + pos[0]]

    def __setitem__(self, pos: Vec2, value: int) -> None:
        self._data[pos[1] * self.width + pos[0]] = value

    def get(self, pos: Vec2, default: int = -1) -> int:
        if not self.in_bounds(pos):
            return default
        return self._data[pos[1] * self.width + pos[0]]

    def in_bounds(self, pos: Vec2) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    @property
    def bounds(self) -> Rect:
        return Rect(0, 0, self.width, self.height)

    def positions(self) -> tuple[Vec2, ...]:
        """Every position in row-major order (cached)."""
        if self._positions is None:
            self._positions = tuple(
                Vec2(x, y) for y in range(self.height) for x in range(self.width)
            )
        return self._positions

    def fill(self, value: int) -> None:
        self._data = [value] * (self.width * self.height)

    def fill_rect(self, rect: Rect, value: int) -> None:
        for y in range(max(0, rect.y), min(self.height, rect.y2)):
            row = y * self.width
            for x in range(max(0, rect.x), min(self.width, rect.x2)):
                self._data[row + x] = value

    def copy(self) -> "Grid":
        clone = Grid.__new__(Grid)
        clone.width = self.width
        clone.height = self.height
        clone._data = list(self._data)
        clone._positions = self._positions
        return clone

    def count(self, value: int) -> int:
        return self._data.count(value)

    def find_all(self, value: int) -> list[Vec2]:
        return [p for p in self.positions() if self[p] == value]

    def rows(self) -> list[list[int]]:
        return [
            self._data[y * self.width : (y + 1) * self.width]
            for y in range(self.height)
        ]

    def to_list(self) -> list[int]:
        """Flat row-major copy -- the handoff format for future array backends."""
        return list(self._data)

    # -- neighbourhood queries -------------------------------------------

    def neighbors4(self, pos: Vec2) -> Iterator[Vec2]:
        x, y = pos
        if x + 1 < self.width:
            yield Vec2(x + 1, y)
        if x > 0:
            yield Vec2(x - 1, y)
        if y + 1 < self.height:
            yield Vec2(x, y + 1)
        if y > 0:
            yield Vec2(x, y - 1)

    def neighbors8(self, pos: Vec2) -> Iterator[Vec2]:
        x, y = pos
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    yield Vec2(nx, ny)


Predicate = Callable[[Vec2], bool]


def flood_fill(start: Vec2, passable: Predicate, bounds: Rect) -> set[Vec2]:
    """Return every position 4-connected to `start` through `passable` tiles."""
    if not bounds.contains(start) or not passable(start):
        return set()
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for n in current.neighbors4():
            if n in seen or not bounds.contains(n) or not passable(n):
                continue
            seen.add(n)
            queue.append(n)
    return seen


def connected_components(
    candidates: Iterable[Vec2], passable: Predicate, bounds: Rect
) -> list[set[Vec2]]:
    """Partition `candidates` into 4-connected components.

    Results are ordered largest-first, then by top-left position, so the output
    is stable regardless of the iteration order of the input.
    """
    remaining = {p for p in candidates if passable(p)}
    components: list[set[Vec2]] = []
    while remaining:
        seed = min(remaining, key=lambda p: (p[1], p[0]))
        component = flood_fill(seed, lambda p: p in remaining, bounds)
        if not component:
            remaining.discard(seed)
            continue
        components.append(component)
        remaining -= component
    components.sort(key=lambda c: (-len(c), min((p[1], p[0]) for p in c)))
    return components


def distance_field(sources: Iterable[Vec2], passable: Predicate, bounds: Rect) -> dict[Vec2, int]:
    """Breadth-first step counts from `sources`.

    This is a static spatial measurement over the map -- it is used to place
    objects far apart and to score layouts during generation. It is never used
    to move anything.
    """
    dist: dict[Vec2, int] = {}
    queue: deque[Vec2] = deque()
    for s in sources:
        if bounds.contains(s) and passable(s) and s not in dist:
            dist[s] = 0
            queue.append(s)
    while queue:
        current = queue.popleft()
        d = dist[current] + 1
        for n in current.neighbors4():
            if n in dist or not bounds.contains(n) or not passable(n):
                continue
            dist[n] = d
            queue.append(n)
    return dist


def largest_component(
    candidates: Iterable[Vec2], passable: Predicate, bounds: Rect
) -> set[Vec2]:
    components = connected_components(candidates, passable, bounds)
    return components[0] if components else set()

"""Small immutable geometry primitives used across the whole project.

`Vec2` is a NamedTuple so it *is* a tuple: hashable, cheap to copy, and usable
directly as a dict key in the hot loops of generation and validation without
any conversion layer.
"""

from __future__ import annotations

from typing import Iterator, NamedTuple


class Vec2(NamedTuple):
    """An integer grid position or offset."""

    x: int
    y: int

    def __add__(self, other: "Vec2") -> "Vec2":  # type: ignore[override]
        return Vec2(self.x + other[0], self.y + other[1])

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other[0], self.y - other[1])

    def scaled(self, k: int) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    def manhattan(self, other: "Vec2") -> int:
        return abs(self.x - other[0]) + abs(self.y - other[1])

    def chebyshev(self, other: "Vec2") -> int:
        return max(abs(self.x - other[0]), abs(self.y - other[1]))

    def neighbors4(self) -> tuple["Vec2", "Vec2", "Vec2", "Vec2"]:
        return (
            Vec2(self.x + 1, self.y),
            Vec2(self.x - 1, self.y),
            Vec2(self.x, self.y + 1),
            Vec2(self.x, self.y - 1),
        )

    def neighbors8(self) -> tuple["Vec2", ...]:
        return tuple(
            Vec2(self.x + dx, self.y + dy)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dx or dy
        )


NORTH = Vec2(0, -1)
SOUTH = Vec2(0, 1)
EAST = Vec2(1, 0)
WEST = Vec2(-1, 0)
DIRECTIONS4 = (NORTH, EAST, SOUTH, WEST)


class Rect(NamedTuple):
    """An axis-aligned rectangle with an inclusive origin and exclusive extent."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> Vec2:
        return Vec2(self.x + self.w // 2, self.y + self.h // 2)

    def contains(self, p: Vec2) -> bool:
        return self.x <= p[0] < self.x2 and self.y <= p[1] < self.y2

    def positions(self) -> Iterator[Vec2]:
        for y in range(self.y, self.y2):
            for x in range(self.x, self.x2):
                yield Vec2(x, y)

    def inset(self, margin: int) -> "Rect":
        return Rect(
            self.x + margin,
            self.y + margin,
            max(0, self.w - 2 * margin),
            max(0, self.h - 2 * margin),
        )

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.x2 <= other.x
            or other.x2 <= self.x
            or self.y2 <= other.y
            or other.y2 <= self.y
        )


def line_between(a: Vec2, b: Vec2) -> list[Vec2]:
    """Axis-aligned L-shaped run of tiles from `a` to `b` (horizontal leg first).

    Used purely as a geometry helper for carving corridors and platform tracks.
    """
    tiles: list[Vec2] = []
    x, y = a
    step = 1 if b.x >= x else -1
    for cx in range(x, b.x + step, step):
        tiles.append(Vec2(cx, y))
    step = 1 if b.y >= y else -1
    for cy in range(y, b.y + step, step):
        tiles.append(Vec2(b.x, cy))
    seen: set[Vec2] = set()
    unique: list[Vec2] = []
    for t in tiles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique

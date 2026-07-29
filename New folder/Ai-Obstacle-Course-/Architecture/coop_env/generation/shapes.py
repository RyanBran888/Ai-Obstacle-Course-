"""Room silhouettes.

Stage 1 of generation: decide the outline of the play area. Each builder
returns the set of floor tiles inside a bounding rectangle; walls are added
later by the generator. Every builder result is reduced to its largest
4-connected component, so a silhouette can never hand a disconnected blob to
the stages downstream.

To add a new silhouette: write a builder, register it in `SHAPE_BUILDERS`, and
add the enum member in `config.RoomShape`. Nothing else changes.
"""

from __future__ import annotations

from typing import Callable

from ..config import RoomShape
from ..rng import SeededRandom
from ..utils.geometry import Rect, Vec2
from ..utils.grid import largest_component

ShapeBuilder = Callable[[Rect, SeededRandom], set[Vec2]]

#: Below this fraction of the bounding box, a silhouette is considered a dud
#: and the generator falls back to a plain rectangle.
MIN_FILL_RATIO = 0.28


def _rectangle(area: Rect, rng: SeededRandom) -> set[Vec2]:
    return set(area.positions())


def _l_shape(area: Rect, rng: SeededRandom) -> set[Vec2]:
    cut_w = rng.randint(area.w // 3, max(area.w // 3, (area.w * 2) // 3))
    cut_h = rng.randint(area.h // 3, max(area.h // 3, (area.h * 2) // 3))
    corner = rng.choice(("nw", "ne", "sw", "se"))
    x0 = area.x if corner in ("nw", "sw") else area.x2 - cut_w
    y0 = area.y if corner in ("nw", "ne") else area.y2 - cut_h
    removed = set(Rect(x0, y0, cut_w, cut_h).positions())
    return set(area.positions()) - removed


def _t_shape(area: Rect, rng: SeededRandom) -> set[Vec2]:
    band_h = max(3, rng.randint(area.h // 3, area.h // 2))
    stem_w = max(3, rng.randint(area.w // 3, area.w // 2))
    stem_x = area.x + (area.w - stem_w) // 2
    flip = rng.chance(0.5)
    band_y = area.y if not flip else area.y2 - band_h
    stem_y = area.y + band_h if not flip else area.y
    stem_h = area.h - band_h
    tiles = set(Rect(area.x, band_y, area.w, band_h).positions())
    tiles |= set(Rect(stem_x, stem_y, stem_w, stem_h).positions())
    return tiles


def _plus(area: Rect, rng: SeededRandom) -> set[Vec2]:
    band_h = max(3, rng.randint(area.h // 3, (area.h * 3) // 5))
    band_w = max(3, rng.randint(area.w // 3, (area.w * 3) // 5))
    bx = area.x + (area.w - band_w) // 2
    by = area.y + (area.h - band_h) // 2
    tiles = set(Rect(area.x, by, area.w, band_h).positions())
    tiles |= set(Rect(bx, area.y, band_w, area.h).positions())
    return tiles


def _donut(area: Rect, rng: SeededRandom) -> set[Vec2]:
    hole_w = max(2, rng.randint(area.w // 4, area.w // 2))
    hole_h = max(2, rng.randint(area.h // 4, area.h // 2))
    # keep at least a two-tile ring so the loop stays walkable
    hole_w = min(hole_w, area.w - 6)
    hole_h = min(hole_h, area.h - 6)
    if hole_w < 2 or hole_h < 2:
        return _rectangle(area, rng)
    hx = area.x + rng.randint(2, area.w - hole_w - 2)
    hy = area.y + rng.randint(2, area.h - hole_h - 2)
    return set(area.positions()) - set(Rect(hx, hy, hole_w, hole_h).positions())


def _diamond(area: Rect, rng: SeededRandom) -> set[Vec2]:
    cx, cy = area.center
    rx = max(1, area.w // 2)
    ry = max(1, area.h // 2)
    slack = rng.uniform(1.05, 1.35)
    tiles: set[Vec2] = set()
    for p in area.positions():
        nx = abs(p[0] - cx) / rx
        ny = abs(p[1] - cy) / ry
        if nx + ny <= slack:
            tiles.add(p)
    return tiles


def _terrace(area: Rect, rng: SeededRandom) -> set[Vec2]:
    """A stepped silhouette: each horizontal band is inset a little differently."""
    steps = rng.randint(3, 5)
    band_h = max(2, area.h // steps)
    tiles: set[Vec2] = set()
    offset = 0
    for i in range(steps):
        y = area.y + i * band_h
        h = band_h if i < steps - 1 else area.y2 - y
        if h <= 0:
            break
        offset = max(0, min(area.w // 3, offset + rng.randint(-2, 3)))
        width = area.w - offset
        if width < 4:
            width = 4
            offset = area.w - width
        tiles |= set(Rect(area.x + offset, y, width, h).positions())
    return tiles


def _cavern(area: Rect, rng: SeededRandom) -> set[Vec2]:
    """Random noise smoothed by a cellular automaton into an organic blob."""
    fill_probability = rng.uniform(0.42, 0.50)
    solid = {p for p in area.positions() if rng.random() < fill_probability}
    for _ in range(4):
        next_solid: set[Vec2] = set()
        for p in area.positions():
            neighbours = sum(
                1
                for n in p.neighbors8()
                if not area.contains(n) or n in solid
            )
            if neighbours >= 5:
                next_solid.add(p)
        solid = next_solid
    tiles = set(area.positions()) - solid
    # punch a channel through the middle so the blob rarely collapses
    cy = area.center[1]
    for x in range(area.x, area.x2):
        for dy in (-1, 0, 1):
            tiles.add(Vec2(x, max(area.y, min(area.y2 - 1, cy + dy))))
    return tiles


SHAPE_BUILDERS: dict[RoomShape, ShapeBuilder] = {
    RoomShape.RECTANGLE: _rectangle,
    RoomShape.L_SHAPE: _l_shape,
    RoomShape.T_SHAPE: _t_shape,
    RoomShape.PLUS: _plus,
    RoomShape.DONUT: _donut,
    RoomShape.DIAMOND: _diamond,
    RoomShape.CAVERN: _cavern,
    RoomShape.TERRACE: _terrace,
}


def build_silhouette(shape: RoomShape, area: Rect, rng: SeededRandom) -> set[Vec2]:
    """Produce the floor tiles for `shape` inside `area`.

    The result is always a single connected region of at least
    `MIN_FILL_RATIO` of the bounding box; degenerate results fall back to a
    plain rectangle so generation cannot stall on an unlucky draw.
    """
    builder = SHAPE_BUILDERS.get(shape, _rectangle)
    tiles = builder(area, rng)
    tiles = {p for p in tiles if area.contains(p)}
    if tiles:
        tiles = largest_component(tiles, lambda p: p in tiles, area)
    if len(tiles) < MIN_FILL_RATIO * area.area:
        return set(area.positions())
    return tiles

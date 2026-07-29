"""Binary space partitioning of the play area into sub-areas.

Stage 2 of generation. The area is split recursively, and every split reserves
a one-tile-thick divider line that becomes a wall. Those walls are what turn a
single open floor into a set of rooms joined by doorways -- which in turn is
what gives the puzzle layer something to gate.

Splitting is purely geometric; it knows nothing about mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..rng import SeededRandom
from ..utils.geometry import Rect


@dataclass(frozen=True, slots=True)
class Divider:
    """The one-tile wall line left behind by a split."""

    rect: Rect
    vertical: bool


@dataclass(frozen=True, slots=True)
class Partition:
    leaves: tuple[Rect, ...]
    dividers: tuple[Divider, ...]


def _can_split(rect: Rect, min_span: int) -> tuple[bool, bool]:
    """Whether the rect can be split vertically / horizontally."""
    needed = 2 * min_span + 1
    return rect.w >= needed, rect.h >= needed


def _split_once(
    rect: Rect, rng: SeededRandom, min_span: int
) -> tuple[Rect, Rect, Divider] | None:
    can_v, can_h = _can_split(rect, min_span)
    if not can_v and not can_h:
        return None
    if can_v and can_h:
        # bias toward cutting the longer axis so leaves stay chunky
        if rect.w > rect.h * 1.25:
            vertical = True
        elif rect.h > rect.w * 1.25:
            vertical = False
        else:
            vertical = rng.chance(0.5)
    else:
        vertical = can_v

    if vertical:
        cut = rng.randint(rect.x + min_span, rect.x2 - min_span - 1)
        left = Rect(rect.x, rect.y, cut - rect.x, rect.h)
        right = Rect(cut + 1, rect.y, rect.x2 - cut - 1, rect.h)
        divider = Divider(Rect(cut, rect.y, 1, rect.h), vertical=True)
        return left, right, divider

    cut = rng.randint(rect.y + min_span, rect.y2 - min_span - 1)
    top = Rect(rect.x, rect.y, rect.w, cut - rect.y)
    bottom = Rect(rect.x, cut + 1, rect.w, rect.y2 - cut - 1)
    divider = Divider(Rect(rect.x, cut, rect.w, 1), vertical=False)
    return top, bottom, divider


def partition_area(
    area: Rect, rng: SeededRandom, target_leaves: int, min_span: int
) -> Partition:
    """Split `area` until it holds about `target_leaves` sub-rectangles.

    Splits the largest splittable leaf each round, which keeps sub-areas
    similar in size rather than producing one huge room and a row of slivers.
    """
    leaves: list[Rect] = [area]
    dividers: list[Divider] = []
    guard = 0
    while len(leaves) < max(1, target_leaves) and guard < 256:
        guard += 1
        splittable = [
            (i, r) for i, r in enumerate(leaves) if any(_can_split(r, min_span))
        ]
        if not splittable:
            break
        # largest first, with position as a deterministic tie-break
        splittable.sort(key=lambda item: (-item[1].area, item[1].y, item[1].x))
        top = splittable[: max(1, len(splittable) // 2)]
        index, rect = rng.choice(top)
        result = _split_once(rect, rng, min_span)
        if result is None:
            continue
        first, second, divider = result
        leaves[index : index + 1] = [first, second]
        dividers.append(divider)
    leaves.sort(key=lambda r: (r.y, r.x))
    return Partition(tuple(leaves), tuple(dividers))

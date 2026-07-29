"""Dependency-free helpers: geometry, grids, and graphs."""

from .geometry import DIRECTIONS4, EAST, NORTH, SOUTH, WEST, Rect, Vec2, line_between
from .graph import Graph
from .grid import (
    Grid,
    connected_components,
    distance_field,
    flood_fill,
    largest_component,
)

__all__ = [
    "Vec2",
    "Rect",
    "line_between",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "DIRECTIONS4",
    "Grid",
    "flood_fill",
    "connected_components",
    "distance_field",
    "largest_component",
    "Graph",
]

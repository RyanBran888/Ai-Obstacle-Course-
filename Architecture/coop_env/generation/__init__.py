"""Procedural generation pipeline.

Stages run in order and hand off plain data, so any one of them can be swapped
without touching the others:

    shapes      room silhouette
    partition   BSP split into sub-areas
    layout      walls and doorways
    terrain     obstacles and hazards
    topology    region graph derived from the finished terrain
    mechanisms  spawns, gates, triggers, exit, extras
    generator   orchestration, retries, fallback
"""

from .generator import GenerationError, GenerationOutcome, RoomGenerator
from .layout import Layout, build_layout
from .mechanisms import GateKind, MechanismResult, populate_mechanisms
from .partition import Partition, partition_area
from .shapes import SHAPE_BUILDERS, build_silhouette
from .terrain import Decoration, decorate_terrain
from .topology import Topology, build_topology

__all__ = [
    "RoomGenerator",
    "GenerationOutcome",
    "GenerationError",
    "build_silhouette",
    "SHAPE_BUILDERS",
    "partition_area",
    "Partition",
    "build_layout",
    "Layout",
    "decorate_terrain",
    "Decoration",
    "build_topology",
    "Topology",
    "populate_mechanisms",
    "MechanismResult",
    "GateKind",
]

"""The generation pipeline: seed in, validated room out.

    seed -> layout -> terrain -> topology -> mechanisms -> validation
                          ^                                    |
                          +------------ retry -----------------+

Each attempt draws from its own deterministic sub-stream, so a rejected room
never contaminates the next try, and the same seed always produces the same
sequence of attempts. If the retry budget runs out, `_fallback_room` returns a
small hand-shaped room that is solvable by construction, so the caller always
gets something usable unless `raise_on_failure` is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import GenerationConfig, RoomShape
from ..entities import AgentSpawn, Entity, ExitDoor, Key, LockedDoor
from ..requirements import KeyRequirement
from ..rng import SeededRandom, normalize_seed
from ..room import Region, Room, RoomTopology, portal_key
from ..tiles import Tile
from ..utils.geometry import Rect, Vec2
from ..utils.graph import Graph
from ..utils.grid import Grid
from ..validation.validator import ValidationReport, validate_room
from .layout import build_layout
from .mechanisms import COOPERATIVE_GATES, populate_mechanisms
from .terrain import decorate_terrain
from .topology import build_topology


class GenerationError(RuntimeError):
    """Raised when the retry budget is exhausted and `raise_on_failure` is set."""


@dataclass(slots=True)
class AttemptLog:
    index: int
    stage: str
    detail: str


@dataclass(slots=True)
class GenerationOutcome:
    """A generated room plus the paper trail that produced it."""

    room: Room
    report: ValidationReport
    attempts: int
    fallback: bool = False
    rejected: list[AttemptLog] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report.ok


class RoomGenerator:
    """Turns seeds into validated rooms.

    The generator is stateless between calls apart from its config, so the same
    instance can serve many episodes and stays safe to reuse.
    """

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = (config or GenerationConfig()).require_valid()

    # -- public API --------------------------------------------------------

    def generate(self, seed: int | str | None = None) -> Room:
        """Generate one validated room."""
        return self.generate_with_report(seed).room

    def generate_with_report(self, seed: int | str | None = None) -> GenerationOutcome:
        """Generate a room and return the validation report and retry log too."""
        base_seed = normalize_seed(seed if seed is not None else self.config.seed)
        rejected: list[AttemptLog] = []

        for attempt in range(self.config.max_attempts):
            rng = SeededRandom(base_seed).fork(f"attempt:{attempt}")
            try:
                room = self._build_once(base_seed, rng, attempt)
            except _AttemptRejected as rejection:
                rejected.append(AttemptLog(attempt, rejection.stage, rejection.detail))
                continue

            report = validate_room(room)
            if report.ok:
                room.metadata["attempts"] = attempt + 1
                room.metadata["rejected_attempts"] = [
                    {"attempt": r.index, "stage": r.stage, "detail": r.detail}
                    for r in rejected
                ]
                return GenerationOutcome(room, report, attempt + 1, False, rejected)

            rejected.append(
                AttemptLog(attempt, "validation", report.summary())
            )

        if self.config.raise_on_failure:
            details = "; ".join(f"#{r.index} {r.stage}: {r.detail}" for r in rejected[-5:])
            raise GenerationError(
                f"no valid room after {self.config.max_attempts} attempts (seed {base_seed}). "
                f"Recent failures: {details}"
            )

        room = _fallback_room(base_seed, self.config)
        report = validate_room(room)
        room.metadata["attempts"] = self.config.max_attempts
        room.metadata["fallback"] = True
        return GenerationOutcome(room, report, self.config.max_attempts, True, rejected)

    def generate_many(self, count: int, start_seed: int | None = None) -> list[Room]:
        """Convenience batch generator, one room per consecutive seed."""
        base = normalize_seed(start_seed if start_seed is not None else self.config.seed)
        return [self.generate(base + i) for i in range(count)]

    # -- one attempt -------------------------------------------------------

    def _build_once(self, base_seed: int, rng: SeededRandom, attempt: int) -> Room:
        config = self.config

        layout = build_layout(config, rng.derive("layout"))
        if not layout.floor_tiles():
            raise _AttemptRejected("layout", "silhouette produced no floor")

        decoration = decorate_terrain(layout, config, rng.derive("terrain"))
        topology = build_topology(layout, decoration)

        if not topology.regions:
            raise _AttemptRejected("topology", "no walkable regions")
        if not topology.graph.is_connected():
            raise _AttemptRejected(
                "topology", f"region graph split into {len(topology.graph.components())} parts"
            )

        mechanisms = populate_mechanisms(
            layout, topology, decoration, config, rng.derive("mechanisms")
        )
        if not mechanisms.entities:
            raise _AttemptRejected("mechanisms", "nothing could be placed")
        if len([e for e in mechanisms.entities if isinstance(e, AgentSpawn)]) != 2:
            raise _AttemptRejected("mechanisms", "could not place two spawn points")
        if not any(isinstance(e, ExitDoor) for e in mechanisms.entities):
            raise _AttemptRejected("mechanisms", "could not place an exit")

        room_topology = RoomTopology(
            regions=topology.regions,
            graph=topology.graph,
            portals=topology.portals,
            spawn_regions=mechanisms.spawn_regions,
            exit_region=mechanisms.exit_region,
            depths=mechanisms.depths,
            platform_links=tuple(sorted(topology.platform_links)),
        )

        metadata: dict[str, Any] = {
            "attempt": attempt,
            "shape": layout.shape.value,
            "complexity": config.complexity,
            "gates": [
                {
                    "edge": list(gate.edge),
                    "kind": gate.kind.value,
                    "doors": list(gate.door_ids),
                    "triggers": list(gate.trigger_ids),
                    "depth": gate.depth,
                }
                for gate in mechanisms.gates
            ],
            "cooperative_gates": sum(
                1 for gate in mechanisms.gates if gate.kind in COOPERATIVE_GATES
            ),
            "platform_bridges": len(topology.platform_links),
            "hazard_tiles": len(decoration.hazard_tiles),
            "obstacle_tiles": len(decoration.obstacle_tiles),
            **mechanisms.stats,
        }

        return Room(
            seed=base_seed,
            config=config,
            terrain=layout.terrain,
            entities=tuple(mechanisms.entities),
            topology=room_topology,
            shape=layout.shape,
            metadata=metadata,
        )


class _AttemptRejected(Exception):
    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------


def _fallback_room(seed: int, config: GenerationConfig) -> Room:
    """A minimal two-region room, correct by construction.

    Used only when the retry budget is exhausted -- an unlucky config should
    still hand the caller something valid rather than an exception. Rooms
    produced here are tagged `fallback` in metadata so they are easy to spot in
    a dataset.
    """
    width, height = 17, 11
    terrain = Grid(width, height, Tile.WALL)
    for pos in Rect(1, 1, width - 2, height - 2).positions():
        terrain[pos] = Tile.FLOOR

    divider_x = width // 2
    for y in range(1, height - 1):
        terrain[Vec2(divider_x, y)] = Tile.WALL
    doorway = Vec2(divider_x, height // 2)
    terrain[doorway] = Tile.FLOOR

    left = frozenset(
        Vec2(x, y)
        for y in range(1, height - 1)
        for x in range(1, divider_x)
    )
    right = frozenset(
        Vec2(x, y)
        for y in range(1, height - 1)
        for x in range(divider_x + 1, width - 1)
    )

    entities: list[Entity] = [
        AgentSpawn(id="spawn_0", pos=Vec2(2, 2), index=0),
        AgentSpawn(id="spawn_1", pos=Vec2(2, height - 3), index=1),
        Key(id="key_0", pos=Vec2(divider_x - 2, 2), color="gold", opens=("door_0",)),
        LockedDoor(
            id="door_0",
            pos=doorway,
            requirement=KeyRequirement(("key_0",)),
            latching=True,
            horizontal=False,
            region_a=0,
            region_b=1,
        ),
        ExitDoor(id="exit", pos=Vec2(width - 3, height // 2)),
    ]

    graph: Graph[int] = Graph()
    graph.add_edge(0, 1)
    topology = RoomTopology(
        regions={0: Region(0, left), 1: Region(1, right)},
        graph=graph,
        portals={portal_key(0, 1): (doorway,)},
        spawn_regions=(0, 0),
        exit_region=1,
        depths={0: 0, 1: 1},
    )
    return Room(
        seed=seed,
        config=config,
        terrain=terrain,
        entities=tuple(entities),
        topology=topology,
        shape=RoomShape.RECTANGLE,
        metadata={"fallback": True, "shape": RoomShape.RECTANGLE.value},
    )

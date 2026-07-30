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
from ..entities import (
    AgentSpawn,
    Entity,
    ExitDoor,
    Key,
    LockedDoor,
    PushableBlock,
    ResetZone,
    Switch,
    SwitchMode,
    TemporaryBridge,
    WipeoutBall,
    WipeoutBallSize,
)
from ..requirements import KeyRequirement
from ..rng import SeededRandom, normalize_seed
from ..room import Region, Room, RoomTopology, portal_key
from ..tiles import HAZARD_TILES, Tile
from ..utils.geometry import Rect, Vec2
from ..utils.graph import Graph
from ..utils.grid import Grid
from ..validation.validator import ValidationReport, validate_room
from .combined_course import build_combined_course
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
        if self.config.require_combined_course:
            room = build_combined_course(base_seed, self.config)
            report = validate_room(room)
            if not report.ok:
                raise GenerationError(
                    "combined course construction failed: " + report.summary()
                )
            room.metadata["attempts"] = 1
            room.metadata["rejected_attempts"] = []
            return GenerationOutcome(room, report, 1)

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
        for size in ("normal", "big"):
            requested = mechanisms.stats.get(f"requested_{size}_wipeout_balls", 0)
            placed = mechanisms.stats.get(f"placed_{size}_wipeout_balls", 0)
            if placed != requested:
                raise _AttemptRejected(
                    "mechanisms",
                    f"placed {placed}/{requested} {size} wipeout balls",
                )
        if (
            config.require_wipeout_crossing
            and mechanisms.stats.get("required_wipeout_ball_id") is None
        ):
            raise _AttemptRejected(
                "mechanisms", "could not place a required wipeout crossing"
            )
        if config.require_bridge_crossing:
            requested = mechanisms.stats.get("requested_temporary_bridges", 0)
            placed = mechanisms.stats.get("placed_temporary_bridges", 0)
            if placed != requested:
                raise _AttemptRejected(
                    "mechanisms",
                    f"placed {placed}/{requested} temporary bridges",
                )
            if mechanisms.stats.get("required_bridge_id") is None:
                raise _AttemptRejected(
                    "mechanisms", "could not place a required bridge crossing"
                )
        if config.require_reset_detour:
            requested = mechanisms.stats.get("requested_reset_zones", 0)
            placed = mechanisms.stats.get("placed_reset_zones", 0)
            if requested != 1 or placed != 1:
                raise _AttemptRejected(
                    "mechanisms",
                    f"placed {placed}/{requested} reset detours",
                )
            if mechanisms.stats.get("required_reset_zone_id") is None:
                raise _AttemptRejected(
                    "mechanisms", "could not place a required reset detour"
                )
        requested_blocks = mechanisms.stats.get("requested_pushable_blocks", 0)
        placed_blocks = mechanisms.stats.get("placed_pushable_blocks", 0)
        if placed_blocks != requested_blocks:
            raise _AttemptRejected(
                "mechanisms",
                f"placed {placed_blocks}/{requested_blocks} pushable blocks",
            )
        has_hold_switch = any(
            isinstance(entity, Switch) and entity.mode is SwitchMode.HOLD
            for entity in mechanisms.entities
        )
        has_crate_pair = any(
            isinstance(entity, PushableBlock)
            and entity.target_switch_id is not None
            for entity in mechanisms.entities
        )
        if requested_blocks and has_hold_switch and not has_crate_pair:
            raise _AttemptRejected(
                "mechanisms",
                "could not pair a pushable block with a hold switch",
            )
        if config.require_key_for_each_agent:
            owners = {
                key.agent_index
                for key in mechanisms.entities
                if isinstance(key, Key)
            }
            if not {0, 1}.issubset(owners):
                raise _AttemptRejected(
                    "mechanisms", "could not place a key for each agent"
                )

        room_topology = RoomTopology(
            regions=topology.regions,
            graph=topology.graph,
            portals=topology.portals,
            spawn_regions=mechanisms.spawn_regions,
            exit_region=mechanisms.exit_region,
            depths=mechanisms.depths,
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
    normal_count = config.num_normal_wipeout_balls[0]
    big_count = config.num_big_wipeout_balls[0]
    height = max(
        23
        if config.require_reset_detour
        else (19 if config.require_bridge_crossing else 15),
        9 + normal_count * 2 + big_count * 4,
    )
    owned_pair = config.require_key_for_each_agent
    width = 63 if owned_pair else 39
    terrain = Grid(width, height, Tile.WALL)
    for pos in Rect(1, 1, width - 2, height - 2).positions():
        terrain[pos] = Tile.FLOOR

    owned_second_divider = 2 * width // 3 if owned_pair else None
    dividers = (
        (width // 3, owned_second_divider)
        if owned_second_divider is not None
        else (width // 2,)
    )
    for divider in dividers:
        for y in range(1, height - 1):
            terrain[Vec2(divider, y)] = Tile.WALL
    forced_course = (
        config.require_wipeout_crossing
        or config.require_bridge_crossing
        or config.require_reset_detour
    )
    doorway_y = height - 3 if forced_course else height // 2
    doorways = [Vec2(divider, doorway_y) for divider in dividers]
    for doorway in doorways:
        terrain[doorway] = Tile.FLOOR

    if owned_second_divider is not None:
        x_spans = (
            (1, dividers[0]),
            (dividers[0] + 1, owned_second_divider),
            (owned_second_divider + 1, width - 1),
        )
    else:
        x_spans = ((1, dividers[0]), (dividers[0] + 1, width - 1))
    required_track: tuple[Vec2, ...] = ()
    required_size: WipeoutBallSize | None = None
    if config.require_wipeout_crossing:
        required_size = (
            WipeoutBallSize.BIG if big_count else WipeoutBallSize.NORMAL
        )
        length = 11 if required_size is WipeoutBallSize.BIG else 7
        radius = 1 if required_size is WipeoutBallSize.BIG else 0
        start, stop = x_spans[-1]
        swept_width = length + 2 * radius
        opening_x = start + (stop - start - swept_width) // 2
        track_y = 4
        required_track = tuple(
            Vec2(opening_x + radius + offset, track_y)
            for offset in range(length)
        )
        for y in range(track_y - radius, track_y + radius + 1):
            for x in range(start, stop):
                terrain[Vec2(x, y)] = Tile.WALL
        for center in required_track:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    terrain[Vec2(center.x + dx, center.y + dy)] = Tile.FLOOR

    required_bridge_tiles: tuple[Vec2, ...] = ()
    bridge_y: int | None = None
    if config.require_bridge_crossing:
        start, stop = x_spans[-1]
        bridge_y = 7 if required_track else 4
        bridge_length = 3
        bridge_x = start + (stop - start - bridge_length) // 2
        required_bridge_tiles = tuple(
            Vec2(bridge_x + offset, bridge_y)
            for offset in range(bridge_length)
        )
        for x in range(start, stop):
            terrain[Vec2(x, bridge_y)] = Tile.WALL
        hazard = next(
            tile
            for tile in HAZARD_TILES
            if config.hazard_weights.get(tile, 0) > 0
        )
        for tile in required_bridge_tiles:
            terrain[tile] = hazard

    required_reset_tile: Vec2 | None = None
    required_reset_safe_tile: Vec2 | None = None
    reset_barrier: tuple[Vec2, ...] = ()
    if config.require_reset_detour:
        start, stop = x_spans[-1]
        last_barrier = bridge_y
        if last_barrier is None and required_track:
            last_barrier = 4 + (
                1 if required_size is WipeoutBallSize.BIG else 0
            )
        reset_y = 4 if last_barrier is None else last_barrier + 3
        reset_barrier = tuple(Vec2(x, reset_y) for x in range(start, stop))
        required_reset_tile = Vec2(start + 1, reset_y)
        required_reset_safe_tile = Vec2(stop - 2, reset_y)
        for tile in reset_barrier:
            terrain[tile] = Tile.OBSTACLE
        terrain[required_reset_tile] = Tile.FLOOR
        terrain[required_reset_safe_tile] = Tile.FLOOR

    regions = {
        index: Region(
            index,
            frozenset(
                Vec2(x, y)
                for y in range(1, height - 1)
                for x in range(start, stop)
                if terrain[Vec2(x, y)] == Tile.FLOOR
            ),
        )
        for index, (start, stop) in enumerate(x_spans)
    }

    key_count = 2 if owned_pair else 1
    key_positions = [Vec2(dividers[0] - 3, 2)]
    if owned_second_divider is not None:
        key_positions.append(Vec2(owned_second_divider - 3, height - 3))
    keys = [
        Key(
            id=f"key_{index}",
            pos=key_positions[index],
            color=("azure", "crimson")[index] if config.agent_specific_keys else "gold",
            opens=(f"door_{index}",),
            agent_index=index if config.agent_specific_keys else None,
        )
        for index in range(key_count)
    ]
    doors = [
        LockedDoor(
            id=f"door_{index}",
            pos=doorway,
            requirement=KeyRequirement((f"key_{index}",)),
            latching=True,
            horizontal=False,
            region_a=index if owned_pair else 0,
            region_b=index + 1 if owned_pair else 1,
        )
        for index, doorway in enumerate(doorways)
    ]
    entities: list[Entity] = [
        AgentSpawn(id="spawn_0", pos=Vec2(2, 2), index=0),
        AgentSpawn(id="spawn_1", pos=Vec2(2, height - 3), index=1),
        *keys,
        *doors,
        ExitDoor(
            id="exit",
            pos=Vec2(
                x_spans[-1][0] + 1 if required_reset_tile else width - 3,
                2 if forced_course else height // 2,
            ),
        ),
    ]

    required_ball_id: str | None = None
    if required_size is not None:
        assert required_track
        required_ball_id = f"wipeout_{required_size.value}_0"
        entities.append(
            WipeoutBall(
                id=required_ball_id,
                pos=required_track[0],
                track=required_track,
                size=required_size,
            )
        )

    required_bridge_id: str | None = None
    if required_bridge_tiles:
        required_bridge_id = "bridge_0"
        entities.append(
            TemporaryBridge(
                id=required_bridge_id,
                pos=required_bridge_tiles[0],
                tiles=required_bridge_tiles,
                period=12,
                on_ticks=6,
                phase=0,
            )
        )

    required_reset_zone_id: str | None = None
    if required_reset_tile is not None:
        required_reset_zone_id = "reset_0"
        entities.append(
            ResetZone(
                id=required_reset_zone_id,
                pos=required_reset_tile,
                rect=Rect(required_reset_tile.x, required_reset_tile.y, 1, 1),
            )
        )

    ball_row = 5
    big_start = 1 if required_size is WipeoutBallSize.BIG else 0
    for index in range(big_start, big_count):
        track = tuple(Vec2(3 + offset, ball_row) for offset in range(11))
        entities.append(
            WipeoutBall(
                id=f"wipeout_big_{index}",
                pos=track[0],
                track=track,
                size=WipeoutBallSize.BIG,
            )
        )
        ball_row += 4
    normal_start = 1 if required_size is WipeoutBallSize.NORMAL else 0
    for index in range(normal_start, normal_count):
        track = tuple(Vec2(3 + offset, ball_row) for offset in range(7))
        entities.append(
            WipeoutBall(
                id=f"wipeout_normal_{index}",
                pos=track[0],
                track=track,
                size=WipeoutBallSize.NORMAL,
            )
        )
        ball_row += 2

    graph: Graph[int] = Graph()
    for index in range(len(regions) - 1):
        graph.add_edge(index, index + 1)
    topology = RoomTopology(
        regions=regions,
        graph=graph,
        portals={
            portal_key(index, index + 1): (doorway,)
            for index, doorway in enumerate(doorways)
        },
        spawn_regions=(0, 0),
        exit_region=len(regions) - 1,
        depths={index: index for index in regions},
    )
    return Room(
        seed=seed,
        config=config,
        terrain=terrain,
        entities=tuple(entities),
        topology=topology,
        shape=RoomShape.RECTANGLE,
        metadata={
            "fallback": True,
            "shape": RoomShape.RECTANGLE.value,
            "requested_normal_wipeout_balls": normal_count,
            "requested_big_wipeout_balls": big_count,
            "required_wipeout_ball_id": required_ball_id,
            "requested_temporary_bridges": (
                1 if config.require_bridge_crossing else 0
            ),
            "placed_temporary_bridges": (
                1 if config.require_bridge_crossing else 0
            ),
            "required_bridge_id": required_bridge_id,
            "hazard_tiles": len(required_bridge_tiles),
            "requested_reset_zones": 1 if config.require_reset_detour else 0,
            "placed_reset_zones": 1 if config.require_reset_detour else 0,
            "required_reset_zone_id": required_reset_zone_id,
            "obstacle_tiles": max(0, len(reset_barrier) - 2),
        },
    )

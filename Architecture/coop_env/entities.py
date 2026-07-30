"""Immutable room objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

from .requirements import AlwaysOpen, Requirement
from .utils.geometry import Rect, Vec2


class EntityKind(IntEnum):
    """Stable entity type IDs."""

    AGENT_SPAWN = 1
    EXIT_DOOR = 2
    KEY = 3
    LOCKED_DOOR = 4
    SWITCH = 5
    # 6 (pressure plates) and 7 (moving platforms) are retired. Values are
    # stable for encoding, so the gaps stay rather than renumbering.
    PUSHABLE_BLOCK = 8
    CHECKPOINT = 9
    RESET_ZONE = 10
    TEMPORARY_BRIDGE = 11
    WIPEOUT_BALL = 12


class SwitchMode(str, Enum):
    TOGGLE = "toggle"  # flips and stays flipped
    HOLD = "hold"      # active only while something rests on it
    ONESHOT = "oneshot"  # can be turned on once, never off


class WipeoutBallSize(str, Enum):
    NORMAL = "normal"
    BIG = "big"


@dataclass(frozen=True, slots=True)
class Entity:
    """Base blueprint: a stable id plus a home position."""

    id: str
    pos: Vec2

    @property
    def kind(self) -> EntityKind:  # pragma: no cover - overridden everywhere
        raise NotImplementedError

    def footprint(self) -> tuple[Vec2, ...]:
        """Every tile this entity can ever occupy."""
        return (self.pos,)

    def describe(self) -> str:
        return f"{self.kind.name.lower()} {self.id} at {tuple(self.pos)}"


@dataclass(frozen=True, slots=True)
class AgentSpawn(Entity):
    """A start tile for one agent."""

    index: int = 0

    @property
    def kind(self) -> EntityKind:
        return EntityKind.AGENT_SPAWN


@dataclass(frozen=True, slots=True)
class Key(Entity):
    """A portable unlock token."""

    color: str = "gold"
    opens: tuple[str, ...] = ()  # ids of doors this key is intended for
    agent_index: int | None = None

    @property
    def kind(self) -> EntityKind:
        return EntityKind.KEY


@dataclass(frozen=True, slots=True)
class LockedDoor(Entity):
    """A blocking tile that opens when `requirement` is satisfied.

    `latching` doors stay open once opened. Non-latching doors track the live
    requirement, which is what makes hold-switch mechanics meaningful. `timer`
    (when set) is how many ticks a latching door stays open before re-locking --
    the timed-door mechanic.
    """

    requirement: Requirement = field(default_factory=AlwaysOpen)
    latching: bool = True
    timer: int | None = None
    horizontal: bool = True
    region_a: int = -1
    region_b: int = -1

    @property
    def kind(self) -> EntityKind:
        return EntityKind.LOCKED_DOOR

    def describe(self) -> str:
        flavour = "timed" if self.timer else ("latching" if self.latching else "held")
        return f"door {self.id} at {tuple(self.pos)} [{flavour}] needs {self.requirement.describe()}"


@dataclass(frozen=True, slots=True)
class ExitDoor(Entity):
    """The episode objective tile.

    Reaching it is the point of the room; the environment only reports whether
    it is open and where it is. No reward is attached to it here -- that is for
    a future training layer to define.
    """

    requirement: Requirement = field(default_factory=AlwaysOpen)

    @property
    def kind(self) -> EntityKind:
        return EntityKind.EXIT_DOOR

    def describe(self) -> str:
        return f"exit {self.id} at {tuple(self.pos)} needs {self.requirement.describe()}"


@dataclass(frozen=True, slots=True)
class Switch(Entity):
    """A lever. HOLD switches need continuous occupancy to stay active.

    A pair of HOLD switches sharing a `group` behind a SIMULTANEOUS requirement
    is the two-agent lock: both must be weighed down at the same instant.
    """

    mode: SwitchMode = SwitchMode.TOGGLE
    group: str = ""
    controls: tuple[str, ...] = ()  # door ids this switch feeds

    @property
    def kind(self) -> EntityKind:
        return EntityKind.SWITCH


@dataclass(frozen=True, slots=True)
class PushableBlock(Entity):
    """A crate. The environment tracks where it is; nothing here decides to push it.

    Crates weigh down a HOLD switch, which is what lets one be parked on a lever
    to free up an agent.
    """

    heavy: bool = False
    target_switch_id: str | None = None
    push_from: Vec2 | None = None

    @property
    def kind(self) -> EntityKind:
        return EntityKind.PUSHABLE_BLOCK

    def describe(self) -> str:
        if self.target_switch_id is None:
            return Entity.describe(self)
        return (
            f"crate {self.id} at {tuple(self.pos)} pushes from "
            f"{tuple(self.push_from) if self.push_from is not None else '?'} "
            f"onto {self.target_switch_id}"
        )


@dataclass(frozen=True, slots=True)
class Checkpoint(Entity):
    """A progress marker. May also feed a door requirement (sequential puzzles)."""

    order: int = 0
    group: str = ""

    @property
    def kind(self) -> EntityKind:
        return EntityKind.CHECKPOINT


@dataclass(frozen=True, slots=True)
class ResetZone(Entity):
    """An area that returns whatever enters it to a spawn or checkpoint."""

    rect: Rect = Rect(0, 0, 1, 1)
    returns_to: str = ""  # checkpoint/spawn id, empty means nearest spawn

    @property
    def kind(self) -> EntityKind:
        return EntityKind.RESET_ZONE

    def footprint(self) -> tuple[Vec2, ...]:
        return tuple(self.rect.positions())


@dataclass(frozen=True, slots=True)
class TemporaryBridge(Entity):
    """Tiles that phase in and out on a fixed cycle -- a timed crossing.

    Its state is a pure function of the tick, so no simulation loop is needed.
    """

    tiles: tuple[Vec2, ...] = ()
    period: int = 12
    on_ticks: int = 6
    phase: int = 0

    @property
    def kind(self) -> EntityKind:
        return EntityKind.TEMPORARY_BRIDGE

    def footprint(self) -> tuple[Vec2, ...]:
        return self.tiles or (self.pos,)

    def is_solid_at(self, tick: int) -> bool:
        return ((tick + self.phase) % max(1, self.period)) < self.on_ticks


@dataclass(frozen=True, slots=True)
class WipeoutBall(Entity):
    """A lethal ball that advances one tile per environment tick."""

    track: tuple[Vec2, ...] = ()
    size: WipeoutBallSize = WipeoutBallSize.NORMAL

    @property
    def kind(self) -> EntityKind:
        return EntityKind.WIPEOUT_BALL

    @property
    def expected_track_length(self) -> int:
        return 7 if self.size is WipeoutBallSize.NORMAL else 11

    @property
    def collision_radius(self) -> int:
        return 0 if self.size is WipeoutBallSize.NORMAL else 1

    def position_at(self, tick: int) -> Vec2:
        path = self.track or (self.pos,)
        if len(path) == 1:
            return path[0]
        cycle = 2 * (len(path) - 1)
        offset = tick % cycle
        index = offset if offset < len(path) else cycle - offset
        return path[index]

    def collision_tiles_at(self, tick: int) -> tuple[Vec2, ...]:
        center = self.position_at(tick)
        radius = self.collision_radius
        return tuple(
            Vec2(center.x + dx, center.y + dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        )

    def footprint(self) -> tuple[Vec2, ...]:
        path = self.track or (self.pos,)
        tiles = {
            tile
            for tick in range(len(path))
            for tile in self.collision_tiles_at(tick)
        }
        return tuple(sorted(tiles, key=lambda tile: (tile.y, tile.x)))


#: Entity types that block movement while in their default state. Used by the
#: renderer and the validator to reason about occupancy without isinstance
#: chains scattered around the codebase.
BLOCKING_KINDS: frozenset[EntityKind] = frozenset(
    {EntityKind.LOCKED_DOOR, EntityKind.PUSHABLE_BLOCK}
)

"""Interactive object definitions.

Everything here is an immutable *blueprint*: the shape of the mechanism, where
it sits, and what it needs. Mutable per-episode values (is the door open, has
the key been taken) live in `EpisodeState`, which is rebuilt from these
definitions on every reset. That split is what makes reset trivially correct --
there is no in-place mutation to undo.

Adding a new mechanic means adding a dataclass here, a placement rule in
`generation/mechanisms.py`, and a glyph in the renderers. Nothing else needs to
know about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

from .requirements import AlwaysOpen, Requirement
from .utils.geometry import Rect, Vec2


class EntityKind(IntEnum):
    """Stable integer ids, suitable for a future observation encoder."""

    AGENT_SPAWN = 1
    EXIT_DOOR = 2
    KEY = 3
    LOCKED_DOOR = 4
    SWITCH = 5
    # 6 is retired (pressure plates). Values are stable for encoding, so the
    # gap stays rather than renumbering everything below it.
    MOVING_PLATFORM = 7
    PUSHABLE_BLOCK = 8
    CHECKPOINT = 9
    RESET_ZONE = 10
    TEMPORARY_BRIDGE = 11


class SwitchMode(str, Enum):
    TOGGLE = "toggle"  # flips and stays flipped
    HOLD = "hold"      # active only while something rests on it
    ONESHOT = "oneshot"  # can be turned on once, never off


class PlatformCycle(str, Enum):
    LOOP = "loop"          # ... -> last -> first -> ...
    PINGPONG = "pingpong"  # ... -> last -> last-1 -> ... -> first -> ...


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
    """A start tile reserved for one of the two future agents.

    This marks a location only. No agent object, controller, or observation is
    created here or anywhere else in the project.
    """

    index: int = 0

    @property
    def kind(self) -> EntityKind:
        return EntityKind.AGENT_SPAWN


@dataclass(frozen=True, slots=True)
class Key(Entity):
    """A portable unlock token."""

    color: str = "gold"
    opens: tuple[str, ...] = ()  # ids of doors this key is intended for

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
class MovingPlatform(Entity):
    """A tile that shuttles along a fixed track, usually over a hazard.

    Its position is a pure function of the tick (see `position_at`), so the
    whole mechanic is deterministic and needs no simulation loop.
    """

    path: tuple[Vec2, ...] = ()
    ticks_per_step: int = 2
    phase: int = 0
    cycle: PlatformCycle = PlatformCycle.PINGPONG

    @property
    def kind(self) -> EntityKind:
        return EntityKind.MOVING_PLATFORM

    def footprint(self) -> tuple[Vec2, ...]:
        return self.path or (self.pos,)

    def position_at(self, tick: int) -> Vec2:
        """Where the platform sits at `tick`. Pure, total, and reproducible."""
        if not self.path:
            return self.pos
        if len(self.path) == 1:
            return self.path[0]
        step = (tick // max(1, self.ticks_per_step)) + self.phase
        n = len(self.path)
        if self.cycle is PlatformCycle.LOOP:
            return self.path[step % n]
        span = 2 * (n - 1)
        offset = step % span
        return self.path[offset] if offset < n else self.path[span - offset]

    @property
    def period(self) -> int:
        """Full cycle length in ticks -- handy for future timing features."""
        n = len(self.path)
        if n <= 1:
            return 1
        steps = n if self.cycle is PlatformCycle.LOOP else 2 * (n - 1)
        return steps * max(1, self.ticks_per_step)


@dataclass(frozen=True, slots=True)
class PushableBlock(Entity):
    """A crate. The environment tracks where it is; nothing here decides to push it.

    Crates weigh down a HOLD switch, which is what lets one be parked on a lever
    to free up an agent.
    """

    heavy: bool = False

    @property
    def kind(self) -> EntityKind:
        return EntityKind.PUSHABLE_BLOCK


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

    Like `MovingPlatform`, its state is a pure function of the tick.
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


#: Entity types that block movement while in their default state. Used by the
#: renderer and the validator to reason about occupancy without isinstance
#: chains scattered around the codebase.
BLOCKING_KINDS: frozenset[EntityKind] = frozenset(
    {EntityKind.LOCKED_DOOR, EntityKind.PUSHABLE_BLOCK}
)

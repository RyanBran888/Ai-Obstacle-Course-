"""Unlock conditions for doors and for the exit.

A `Requirement` is a declarative description of what must be true for a
mechanism to open. It is evaluated in two very different places:

* `is_satisfied()` reads the live episode state -- this is plain mechanism
  bookkeeping, the same way a physical door reads its own latch.
* the validator reads the *structure* (which entity ids are referenced, and
  whether they must happen at the same time) to prove a room is completable
  without ever simulating anyone.

Requirements are immutable and carry no solution order. The generator never
records "how" a room is solved, only what each mechanism needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol, runtime_checkable


class TriggerMode(str, Enum):
    """How multiple triggers combine."""

    ANY = "any"                    # one of them is enough
    ALL = "all"                    # all of them, in any order, at any time
    SIMULTANEOUS = "simultaneous"  # all of them held down at the same instant


@runtime_checkable
class StateView(Protocol):
    """The read-only slice of episode state a requirement may consult."""

    def is_key_collected(self, key_id: str) -> bool: ...
    def is_switch_active(self, switch_id: str) -> bool: ...
    def is_plate_pressed(self, plate_id: str) -> bool: ...
    def is_checkpoint_reached(self, checkpoint_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class Requirement:
    """Base class. The default requirement is 'always open'."""

    def is_satisfied(self, state: StateView) -> bool:
        return True

    def referenced_ids(self) -> frozenset[str]:
        """Every entity id this requirement depends on."""
        return frozenset()

    def needs_simultaneity(self) -> bool:
        """True when two things must be triggered at the same moment."""
        return False

    def describe(self) -> str:
        return "unlocked"


@dataclass(frozen=True, slots=True)
class AlwaysOpen(Requirement):
    def describe(self) -> str:
        return "always open"


@dataclass(frozen=True, slots=True)
class NeverOpen(Requirement):
    """A permanently sealed mechanism (used for decorative walls-as-doors)."""

    def is_satisfied(self, state: StateView) -> bool:
        return False

    def describe(self) -> str:
        return "sealed"


@dataclass(frozen=True, slots=True)
class KeyRequirement(Requirement):
    key_ids: tuple[str, ...]
    mode: TriggerMode = TriggerMode.ALL

    def is_satisfied(self, state: StateView) -> bool:
        checks = (state.is_key_collected(k) for k in self.key_ids)
        return any(checks) if self.mode is TriggerMode.ANY else all(checks)

    def referenced_ids(self) -> frozenset[str]:
        return frozenset(self.key_ids)

    def describe(self) -> str:
        joiner = " or " if self.mode is TriggerMode.ANY else " + "
        return f"key({joiner.join(self.key_ids)})"


@dataclass(frozen=True, slots=True)
class SwitchRequirement(Requirement):
    switch_ids: tuple[str, ...]
    mode: TriggerMode = TriggerMode.ALL

    def is_satisfied(self, state: StateView) -> bool:
        checks = [state.is_switch_active(s) for s in self.switch_ids]
        if self.mode is TriggerMode.ANY:
            return any(checks)
        return all(checks)

    def referenced_ids(self) -> frozenset[str]:
        return frozenset(self.switch_ids)

    def needs_simultaneity(self) -> bool:
        return self.mode is TriggerMode.SIMULTANEOUS

    def describe(self) -> str:
        joiner = {
            TriggerMode.ANY: " or ",
            TriggerMode.ALL: " + ",
            TriggerMode.SIMULTANEOUS: " & ",
        }[self.mode]
        return f"switch({joiner.join(self.switch_ids)})"


@dataclass(frozen=True, slots=True)
class PlateRequirement(Requirement):
    """Pressure plates. SIMULTANEOUS plates are the classic two-agent lock."""

    plate_ids: tuple[str, ...]
    mode: TriggerMode = TriggerMode.SIMULTANEOUS

    def is_satisfied(self, state: StateView) -> bool:
        checks = [state.is_plate_pressed(p) for p in self.plate_ids]
        if self.mode is TriggerMode.ANY:
            return any(checks)
        return all(checks)

    def referenced_ids(self) -> frozenset[str]:
        return frozenset(self.plate_ids)

    def needs_simultaneity(self) -> bool:
        return self.mode is TriggerMode.SIMULTANEOUS and len(self.plate_ids) > 1

    def describe(self) -> str:
        joiner = " & " if self.mode is TriggerMode.SIMULTANEOUS else " + "
        return f"plate({joiner.join(self.plate_ids)})"


@dataclass(frozen=True, slots=True)
class CheckpointRequirement(Requirement):
    checkpoint_ids: tuple[str, ...]

    def is_satisfied(self, state: StateView) -> bool:
        return all(state.is_checkpoint_reached(c) for c in self.checkpoint_ids)

    def referenced_ids(self) -> frozenset[str]:
        return frozenset(self.checkpoint_ids)

    def describe(self) -> str:
        return f"checkpoint({' + '.join(self.checkpoint_ids)})"


@dataclass(frozen=True, slots=True)
class CompositeRequirement(Requirement):
    """Conjunction (or disjunction) of other requirements.

    This is what lets the exit demand, say, two keys *and* a switch -- the
    "multiple objectives before the exit opens" case.
    """

    parts: tuple[Requirement, ...]
    mode: TriggerMode = TriggerMode.ALL

    def is_satisfied(self, state: StateView) -> bool:
        checks = [p.is_satisfied(state) for p in self.parts]
        if self.mode is TriggerMode.ANY:
            return any(checks)
        return all(checks)

    def referenced_ids(self) -> frozenset[str]:
        out: frozenset[str] = frozenset()
        for part in self.parts:
            out |= part.referenced_ids()
        return out

    def needs_simultaneity(self) -> bool:
        return any(p.needs_simultaneity() for p in self.parts)

    def describe(self) -> str:
        joiner = " OR " if self.mode is TriggerMode.ANY else " AND "
        return "(" + joiner.join(p.describe() for p in self.parts) + ")"


def combine(parts: Iterable[Requirement], mode: TriggerMode = TriggerMode.ALL) -> Requirement:
    """Fold several requirements into one, collapsing trivial cases."""
    real = [p for p in parts if not isinstance(p, AlwaysOpen)]
    if not real:
        return AlwaysOpen()
    if len(real) == 1:
        return real[0]
    return CompositeRequirement(tuple(real), mode)

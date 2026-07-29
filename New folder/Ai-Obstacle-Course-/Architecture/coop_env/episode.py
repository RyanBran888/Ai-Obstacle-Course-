"""Episode lifecycle: which room is loaded, and how it gets reset.

`EnvironmentSession` is the object a future training loop would hold. It owns a
generator, the current room, and the current state, and it offers the four
reset behaviours the environment needs:

    session.reset()                 # brand new room from a fresh seed
    session.reset(seed=1234)        # rebuild a specific room
    session.reset(same_room=True)   # same room, mechanisms back to start
    session.reset(reroll=True)      # same seed lineage, new puzzle draw

Seeds come from a master `SeededRandom` stream, so a session started with a
master seed replays its entire sequence of episodes identically -- useful for
reproducing a training run, and the reason seed handling lives here rather than
being left to the caller.

There is no `step()`. Stepping means agents acting, and this project has none.
`advance_time()` is provided for the mechanics that are functions of the clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import GenerationConfig
from .generation.generator import GenerationOutcome, RoomGenerator
from .rng import SeededRandom, normalize_seed
from .room import Room
from .state import EpisodeState
from .validation.validator import ValidationReport


@dataclass(slots=True)
class EpisodeRecord:
    """What happened on one reset -- kept so a run can be audited later."""

    index: int
    seed: int
    attempts: int
    fallback: bool
    valid: bool
    summary: str


class EnvironmentSession:
    """Holds the current room and drives episode resets."""

    def __init__(
        self,
        config: GenerationConfig | None = None,
        master_seed: int | str | None = None,
        generator: RoomGenerator | None = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self.generator = generator or RoomGenerator(self.config)
        self._seed_stream = SeededRandom(
            master_seed if master_seed is not None else self.config.seed,
            label="episodes",
        )
        self._room: Room | None = None
        self._state: EpisodeState | None = None
        self._report: ValidationReport | None = None
        self._episode_index = -1
        self.history: list[EpisodeRecord] = []
        self.on_reset: list[Callable[[Room, EpisodeState], None]] = []
        """Observer hooks. A future training wrapper can subscribe here."""

    # -- current episode ---------------------------------------------------

    @property
    def room(self) -> Room:
        if self._room is None:
            raise RuntimeError("no room loaded yet -- call reset() first")
        return self._room

    @property
    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("no episode state yet -- call reset() first")
        return self._state

    @property
    def report(self) -> ValidationReport:
        if self._report is None:
            raise RuntimeError("no validation report yet -- call reset() first")
        return self._report

    @property
    def episode_index(self) -> int:
        return self._episode_index

    @property
    def seed(self) -> int:
        return self.room.seed

    @property
    def master_seed(self) -> int:
        return self._seed_stream.seed

    # -- resets ------------------------------------------------------------

    def reset(
        self,
        seed: int | str | None = None,
        *,
        same_room: bool = False,
        reroll: bool = False,
    ) -> EpisodeState:
        """Start a new episode.

        seed        rebuild exactly this room
        same_room   keep the current room, return mechanisms to their defaults
        reroll      regenerate from the current room's seed (same lineage)
        """
        if same_room:
            if self._room is None:
                raise RuntimeError("same_room reset requested before any room was generated")
            self._episode_index += 1
            self._state = EpisodeState.from_room(self._room)
            self._notify()
            return self._state

        if reroll:
            if self._room is None:
                raise RuntimeError("reroll requested before any room was generated")
            seed = self._room.seed

        chosen = normalize_seed(seed) if seed is not None else self.next_seed()
        outcome = self.generator.generate_with_report(chosen)
        self._adopt(outcome)
        return self.state

    def reset_state(self) -> EpisodeState:
        """Return every mechanism to its blueprint default, same room, same episode."""
        self.state.reset()
        return self.state

    def next_seed(self) -> int:
        """Draw the next seed from the session's master stream."""
        return self._seed_stream.randint(0, 2**63 - 2)

    def load(self, room: Room, report: ValidationReport | None = None) -> EpisodeState:
        """Adopt an externally supplied room (a saved level, a fixture, a test case)."""
        self._room = room
        self._state = EpisodeState.from_room(room)
        self._report = report
        self._episode_index += 1
        self._notify()
        return self._state

    def _adopt(self, outcome: GenerationOutcome) -> None:
        self._room = outcome.room
        self._report = outcome.report
        self._state = EpisodeState.from_room(outcome.room)
        self._episode_index += 1
        self.history.append(
            EpisodeRecord(
                index=self._episode_index,
                seed=outcome.room.seed,
                attempts=outcome.attempts,
                fallback=outcome.fallback,
                valid=outcome.report.ok,
                summary=outcome.room.summary(),
            )
        )
        self._notify()

    def _notify(self) -> None:
        for hook in self.on_reset:
            hook(self.room, self.state)

    # -- clock -------------------------------------------------------------

    def advance_time(self, ticks: int = 1) -> EpisodeState:
        """Advance only the time-driven mechanics (platforms, bridges, timers)."""
        return self.state.advance(ticks)

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        if self._room is None:
            return "EnvironmentSession(no room loaded)"
        return (
            f"EnvironmentSession(episode={self._episode_index}, "
            f"seed={self._room.seed}, {self._room.summary()})"
        )

    def stats(self) -> dict[str, Any]:
        return {
            "episodes": len(self.history),
            "fallbacks": sum(1 for r in self.history if r.fallback),
            "mean_attempts": (
                sum(r.attempts for r in self.history) / len(self.history)
                if self.history
                else 0.0
            ),
            "master_seed": self.master_seed,
        }

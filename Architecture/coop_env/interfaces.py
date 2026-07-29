"""Placeholders for the future agent-facing layer.

Nothing in this module is implemented, and that is deliberate. The project's
scope stops at generating, validating, rendering, and resetting environments.
What lives here is the *shape* of the seam a reinforcement-learning integration
would attach to later, written down so that the environment side can be
designed against it without anyone having to guess.

Every method below raises `NotImplementedError`. There are no observations, no
action handling, no rewards, no termination logic, and no agent objects
anywhere in this package.

Sketch of how the pieces would connect, for whoever picks this up:

    Gymnasium / PettingZoo
        wrap `EnvironmentSession`; `reset()` maps onto `session.reset()`, and a
        `step()` would read actions, apply them through the mutator methods on
        `EpisodeState` (`collect_key`, `set_switch`, `place_block`, ...), then
        call `session.advance_time(1)`.

    Unity ML-Agents
        treat `Room` as the level description to instantiate on the C# side;
        `Room.terrain.to_list()` plus `Room.entities` is enough to rebuild the
        scene, and `Room.seed` keeps the two sides in sync.

Where the environment already meets you halfway:

    Room.terrain.to_list()      flat row-major terrain, ready for an array
    Tile / EntityKind           stable integer ids for one-hot encoding
    Room.spawns                 the two start tiles
    EpisodeState.snapshot()     serialisable episode state
    EpisodeState.is_walkable()  the movement rule a physics layer would enforce
    EnvironmentSession.on_reset observer hook for a wrapper to subscribe to
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .room import Room
from .state import EpisodeState

#: The number of agent slots every generated room provides tiles for.
AGENT_SLOTS = 2

_NOT_IMPLEMENTED = (
    "Not implemented by design: this project delivers the environment only. "
    "See coop_env/interfaces.py for how a future RL layer is expected to attach."
)


@runtime_checkable
class ObservationEncoder(Protocol):
    """Turns a room and its state into whatever a learner wants to consume."""

    def encode(self, room: Room, state: EpisodeState, agent_slot: int) -> Any:
        """Build the observation for one agent slot."""
        ...

    def space(self, room: Room) -> Any:
        """Describe the observation shape (a Gym space, a dict of shapes, ...)."""
        ...


@runtime_checkable
class ActionApplier(Protocol):
    """Applies a decoded action to the environment.

    An implementation would translate an action into calls on `EpisodeState`,
    honouring `EpisodeState.is_walkable` for movement. This project supplies
    the mechanisms and the walkability rule; it does not supply the mover.
    """

    def apply(self, state: EpisodeState, agent_slot: int, action: Any) -> None: ...

    def space(self, room: Room) -> Any: ...


class MultiAgentEnvironmentAdapter:
    """Placeholder base class for a future Gymnasium/PettingZoo adapter.

    Subclass this when integrating a training framework. The environment side
    (`EnvironmentSession`, `Room`, `EpisodeState`) is complete and does not need
    changes to support it.
    """

    agent_slots: int = AGENT_SLOTS

    def observation_space(self, agent_slot: int) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def action_space(self, agent_slot: int) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def reset(self, seed: int | None = None) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def step(self, actions: Any) -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def render(self, mode: str = "ascii") -> Any:
        raise NotImplementedError(_NOT_IMPLEMENTED)


class RewardFunction:
    """Placeholder. No reward shaping is defined by this project.

    Termination, scoring, and credit assignment are training concerns and are
    left entirely to whoever builds the learning layer. The environment reports
    facts -- `EpisodeState.exit_open`, `objectives_remaining()`,
    `is_hazardous()` -- and takes no view on what they are worth.
    """

    def __call__(self, room: Room, state: EpisodeState, agent_slot: int) -> float:
        raise NotImplementedError(_NOT_IMPLEMENTED)

"""coop_env -- a procedurally generated 2D environment for future multi-agent RL.

This package generates, validates, renders, and resets randomized cooperative
rooms. It deliberately contains **no agents**: no policies, no rewards, no
training loops, no pathfinding, and no controls. See `coop_env.interfaces` for
the placeholders where a learning framework would later attach.

Typical use:

    from coop_env import EnvironmentSession, GenerationConfig

    session = EnvironmentSession(GenerationConfig.preset("standard"), master_seed=7)
    session.reset()                 # new room
    print(session.room.summary())
    session.reset(seed=1234)        # exact rebuild
    session.reset(same_room=True)   # mechanisms back to defaults

Or go straight to the generator:

    from coop_env import RoomGenerator, GenerationConfig
    room = RoomGenerator(GenerationConfig.from_complexity(0.6)).generate(seed=99)
"""

from __future__ import annotations

from .config import PRESET_COMPLEXITY, GenerationConfig, RoomShape
from .entities import (
    AgentSpawn,
    Checkpoint,
    Entity,
    EntityKind,
    ExitDoor,
    Key,
    LockedDoor,
    MovingPlatform,
    PlatformCycle,
    PushableBlock,
    ResetZone,
    Switch,
    SwitchMode,
    TemporaryBridge,
)
from .episode import EnvironmentSession, EpisodeRecord
from .generation.generator import GenerationError, GenerationOutcome, RoomGenerator
from .interfaces import (
    AGENT_SLOTS,
    ActionApplier,
    MultiAgentEnvironmentAdapter,
    ObservationEncoder,
    RewardFunction,
)
from .requirements import (
    AlwaysOpen,
    CheckpointRequirement,
    CompositeRequirement,
    KeyRequirement,
    Requirement,
    SwitchRequirement,
    TriggerMode,
)
from .rng import SeededRandom
from .room import Region, Room, RoomTopology
from .state import EpisodeState
from .tiles import Tile, is_hazard, is_walkable
from .utils.geometry import Rect, Vec2
from .validation.validator import ValidationReport, validate_room

__version__ = "1.0.0"

__all__ = [
    # configuration
    "GenerationConfig",
    "RoomShape",
    "PRESET_COMPLEXITY",
    # generation
    "RoomGenerator",
    "GenerationOutcome",
    "GenerationError",
    "SeededRandom",
    # world model
    "Room",
    "Region",
    "RoomTopology",
    "EpisodeState",
    "Tile",
    "is_walkable",
    "is_hazard",
    "Vec2",
    "Rect",
    # entities
    "Entity",
    "EntityKind",
    "AgentSpawn",
    "ExitDoor",
    "Key",
    "LockedDoor",
    "Switch",
    "SwitchMode",
    "MovingPlatform",
    "PlatformCycle",
    "PushableBlock",
    "Checkpoint",
    "ResetZone",
    "TemporaryBridge",
    # requirements
    "Requirement",
    "AlwaysOpen",
    "KeyRequirement",
    "SwitchRequirement",
    "CheckpointRequirement",
    "CompositeRequirement",
    "TriggerMode",
    # lifecycle
    "EnvironmentSession",
    "EpisodeRecord",
    # validation
    "validate_room",
    "ValidationReport",
    # future integration placeholders
    "AGENT_SLOTS",
    "ObservationEncoder",
    "ActionApplier",
    "MultiAgentEnvironmentAdapter",
    "RewardFunction",
]

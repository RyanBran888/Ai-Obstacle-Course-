"""Generation parameters and difficulty scaling.

`GenerationConfig` is the single knob-box for the generator. Every stage reads
its settings from here and nothing else, so a config plus a seed fully
determines a room.

Two ways to use it:

    GenerationConfig(seed=7, width=(20, 30), num_keys=(2, 3))   # explicit
    GenerationConfig.from_complexity(0.75, seed=7)              # one scalar

`from_complexity` interpolates every parameter along a single 0..1 axis, which
is the hook a future curriculum would drive. Explicit keyword overrides always
win over the derived values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

from .tiles import HAZARD_TILES, Tile

IntRange = tuple[int, int]


class RoomShape(str, Enum):
    """Silhouette templates for the room outline."""

    RECTANGLE = "rectangle"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    PLUS = "plus"
    DONUT = "donut"
    DIAMOND = "diamond"
    CAVERN = "cavern"      # cellular-automata smoothed blob
    TERRACE = "terrace"    # stepped / staircase silhouette


def _default_shape_weights() -> dict[RoomShape, float]:
    return {
        RoomShape.RECTANGLE: 1.6,
        RoomShape.L_SHAPE: 1.2,
        RoomShape.T_SHAPE: 1.0,
        RoomShape.PLUS: 0.9,
        RoomShape.DONUT: 0.7,
        RoomShape.DIAMOND: 0.7,
        RoomShape.CAVERN: 1.1,
        RoomShape.TERRACE: 0.8,
    }


def _default_hazard_weights() -> dict[Tile, float]:
    return {
        Tile.HAZARD_LAVA: 1.0,
        Tile.HAZARD_SPIKES: 1.0,
        Tile.HAZARD_WATER: 0.8,
        Tile.HAZARD_PIT: 0.9,
    }


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_int(a: int, b: int, t: float) -> int:
    return int(round(_lerp(a, b, t)))


def _lerp_range(a: IntRange, b: IntRange, t: float) -> IntRange:
    return (_lerp_int(a[0], b[0], t), _lerp_int(a[1], b[1], t))


@dataclass(slots=True)
class GenerationConfig:
    """Everything the generator needs, beyond the seed itself."""

    # -- reproducibility ---------------------------------------------------
    seed: int | None = None
    """Base seed. `None` means 'pick a fresh random one at generation time'."""

    # -- room shape and size ----------------------------------------------
    width: IntRange = (20, 32)
    height: IntRange = (16, 26)
    shape_weights: dict[RoomShape, float] = field(default_factory=_default_shape_weights)

    # -- region topology ---------------------------------------------------
    region_count: IntRange = (4, 8)
    """Target number of BSP sub-areas before they are merged by connectivity."""
    min_region_span: int = 5
    """Smallest allowed width/height of a BSP leaf, in tiles."""
    branching_factor: float = 0.3
    """0 = strict tree of regions (one route), 1 = keep every possible link."""
    corridor_width: IntRange = (1, 2)

    # -- terrain density ---------------------------------------------------
    obstacle_density: float = 0.06
    """Fraction of open floor turned into static blockers."""
    hazard_density: float = 0.07
    """Fraction of open floor turned into hazard surface."""
    hazard_weights: dict[Tile, float] = field(default_factory=_default_hazard_weights)
    hazard_blob_size: IntRange = (2, 7)

    # -- mechanism budget --------------------------------------------------
    num_keys: IntRange = (1, 3)
    num_locked_doors: IntRange = (1, 3)
    num_switches: IntRange = (1, 3)
    num_pushable_blocks: IntRange = (0, 2)
    num_checkpoints: IntRange = (0, 2)
    num_reset_zones: IntRange = (0, 1)
    num_temporary_bridges: IntRange = (0, 1)
    num_normal_wipeout_balls: IntRange = (0, 0)
    num_big_wipeout_balls: IntRange = (0, 0)
    require_wipeout_crossing: bool = False
    """Require one ball track to lie on every spawn-to-exit route."""
    require_bridge_crossing: bool = False
    """Require one phasing bridge to lie on every spawn-to-exit route."""
    require_reset_detour: bool = False
    """Put one reset tile on a shorter route while keeping a safe detour."""
    require_combined_course: bool = False
    """Build the serial, fully contracted cooperative course."""

    # -- puzzle structure --------------------------------------------------
    puzzle_chain_length: int = 2
    """Target depth of the lock/unlock dependency chain from spawn to exit."""
    exit_objective_count: int = 1
    """How many separate objectives the exit door demands."""
    required_cooperative_actions: int = 1
    """Gates that structurally need two agents (hold-lever, or paired levers)."""
    timed_door_probability: float = 0.15
    separate_spawns_probability: float = 0.3
    """Chance the two spawn points are placed in different regions."""
    exit_requires_both_agents: bool = False
    """When true, the validator demands both agents can stand at the exit."""
    agent_specific_keys: bool = False
    """Assign every generated key to one of the two agent slots."""
    allow_shared_keys: bool = True
    """Allow one key to unlock more than one logical doorway."""
    require_key_for_each_agent: bool = False
    """Require at least one generated key for both agent slots."""

    # -- generation control ------------------------------------------------
    max_attempts: int = 24
    """Regeneration budget before falling back to a guaranteed-simple room."""
    raise_on_failure: bool = False
    """Raise instead of emitting the fallback room when the budget runs out."""
    complexity: float | None = None
    """Set by `from_complexity`; informational once the config is built."""

    # -- housekeeping ------------------------------------------------------

    def __post_init__(self) -> None:
        self.width = _ordered(self.width)
        self.height = _ordered(self.height)
        self.region_count = _ordered(self.region_count)
        self.corridor_width = _ordered(self.corridor_width)
        self.hazard_blob_size = _ordered(self.hazard_blob_size)
        for name in (
            "num_keys",
            "num_locked_doors",
            "num_switches",
            "num_pushable_blocks",
            "num_checkpoints",
            "num_reset_zones",
            "num_temporary_bridges",
            "num_normal_wipeout_balls",
            "num_big_wipeout_balls",
        ):
            setattr(self, name, _ordered(getattr(self, name)))

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means the config is usable."""
        problems: list[str] = []
        if self.width[0] < 8 or self.height[0] < 8:
            problems.append("minimum room dimension is 8 tiles")
        if self.width[1] > 200 or self.height[1] > 200:
            problems.append("maximum room dimension is 200 tiles")
        for name in ("obstacle_density", "hazard_density"):
            value = getattr(self, name)
            if not 0.0 <= value <= 0.6:
                problems.append(f"{name} must be within 0.0..0.6 (got {value})")
        for name in (
            "branching_factor",
            "timed_door_probability",
            "separate_spawns_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                problems.append(f"{name} must be within 0.0..1.0 (got {value})")
        if self.min_region_span < 3:
            problems.append("min_region_span must be at least 3")
        if self.puzzle_chain_length < 0:
            problems.append("puzzle_chain_length must be >= 0")
        if self.required_cooperative_actions < 0:
            problems.append("required_cooperative_actions must be >= 0")
        if self.exit_objective_count < 0:
            problems.append("exit_objective_count must be >= 0")
        if self.max_attempts < 1:
            problems.append("max_attempts must be at least 1")
        for name in (
            "num_keys",
            "num_locked_doors",
            "num_switches",
            "num_pushable_blocks",
            "num_checkpoints",
            "num_reset_zones",
            "num_temporary_bridges",
            "num_normal_wipeout_balls",
            "num_big_wipeout_balls",
        ):
            if getattr(self, name)[0] < 0:
                problems.append(f"{name} must be nonnegative")
        if self.require_key_for_each_agent and not self.agent_specific_keys:
            problems.append("require_key_for_each_agent needs agent_specific_keys")
        if self.require_key_for_each_agent and self.num_keys[0] < 2:
            problems.append("require_key_for_each_agent needs at least 2 keys")
        if self.require_key_for_each_agent and self.num_locked_doors[0] < 2:
            problems.append("require_key_for_each_agent needs at least 2 locked doors")
        if self.require_key_for_each_agent and self.region_count[0] < 3:
            problems.append("require_key_for_each_agent needs at least 3 regions")
        if (
            self.require_wipeout_crossing
            and self.num_normal_wipeout_balls[0] + self.num_big_wipeout_balls[0] < 1
        ):
            problems.append("require_wipeout_crossing needs at least 1 wipeout ball")
        if self.require_bridge_crossing and self.num_temporary_bridges[0] < 1:
            problems.append("require_bridge_crossing needs at least 1 temporary bridge")
        if self.require_bridge_crossing and self.num_temporary_bridges[1] != 1:
            problems.append("require_bridge_crossing needs exactly 1 temporary bridge")
        if self.require_bridge_crossing and not any(
            weight > 0 and tile in HAZARD_TILES
            for tile, weight in self.hazard_weights.items()
        ):
            problems.append("require_bridge_crossing needs a positive hazard weight")
        if self.require_reset_detour and self.num_reset_zones != (1, 1):
            problems.append("require_reset_detour needs exactly 1 reset zone")
        if self.require_combined_course:
            if (
                self.width[1] < 37
                or self.width[0] > 38
                or self.height[1] < 25
                or self.height[0] > 32
            ):
                problems.append(
                    "require_combined_course needs dimensions overlapping 37..38 x 25..32"
                )
            if not self.agent_specific_keys or not self.require_key_for_each_agent:
                problems.append("require_combined_course needs owned keys for both agents")
            if self.allow_shared_keys:
                problems.append("require_combined_course does not allow shared keys")
            for name, wanted in (
                ("num_keys", 2),
                ("num_locked_doors", 3),
                ("num_switches", 1),
                ("num_pushable_blocks", 1),
                ("num_reset_zones", 1),
                ("num_temporary_bridges", 1),
                ("num_normal_wipeout_balls", 1),
                ("num_big_wipeout_balls", 1),
            ):
                low, high = getattr(self, name)
                if not low <= wanted <= high:
                    problems.append(
                        f"require_combined_course needs {name} to include {wanted}"
                    )
            if self.exit_objective_count != 1:
                problems.append("require_combined_course needs one exit objective")
            if self.num_checkpoints != (0, 0):
                problems.append(
                    "require_combined_course creates its checkpoint from the exit objective"
                )
            if self.required_cooperative_actions != 1:
                problems.append("require_combined_course needs one cooperative action")
            if not self.exit_requires_both_agents:
                problems.append("require_combined_course needs both agents at the exit")
            if not any(
                weight > 0 and tile in HAZARD_TILES
                for tile, weight in self.hazard_weights.items()
            ):
                problems.append("require_combined_course needs a positive hazard weight")
        if not any(w > 0 for w in self.shape_weights.values()):
            problems.append("at least one room shape needs a positive weight")
        if self.hazard_density > 0 and not any(w > 0 for w in self.hazard_weights.values()):
            problems.append("hazard_density > 0 requires at least one hazard weight")
        return problems

    def require_valid(self) -> "GenerationConfig":
        problems = self.validate()
        if problems:
            raise ValueError("invalid GenerationConfig: " + "; ".join(problems))
        return self

    def with_seed(self, seed: int | None) -> "GenerationConfig":
        return replace(self, seed=seed)

    def with_overrides(self, **overrides: Any) -> "GenerationConfig":
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape_weights"] = {k.value: v for k, v in self.shape_weights.items()}
        data["hazard_weights"] = {int(k): v for k, v in self.hazard_weights.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationConfig":
        payload = dict(data)
        if "shape_weights" in payload:
            payload["shape_weights"] = {
                RoomShape(k): float(v) for k, v in payload["shape_weights"].items()
            }
        if "hazard_weights" in payload:
            payload["hazard_weights"] = {
                Tile(int(k)): float(v) for k, v in payload["hazard_weights"].items()
            }
        for key in ("width", "height", "region_count", "corridor_width", "hazard_blob_size"):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(payload[key])
        for key in list(payload):
            if key.startswith("num_") and payload[key] is not None:
                payload[key] = tuple(payload[key])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    # -- difficulty scaling ------------------------------------------------

    @classmethod
    def from_complexity(
        cls, complexity: float, seed: int | None = None, **overrides: Any
    ) -> "GenerationConfig":
        """Derive a full config from one 0..1 difficulty scalar.

        Anchors are the `EASY` end at 0.0 and the `BRUTAL` end at 1.0. Any
        keyword in `overrides` replaces the derived value.
        """
        t = max(0.0, min(1.0, float(complexity)))
        derived = cls(
            seed=seed,
            width=_lerp_range((14, 18), (40, 56), t),
            height=_lerp_range((12, 16), (30, 42), t),
            region_count=_lerp_range((2, 3), (8, 14), t),
            min_region_span=max(4, _lerp_int(6, 5, t)),
            branching_factor=_lerp(0.10, 0.55, t),
            obstacle_density=_lerp(0.02, 0.13, t),
            hazard_density=_lerp(0.01, 0.18, t),
            hazard_blob_size=_lerp_range((2, 4), (3, 10), t),
            num_keys=_lerp_range((0, 1), (3, 5), t),
            num_locked_doors=_lerp_range((0, 1), (4, 6), t),
            num_switches=_lerp_range((0, 1), (3, 5), t),
            num_pushable_blocks=_lerp_range((0, 1), (1, 3), t),
            num_checkpoints=_lerp_range((0, 0), (2, 3), t),
            num_reset_zones=_lerp_range((0, 0), (1, 2), t),
            num_temporary_bridges=_lerp_range((0, 0), (1, 2), t),
            num_normal_wipeout_balls=(0, 0),
            num_big_wipeout_balls=(0, 0),
            puzzle_chain_length=_lerp_int(1, 5, t),
            exit_objective_count=_lerp_int(1, 3, t),
            required_cooperative_actions=_lerp_int(0, 3, t),
            timed_door_probability=_lerp(0.05, 0.35, t),
            separate_spawns_probability=_lerp(0.10, 0.60, t),
            agent_specific_keys=t >= 0.20,
            allow_shared_keys=t < 0.20,
            require_key_for_each_agent=t >= 0.60,
            complexity=t,
        )
        return replace(derived, **overrides) if overrides else derived

    @classmethod
    def preset(cls, name: str, seed: int | None = None, **overrides: Any) -> "GenerationConfig":
        """Look up a named difficulty preset."""
        key = name.strip().lower()
        if key not in PRESET_COMPLEXITY:
            options = ", ".join(sorted(PRESET_COMPLEXITY))
            raise KeyError(f"unknown preset {name!r}; available presets: {options}")
        return cls.from_complexity(PRESET_COMPLEXITY[key], seed=seed, **overrides)


def _ordered(span: IntRange) -> IntRange:
    lo, hi = int(span[0]), int(span[1])
    return (lo, hi) if lo <= hi else (hi, lo)


#: Named points along the complexity axis.
PRESET_COMPLEXITY: dict[str, float] = {
    "tutorial": 0.0,
    "easy": 0.2,
    "standard": 0.45,
    "hard": 0.7,
    "brutal": 1.0,
}

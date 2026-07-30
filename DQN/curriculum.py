from __future__ import annotations

import random
from array import array
from collections.abc import Callable, Sequence
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from env_bridge import CoopEnvBridge, GenerationConfig
from DQN.DQN_train import (
    Config,
    Evaluation,
    EvaluationEpisode,
    Trainer,
    evaluate,
    evaluate_detailed,
)
from room_manifest import (
    CurriculumRoomManifest,
    LazyRoomManifestBuilder,
    verify_training_coverage,
)

from coop_env import (
    AlwaysOpen,
    CheckpointRequirement,
    KeyRequirement,
    LockedDoor,
    Room,
    RoomShape,
    SwitchRequirement,
    SwitchMode,
    Tile,
    WipeoutBall,
    WipeoutBallSize,
)
from coop_env.rng import derive_seed
from coop_env.tiles import is_hazard, is_walkable
from coop_env.utils.geometry import Vec2

if TYPE_CHECKING:
    from DQN.DQN_rewards import CurriculumPlot


RoomCheck = Callable[[Room], bool]
EvaluationCheck = Callable[[Evaluation], bool]
FULL_COURSE_HORIZON = 200


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    config: GenerationConfig
    accepts: RoomCheck
    train_threshold: float = 0.90
    validation_threshold: float = 0.80
    max_wipeout_death_rate: float = 1.0
    lesson: str = ""
    pool_sizes: tuple[int, ...] | None = None
    objective_check: EvaluationCheck | None = None
    objective: str = ""
    required_features: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StageResult:
    stage: str
    pool_size: int
    rounds: int
    training: Evaluation
    validation: Evaluation | None
    promoted: bool
    retention: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class StageTestResult:
    stage: str
    evaluation: Evaluation
    episodes: tuple[EvaluationEpisode, ...]


def _base_config() -> GenerationConfig:
    return GenerationConfig(
        width=(10, 16),
        height=(9, 14),
        shape_weights={RoomShape.RECTANGLE: 1.0},
        region_count=(2, 3),
        min_region_span=4,
        branching_factor=0.15,
        corridor_width=(1, 1),
        obstacle_density=0.0,
        hazard_density=0.0,
        num_keys=(0, 0),
        num_locked_doors=(0, 0),
        num_switches=(0, 0),
        num_pushable_blocks=(0, 0),
        num_checkpoints=(0, 0),
        num_reset_zones=(0, 0),
        num_temporary_bridges=(0, 0),
        num_normal_wipeout_balls=(0, 0),
        num_big_wipeout_balls=(0, 0),
        require_wipeout_crossing=False,
        require_bridge_crossing=False,
        require_reset_detour=False,
        puzzle_chain_length=0,
        exit_objective_count=0,
        required_cooperative_actions=0,
        timed_door_probability=0.0,
        separate_spawns_probability=0.0,
        exit_requires_both_agents=False,
        agent_specific_keys=False,
        allow_shared_keys=True,
        require_key_for_each_agent=False,
    )


def _gate_kinds(room: Room) -> tuple[str, ...]:
    return tuple(str(gate.get("kind")) for gate in room.metadata.get("gates", ()))


def _has_gate_kinds(room: Room, **wanted: int) -> bool:
    return Counter(_gate_kinds(room)) == Counter(wanted)


def _gate_groups_are_required(room: Room) -> bool:
    gates = room.metadata.get("gates", ())
    return bool(gates) and all(_gate_group_is_required(room, gate) for gate in gates)


def _gate_group_is_required(room: Room, gate: dict) -> bool:
    door_ids = gate.get("doors", ())
    blocked: set[Vec2] = set()
    for door_id in door_ids:
        door = room.find(door_id)
        if not isinstance(door, LockedDoor):
            return False
        blocked.add(door.pos)

    floor = {
        pos
        for pos in room.terrain.positions()
        if is_walkable(room.terrain[pos])
    }
    floor.difference_update(block.pos for block in room.blocks)
    if not all(_connected(spawn.pos, room.exit.pos, floor) for spawn in room.spawns):
        return False
    return all(
        not _connected(spawn.pos, room.exit.pos, floor - blocked)
        for spawn in room.spawns
    )


def _connected(start: Vec2, goal: Vec2, floor: set[Vec2]) -> bool:
    return _shortest_distance(start, goal, floor) is not None


def _shortest_distance(
    start: Vec2,
    goal: Vec2,
    floor: set[Vec2],
) -> int | None:
    if start not in floor or goal not in floor:
        return None
    seen = {start: 0}
    pending = [start]
    for current in pending:
        if current == goal:
            return seen[current]
        for neighbor in current.neighbors4():
            if neighbor in floor and neighbor not in seen:
                seen[neighbor] = seen[current] + 1
                pending.append(neighbor)
    return None


def _forces_detour(room: Room, restored_tiles: set[Vec2]) -> bool:
    safe = {
        pos
        for pos in room.terrain.positions()
        if is_walkable(room.terrain[pos]) and not is_hazard(room.terrain[pos])
    }
    direct = safe | restored_tiles
    for spawn in room.spawns:
        safe_steps = _shortest_distance(spawn.pos, room.exit.pos, safe)
        direct_steps = _shortest_distance(spawn.pos, room.exit.pos, direct)
        if (
            safe_steps is None
            or direct_steps is None
            or safe_steps <= direct_steps
        ):
            return False
    return True


def default_stages() -> tuple[CurriculumStage, ...]:
    all_shapes = {shape: 1.0 for shape in RoomShape}
    all_shape_features = tuple(f"shape:{shape.value}" for shape in RoomShape)
    open_config = _base_config()
    layout_config = _base_config().with_overrides(
        width=(14, 22),
        height=(12, 18),
        shape_weights=all_shapes,
        region_count=(2, 4),
    )
    obstacle_config = _base_config().with_overrides(
        width=(12, 18),
        height=(10, 15),
        region_count=(2, 4),
        obstacle_density=0.08,
    )
    hazard_config = _base_config().with_overrides(
        width=(14, 20),
        height=(11, 16),
        region_count=(2, 4),
        obstacle_density=0.03,
        hazard_density=0.08,
        hazard_blob_size=(1, 3),
    )
    checkpoint_exit_config = _base_config().with_overrides(
        exit_objective_count=1,
    )
    key_config = _base_config().with_overrides(
        num_keys=(1, 1),
        num_locked_doors=(1, 1),
        puzzle_chain_length=1,
    )
    shared_key_config = _base_config().with_overrides(
        width=(14, 20),
        height=(10, 15),
        region_count=(3, 4),
        num_keys=(1, 1),
        num_locked_doors=(2, 2),
        puzzle_chain_length=2,
        allow_shared_keys=True,
    )
    owned_key_config = _base_config().with_overrides(
        width=(16, 20),
        height=(11, 15),
        region_count=(3, 4),
        num_keys=(2, 2),
        num_locked_doors=(2, 3),
        puzzle_chain_length=2,
        agent_specific_keys=True,
        allow_shared_keys=False,
        require_key_for_each_agent=True,
    )
    normal_ball_config = _base_config().with_overrides(
        width=(9, 9),
        height=(12, 14),
        region_count=(1, 1),
        num_normal_wipeout_balls=(1, 1),
        require_wipeout_crossing=True,
    )
    normal_crossing_config = _base_config().with_overrides(
        width=(9, 9),
        height=(16, 18),
        region_count=(1, 1),
        num_normal_wipeout_balls=(1, 1),
        require_wipeout_crossing=True,
        exit_requires_both_agents=True,
    )
    big_ball_config = _base_config().with_overrides(
        width=(15, 15),
        height=(14, 16),
        region_count=(1, 1),
        num_big_wipeout_balls=(1, 1),
        require_wipeout_crossing=True,
    )
    big_crossing_config = _base_config().with_overrides(
        width=(15, 15),
        height=(18, 20),
        region_count=(1, 1),
        num_big_wipeout_balls=(1, 1),
        require_wipeout_crossing=True,
        exit_requires_both_agents=True,
    )
    switch_config = _base_config().with_overrides(
        num_switches=(1, 1),
        num_locked_doors=(1, 1),
        puzzle_chain_length=1,
    )
    switch_exit_config = _base_config().with_overrides(
        num_switches=(1, 1),
        exit_objective_count=1,
    )
    timed_switch_config = switch_config.with_overrides(
        timed_door_probability=1.0,
    )
    hold_switch_config = _base_config().with_overrides(
        width=(14, 18),
        height=(10, 15),
        region_count=(2, 3),
        num_switches=(1, 1),
        num_locked_doors=(1, 1),
        puzzle_chain_length=1,
        required_cooperative_actions=1,
    )
    paired_switch_config = _base_config().with_overrides(
        width=(16, 20),
        height=(12, 16),
        region_count=(2, 3),
        num_switches=(2, 2),
        num_locked_doors=(1, 1),
        puzzle_chain_length=1,
        required_cooperative_actions=1,
        exit_requires_both_agents=True,
    )
    crate_hold_config = hold_switch_config.with_overrides(
        num_pushable_blocks=(1, 1),
    )
    reset_config = _base_config().with_overrides(
        width=(10, 10),
        height=(16, 18),
        region_count=(1, 1),
        num_reset_zones=(1, 1),
        require_reset_detour=True,
    )
    bridge_config = _base_config().with_overrides(
        width=(8, 8),
        height=(16, 18),
        region_count=(1, 1),
        num_temporary_bridges=(1, 1),
        require_bridge_crossing=True,
        exit_requires_both_agents=True,
    )
    checkpoint_config = owned_key_config.with_overrides(exit_objective_count=1)
    static_layout_mix_config = owned_key_config.with_overrides(
        width=(22, 30),
        height=(16, 22),
        shape_weights=all_shapes,
        region_count=(3, 5),
        obstacle_density=0.05,
        hazard_density=0.04,
        hazard_blob_size=(1, 4),
        exit_objective_count=1,
    )
    owned_normal_config = _base_config().with_overrides(
        width=(9, 9),
        height=(24, 28),
        region_count=(3, 3),
        num_keys=(2, 2),
        num_locked_doors=(2, 2),
        puzzle_chain_length=2,
        agent_specific_keys=True,
        allow_shared_keys=False,
        require_key_for_each_agent=True,
        num_normal_wipeout_balls=(1, 1),
        require_wipeout_crossing=True,
        exit_requires_both_agents=True,
    )
    owned_both_balls_config = owned_normal_config.with_overrides(
        width=(15, 15),
        height=(32, 38),
        num_big_wipeout_balls=(1, 1),
    )
    cooperation_mix_config = _base_config().with_overrides(
        width=(22, 28),
        height=(16, 22),
        region_count=(4, 5),
        num_keys=(2, 2),
        num_locked_doors=(3, 3),
        num_switches=(2, 2),
        puzzle_chain_length=3,
        required_cooperative_actions=1,
        exit_requires_both_agents=True,
        agent_specific_keys=True,
        allow_shared_keys=False,
        require_key_for_each_agent=True,
    )
    normal_layout_mix_config = owned_normal_config.with_overrides(
        width=(14, 22),
        height=(24, 34),
        shape_weights=all_shapes,
        region_count=(3, 5),
    )
    both_balls_layout_mix_config = owned_both_balls_config.with_overrides(
        width=(18, 26),
        height=(28, 38),
        shape_weights=all_shapes,
        region_count=(3, 5),
    )
    cooperation_layout_mix_config = cooperation_mix_config.with_overrides(
        width=(26, 34),
        height=(20, 28),
        shape_weights=all_shapes,
        region_count=(4, 6),
    )
    tutorial_config = GenerationConfig.preset(
        "tutorial",
        required_cooperative_actions=0,
        timed_door_probability=0.0,
        num_pushable_blocks=(0, 0),
        num_reset_zones=(0, 0),
        num_temporary_bridges=(0, 0),
        width=(24, 30),
        height=(16, 22),
        shape_weights={RoomShape.RECTANGLE: 1.0},
        obstacle_density=0.0,
        hazard_density=0.0,
        region_count=(3, 4),
        num_keys=(2, 3),
        num_locked_doors=(2, 3),
        puzzle_chain_length=2,
        agent_specific_keys=True,
        allow_shared_keys=False,
        require_key_for_each_agent=True,
        num_normal_wipeout_balls=(1, 1),
        num_big_wipeout_balls=(1, 1),
    )
    full_course_config = _base_config().with_overrides(
        width=(37, 38),
        height=(25, 32),
        region_count=(5, 5),
        obstacle_density=0.04,
        hazard_density=0.04,
        hazard_blob_size=(1, 4),
        num_keys=(2, 2),
        num_locked_doors=(3, 3),
        num_switches=(1, 1),
        num_pushable_blocks=(1, 1),
        num_checkpoints=(0, 0),
        num_reset_zones=(1, 1),
        num_temporary_bridges=(1, 1),
        num_normal_wipeout_balls=(1, 1),
        num_big_wipeout_balls=(1, 1),
        puzzle_chain_length=3,
        exit_objective_count=1,
        required_cooperative_actions=1,
        exit_requires_both_agents=True,
        agent_specific_keys=True,
        allow_shared_keys=False,
        require_key_for_each_agent=True,
        require_combined_course=True,
    )

    def open_room(room: Room) -> bool:
        return (
            not room.keys
            and not room.doors
            and not room.switches
            and not room.checkpoints
            and not room.blocks
            and not room.reset_zones
            and not room.bridges
            and not room.wipeout_balls
            and isinstance(room.exit.requirement, AlwaysOpen)
        )

    def obstacle_room(room: Room) -> bool:
        obstacles = {
            pos
            for pos in room.terrain.positions()
            if room.terrain[pos] is Tile.OBSTACLE
        }
        return open_room(room) and bool(obstacles) and _forces_detour(room, obstacles)

    def hazard_room(room: Room) -> bool:
        hazards = {
            pos
            for pos in room.terrain.positions()
            if is_hazard(room.terrain[pos])
        }
        return (
            open_room(room)
            and bool(hazards)
            and _forces_detour(room, hazards)
        )

    def checkpoint_exit(room: Room) -> bool:
        return (
            not room.keys
            and not room.doors
            and not room.switches
            and len(room.checkpoints) == 1
            and isinstance(room.exit.requirement, CheckpointRequirement)
        )

    def key_door(room: Room) -> bool:
        return (
            len(room.keys) == 1
            and _has_gate_kinds(room, key=1)
            and _gate_groups_are_required(room)
            and all(isinstance(door.requirement, KeyRequirement) for door in room.doors)
            and not room.switches
            and not room.checkpoints
            and isinstance(room.exit.requirement, AlwaysOpen)
        )

    def shared_key_chain(room: Room) -> bool:
        return (
            len(room.keys) == 1
            and _has_gate_kinds(room, key=1, shared_key=1)
            and _gate_groups_are_required(room)
            and len(room.keys[0].opens) >= 2
            and all(isinstance(door.requirement, KeyRequirement) for door in room.doors)
            and isinstance(room.exit.requirement, AlwaysOpen)
        )

    def key_door_checkpoint(room: Room) -> bool:
        return (
            len(room.keys) == 2
            and _has_gate_kinds(room, key=2)
            and _gate_groups_are_required(room)
            and all(isinstance(door.requirement, KeyRequirement) for door in room.doors)
            and not room.switches
            and len(room.checkpoints) == 1
            and isinstance(room.exit.requirement, CheckpointRequirement)
        )

    def owned_key_doors(room: Room) -> bool:
        return (
            len(room.keys) == 2
            and {key.agent_index for key in room.keys} == {0, 1}
            and _has_gate_kinds(room, key=2)
            and _gate_groups_are_required(room)
            and all(isinstance(door.requirement, KeyRequirement) for door in room.doors)
            and all(len(key.opens) >= 1 for key in room.keys)
            and not room.switches
            and not room.checkpoints
            and isinstance(room.exit.requirement, AlwaysOpen)
        )

    def normal_wipeout(room: Room) -> bool:
        return (
            len(room.wipeout_balls) == 1
            and room.wipeout_balls[0].size is WipeoutBallSize.NORMAL
            and required_wipeout(room, WipeoutBallSize.NORMAL)
        )

    def big_wipeout(room: Room) -> bool:
        return (
            len(room.wipeout_balls) == 1
            and room.wipeout_balls[0].size is WipeoutBallSize.BIG
            and required_wipeout(room, WipeoutBallSize.BIG)
        )

    def required_wipeout(room: Room, size: WipeoutBallSize) -> bool:
        required = room.find(room.metadata.get("required_wipeout_ball_id", ""))
        return (
            len(room.wipeout_balls) >= 1
            and isinstance(required, WipeoutBall)
            and required.size is size
        )

    def switch_door(room: Room) -> bool:
        return (
            not room.keys
            and _has_gate_kinds(room, switch=1)
            and _gate_groups_are_required(room)
            and all(
                isinstance(door.requirement, SwitchRequirement)
                for door in room.doors
            )
            and len(room.switches) == 1
            and room.switches[0].mode is SwitchMode.TOGGLE
            and not room.checkpoints
            and isinstance(room.exit.requirement, AlwaysOpen)
        )

    def switch_exit(room: Room) -> bool:
        return (
            not room.keys
            and not room.doors
            and len(room.switches) == 1
            and room.switches[0].mode is SwitchMode.ONESHOT
            and isinstance(room.exit.requirement, SwitchRequirement)
        )

    def timed_switch_door(room: Room) -> bool:
        return (
            not room.keys
            and _has_gate_kinds(room, timed_switch=1)
            and _gate_groups_are_required(room)
            and len(room.switches) == 1
            and room.switches[0].mode is SwitchMode.TOGGLE
            and all(door.timer is not None for door in room.doors)
        )

    def hold_switch_door(room: Room) -> bool:
        return (
            _has_gate_kinds(room, hold_switch=1)
            and _gate_groups_are_required(room)
            and len(room.switches) == 1
            and room.switches[0].mode is SwitchMode.HOLD
            and all(not door.latching for door in room.doors)
        )

    def paired_levers(room: Room) -> bool:
        groups = {switch.group for switch in room.switches}
        return (
            _has_gate_kinds(room, paired_levers=1)
            and _gate_groups_are_required(room)
            and len(room.switches) == 2
            and all(switch.mode is SwitchMode.HOLD for switch in room.switches)
            and len(groups) == 1
            and "" not in groups
            and all(
                isinstance(door.requirement, SwitchRequirement)
                and door.requirement.needs_simultaneity()
                and door.latching
                for door in room.doors
            )
            and room.config.exit_requires_both_agents
        )

    def crate_hold(room: Room) -> bool:
        return (
            hold_switch_door(room)
            and len(room.blocks) == 1
            and room.blocks[0].target_switch_id == room.switches[0].id
            and room.blocks[0].push_from is not None
            and bool(room.metadata.get("crate_switch_pairs"))
        )

    def reset_detour(room: Room) -> bool:
        return (
            len(room.reset_zones) == 1
            and room.reset_zones[0].id
            == room.metadata.get("required_reset_zone_id")
            and room.config.require_reset_detour
        )

    def required_bridge(room: Room) -> bool:
        required_id = room.metadata.get("required_bridge_id")
        return (
            len(room.bridges) == 1
            and room.bridges[0].id == required_id
            and room.config.require_bridge_crossing
        )

    def static_layout_mix(room: Room) -> bool:
        return (
            key_door_checkpoint(room)
            and int(room.metadata.get("obstacle_tiles", 0)) > 0
            and int(room.metadata.get("hazard_tiles", 0)) > 0
        )

    def owned_normal_mix(room: Room) -> bool:
        return (
            owned_key_doors(room)
            and len(room.wipeout_balls) == 1
            and required_wipeout(room, WipeoutBallSize.NORMAL)
        )

    def owned_both_balls_mix(room: Room) -> bool:
        return (
            owned_key_doors(room)
            and Counter(ball.size for ball in room.wipeout_balls)
            == Counter(
                {
                    WipeoutBallSize.NORMAL: 1,
                    WipeoutBallSize.BIG: 1,
                }
            )
            and required_wipeout(room, WipeoutBallSize.BIG)
        )

    def cooperation_mix(room: Room) -> bool:
        return (
            len(room.keys) == 2
            and {key.agent_index for key in room.keys} == {0, 1}
            and _has_gate_kinds(room, key=2, paired_levers=1)
            and _gate_groups_are_required(room)
            and len(room.switches) == 2
            and all(switch.mode is SwitchMode.HOLD for switch in room.switches)
            and room.config.exit_requires_both_agents
        )

    def tutorial_mix(room: Room) -> bool:
        sizes = Counter(ball.size for ball in room.wipeout_balls)
        return (
            {key.agent_index for key in room.keys} == {0, 1}
            and len(room.keys) >= 2
            and len(room.metadata.get("gates", ())) >= 2
            and _gate_groups_are_required(room)
            and sizes[WipeoutBallSize.NORMAL] == 1
            and sizes[WipeoutBallSize.BIG] == 1
        )

    def full_course(room: Room) -> bool:
        sizes = Counter(ball.size for ball in room.wipeout_balls)
        course = room.metadata.get("combined_course")
        sections = (
            tuple(section.get("id") for section in course.get("sections", ()))
            if isinstance(course, dict)
            else ()
        )
        return (
            room.config.require_combined_course
            and isinstance(course, dict)
            and sections
            == (
                "owned_key_0",
                "owned_key_1",
                "crate_hold",
                "wipeout_cut",
                "bridge_cut",
                "reset_detour",
                "checkpoint_exit",
            )
            and {key.agent_index for key in room.keys} == {0, 1}
            and _has_gate_kinds(room, key=2, hold_switch=1)
            and len(room.switches) == 1
            and room.switches[0].mode is SwitchMode.HOLD
            and len(room.blocks) == 1
            and room.blocks[0].target_switch_id == room.switches[0].id
            and room.blocks[0].push_from is not None
            and len(room.checkpoints) == 1
            and isinstance(room.exit.requirement, CheckpointRequirement)
            and len(room.reset_zones) == 1
            and len(room.bridges) == 1
            and sizes[WipeoutBallSize.NORMAL] == 1
            and sizes[WipeoutBallSize.BIG] == 1
            and int(room.metadata.get("obstacle_tiles", 0)) > 0
            and int(room.metadata.get("hazard_tiles", 0)) > 0
            and course.get("required_ball_size") in {"normal", "big"}
        )

    return (
        CurriculumStage(
            "open_navigation",
            open_config,
            open_room,
            0.95,
            0.90,
            lesson="Reach the exit on the original open procedural rooms.",
        ),
        CurriculumStage(
            "layout_variation",
            layout_config,
            open_room,
            0.95,
            0.90,
            lesson="Generalize navigation across every room silhouette.",
            required_features=all_shape_features,
        ),
        CurriculumStage(
            "obstacle_navigation",
            obstacle_config,
            obstacle_room,
            0.93,
            0.86,
            lesson="Route both agents around obstacles that force a detour.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "hazard_avoidance",
            hazard_config,
            hazard_room,
            0.92,
            0.84,
            lesson="Route both agents around hazards that force a safe detour.",
            pool_sizes=(1, 4, 16),
            objective_check=lambda result: result.hazard_entry_rate <= 0.10,
            objective="hazard_entry_rate <= 10%",
        ),
        CurriculumStage(
            "checkpoint_exit",
            checkpoint_exit_config,
            checkpoint_exit,
            0.92,
            0.84,
            lesson="Activate a checkpoint before exiting.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "key_door",
            key_config,
            key_door,
            0.90,
            0.80,
            lesson="Collect one key, open its required door, then exit.",
        ),
        CurriculumStage(
            "shared_key_chain",
            shared_key_config,
            shared_key_chain,
            0.90,
            0.80,
            lesson="Reuse one shared key through two required doors.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "owned_key_doors",
            owned_key_config,
            owned_key_doors,
            0.88,
            0.78,
            lesson="Each agent collects its own key through a two-door chain.",
        ),
        CurriculumStage(
            "normal_wipeout",
            normal_ball_config,
            normal_wipeout,
            0.85,
            0.75,
            0.10,
            lesson="Let one agent learn a required 7x1 moving-ball crossing.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "normal_wipeout_crossing",
            normal_crossing_config,
            lambda room: required_wipeout(room, WipeoutBallSize.NORMAL),
            0.82,
            0.72,
            0.12,
            lesson="Coordinate both agents through the required 7x1 ball lane.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "big_wipeout",
            big_ball_config,
            big_wipeout,
            0.80,
            0.70,
            0.12,
            lesson="Let one agent learn the big ball's required 3x3 crossing.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "big_wipeout_crossing",
            big_crossing_config,
            lambda room: required_wipeout(room, WipeoutBallSize.BIG),
            0.78,
            0.68,
            0.15,
            lesson="Coordinate both agents through the required big-ball lane.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "oneshot_switch_exit",
            switch_exit_config,
            switch_exit,
            0.92,
            0.84,
            lesson="Activate a one-shot switch before reaching the exit.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "switch_door",
            switch_config,
            switch_door,
            0.90,
            0.80,
            lesson="Toggle one switch to open its required door.",
        ),
        CurriculumStage(
            "timed_switch_door",
            timed_switch_config,
            timed_switch_door,
            0.86,
            0.76,
            lesson="Open and cross a timed door before it closes; re-arm after mistakes.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "hold_switch_door",
            hold_switch_config,
            hold_switch_door,
            0.84,
            0.74,
            lesson="Leave one agent holding a lever while the other exits.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "paired_levers",
            paired_switch_config,
            paired_levers,
            0.84,
            0.74,
            lesson="Assign one agent to each simultaneous lever.",
            pool_sizes=(1, 4, 16),
        ),
        CurriculumStage(
            "crate_hold_switch",
            crate_hold_config,
            crate_hold,
            0.82,
            0.72,
            lesson="Push the aligned crate onto a HOLD switch.",
            pool_sizes=(1, 4, 16),
            objective_check=lambda result: result.crate_switch_rate >= 0.70,
            objective="crate_switch_rate >= 70%",
        ),
        CurriculumStage(
            "reset_detour",
            reset_config,
            reset_detour,
            0.90,
            0.82,
            lesson="Recognize the reset shortcut and take the safe detour.",
            pool_sizes=(1, 4, 16),
            objective_check=lambda result: result.mean_reset_entries <= 0.15,
            objective="mean_reset_entries <= 0.15",
        ),
        CurriculumStage(
            "temporary_bridge",
            bridge_config,
            required_bridge,
            0.82,
            0.72,
            lesson="Wait for a required temporary bridge and cross safely.",
            pool_sizes=(1, 4, 16),
            objective_check=lambda result: result.bridge_fall_rate <= 0.10,
            objective="bridge_fall_rate <= 10%",
        ),
        CurriculumStage(
            "key_door_checkpoint",
            checkpoint_config,
            key_door_checkpoint,
            0.90,
            0.80,
            lesson="Complete both owned key doors and the exit checkpoint.",
        ),
        CurriculumStage(
            "static_layout_mix",
            static_layout_mix_config,
            static_layout_mix,
            0.84,
            0.74,
            lesson="Transfer owned-key and checkpoint skills across varied layouts.",
            required_features=all_shape_features,
        ),
        CurriculumStage(
            "owned_keys_normal_wipeout",
            owned_normal_config,
            owned_normal_mix,
            0.80,
            0.70,
            0.12,
            lesson="Combine owned keys with a required normal-ball crossing.",
        ),
        CurriculumStage(
            "owned_keys_both_wipeouts",
            owned_both_balls_config,
            owned_both_balls_mix,
            0.76,
            0.66,
            0.15,
            lesson="Cross the required big ball while tracking a normal-ball distractor.",
        ),
        CurriculumStage(
            "cooperation_mix",
            cooperation_mix_config,
            cooperation_mix,
            0.78,
            0.68,
            lesson="Combine both owned keys with a required paired-lever gate.",
        ),
        CurriculumStage(
            "tutorial_mix",
            tutorial_config,
            tutorial_mix,
            0.78,
            0.68,
            0.15,
            lesson="Rehearse the procedural tutorial distribution.",
        ),
        CurriculumStage(
            "normal_wipeout_layout_mix",
            normal_layout_mix_config,
            owned_normal_mix,
            0.74,
            0.64,
            0.15,
            lesson="Generalize owned keys and a required normal-ball crossing across layouts.",
            required_features=all_shape_features,
        ),
        CurriculumStage(
            "both_wipeouts_layout_mix",
            both_balls_layout_mix_config,
            owned_both_balls_mix,
            0.72,
            0.62,
            0.17,
            lesson="Generalize a required big-ball crossing with a normal ball across layouts.",
            required_features=all_shape_features,
        ),
        CurriculumStage(
            "cooperation_layout_mix",
            cooperation_layout_mix_config,
            cooperation_mix,
            0.72,
            0.62,
            lesson="Generalize owned keys and paired cooperation across every layout.",
            required_features=all_shape_features,
        ),
        CurriculumStage(
            "full_course_mix",
            full_course_config,
            full_course,
            0.72,
            0.62,
            0.18,
            lesson="Solve the serial course where every generated object has a defined role.",
            objective_check=lambda result: (
                result.crate_switch_rate >= 0.70
                and result.bridge_fall_rate <= 0.10
                and result.mean_reset_entries <= 0.15
                and result.hazard_entry_rate <= 0.15
            ),
            objective=(
                "crate >= 70%, bridge falls <= 10%, "
                "reset entries <= 0.15, hazard entries <= 15%"
            ),
            required_features=(
                "contract:combined_course",
                "contract:required_wipeout:normal",
                "contract:required_wipeout:big",
                "tile:lava",
                "tile:spikes",
                "tile:water",
                "tile:pit",
            ),
        ),
    )


class CurriculumRunner:
    def __init__(
        self,
        trainer: Trainer,
        *,
        stages: Sequence[CurriculumStage] | None = None,
        pool_sizes: Sequence[int] = (1, 4, 16, 64),
        validation_size: int = 64,
        test_size: int = 256,
        episodes_per_seed: int = 50,
        max_rounds: int = 8,
        run_seed: int = 0,
        data_seed: int = 0,
        live: bool = False,
        plot_every: int = 100,
        plot_max_points: int = 5_000,
        graph_path: str | None = "curriculum_training.png",
        promotion_passes: int = 1,
        require_full_coverage: bool = True,
        retention_size: int = 8,
        retention_margin: float = 0.15,
    ) -> None:
        if not pool_sizes or any(size < 1 for size in pool_sizes):
            raise ValueError("pool_sizes must contain positive values")
        if (
            validation_size < 1
            or test_size < 1
            or episodes_per_seed < 1
            or max_rounds < 1
        ):
            raise ValueError("curriculum sizes and rounds must be positive")
        if plot_every < 1 or plot_max_points < 1:
            raise ValueError("plot settings must be positive")
        if promotion_passes < 1:
            raise ValueError("promotion_passes must be positive")
        if retention_size < 1 or not 0.0 <= retention_margin <= 1.0:
            raise ValueError("retention settings are invalid")
        self.trainer = trainer
        self.stages = tuple(stages or default_stages())
        self.pool_sizes = tuple(sorted(set(pool_sizes)))
        for stage in self.stages:
            sizes = stage.pool_sizes or self.pool_sizes
            if not sizes or any(size < 1 for size in sizes):
                raise ValueError(f"{stage.name} has invalid pool sizes")
        self.validation_size = validation_size
        self.test_size = test_size
        self.episodes_per_seed = episodes_per_seed
        self.max_rounds = max_rounds
        self.run_seed = run_seed
        self.data_seed = data_seed
        self.live = live
        self.plot_every = plot_every
        self.plot_max_points = plot_max_points
        self.graph_path = graph_path
        self.promotion_passes = promotion_passes
        self.require_full_coverage = require_full_coverage
        self.retention_size = retention_size
        self.retention_margin = retention_margin
        self.results: list[StageResult] = []
        self.test_results: list[StageTestResult] = []
        self._manifest_builder = LazyRoomManifestBuilder(
            self.stages,
            data_seed=self.data_seed,
        )
        self.room_manifest: CurriculumRoomManifest = (
            self._manifest_builder.snapshot()
        )
        self.training_features: tuple[str, ...] = ()
        self._test_started = False
        self._plot: CurriculumPlot | None = None

    @property
    def completed(self) -> bool:
        expected = sum(
            len(stage.pool_sizes or self.pool_sizes) for stage in self.stages
        )
        return (
            len(self.results) == expected
            and all(result.promoted for result in self.results)
        )

    def prepare_room_manifest(self) -> CurriculumRoomManifest:
        self.room_manifest = self._manifest_builder.snapshot()
        return self.room_manifest

    def run(self) -> list[StageResult]:
        from DQN.DQN_rewards import CurriculumPlot

        if self.results:
            raise RuntimeError("this curriculum runner has already trained")
        self.prepare_room_manifest()
        plot = self._plot
        created_plot = False
        if plot is None and self.live:
            plot = CurriculumPlot(
                interactive=self.live,
                every=self.plot_every,
                max_points=self.plot_max_points,
            )
            self._plot = plot
            created_plot = True
        plot_returns = array("f")
        plot_completed = array("f")
        plot_deaths = array("f")
        plot_hazards = array("f")
        plot_bridge_falls = array("f")
        plot_crate_switches = array("f")
        plot_resets = array("f")
        plot_steps = array("f")
        plot_epsilons = array("f")
        stage_marks: list[tuple[int, str]] = []
        evaluation_marks: list[
            tuple[int, float, float | None, float | None]
        ] = []
        if plot is not None and self.live and created_plot:
            status = "opened" if plot.visible else "unavailable"
            print(f"Live dashboard {status} ({plot.backend})", flush=True)

        def finish_plot() -> None:
            nonlocal plot
            if plot is None and self.graph_path:
                plot = CurriculumPlot(
                    interactive=False,
                    every=self.plot_every,
                    max_points=self.plot_max_points,
                )
                for episode, name in stage_marks:
                    plot.mark_stage(episode, name)
                for evaluation in evaluation_marks:
                    plot.add_evaluation(*evaluation)
            if plot is None:
                return
            plot.update(
                plot_returns,
                plot_completed,
                plot_deaths,
                plot_steps,
                plot_epsilons,
                hazards=plot_hazards,
                bridge_falls=plot_bridge_falls,
                crate_switches=plot_crate_switches,
                resets=plot_resets,
                force=True,
            )
            if self.graph_path:
                plot.save(self.graph_path)
                print(f"Dashboard saved to {self.graph_path}", flush=True)
            plot.close()
            self._plot = None

        def stage_rooms(
            stage: CurriculumStage,
            split: str,
            count: int,
            feature_target: int,
        ):
            def show_progress(message: str) -> None:
                print(message, flush=True)
                if plot is not None and plot.visible:
                    plot.set_status(f"Staging rooms: {message.strip()}")

            records = self._manifest_builder.ensure(
                stage.name,
                split,
                count,
                feature_target=feature_target,
                progress=show_progress,
            )
            generated = self._manifest_builder.take_rooms(stage.name, split)
            return (
                tuple(record.seed for record in records),
                tuple(room for _, room in generated),
            )

        def finish_manifest(require_all: bool) -> None:
            self.room_manifest = self._manifest_builder.snapshot()
            self.training_features = verify_training_coverage(
                self.room_manifest,
                {
                    stage.name: len(
                        self.room_manifest.stage(stage.name).train
                    )
                    for stage in self.stages
                },
                require_all=require_all,
            )

        rehearsal: list[tuple[CoopEnvBridge, tuple[int, ...]]] = []
        retention_sets: list[
            tuple[str, CoopEnvBridge, tuple[int, ...]]
        ] = []
        for stage_index, stage in enumerate(self.stages):
            stage_pool_sizes = tuple(
                sorted(set(stage.pool_sizes or self.pool_sizes))
            )
            largest_pool = stage_pool_sizes[-1]
            stage_marks.append((len(plot_returns), stage.name))
            if plot is not None:
                plot.mark_stage(len(plot_returns), stage.name)
            if stage.lesson:
                print(f"\n{stage.name}: {stage.lesson}", flush=True)
            if stage.objective:
                print(f"Promotion objective: {stage.objective}", flush=True)
            stage_config = stage.config
            train_seeds: tuple[int, ...] = ()
            validation_seeds: tuple[int, ...] = ()

            train_env = CoopEnvBridge(
                stage_config,
                seed=self.data_seed,
                max_steps=self.trainer.cfg.max_steps,
                shaping_gamma=self.trainer.cfg.gamma,
                record_metrics=False,
            )
            train_env.set_room_cache_limit(largest_pool)
            training_eval_env = CoopEnvBridge(
                stage_config,
                seed=self.data_seed,
                max_steps=self.trainer.cfg.max_steps,
                shaping_gamma=self.trainer.cfg.gamma,
                record_metrics=False,
            )
            training_eval_env.set_room_cache_limit(largest_pool)
            validation_eval_env = CoopEnvBridge(
                stage_config,
                seed=self.data_seed,
                max_steps=self.trainer.cfg.max_steps,
                shaping_gamma=self.trainer.cfg.gamma,
                record_metrics=False,
            )
            validation_eval_env.set_room_cache_limit(self.validation_size)
            self.trainer.set_env(train_env, clear_replay=stage_index > 0)
            if stage_index > 0:
                self.trainer.reheat_exploration()

            for pool_size in stage_pool_sizes:
                train_seeds, new_train_rooms = stage_rooms(
                    stage,
                    "train",
                    pool_size,
                    largest_pool,
                )
                train_env.cache_rooms(new_train_rooms)
                training_eval_env.cache_rooms(new_train_rooms)
                if pool_size == largest_pool:
                    validation_seeds, new_validation_rooms = stage_rooms(
                        stage,
                        "validation",
                        self.validation_size,
                        self.validation_size,
                    )
                    validation_eval_env.cache_rooms(new_validation_rooms)
                if plot is not None and plot.visible:
                    plot.set_context(stage.name, pool_size)
                pool = train_seeds[:pool_size]
                rng = random.Random(
                    derive_seed(self.run_seed, f"{stage.name}:pool:{pool_size}")
                )
                streak = 0
                round_index = 0
                training_eval: Evaluation | None = None
                validation_eval: Evaluation | None = None
                retention_eval: tuple[tuple[str, float], ...] = ()

                for round_index in range(1, self.max_rounds + 1):
                    episodes = max(50, self.episodes_per_seed * pool_size)
                    for _ in range(episodes):
                        if rehearsal and rng.random() < 0.20:
                            old_env, old_seeds = rng.choice(rehearsal)
                            outcome = self.trainer.run_episode(
                                seed=rng.choice(old_seeds),
                                env=old_env,
                            )
                        else:
                            outcome = self.trainer.run_episode(seed=rng.choice(pool))
                        plot_returns.append(outcome.reward)
                        plot_completed.append(float(outcome.completed))
                        plot_deaths.append(
                            float(outcome.metrics["wipeout_deaths"] > 0)
                        )
                        plot_hazards.append(
                            float(outcome.metrics["hazards"] > 0)
                        )
                        plot_bridge_falls.append(
                            float(outcome.metrics["bridge_falls"] > 0)
                        )
                        plot_crate_switches.append(
                            float(outcome.metrics["crate_switches_solved"] > 0)
                        )
                        plot_resets.append(
                            float(outcome.metrics["reset_zones"] > 0)
                        )
                        plot_steps.append(float(outcome.steps))
                        plot_epsilons.append(self.trainer.epsilon())
                        if plot is not None and plot.visible:
                            plot.update(
                                plot_returns,
                                plot_completed,
                                plot_deaths,
                                plot_steps,
                                plot_epsilons,
                                hazards=plot_hazards,
                                bridge_falls=plot_bridge_falls,
                                crate_switches=plot_crate_switches,
                                resets=plot_resets,
                            )

                    training_eval = evaluate(
                        self.trainer.agents,
                        training_eval_env,
                        pool,
                    )
                    validation_eval = None
                    passed = (
                        training_eval.success_rate >= stage.train_threshold
                        and training_eval.wipeout_death_rate
                        <= stage.max_wipeout_death_rate
                        and (
                            stage.objective_check is None
                            or stage.objective_check(training_eval)
                        )
                    )
                    if pool_size == largest_pool:
                        validation_eval = evaluate(
                            self.trainer.agents,
                            validation_eval_env,
                            validation_seeds,
                        )
                        passed = (
                            passed
                            and validation_eval.success_rate
                            >= stage.validation_threshold
                            and validation_eval.wipeout_death_rate
                            <= stage.max_wipeout_death_rate
                            and (
                                stage.objective_check is None
                                or stage.objective_check(validation_eval)
                            )
                        )
                        retention_eval = self._evaluate_retention(retention_sets)
                        passed = passed and all(
                            rate
                            >= max(
                                0.50,
                                self.stages[index].validation_threshold
                                - self.retention_margin,
                            )
                            for index, (_, rate) in enumerate(retention_eval)
                        )
                    evaluation_mark = (
                        len(plot_returns),
                        training_eval.success_rate,
                        (
                            validation_eval.success_rate
                            if validation_eval is not None
                            else None
                        ),
                        (
                            min(rate for _, rate in retention_eval)
                            if retention_eval
                            else None
                        ),
                    )
                    evaluation_marks.append(evaluation_mark)
                    if plot is not None:
                        plot.add_evaluation(
                            *evaluation_mark,
                        )
                        if plot.visible:
                            plot.update(
                                plot_returns,
                                plot_completed,
                                plot_deaths,
                                plot_steps,
                                plot_epsilons,
                                hazards=plot_hazards,
                                bridge_falls=plot_bridge_falls,
                                crate_switches=plot_crate_switches,
                                resets=plot_resets,
                                force=True,
                            )

                    streak = streak + 1 if passed else 0
                    self._print_round(
                        stage,
                        pool_size,
                        round_index,
                        training_eval,
                        validation_eval,
                        retention_eval,
                    )
                    if streak >= self.promotion_passes:
                        break

                assert training_eval is not None
                promoted = streak >= self.promotion_passes
                result = StageResult(
                    stage=stage.name,
                    pool_size=pool_size,
                    rounds=round_index,
                    training=training_eval,
                    validation=validation_eval,
                    promoted=promoted,
                    retention=retention_eval,
                )
                self.results.append(result)
                if not promoted:
                    finish_manifest(False)
                    finish_plot()
                    return self.results
            train_env.set_room_cache_limit(min(4, largest_pool))
            rehearsal.append((train_env, train_seeds[:largest_pool]))
            retention_seeds = validation_seeds[: self.retention_size]
            for seed in retention_seeds:
                validation_eval_env.reset(seed=seed)
            validation_eval_env.set_room_cache_limit(len(retention_seeds))
            retention_sets.append(
                (stage.name, validation_eval_env, retention_seeds)
            )

        finish_manifest(self.require_full_coverage)
        finish_plot()
        return self.results

    def evaluate_final_test(self) -> list[StageTestResult]:
        if self._test_started:
            raise RuntimeError("the final test has already been started")
        if not self.completed:
            raise RuntimeError("the final test requires a completed curriculum")

        self._test_started = True
        print("\nFinal greedy test on untouched rooms", flush=True)
        for stage in self.stages:
            records = self._manifest_builder.ensure(
                stage.name,
                "test",
                self.test_size,
                feature_target=self.test_size,
                progress=lambda message: print(message, flush=True),
            )
            generated = self._manifest_builder.take_rooms(stage.name, "test")
            seeds = tuple(record.seed for record in records)
            test_env = CoopEnvBridge(
                stage.config,
                seed=self.data_seed,
                max_steps=self.trainer.cfg.max_steps,
                shaping_gamma=self.trainer.cfg.gamma,
                record_metrics=False,
            )
            test_env.set_room_cache_limit(self.test_size)
            test_env.cache_rooms(room for _, room in generated)
            evaluation, episodes = evaluate_detailed(
                self.trainer.agents,
                test_env,
                seeds,
            )
            result = StageTestResult(stage.name, evaluation, episodes)
            self.test_results.append(result)
            self._print_test(result)
        self.room_manifest = self._manifest_builder.snapshot()
        return list(self.test_results)

    def _evaluate_retention(
        self,
        retention_sets: Sequence[
            tuple[str, CoopEnvBridge, tuple[int, ...]]
        ],
    ) -> tuple[tuple[str, float], ...]:
        results: list[tuple[str, float]] = []
        for stage_name, env, seeds in retention_sets:
            result = evaluate(
                self.trainer.agents,
                env,
                seeds,
            )
            results.append((stage_name, result.success_rate))
        return tuple(results)

    @staticmethod
    def _print_round(
        stage: CurriculumStage,
        pool_size: int,
        round_index: int,
        training: Evaluation,
        validation: Evaluation | None,
        retention: tuple[tuple[str, float], ...],
    ) -> None:
        message = (
            f"{stage.name:22} pool={pool_size:>2} round={round_index:>2} "
            f"train={training.success_rate:>6.1%}"
        )
        if validation is not None:
            message += f" validation={validation.success_rate:>6.1%}"
        if retention:
            message += f" retention={min(rate for _, rate in retention):>6.1%}"
        print(message, flush=True)
        print(
            "  train metrics: "
            + CurriculumRunner._metric_summary(training),
            flush=True,
        )
        if validation is not None:
            print(
                "  validation metrics: "
                + CurriculumRunner._metric_summary(validation),
                flush=True,
            )

    @staticmethod
    def _metric_summary(evaluation: Evaluation) -> str:
        return (
            f"reward={evaluation.mean_return:.2f} "
            f"episode_steps={evaluation.mean_episode_steps:.1f} "
            f"success_steps={evaluation.mean_steps:.1f} "
            f"keys={evaluation.mean_keys:.2f} "
            f"doors={evaluation.mean_doors:.2f} "
            f"switches={evaluation.mean_switches:.2f} "
            f"checkpoints={evaluation.mean_checkpoints:.2f} "
            f"wipeout={evaluation.wipeout_death_rate:.1%} "
            f"hazard={evaluation.hazard_entry_rate:.1%} "
            f"bridge_fall={evaluation.bridge_fall_rate:.1%} "
            f"crate={evaluation.crate_switch_rate:.1%} "
            f"reset_mean={evaluation.mean_reset_entries:.2f}"
        )

    @staticmethod
    def _print_test(result: StageTestResult) -> None:
        evaluation = result.evaluation
        print(
            f"{result.stage:22} test={evaluation.success_rate:>6.1%} "
            f"success={evaluation.completed}/{evaluation.episodes} "
            f"reward={evaluation.mean_return:.2f} "
            f"timeouts={evaluation.timeouts} "
            f"episode_steps={evaluation.mean_episode_steps:.1f} "
            f"success_steps={evaluation.mean_steps:.1f}",
            flush=True,
        )


def make_runner(
    cfg: Config | None = None,
    *,
    stages: Sequence[CurriculumStage] | None = None,
    **kwargs,
) -> CurriculumRunner:
    training_config = cfg or Config()
    if stages is None and training_config.max_steps < FULL_COURSE_HORIZON:
        raise ValueError(
            f"default curriculum needs max_steps >= {FULL_COURSE_HORIZON}"
        )
    selected_stages = tuple(stages or default_stages())
    first_stage = selected_stages[0]
    env = CoopEnvBridge(
        first_stage.config,
        seed=training_config.seed,
        max_steps=training_config.max_steps,
        shaping_gamma=training_config.gamma,
        record_metrics=False,
    )
    trainer = Trainer(env, training_config)
    kwargs.setdefault("require_full_coverage", stages is None)
    return CurriculumRunner(trainer, stages=selected_stages, **kwargs)

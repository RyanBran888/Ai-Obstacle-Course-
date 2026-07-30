from __future__ import annotations

import hashlib
import json
import os
import random
import time
from array import array
from collections.abc import Callable, Mapping, Sequence
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from env_bridge import CoopEnvBridge, GenerationConfig
from DQN.DQN_model import LEGACY_POLICY_CONTRACT, POLICY_CONTRACT
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
    manifest_from_dict,
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
EvaluationDeficit = Callable[[Evaluation], float]
FULL_COURSE_HORIZON = 200
CONSOLIDATION_ROLLBACK_PATIENCE = 8
REPAIR_BRIDGE_INDEX = 13
REPAIR_BRIDGE_STAGE = "oneshot_hazard_detour"


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
    objective_deficit: EvaluationDeficit | None = None
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
    scheduled_pool_size: int | None = None
    recovery_rounds: int = 0
    best_round: int = 0
    expansions: tuple[int, ...] = ()
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StageTestResult:
    stage: str
    evaluation: Evaluation
    episodes: tuple[EvaluationEpisode, ...]


def _evaluation_payload(evaluation: Evaluation | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        field.name: getattr(evaluation, field.name)
        for field in fields(Evaluation)
    }


def _evaluation_from_payload(payload: Mapping[str, Any] | None) -> Evaluation | None:
    if payload is None:
        return None
    return Evaluation(
        **{
            field.name: payload[field.name]
            for field in fields(Evaluation)
        }
    )


def _result_payload(result: StageResult) -> dict[str, Any]:
    return {
        "stage": result.stage,
        "pool_size": result.pool_size,
        "scheduled_pool_size": result.scheduled_pool_size,
        "rounds": result.rounds,
        "recovery_rounds": result.recovery_rounds,
        "best_round": result.best_round,
        "expansions": list(result.expansions),
        "training": _evaluation_payload(result.training),
        "validation": _evaluation_payload(result.validation),
        "promoted": result.promoted,
        "retention": [list(item) for item in result.retention],
        "failure_reasons": list(result.failure_reasons),
    }


def _result_from_payload(payload: Mapping[str, Any]) -> StageResult:
    training = _evaluation_from_payload(payload["training"])
    if training is None:
        raise ValueError("recovery result is missing its training evaluation")
    return StageResult(
        stage=str(payload["stage"]),
        pool_size=int(payload["pool_size"]),
        scheduled_pool_size=(
            int(payload["scheduled_pool_size"])
            if payload.get("scheduled_pool_size") is not None
            else None
        ),
        rounds=int(payload["rounds"]),
        recovery_rounds=int(payload.get("recovery_rounds", 0)),
        best_round=int(payload.get("best_round", 0)),
        expansions=tuple(int(value) for value in payload.get("expansions", ())),
        training=training,
        validation=_evaluation_from_payload(payload.get("validation")),
        promoted=bool(payload["promoted"]),
        retention=tuple(
            (str(name), float(rate))
            for name, rate in payload.get("retention", ())
        ),
        failure_reasons=tuple(
            str(reason) for reason in payload.get("failure_reasons", ())
        ),
    )


def _test_result_payload(result: StageTestResult) -> dict[str, Any]:
    return {
        "stage": result.stage,
        "evaluation": _evaluation_payload(result.evaluation),
        "episodes": [
            {
                field.name: getattr(episode, field.name)
                for field in fields(EvaluationEpisode)
            }
            for episode in result.episodes
        ],
    }


def _test_result_from_payload(payload: Mapping[str, Any]) -> StageTestResult:
    evaluation = _evaluation_from_payload(payload["evaluation"])
    if evaluation is None:
        raise ValueError("recovery test result is missing its evaluation")
    episodes = tuple(
        EvaluationEpisode(
            **{
                field.name: episode[field.name]
                for field in fields(EvaluationEpisode)
            }
        )
        for episode in payload.get("episodes", ())
    )
    return StageTestResult(
        stage=str(payload["stage"]),
        evaluation=evaluation,
        episodes=episodes,
    )


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    oneshot_hazard_config = switch_exit_config.with_overrides(
        width=(14, 20),
        height=(11, 16),
        region_count=(2, 4),
        obstacle_density=0.0,
        hazard_density=0.08,
        hazard_blob_size=(1, 3),
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

    def oneshot_hazard_detour(room: Room) -> bool:
        hazards = {
            pos
            for pos in room.terrain.positions()
            if is_hazard(room.terrain[pos])
        }
        return (
            switch_exit(room)
            and bool(hazards)
            and _forces_detour(room, hazards)
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
            objective_deficit=lambda result: max(
                0.0,
                result.hazard_entry_rate - 0.10,
            ) / 0.10,
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
            "oneshot_hazard_detour",
            oneshot_hazard_config,
            oneshot_hazard_detour,
            0.90,
            0.80,
            lesson="Activate a one-shot switch while routing around hazards.",
            pool_sizes=(1, 4, 16),
            objective_check=lambda result: result.hazard_entry_rate <= 0.10,
            objective_deficit=lambda result: max(
                0.0,
                result.hazard_entry_rate - 0.10,
            ) / 0.10,
            objective="hazard_entry_rate <= 10%",
            required_features=(
                "switch_mode:oneshot",
                "tile:lava",
                "tile:spikes",
                "tile:water",
                "tile:pit",
            ),
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
            objective_deficit=lambda result: max(
                0.0,
                0.70 - result.crate_switch_rate,
            ) / 0.70,
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
            objective_deficit=lambda result: max(
                0.0,
                result.mean_reset_entries - 0.15,
            ) / 0.15,
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
            objective_deficit=lambda result: max(
                0.0,
                result.bridge_fall_rate - 0.10,
            ) / 0.10,
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
            objective_deficit=lambda result: max(
                max(0.0, 0.70 - result.crate_switch_rate) / 0.70,
                max(0.0, result.bridge_fall_rate - 0.10) / 0.10,
                max(0.0, result.mean_reset_entries - 0.15) / 0.15,
                max(0.0, result.hazard_entry_rate - 0.15) / 0.15,
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
        recovery_rounds: int = 8,
        recovery_pool_max: int = 128,
        recovery_expansions: int = 2,
        progress_path: str | None = None,
        resume_from: str | None = None,
        progress_contract: Mapping[str, Any] | None = None,
        extend_stopped_rounds: int = 0,
        retention_upgrade: bool = False,
        repair_upgrade: bool = False,
        policy_upgrade: bool = False,
        planner_upgrade: bool = False,
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
        if (
            retention_size < 1
            or retention_size > validation_size
            or not 0.0 <= retention_margin <= 1.0
        ):
            raise ValueError("retention settings are invalid")
        if (
            recovery_rounds < 0
            or recovery_pool_max < 1
            or recovery_expansions < 0
            or extend_stopped_rounds < 0
        ):
            raise ValueError("recovery settings are invalid")
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
        self.recovery_rounds = recovery_rounds
        self.recovery_pool_max = recovery_pool_max
        self.recovery_expansions = recovery_expansions
        self.progress_path = (
            Path(progress_path).expanduser() if progress_path else None
        )
        self._external_progress_contract = dict(progress_contract or {})
        self.extend_stopped_rounds = extend_stopped_rounds
        if sum(
            bool(value)
            for value in (
                retention_upgrade,
                repair_upgrade,
                policy_upgrade,
                planner_upgrade,
            )
        ) > 1:
            raise ValueError("only one recovery upgrade may be selected")
        self._retention_upgrade = bool(retention_upgrade)
        self._repair_upgrade = bool(repair_upgrade)
        self._policy_upgrade = bool(policy_upgrade)
        self._planner_upgrade = bool(planner_upgrade)
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
        self._resume_active: dict[str, Any] | None = None
        self._resume_status: str | None = None
        self._test_model_sha256: str | None = None
        self._force_replay_refill = False
        if resume_from is not None:
            self._load_progress(Path(resume_from).expanduser())

    @property
    def completed(self) -> bool:
        expected = sum(
            len(stage.pool_sizes or self.pool_sizes) for stage in self.stages
        )
        return (
            len(self.results) == expected
            and all(result.promoted for result in self.results)
        )

    @property
    def recovery_status(self) -> str | None:
        return self._resume_status

    @property
    def model_sha256(self) -> str:
        return self._model_sha256()

    def prepare_room_manifest(self) -> CurriculumRoomManifest:
        self.room_manifest = self._manifest_builder.snapshot()
        return self.room_manifest

    def _progress_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trainer": asdict(self.trainer.cfg),
            "device": self.trainer.device.type,
            "cpu_threads": (
                torch.get_num_threads()
                if self.trainer.device.type == "cpu"
                else None
            ),
            "run_seed": self.run_seed,
            "data_seed": self.data_seed,
            "pool_sizes": list(self.pool_sizes),
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "episodes_per_seed": self.episodes_per_seed,
            "max_rounds": self.max_rounds,
            "promotion_passes": self.promotion_passes,
            "retention_size": self.retention_size,
            "retention_margin": self.retention_margin,
            "recovery_rounds": self.recovery_rounds,
            "recovery_pool_max": self.recovery_pool_max,
            "recovery_expansions": self.recovery_expansions,
            "stages": [
                {
                    "name": stage.name,
                    "config": stage.config.to_dict(),
                    "pool_sizes": list(stage.pool_sizes or self.pool_sizes),
                    "train_threshold": stage.train_threshold,
                    "validation_threshold": stage.validation_threshold,
                    "max_wipeout_death_rate": stage.max_wipeout_death_rate,
                    "objective": stage.objective,
                    "required_features": list(stage.required_features),
                }
                for stage in self.stages
            ],
            "external": self._external_progress_contract,
        }

    def _model_sha256(self) -> str:
        digest = hashlib.sha256()
        for learner_index, learner in enumerate(self.trainer.learners):
            digest.update(str(learner_index).encode("ascii"))
            for name, tensor in learner.net.state_dict().items():
                value = tensor.detach().cpu().contiguous()
                digest.update(name.encode("utf-8"))
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(str(tuple(value.shape)).encode("ascii"))
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _active_payload(active: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if active is None:
            return None
        return {
            "stage_index": int(active["stage_index"]),
            "pool_index": int(active["pool_index"]),
            "scheduled_pool_size": int(active["scheduled_pool_size"]),
            "active_pool_size": int(active["active_pool_size"]),
            "total_rounds": int(active["total_rounds"]),
            "normal_rounds": int(active["normal_rounds"]),
            "recovery_rounds": int(active["recovery_rounds"]),
            "phase": str(active["phase"]),
            "phase_rounds": int(active["phase_rounds"]),
            "streak": int(active["streak"]),
            "expansions": list(active["expansions"]),
            "rng_state": active["rng_state"],
            "best_rank": list(active["best_rank"]),
            "best_round": int(active["best_round"]),
            "best_training": _evaluation_payload(active["best_training"]),
            "best_validation": _evaluation_payload(active["best_validation"]),
            "best_retention": [
                list(item) for item in active["best_retention"]
            ],
            "best_failures": list(active["best_failures"]),
            "best_retention_deficits": dict(
                active.get("best_retention_deficits", {})
            ),
            "latest_retention": [
                list(item) for item in active["latest_retention"]
            ],
            "latest_failures": list(active["latest_failures"]),
            "latest_retention_deficits": dict(
                active.get("latest_retention_deficits", {})
            ),
            "best_learner_state": active["best_learner_state"],
            "phase_limit": (
                int(active["phase_limit"])
                if active.get("phase_limit") is not None
                else None
            ),
            "needs_replay_refill": bool(active["needs_replay_refill"]),
            "needs_baseline_assessment": bool(
                active.get("needs_baseline_assessment", False)
            ),
            "consolidation_bad_rounds": int(
                active.get("consolidation_bad_rounds", 0)
            ),
        }

    @staticmethod
    def _active_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        training = _evaluation_from_payload(payload["best_training"])
        if training is None:
            raise ValueError("active recovery state is missing training metrics")
        return {
            "stage_index": int(payload["stage_index"]),
            "pool_index": int(payload["pool_index"]),
            "scheduled_pool_size": int(payload["scheduled_pool_size"]),
            "active_pool_size": int(payload["active_pool_size"]),
            "total_rounds": int(payload["total_rounds"]),
            "normal_rounds": int(payload["normal_rounds"]),
            "recovery_rounds": int(payload["recovery_rounds"]),
            "phase": str(payload["phase"]),
            "phase_rounds": int(payload["phase_rounds"]),
            "streak": int(payload["streak"]),
            "expansions": [
                int(value) for value in payload.get("expansions", ())
            ],
            "rng_state": payload["rng_state"],
            "best_rank": tuple(
                float(value) for value in payload["best_rank"]
            ),
            "best_round": int(payload["best_round"]),
            "best_training": training,
            "best_validation": _evaluation_from_payload(
                payload.get("best_validation")
            ),
            "best_retention": tuple(
                (str(name), float(rate))
                for name, rate in payload.get("best_retention", ())
            ),
            "best_failures": tuple(
                str(reason) for reason in payload.get("best_failures", ())
            ),
            "best_retention_deficits": {
                str(name): float(value)
                for name, value in payload.get(
                    "best_retention_deficits",
                    {},
                ).items()
            },
            "latest_retention": tuple(
                (str(name), float(rate))
                for name, rate in payload.get("latest_retention", ())
            ),
            "latest_failures": tuple(
                str(reason) for reason in payload.get("latest_failures", ())
            ),
            "latest_retention_deficits": {
                str(name): float(value)
                for name, value in payload.get(
                    "latest_retention_deficits",
                    {},
                ).items()
            },
            "best_learner_state": payload["best_learner_state"],
            "phase_limit": (
                int(payload["phase_limit"])
                if payload.get("phase_limit") is not None
                else None
            ),
            "needs_replay_refill": bool(
                payload.get("needs_replay_refill", False)
            ),
            "needs_baseline_assessment": bool(
                payload.get("needs_baseline_assessment", False)
            ),
            "consolidation_bad_rounds": int(
                payload.get("consolidation_bad_rounds", 0)
            ),
        }

    def _save_progress(
        self,
        status: str,
        active: Mapping[str, Any] | None,
    ) -> float:
        if self.progress_path is None:
            return 0.0
        started = time.perf_counter()
        contract = self._progress_contract()
        payload = {
            "schema_version": 1,
            "status": status,
            "contract": contract,
            "contract_sha256": _hash_payload(contract),
            "results": [_result_payload(result) for result in self.results],
            "test_results": [
                _test_result_payload(result) for result in self.test_results
            ],
            "test_model_sha256": self._test_model_sha256,
            "active": self._active_payload(active),
            "manifest": self._manifest_builder.snapshot().as_dict(),
            "trainer": self.trainer.recovery_state(),
        }
        target = self.progress_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(target)
        try:
            directory = os.open(target.parent, os.O_RDONLY)
        except OSError:
            return time.perf_counter() - started
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return time.perf_counter() - started

    def _load_progress(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"recovery state does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError("curriculum recovery schema does not match")
        saved_contract = payload.get("contract")
        if not isinstance(saved_contract, dict):
            raise ValueError("curriculum recovery contract is missing")
        if payload.get("contract_sha256") != _hash_payload(saved_contract):
            raise ValueError("curriculum recovery contract is corrupted")
        contract = self._progress_contract()
        exact_contract = saved_contract == contract
        if not exact_contract and not self._compatible_source_upgrade(
            saved_contract,
            contract,
        ):
            raise ValueError(
                "recovery state does not match this code, seed, or training config"
            )
        if not exact_contract:
            label = (
                "training-repair-v4"
                if self._repair_upgrade
                else (
                    "policy-v2"
                    if self._policy_upgrade
                    else (
                        "planner-v3"
                        if self._planner_upgrade
                        else "retention-v2"
                    )
                )
            )
            print(
                f"Accepted the {label} source upgrade; the next recovery "
                "checkpoint will use the new contract.",
                flush=True,
            )
        status = str(payload.get("status"))
        if status not in {
            "training",
            "stopped",
            "completed",
            "test_started",
            "tested",
        }:
            raise ValueError(f"unsupported recovery status {status!r}")
        if self.extend_stopped_rounds and status != "stopped":
            raise ValueError(
                "--extend-stopped-rounds only applies to a stopped state"
            )
        if self._repair_upgrade and not exact_contract:
            self._validate_repair_cursor(payload)
            manifest_payload = self._repair_manifest(payload["manifest"])
        else:
            manifest_payload = payload["manifest"]
        manifest = manifest_from_dict(manifest_payload)
        self._manifest_builder = LazyRoomManifestBuilder(
            self.stages,
            data_seed=self.data_seed,
            initial=manifest,
        )
        self.room_manifest = manifest
        self.results = [
            _result_from_payload(result)
            for result in payload.get("results", ())
        ]
        self.test_results = [
            _test_result_from_payload(result)
            for result in payload.get("test_results", ())
        ]
        if self._policy_upgrade and not exact_contract:
            self._upgrade_policy_states(payload)
        self.trainer.load_recovery_state(payload["trainer"])
        self._test_model_sha256 = payload.get("test_model_sha256")
        if (
            status in {"test_started", "tested"}
            and self._test_model_sha256 != self._model_sha256()
        ):
            raise ValueError("sealed final-test model hash does not match")
        active_payload = payload.get("active")
        self._resume_active = (
            self._active_from_payload(active_payload)
            if active_payload is not None
            else None
        )
        if self._resume_active is not None and not exact_contract:
            self._resume_active["best_rank"] = ()
            self._resume_active["needs_baseline_assessment"] = True
            if self._planner_upgrade:
                self.trainer.load_learner_state(
                    self._resume_active["best_learner_state"]
                )
        if status == "stopped":
            if self._resume_active is None:
                raise ValueError("stopped recovery state has no active pool")
            if self.extend_stopped_rounds < 1:
                raise RuntimeError(
                    "recovery budget is exhausted; pass "
                    "--extend-stopped-rounds N to authorize more tuning"
                )
            self._resume_active["phase"] = "extension"
            self._resume_active["phase_limit"] = self.extend_stopped_rounds
            self._resume_active["phase_rounds"] = 0
            self._resume_active["streak"] = 0
            self._resume_active["consolidation_bad_rounds"] = 0
            self.trainer.load_learner_state(
                self._resume_active["best_learner_state"]
            )
            self._resume_active["latest_retention"] = (
                self._resume_active["best_retention"]
            )
            self._resume_active["latest_failures"] = (
                self._resume_active["best_failures"]
            )
            self._resume_active["latest_retention_deficits"] = dict(
                self._resume_active["best_retention_deficits"]
            )
        if status in {"test_started", "tested"}:
            self._test_started = True
            if self._resume_active is not None:
                raise ValueError("sealed final-test state has an active training pool")
        self._resume_status = status
        if self.progress_path is None:
            self.progress_path = path
        print(
            f"Resumed {status} curriculum state from {path}.",
            flush=True,
        )
        if status in {"training", "stopped"}:
            if self._resume_active is not None:
                self._resume_active["needs_replay_refill"] = True
            else:
                self._force_replay_refill = True
            print("The replay buffer will warm up again.", flush=True)

    @staticmethod
    def _upgrade_policy_states(payload: Mapping[str, Any]) -> None:
        def upgrade(states: Any, label: str) -> None:
            if not isinstance(states, (list, tuple)) or not states:
                raise ValueError(f"{label} learner state is missing")
            for state in states:
                if not isinstance(state, dict):
                    raise ValueError(f"{label} learner state is invalid")
                if state.get("policy") != LEGACY_POLICY_CONTRACT:
                    raise ValueError(
                        f"{label} does not use the exact policy-v1 contract"
                    )
                state["policy"] = dict(POLICY_CONTRACT)

        trainer = payload.get("trainer")
        if not isinstance(trainer, Mapping):
            raise ValueError("policy upgrade trainer state is missing")
        upgrade(trainer.get("learners"), "trainer")

        active = payload.get("active")
        if isinstance(active, Mapping):
            best = active.get("best_learner_state")
            if best is not None:
                upgrade(best, "best checkpoint")

    def _validate_repair_cursor(self, payload: Mapping[str, Any]) -> None:
        if (
            payload.get("status") != "stopped"
            or payload.get("test_results")
            or payload.get("test_model_sha256") is not None
        ):
            raise ValueError("training repair requires an untested stopped state")
        active = payload.get("active")
        if not isinstance(active, Mapping):
            raise ValueError("training repair requires an active pool")
        if (
            int(active.get("stage_index", -1)) != REPAIR_BRIDGE_INDEX - 1
            or int(active.get("pool_index", -1)) != 2
            or int(active.get("scheduled_pool_size", -1)) != 16
        ):
            raise ValueError("training repair source cursor is not supported")

        expected: list[tuple[str, int]] = []
        for stage_index, stage in enumerate(
            self.stages[:REPAIR_BRIDGE_INDEX]
        ):
            pools = tuple(sorted(set(stage.pool_sizes or self.pool_sizes)))
            limit = 2 if stage_index == REPAIR_BRIDGE_INDEX - 1 else len(pools)
            expected.extend((stage.name, pool) for pool in pools[:limit])
        results = payload.get("results", ())
        if not isinstance(results, (list, tuple)) or len(results) != len(expected):
            raise ValueError("training repair results are not the expected prefix")
        for result, (stage_name, pool_size) in zip(
            results,
            expected,
            strict=True,
        ):
            if not isinstance(result, Mapping):
                raise ValueError("training repair result is invalid")
            scheduled = result.get("scheduled_pool_size")
            if scheduled is None:
                scheduled = result.get("pool_size")
            if (
                result.get("stage") != stage_name
                or int(scheduled) != pool_size
                or not bool(result.get("promoted"))
            ):
                raise ValueError("training repair results are not the expected prefix")

    def _repair_manifest(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        saved_stages = payload.get("stages", ())
        if not isinstance(saved_stages, (list, tuple)) or any(
            not isinstance(stage, Mapping) for stage in saved_stages
        ):
            raise ValueError("training repair room manifest is invalid")
        stages: list[dict[str, Any]] = [
            dict(stage) for stage in saved_stages
        ]
        current_names = [stage.name for stage in self.stages]
        saved_names = [str(stage.get("stage")) for stage in stages]
        expected_saved = (
            current_names[:REPAIR_BRIDGE_INDEX]
            + current_names[REPAIR_BRIDGE_INDEX + 1 :]
        )
        if saved_names != expected_saved:
            raise ValueError("training repair room manifest is not compatible")

        bridge = self.stages[REPAIR_BRIDGE_INDEX]
        config = bridge.config.to_dict()
        config_json = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        )
        stages.insert(
            REPAIR_BRIDGE_INDEX,
            {
                "stage": bridge.name,
                "config": config,
                "config_sha256": hashlib.sha256(
                    config_json.encode("utf-8")
                ).hexdigest(),
                "splits": {
                    "train": [],
                    "validation": [],
                    "test": [],
                },
                "selection_cursors": {
                    "train": 0,
                    "validation": 0,
                    "test": 0,
                },
                "feature_targets": {
                    "train": 0,
                    "validation": 0,
                    "test": 0,
                },
            },
        )
        migrated = dict(payload)
        migrated["stages"] = stages
        return migrated

    def _compatible_source_upgrade(
        self,
        saved: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> bool:
        if not (
            self._retention_upgrade
            or self._repair_upgrade
            or self._policy_upgrade
            or self._planner_upgrade
        ):
            return False

        saved_copy = dict(saved)
        current_copy = dict(current)
        if self._retention_upgrade:
            if int(saved.get("retention_size", -1)) != 8:
                return False
            if int(current.get("retention_size", -1)) != int(
                current.get("validation_size", -2)
            ):
                return False
            saved_copy["retention_size"] = current_copy["retention_size"]
            expected_kind = "retention_v2"
            allowed_changes = {
                "DQN/DQN_train.py",
                "DQN/curriculum.py",
                "DQN/run_curriculum.py",
            }
        elif self._repair_upgrade:
            if saved.get("retention_size") != current.get("retention_size"):
                return False
            saved_stages = saved_copy.get("stages")
            current_stages = current_copy.get("stages")
            if not isinstance(saved_stages, list) or not isinstance(
                current_stages,
                list,
            ):
                return False
            if (
                len(current_stages) != len(saved_stages) + 1
                or current_stages[REPAIR_BRIDGE_INDEX].get("name")
                != REPAIR_BRIDGE_STAGE
                or (
                    current_stages[:REPAIR_BRIDGE_INDEX]
                    + current_stages[REPAIR_BRIDGE_INDEX + 1 :]
                )
                != saved_stages
            ):
                return False
            current_copy["stages"] = saved_stages
            expected_kind = "training_repair_v4"
            allowed_changes = {
                "DQN/DQN_model.py",
                "DQN/DQN_train.py",
                "DQN/curriculum.py",
                "DQN/env_bridge.py",
                "DQN/load_model.py",
                "DQN/run_curriculum.py",
            }
        elif self._policy_upgrade:
            if saved.get("retention_size") != current.get("retention_size"):
                return False
            if saved.get("stages") != current.get("stages"):
                return False
            expected_kind = "policy_v2"
            allowed_changes = {
                "DQN/DQN_model.py",
                "DQN/DQN_train.py",
                "DQN/curriculum.py",
                "DQN/env_bridge.py",
                "DQN/load_model.py",
                "DQN/run_curriculum.py",
            }
        else:
            if saved.get("retention_size") != current.get("retention_size"):
                return False
            if saved.get("stages") != current.get("stages"):
                return False
            expected_kind = "planner_v3"
            allowed_changes = {
                "DQN/curriculum.py",
                "DQN/env_bridge.py",
                "DQN/run_curriculum.py",
            }
        saved_external = dict(saved_copy.get("external", {}))
        current_external = dict(current_copy.get("external", {}))
        current_upgrade = current_external.get("source_upgrade")
        if (
            not isinstance(current_upgrade, Mapping)
            or current_upgrade.get("kind") != expected_kind
            or current_upgrade.get("source_contract_sha256")
            != _hash_payload(saved)
        ):
            return False
        if self._repair_upgrade and (
            current_upgrade.get("inserted_stage") != REPAIR_BRIDGE_STAGE
            or current_upgrade.get("inserted_stage_index")
            != REPAIR_BRIDGE_INDEX
        ):
            return False
        saved_sources = saved_external.pop("source_sha256", None)
        current_sources = current_external.pop("source_sha256", None)
        saved_external.pop("source_upgrade", None)
        current_external.pop("source_upgrade", None)
        saved_copy["external"] = saved_external
        current_copy["external"] = current_external
        if saved_copy != current_copy:
            return False
        if not isinstance(saved_sources, Mapping) or not isinstance(
            current_sources,
            Mapping,
        ):
            return False
        changed = {
            name
            for name in set(saved_sources) | set(current_sources)
            if saved_sources.get(name) != current_sources.get(name)
        }
        return bool(changed) and changed <= allowed_changes

    def _training_failures(
        self,
        stage: CurriculumStage,
        evaluation: Evaluation,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        if evaluation.success_rate < stage.train_threshold:
            failures.append(
                "training success "
                f"{evaluation.success_rate:.1%} < {stage.train_threshold:.1%}"
            )
        if evaluation.wipeout_death_rate > stage.max_wipeout_death_rate:
            failures.append(
                "training wipeout deaths "
                f"{evaluation.wipeout_death_rate:.1%} > "
                f"{stage.max_wipeout_death_rate:.1%}"
            )
        if (
            stage.objective_check is not None
            and not stage.objective_check(evaluation)
        ):
            failures.append(f"training objective unmet: {stage.objective}")
        return tuple(failures)

    def _validation_failures(
        self,
        stage: CurriculumStage,
        evaluation: Evaluation | None,
    ) -> tuple[str, ...]:
        if evaluation is None:
            return ()
        failures: list[str] = []
        if evaluation.success_rate < stage.validation_threshold:
            failures.append(
                "validation success "
                f"{evaluation.success_rate:.1%} < "
                f"{stage.validation_threshold:.1%}"
            )
        if evaluation.wipeout_death_rate > stage.max_wipeout_death_rate:
            failures.append(
                "validation wipeout deaths "
                f"{evaluation.wipeout_death_rate:.1%} > "
                f"{stage.max_wipeout_death_rate:.1%}"
            )
        if (
            stage.objective_check is not None
            and not stage.objective_check(evaluation)
        ):
            failures.append(f"validation objective unmet: {stage.objective}")
        return tuple(failures)

    def _retention_threshold(self, stage_name: str) -> float:
        stage = next(
            (item for item in self.stages if item.name == stage_name),
            None,
        )
        if stage is None:
            raise ValueError(f"unknown retention stage {stage_name!r}")
        return max(
            0.50,
            stage.validation_threshold - self.retention_margin,
        )

    def _retention_failures(
        self,
        retention: Sequence[tuple[str, float]],
        evaluations: Mapping[str, Evaluation] | None = None,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        stage_by_name = {stage.name: stage for stage in self.stages}
        for name, rate in retention:
            threshold = self._retention_threshold(name)
            if rate < threshold:
                failures.append(
                    f"retention {name} {rate:.1%} < {threshold:.1%}"
                )
            if evaluations is None or name not in evaluations:
                continue
            stage = stage_by_name[name]
            evaluation = evaluations[name]
            if evaluation.wipeout_death_rate > stage.max_wipeout_death_rate:
                failures.append(
                    f"retention {name} wipeout deaths "
                    f"{evaluation.wipeout_death_rate:.1%} > "
                    f"{stage.max_wipeout_death_rate:.1%}"
                )
            if (
                stage.objective_check is not None
                and not stage.objective_check(evaluation)
            ):
                failures.append(
                    f"retention {name} objective unmet: {stage.objective}"
                )
        return tuple(failures)

    def _weak_retention_stages(
        self,
        retention: Sequence[tuple[str, float]],
        failures: Sequence[str] = (),
    ) -> tuple[str, ...]:
        weak = {
            name
            for name, rate in retention
            if rate < self._retention_threshold(name)
        }
        for name, _ in retention:
            prefix = f"retention {name} "
            if any(reason.startswith(prefix) for reason in failures):
                weak.add(name)
        return tuple(name for name, _ in retention if name in weak)

    @staticmethod
    def _success_deficit(rate: float, threshold: float) -> float:
        return max(0.0, threshold - rate) / max(threshold, 0.01)

    @staticmethod
    def _safety_deficit(rate: float, maximum: float) -> float:
        return max(0.0, rate - maximum) / max(maximum, 0.05)

    @staticmethod
    def _objective_deficit(
        stage: CurriculumStage,
        evaluation: Evaluation,
    ) -> float:
        if stage.objective_check is None or stage.objective_check(evaluation):
            return 0.0
        if stage.objective_deficit is None:
            return 1.0
        return max(0.0, float(stage.objective_deficit(evaluation)))

    def _retention_stage_deficits(
        self,
        retention: Sequence[tuple[str, float]],
        evaluations: Mapping[str, Evaluation] | None,
    ) -> dict[str, float]:
        stage_by_name = {stage.name: stage for stage in self.stages}
        deficits: dict[str, float] = {}
        for name, rate in retention:
            values = [
                self._success_deficit(
                    rate,
                    self._retention_threshold(name),
                )
            ]
            if evaluations is not None and name in evaluations:
                stage = stage_by_name[name]
                evaluation = evaluations[name]
                values.append(
                    self._safety_deficit(
                        evaluation.wipeout_death_rate,
                        stage.max_wipeout_death_rate,
                    )
                )
                values.append(self._objective_deficit(stage, evaluation))
            deficits[name] = max(values)
        return deficits

    def _assessment(
        self,
        stage: CurriculumStage,
        training: Evaluation,
        validation: Evaluation | None,
        retention: Sequence[tuple[str, float]],
        retention_evaluations: Mapping[str, Evaluation] | None = None,
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        training_failures = self._training_failures(stage, training)
        validation_failures = self._validation_failures(stage, validation)
        retention_failures = self._retention_failures(
            retention,
            retention_evaluations,
        )
        failures = (
            *training_failures,
            *validation_failures,
            *retention_failures,
        )
        deficits = [
            self._success_deficit(
                training.success_rate,
                stage.train_threshold,
            ),
            self._safety_deficit(
                training.wipeout_death_rate,
                stage.max_wipeout_death_rate,
            ),
        ]
        deficits.append(self._objective_deficit(stage, training))
        generalization = training.success_rate
        if validation is not None:
            deficits.extend(
                (
                    self._success_deficit(
                        validation.success_rate,
                        stage.validation_threshold,
                    ),
                    self._safety_deficit(
                        validation.wipeout_death_rate,
                        stage.max_wipeout_death_rate,
                    ),
                )
            )
            deficits.append(self._objective_deficit(stage, validation))
            generalization = validation.success_rate
        stage_by_name = {item.name: item for item in self.stages}
        for name, rate in retention:
            deficits.append(
                self._success_deficit(
                    rate,
                    self._retention_threshold(name),
                )
            )
            if retention_evaluations is None:
                continue
            evaluation = retention_evaluations.get(name)
            if evaluation is None:
                continue
            retained_stage = stage_by_name[name]
            deficits.append(
                self._safety_deficit(
                    evaluation.wipeout_death_rate,
                    retained_stage.max_wipeout_death_rate,
                )
            )
            deficits.append(
                self._objective_deficit(retained_stage, evaluation)
            )
        if retention:
            generalization = min(
                generalization,
                min(rate for _, rate in retention),
            )
        worst_deficit = max(deficits, default=0.0)
        total_deficit = sum(deficits)
        rank = (
            float(not failures),
            -worst_deficit,
            -total_deficit,
            generalization,
            validation.success_rate if validation is not None else 0.0,
            training.success_rate,
            -training.wipeout_death_rate,
            training.mean_return,
        )
        return tuple(failures), rank

    def _should_expand(
        self,
        stage: CurriculumStage,
        training: Evaluation,
        validation: Evaluation | None,
        retention: Sequence[tuple[str, float]],
        failures: Sequence[str] = (),
    ) -> bool:
        return (
            not self._training_failures(stage, training)
            and bool(self._validation_failures(stage, validation))
            and not self._retention_failures(retention)
            and not any(
                reason.startswith("retention ") for reason in failures
            )
        )

    @staticmethod
    def _allocate_episode_counts(
        total: int,
        weights: Mapping[int, float],
    ) -> dict[int, int]:
        positive = {
            group: float(weight)
            for group, weight in weights.items()
            if weight > 0.0
        }
        if total < 1 or not positive:
            raise ValueError("episode allocation requires a positive total")
        if total < len(positive):
            raise ValueError("episode allocation cannot cover every source")
        counts = {group: 1 for group in positive}
        remaining_total = total - len(positive)
        weight_sum = sum(positive.values())
        exact = {
            group: remaining_total * weight / weight_sum
            for group, weight in positive.items()
        }
        for group, value in exact.items():
            counts[group] += int(value)
        remainder = total - sum(counts.values())
        order = sorted(
            positive,
            key=lambda group: (
                -(exact[group] - int(exact[group])),
                group,
            ),
        )
        for group in order[:remainder]:
            counts[group] += 1
        return counts

    @staticmethod
    def _capped_allocation(
        total: float,
        scores: Mapping[str, float],
        cap: float,
    ) -> tuple[dict[str, float], float]:
        allocations = {name: 0.0 for name in scores}
        available = set(scores)
        remaining = total
        while available and remaining > 1e-12:
            score_sum = sum(scores[name] for name in available)
            if score_sum <= 0.0:
                break
            saturated = [
                name
                for name in available
                if remaining * scores[name] / score_sum
                >= cap - allocations[name]
            ]
            if not saturated:
                for name in available:
                    allocations[name] += (
                        remaining * scores[name] / score_sum
                    )
                remaining = 0.0
                break
            for name in saturated:
                addition = cap - allocations[name]
                allocations[name] += addition
                remaining -= addition
                available.remove(name)
        return allocations, max(0.0, remaining)

    def _rehearsal_weights(
        self,
        current_group: int,
        rehearsal: Sequence[
            tuple[str, CoopEnvBridge, tuple[int, ...], int]
        ],
        active: Mapping[str, Any],
    ) -> tuple[dict[int, float], tuple[str, ...], bool, float]:
        if not rehearsal:
            return {current_group: 1.0}, (), False, 1.0

        latest_retention = tuple(active["latest_retention"])
        best_retention = tuple(active["best_retention"])
        latest_failures = tuple(active["latest_failures"])
        best_failures = tuple(active["best_failures"])
        latest_deficits = {
            str(name): float(value)
            for name, value in active.get(
                "latest_retention_deficits",
                {},
            ).items()
        }
        best_deficits = {
            str(name): float(value)
            for name, value in active.get(
                "best_retention_deficits",
                {},
            ).items()
        }
        weak = set(
            self._weak_retention_stages(
                latest_retention,
                latest_failures,
            )
        )
        weak.update(
            self._weak_retention_stages(
                best_retention,
                best_failures,
            )
        )
        weak.update(
            name
            for name, value in {
                **best_deficits,
                **latest_deficits,
            }.items()
            if value > 0.0
        )
        current_failed = any(
            reason.startswith(("training ", "validation "))
            for reason in latest_failures
        )
        phase = str(active["phase"])
        consolidation = (
            bool(weak)
            and (not current_failed or phase in {"recovery", "extension"})
        )
        if consolidation:
            current_share = 0.50 if current_failed else 0.25
            maintenance = 0.15 if current_failed else 0.25
            learning_rate_scale = 0.50 if current_failed else 0.25
        else:
            current_share = 0.80
            maintenance = 0.04 if weak else 0.20
            learning_rate_scale = 1.0
        rehearsal_share = 1.0 - current_share

        correction: dict[str, float] = {}
        if weak:
            scores: dict[str, float] = {}
            for name in weak:
                prefix = f"retention {name} "
                legacy_failure = any(
                    reason.startswith(prefix)
                    for reason in (*latest_failures, *best_failures)
                )
                scores[name] = (
                    0.02
                    + max(
                        latest_deficits.get(name, 0.0),
                        best_deficits.get(name, 0.0),
                        float(legacy_failure)
                        if not latest_deficits and not best_deficits
                        else 0.0,
                    )
                )
            corrective = rehearsal_share - maintenance
            correction, unused = self._capped_allocation(
                corrective,
                scores,
                0.35 if current_failed else 0.50,
            )
            maintenance += unused
            group_by_name = {
                name: group for name, _, _, group in rehearsal
            }
            missing = sorted(weak - set(group_by_name))
            if missing:
                raise RuntimeError(
                    "retention rehearsal is missing stage "
                    + ", ".join(missing)
                )
        else:
            group_by_name = {}

        weights = {current_group: current_share}
        for _, _, _, group in rehearsal:
            weights[group] = maintenance / len(rehearsal)
        for name, value in correction.items():
            weights[group_by_name[name]] += value

        return (
            weights,
            tuple(name for name, _, _, _ in rehearsal if name in weak),
            consolidation,
            learning_rate_scale,
        )

    def run(self) -> list[StageResult]:
        from DQN.DQN_rewards import CurriculumPlot

        if self.results and self._resume_status is None:
            raise RuntimeError("this curriculum runner has already trained")
        if (
            self._resume_status in {"completed", "test_started", "tested"}
            and self.completed
        ):
            self.room_manifest = self._manifest_builder.snapshot()
            train_limits = {stage.name: 0 for stage in self.stages}
            for result in self.results:
                train_limits[result.stage] = max(
                    train_limits[result.stage],
                    result.pool_size,
                )
            self.training_features = verify_training_coverage(
                self.room_manifest,
                train_limits,
                require_all=self.require_full_coverage,
            )
            print("Recovered curriculum is already complete.", flush=True)
            return list(self.results)
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
            started = time.perf_counter()

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
            elapsed = time.perf_counter() - started
            if elapsed >= 0.05:
                print(
                    f"  staged {stage.name} {split} in {elapsed:.2f}s",
                    flush=True,
                )
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

        def record_outcome(outcome) -> None:
            plot_returns.append(outcome.reward)
            plot_completed.append(float(outcome.completed))
            plot_deaths.append(
                float(outcome.metrics["wipeout_deaths"] > 0)
            )
            plot_hazards.append(float(outcome.metrics["hazards"] > 0))
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
                pump_every = min(50, self.plot_every)
                if (
                    len(plot_returns) % pump_every == 0
                    and len(plot_returns) % self.plot_every != 0
                ):
                    plot.pump()

        def print_round_timing(
            timing: Mapping[str, float],
            checkpoint_seconds: float,
        ) -> None:
            print(
                "  timing: "
                + " ".join(
                    f"{name}={seconds:.2f}s"
                    for name, seconds in timing.items()
                )
                + f" checkpoint={checkpoint_seconds:.2f}s",
                flush=True,
            )

        rehearsal: list[
            tuple[str, CoopEnvBridge, tuple[int, ...], int]
        ] = []
        retention_sets: list[
            tuple[str, CoopEnvBridge, tuple[int, ...]]
        ] = []

        def expand_rehearsal_sources(names: Sequence[str]) -> None:
            requested = set(names)
            for index, (name, old_env, old_seeds, group) in enumerate(
                rehearsal
            ):
                if name not in requested:
                    continue
                target = max(
                    len(old_seeds),
                    min(self.validation_size, self.recovery_pool_max),
                )
                if len(old_seeds) >= target:
                    continue
                saved = self._manifest_builder.snapshot().stage(name)
                feature_target = saved.feature_target("train")
                expanded, new_rooms = stage_rooms(
                    self.stages[group],
                    "train",
                    target,
                    feature_target,
                )
                old_env.set_room_cache_limit(target)
                old_env.cache_rooms(new_rooms)
                rehearsal[index] = (name, old_env, expanded, group)
                print(
                    f"  repair pool: {name} train rooms "
                    f"{len(old_seeds)} -> {len(expanded)}.",
                    flush=True,
                )

        result_ordinal = 0
        resume_active = self._resume_active
        resume_warmup_pending = self._resume_status in {"training", "stopped"}
        resume_graph_boundary = self._resume_status is not None
        if resume_graph_boundary:
            print(
                "The dashboard starts at this resume boundary; prior greedy "
                "pool metrics remain in the recovery state and final report.",
                flush=True,
            )
        try:
            for stage_index, stage in enumerate(self.stages):
                stage_pool_sizes = tuple(
                    sorted(set(stage.pool_sizes or self.pool_sizes))
                )
                largest_pool = stage_pool_sizes[-1]
                cache_limit = max(largest_pool, self.recovery_pool_max)

                train_env = CoopEnvBridge(
                    stage.config,
                    seed=self.data_seed,
                    max_steps=self.trainer.cfg.max_steps,
                    shaping_gamma=self.trainer.cfg.gamma,
                    record_metrics=False,
                )
                train_env.set_room_cache_limit(cache_limit)
                training_eval_env = CoopEnvBridge(
                    stage.config,
                    seed=self.data_seed,
                    max_steps=self.trainer.cfg.max_steps,
                    shaping_gamma=self.trainer.cfg.gamma,
                    record_metrics=False,
                )
                training_eval_env.set_room_cache_limit(cache_limit)
                validation_eval_env = CoopEnvBridge(
                    stage.config,
                    seed=self.data_seed,
                    max_steps=self.trainer.cfg.max_steps,
                    shaping_gamma=self.trainer.cfg.gamma,
                    record_metrics=False,
                )
                validation_eval_env.set_room_cache_limit(self.validation_size)
                saved_stage = self._manifest_builder.snapshot().stage(stage.name)
                train_seeds = saved_stage.seeds("train")
                validation_seeds = saved_stage.seeds("validation")
                trainer_attached = False
                stage_announced = False

                for pool_index, scheduled_pool_size in enumerate(stage_pool_sizes):
                    if result_ordinal < len(self.results):
                        saved_result = self.results[result_ordinal]
                        expected_pool = (
                            saved_result.scheduled_pool_size
                            if saved_result.scheduled_pool_size is not None
                            else saved_result.pool_size
                        )
                        if (
                            saved_result.stage != stage.name
                            or expected_pool != scheduled_pool_size
                            or not saved_result.promoted
                        ):
                            raise ValueError(
                                "recovery results do not follow the curriculum order"
                            )
                        result_ordinal += 1
                        continue

                    if not stage_announced:
                        label = (
                            f"resume:{stage.name}"
                            if resume_graph_boundary
                            else stage.name
                        )
                        resume_graph_boundary = False
                        stage_marks.append((len(plot_returns), label))
                        if plot is not None:
                            plot.mark_stage(len(plot_returns), label)
                        if stage.lesson:
                            print(f"\n{stage.name}: {stage.lesson}", flush=True)
                        if stage.objective:
                            print(
                                f"Promotion objective: {stage.objective}",
                                flush=True,
                            )
                        stage_announced = True
                    if not trainer_attached:
                        resumed_here = resume_warmup_pending
                        clear_replay = stage_index > 0 or resumed_here
                        self.trainer.set_env(
                            train_env,
                            clear_replay=clear_replay,
                        )
                        if clear_replay:
                            self.trainer.reheat_exploration()
                            self._force_replay_refill = True
                        resume_warmup_pending = False
                        trainer_attached = True

                    active = None
                    if resume_active is not None:
                        if (
                            int(resume_active["stage_index"]) != stage_index
                            or int(resume_active["pool_index"]) != pool_index
                            or int(resume_active["scheduled_pool_size"])
                            != scheduled_pool_size
                        ):
                            raise ValueError(
                                "recovery cursor does not match completed results"
                            )
                        active = resume_active
                        resume_active = None
                        print(
                            f"Continuing {stage.name} pool="
                            f"{active['active_pool_size']} after round "
                            f"{active['total_rounds']}.",
                            flush=True,
                        )

                    active_pool_size = (
                        int(active["active_pool_size"])
                        if active is not None
                        else scheduled_pool_size
                    )
                    train_seeds, new_train_rooms = stage_rooms(
                        stage,
                        "train",
                        active_pool_size,
                        largest_pool,
                    )
                    train_env.cache_rooms(new_train_rooms)
                    training_eval_env.cache_rooms(new_train_rooms)
                    if scheduled_pool_size == largest_pool:
                        validation_seeds, new_validation_rooms = stage_rooms(
                            stage,
                            "validation",
                            self.validation_size,
                            self.validation_size,
                        )
                        validation_eval_env.cache_rooms(new_validation_rooms)
                    if plot is not None and plot.visible:
                        plot.set_context(stage.name, active_pool_size)

                    rng = random.Random(
                        derive_seed(
                            self.run_seed,
                            f"{stage.name}:pool:{scheduled_pool_size}",
                        )
                    )
                    if active is not None:
                        rng.setstate(active["rng_state"])
                    else:
                        active = {
                            "stage_index": stage_index,
                            "pool_index": pool_index,
                            "scheduled_pool_size": scheduled_pool_size,
                            "active_pool_size": active_pool_size,
                            "total_rounds": 0,
                            "normal_rounds": 0,
                            "recovery_rounds": 0,
                            "phase": "normal",
                            "phase_limit": None,
                            "phase_rounds": 0,
                            "streak": 0,
                            "expansions": [],
                            "rng_state": rng.getstate(),
                            "best_rank": (),
                            "best_round": 0,
                            "best_training": None,
                            "best_validation": None,
                            "best_retention": (),
                            "best_failures": (),
                            "best_retention_deficits": {},
                            "latest_retention": (),
                            "latest_failures": (),
                            "latest_retention_deficits": {},
                            "best_learner_state": None,
                            "needs_replay_refill": False,
                            "needs_baseline_assessment": True,
                            "consolidation_bad_rounds": 0,
                        }
                    if self._force_replay_refill:
                        active["needs_replay_refill"] = True
                        self._force_replay_refill = False
                    if active["best_training"] is None:
                        self._save_progress("training", None)

                    if active["needs_baseline_assessment"]:
                        pool = train_seeds[: int(active["active_pool_size"])]
                        print(
                            "  baseline: evaluating before any training.",
                            flush=True,
                        )
                        training_eval = evaluate(
                            self.trainer.agents,
                            training_eval_env,
                            pool,
                        )
                        validation_eval = None
                        retention_eval: tuple[tuple[str, float], ...] = ()
                        retention_details: dict[str, Evaluation] = {}
                        if scheduled_pool_size == largest_pool:
                            validation_eval = evaluate(
                                self.trainer.agents,
                                validation_eval_env,
                                validation_seeds,
                            )
                            (
                                retention_eval,
                                retention_details,
                            ) = self._evaluate_retention_detailed(
                                retention_sets
                            )
                        failures, rank = self._assessment(
                            stage,
                            training_eval,
                            validation_eval,
                            retention_eval,
                            retention_details,
                        )
                        active["best_rank"] = rank
                        active["best_training"] = training_eval
                        active["best_validation"] = validation_eval
                        active["best_retention"] = retention_eval
                        active["best_failures"] = failures
                        retention_deficits = (
                            self._retention_stage_deficits(
                                retention_eval,
                                retention_details,
                            )
                        )
                        active["best_retention_deficits"] = (
                            retention_deficits
                        )
                        active["latest_retention"] = retention_eval
                        active["latest_failures"] = failures
                        active["latest_retention_deficits"] = (
                            retention_deficits
                        )
                        active["best_learner_state"] = (
                            self.trainer.learner_state()
                        )
                        active["streak"] = 1 if not failures else 0
                        active["consolidation_bad_rounds"] = 0
                        active["needs_baseline_assessment"] = False
                        self._print_round(
                            stage,
                            int(active["active_pool_size"]),
                            int(active["total_rounds"]),
                            training_eval,
                            validation_eval,
                            retention_eval,
                            failures,
                            "baseline",
                        )
                        if int(active["streak"]) >= self.promotion_passes:
                            result = StageResult(
                                stage=stage.name,
                                pool_size=int(active["active_pool_size"]),
                                scheduled_pool_size=scheduled_pool_size,
                                rounds=int(active["total_rounds"]),
                                recovery_rounds=int(active["recovery_rounds"]),
                                best_round=int(active["best_round"]),
                                expansions=tuple(active["expansions"]),
                                training=training_eval,
                                validation=validation_eval,
                                promoted=True,
                                retention=retention_eval,
                            )
                            self.results.append(result)
                            result_ordinal += 1
                            self._resume_active = None
                            self._force_replay_refill = True
                            self._save_progress("training", None)
                            print(
                                "  baseline: the saved model already "
                                "meets every promotion requirement.",
                                flush=True,
                            )
                            continue
                        self._save_progress("training", active)

                    while True:
                        pool = train_seeds[: int(active["active_pool_size"])]
                        phase = str(active["phase"])
                        episodes = max(
                            50,
                            self.episodes_per_seed
                            * int(active["active_pool_size"]),
                        )
                        (
                            replay_weights,
                            weak_stages,
                            consolidation,
                            learning_rate_scale,
                        ) = self._rehearsal_weights(
                            stage_index,
                            rehearsal,
                            active,
                        )
                        expand_rehearsal_sources(weak_stages)
                        self.trainer.set_replay_group_weights(replay_weights)
                        self.trainer.set_learning_rate(
                            self.trainer.cfg.lr * learning_rate_scale
                        )
                        sources: dict[
                            int,
                            tuple[CoopEnvBridge | None, tuple[int, ...]],
                        ] = {
                            stage_index: (None, pool),
                            **{
                                group: (old_env, old_seeds)
                                for _, old_env, old_seeds, group in rehearsal
                            },
                        }

                        def episode_plan() -> list[int]:
                            counts = self._allocate_episode_counts(
                                episodes,
                                replay_weights,
                            )
                            plan = [
                                group
                                for group, count in counts.items()
                                for _ in range(count)
                            ]
                            rng.shuffle(plan)
                            return plan

                        def run_source(
                            group: int,
                            *,
                            optimize: bool = True,
                        ):
                            source_env, source_seeds = sources[group]
                            return self.trainer.run_episode(
                                seed=rng.choice(source_seeds),
                                env=source_env,
                                replay_group=group,
                                optimize=optimize,
                            )

                        if rehearsal:
                            current_share = replay_weights[stage_index]
                            prior_share = 1.0 - current_share
                            label = (
                                "consolidation"
                                if consolidation
                                else "rehearsal"
                            )
                            print(
                                f"  {label}: current={current_share:.0%} "
                                f"prior={prior_share:.0%} "
                                f"lr={learning_rate_scale:.0%}"
                                + (
                                    "; weak=" + ", ".join(weak_stages)
                                    if weak_stages
                                    else ""
                                ),
                                flush=True,
                            )
                        training_started = time.perf_counter()
                        if active["needs_replay_refill"]:
                            ready = max(
                                self.trainer.cfg.batch_size,
                                self.trainer.cfg.replay_warmup,
                            )
                            refill_episodes = 0
                            print(
                                f"  recovery: refilling replay to {ready} "
                                "transitions before counting another round.",
                                flush=True,
                            )

                            def replay_ready() -> bool:
                                replay = self.trainer.learners[0].replay
                                if len(replay) < ready:
                                    return False
                                counts = replay.group_counts()
                                return all(
                                    sum(counts.get(group, (0, 0))) >= 1
                                    for group, weight in replay_weights.items()
                                    if weight > 0.0
                                )

                            refill_plan = list(replay_weights)
                            rng.shuffle(refill_plan)
                            while not replay_ready():
                                if not refill_plan:
                                    refill_plan = episode_plan()
                                outcome = run_source(
                                    refill_plan.pop(),
                                    optimize=False,
                                )
                                record_outcome(outcome)
                                refill_episodes += 1
                            active["needs_replay_refill"] = False
                            active["rng_state"] = rng.getstate()
                            if active["best_training"] is not None:
                                self._save_progress("training", active)
                            print(
                                f"  recovery: replay ready after "
                                f"{refill_episodes} refill episodes.",
                                flush=True,
                            )
                        for group in episode_plan():
                            record_outcome(run_source(group))

                        self.trainer.synchronize()
                        training_seconds = (
                            time.perf_counter() - training_started
                        )
                        train_eval_started = time.perf_counter()
                        training_eval = evaluate(
                            self.trainer.agents,
                            training_eval_env,
                            pool,
                        )
                        train_eval_seconds = (
                            time.perf_counter() - train_eval_started
                        )
                        validation_eval = None
                        retention_eval: tuple[tuple[str, float], ...] = ()
                        retention_details: dict[str, Evaluation] = {}
                        validation_seconds = 0.0
                        retention_seconds = 0.0
                        if scheduled_pool_size == largest_pool:
                            validation_started = time.perf_counter()
                            validation_eval = evaluate(
                                self.trainer.agents,
                                validation_eval_env,
                                validation_seeds,
                            )
                            validation_seconds = (
                                time.perf_counter() - validation_started
                            )
                            retention_started = time.perf_counter()
                            (
                                retention_eval,
                                retention_details,
                            ) = self._evaluate_retention_detailed(
                                retention_sets
                            )
                            retention_seconds = (
                                time.perf_counter() - retention_started
                            )
                        failures, rank = self._assessment(
                            stage,
                            training_eval,
                            validation_eval,
                            retention_eval,
                            retention_details,
                        )
                        active["latest_retention"] = retention_eval
                        active["latest_failures"] = failures
                        retention_deficits = self._retention_stage_deficits(
                            retention_eval,
                            retention_details,
                        )
                        active["latest_retention_deficits"] = (
                            retention_deficits
                        )
                        current_failed = any(
                            reason.startswith(
                                ("training ", "validation ")
                            )
                            for reason in failures
                        )
                        protect_best = (
                            any(
                                value > 0.0
                                for value in retention_deficits.values()
                            )
                            and (
                                not current_failed
                                or phase in {"recovery", "extension"}
                            )
                        )
                        passed = not failures
                        active["total_rounds"] += 1
                        active["phase_rounds"] += 1
                        if phase == "normal":
                            active["normal_rounds"] += 1
                        else:
                            active["recovery_rounds"] += 1
                        active["streak"] = (
                            int(active["streak"]) + 1 if passed else 0
                        )
                        improved = (
                            not active["best_rank"]
                            or rank > tuple(active["best_rank"])
                        )
                        regressed = (
                            bool(active["best_rank"])
                            and rank < tuple(active["best_rank"])
                        )
                        if improved:
                            active["best_rank"] = rank
                            active["best_round"] = active["total_rounds"]
                            active["best_training"] = training_eval
                            active["best_validation"] = validation_eval
                            active["best_retention"] = retention_eval
                            active["best_failures"] = failures
                            active["best_retention_deficits"] = dict(
                                retention_deficits
                            )
                            active["best_learner_state"] = (
                                self.trainer.learner_state()
                            )
                        if protect_best and regressed:
                            active["consolidation_bad_rounds"] = (
                                int(active["consolidation_bad_rounds"]) + 1
                            )
                        else:
                            active["consolidation_bad_rounds"] = 0
                        active["rng_state"] = rng.getstate()

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
                        plot_started = time.perf_counter()
                        if plot is not None:
                            plot.add_evaluation(*evaluation_mark)
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
                        plot_seconds = time.perf_counter() - plot_started
                        round_timing = {
                            "train": training_seconds,
                            "train_eval": train_eval_seconds,
                            "validation": validation_seconds,
                            "retention": retention_seconds,
                            "plot": plot_seconds,
                        }
                        self._print_round(
                            stage,
                            int(active["active_pool_size"]),
                            int(active["total_rounds"]),
                            training_eval,
                            validation_eval,
                            retention_eval,
                            failures,
                            phase,
                        )
                        best_training = active["best_training"]
                        if not isinstance(best_training, Evaluation):
                            raise RuntimeError("best training evaluation was not saved")
                        best_validation = active["best_validation"]
                        best_retention = active["best_retention"]

                        if int(active["streak"]) >= self.promotion_passes:
                            self.trainer.load_learner_state(
                                active["best_learner_state"]
                            )
                            if int(active["best_round"]) != int(
                                active["total_rounds"]
                            ):
                                self.trainer.clear_replay()
                                self.trainer.reheat_exploration()
                                self._force_replay_refill = True
                            result = StageResult(
                                stage=stage.name,
                                pool_size=int(active["active_pool_size"]),
                                scheduled_pool_size=scheduled_pool_size,
                                rounds=int(active["total_rounds"]),
                                recovery_rounds=int(active["recovery_rounds"]),
                                best_round=int(active["best_round"]),
                                expansions=tuple(active["expansions"]),
                                training=best_training,
                                validation=best_validation,
                                promoted=True,
                                retention=best_retention,
                            )
                            self.results.append(result)
                            result_ordinal += 1
                            self._resume_active = None
                            checkpoint_seconds = self._save_progress(
                                "training",
                                None,
                            )
                            print_round_timing(
                                round_timing,
                                checkpoint_seconds,
                            )
                            break

                        phase_limit = active.get("phase_limit")
                        if phase_limit is None:
                            phase_limit = (
                                self.max_rounds
                                if phase == "normal"
                                else self.recovery_rounds
                            )
                        if int(active["phase_rounds"]) < phase_limit:
                            if (
                                int(active["consolidation_bad_rounds"])
                                >= CONSOLIDATION_ROLLBACK_PATIENCE
                            ):
                                self.trainer.load_learner_state(
                                    active["best_learner_state"]
                                )
                                self.trainer.clear_replay()
                                self.trainer.reheat_exploration(
                                    start=0.15,
                                    steps=10_000,
                                )
                                active["needs_replay_refill"] = True
                                active["consolidation_bad_rounds"] = 0
                                print(
                                    "  consolidation: restored the best "
                                    "checkpoint after repeated regressions.",
                                    flush=True,
                                )
                            checkpoint_seconds = self._save_progress(
                                "training",
                                active,
                            )
                            print_round_timing(
                                round_timing,
                                checkpoint_seconds,
                            )
                            continue

                        self.trainer.load_learner_state(
                            active["best_learner_state"]
                        )
                        active["latest_retention"] = best_retention
                        active["latest_failures"] = active["best_failures"]
                        active["latest_retention_deficits"] = dict(
                            active["best_retention_deficits"]
                        )
                        self.trainer.clear_replay()
                        self.trainer.reheat_exploration()
                        active["needs_replay_refill"] = True
                        active["consolidation_bad_rounds"] = 0
                        can_expand = (
                            scheduled_pool_size == largest_pool
                            and int(active["active_pool_size"])
                            < self.recovery_pool_max
                            and len(active["expansions"])
                            < self.recovery_expansions
                            and self._should_expand(
                                stage,
                                best_training,
                                best_validation,
                                best_retention,
                                active["best_failures"],
                            )
                        )
                        begin_recovery = (
                            phase == "normal" and self.recovery_rounds > 0
                        )
                        if begin_recovery or (phase == "recovery" and can_expand):
                            if can_expand:
                                old_size = int(active["active_pool_size"])
                                new_size = min(
                                    self.recovery_pool_max,
                                    max(old_size + 1, old_size * 2),
                                )
                                train_seeds, new_train_rooms = stage_rooms(
                                    stage,
                                    "train",
                                    new_size,
                                    largest_pool,
                                )
                                train_env.cache_rooms(new_train_rooms)
                                training_eval_env.cache_rooms(new_train_rooms)
                                active["active_pool_size"] = new_size
                                active["expansions"].append(new_size)
                                if plot is not None and plot.visible:
                                    plot.set_context(stage.name, new_size)
                                print(
                                    "  recovery: validation generalization "
                                    f"failed, expanding train rooms "
                                    f"{old_size} -> {new_size}.",
                                    flush=True,
                                )
                            else:
                                print(
                                    "  recovery: restoring the best round and "
                                    f"adding {self.recovery_rounds} bounded "
                                    "rounds on the same rooms.",
                                    flush=True,
                                )
                            active["phase"] = "recovery"
                            active["phase_limit"] = None
                            active["phase_rounds"] = 0
                            active["streak"] = 0
                            active["consolidation_bad_rounds"] = 0
                            active["rng_state"] = rng.getstate()
                            checkpoint_seconds = self._save_progress(
                                "training",
                                active,
                            )
                            print_round_timing(
                                round_timing,
                                checkpoint_seconds,
                            )
                            continue

                        active["rng_state"] = rng.getstate()
                        checkpoint_seconds = self._save_progress(
                            "stopped",
                            active,
                        )
                        print_round_timing(
                            round_timing,
                            checkpoint_seconds,
                        )
                        result = StageResult(
                            stage=stage.name,
                            pool_size=int(active["active_pool_size"]),
                            scheduled_pool_size=scheduled_pool_size,
                            rounds=int(active["total_rounds"]),
                            recovery_rounds=int(active["recovery_rounds"]),
                            best_round=int(active["best_round"]),
                            expansions=tuple(active["expansions"]),
                            training=best_training,
                            validation=best_validation,
                            promoted=False,
                            retention=best_retention,
                            failure_reasons=tuple(active["best_failures"]),
                        )
                        self.results.append(result)
                        self._resume_active = active
                        print(
                            "  recovery exhausted without lowering any "
                            "promotion requirement.",
                            flush=True,
                        )
                        finish_manifest(False)
                        finish_plot()
                        return list(self.results)

                saved_stage = self._manifest_builder.snapshot().stage(stage.name)
                train_seeds = saved_stage.seeds("train")
                validation_seeds = saved_stage.seeds("validation")
                train_env.set_room_cache_limit(len(train_seeds))
                rehearsal.append(
                    (stage.name, train_env, train_seeds, stage_index)
                )
                retention_seeds = validation_seeds[: self.retention_size]
                for seed in retention_seeds:
                    validation_eval_env.reset(seed=seed)
                validation_eval_env.set_room_cache_limit(len(retention_seeds))
                retention_sets.append(
                    (stage.name, validation_eval_env, retention_seeds)
                )
        except KeyboardInterrupt:
            finish_manifest(False)
            finish_plot()
            if (
                self.progress_path is not None
                and self.progress_path.is_file()
            ):
                print(
                    f"\nInterrupted safely. Resume from {self.progress_path}",
                    flush=True,
                )
            else:
                print(
                    "\nInterrupted before the first recovery checkpoint; "
                    "restart the run.",
                    flush=True,
                )
            raise

        finish_manifest(self.require_full_coverage)
        finish_plot()
        self._save_progress("completed", None)
        return list(self.results)

    def evaluate_final_test(self) -> list[StageTestResult]:
        if self._resume_status == "tested":
            print("Recovered final test is already complete.", flush=True)
            return list(self.test_results)
        if self._test_started and self._resume_status != "test_started":
            raise RuntimeError("the final test has already been started")
        if not self.completed:
            raise RuntimeError("the final test requires a completed curriculum")

        expected_prefix = tuple(
            stage.name for stage in self.stages[: len(self.test_results)]
        )
        if tuple(result.stage for result in self.test_results) != expected_prefix:
            raise ValueError("recovered final-test results are out of order")
        if not self._test_started:
            self._test_started = True
            self._test_model_sha256 = self._model_sha256()
            self._save_progress("test_started", None)
        if self._test_model_sha256 != self._model_sha256():
            raise RuntimeError("the frozen model changed after final testing began")
        print(
            "\nFinal greedy test on untouched rooms "
            f"({len(self.test_results)}/{len(self.stages)} stages complete)",
            flush=True,
        )
        for stage in self.stages[len(self.test_results) :]:
            staging_started = time.perf_counter()
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
            staging_seconds = time.perf_counter() - staging_started
            evaluation_started = time.perf_counter()
            evaluation, episodes = evaluate_detailed(
                self.trainer.agents,
                test_env,
                seeds,
            )
            evaluation_seconds = time.perf_counter() - evaluation_started
            result = StageTestResult(stage.name, evaluation, episodes)
            self.test_results.append(result)
            self._print_test(result)
            print(
                f"  timing: staging={staging_seconds:.2f}s "
                f"evaluation={evaluation_seconds:.2f}s",
                flush=True,
            )
            self.room_manifest = self._manifest_builder.snapshot()
            if self._test_model_sha256 != self._model_sha256():
                raise RuntimeError("final testing changed the frozen model")
            self._save_progress("test_started", None)
        self.room_manifest = self._manifest_builder.snapshot()
        self._save_progress("tested", None)
        return list(self.test_results)

    def _evaluate_retention_detailed(
        self,
        retention_sets: Sequence[
            tuple[str, CoopEnvBridge, tuple[int, ...]]
        ],
    ) -> tuple[
        tuple[tuple[str, float], ...],
        dict[str, Evaluation],
    ]:
        results: list[tuple[str, float]] = []
        evaluations: dict[str, Evaluation] = {}
        for stage_name, env, seeds in retention_sets:
            result = evaluate(
                self.trainer.agents,
                env,
                seeds,
            )
            evaluations[stage_name] = result
            results.append((stage_name, result.success_rate))
        return tuple(results), evaluations

    @staticmethod
    def _print_round(
        stage: CurriculumStage,
        pool_size: int,
        round_index: int,
        training: Evaluation,
        validation: Evaluation | None,
        retention: tuple[tuple[str, float], ...],
        failures: Sequence[str],
        phase: str,
    ) -> None:
        message = (
            f"{stage.name:22} pool={pool_size:>2} round={round_index:>2} "
            f"phase={phase:<8} train={training.success_rate:>6.1%}"
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
        if failures:
            print(
                "  not promoted: " + "; ".join(failures),
                flush=True,
            )
        else:
            print("  promotion requirements passed.", flush=True)

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

from __future__ import annotations

import operator
import random as _random
import sys
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Architecture"))

from coop_env import EnvironmentSession, GenerationConfig, Vec2
from coop_env.entities import AgentSpawn, ExitDoor
from coop_env.room import Region, RoomTopology
from coop_env.tiles import Tile as _T
from coop_env.utils.graph import Graph
from coop_env.utils.grid import Grid
from coop_env.entities import (
    Checkpoint,
    Key,
    LockedDoor,
    PushableBlock,
    ResetZone,
    Switch,
    SwitchMode,
)
from coop_env.room import Room
from coop_env.state import EpisodeState
from coop_env.tiles import Tile, is_hazard

from DQN.DQN_model import CHANNELS, N_ACTIONS, OBS_DIM, VIEW

N_AGENTS = 2
RADIUS = VIEW // 2
INTERACT = N_ACTIONS - 1
DIRS = (Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0))  # N E S W

# View channels
(
    BLOCKED,
    HAZARD,
    KEY,
    DOOR_CLOSED,
    DOOR_OPEN,
    SWITCH_OFF,
    SWITCH_ON,
    CRATE,
    CHECKPOINT,
    RESET,
    BRIDGE,
    EXIT,
) = range(CHANNELS)

# Rewards
R_COMPLETE = 10.0
R_SPEED_MAX = 2.0
R_EXIT_OPEN = 2.0
R_COOP_DOOR = 3.0
R_DOOR = 2.0
R_TIMED_DOOR = 2.0
R_KEY = 1.0
R_CHECKPOINT = 1.0
R_CRATE_SWITCH = 1.0
R_SWITCH = 1.0
MAX_PROGRESS_REWARD = 5.0

R_TOWARD_EXIT = 1.2
R_TOWARD_OBJECTIVE = 1.0

R_STEP = -0.01
R_BLOCKED = -0.02
R_HAZARD = -0.05
R_RESET_ZONE = -0.5
R_INVALID_ACTION = -0.1
R_TIMEOUT = 0.0

SOLID = (Tile.VOID, Tile.WALL, Tile.OBSTACLE)


def micro_room(size: int = 2, seed: int = 0) -> Room:
    """Build a small open room with two spawns and an open exit."""
    dim = size + 2                      # one tile of wall on every side
    terrain = Grid(dim, dim, _T.WALL)
    tiles = set()
    for y in range(1, dim - 1):
        for x in range(1, dim - 1):
            pos = Vec2(x, y)
            terrain[pos] = _T.FLOOR
            tiles.add(pos)

    ordered = sorted(tiles, key=lambda p: (p[1], p[0]))
    if seed:
        # Vary only the spawns and exit.
        rng = _random.Random(seed)
        picks = rng.sample(ordered, min(3, len(ordered)))
        while len(picks) < 3:
            picks.append(picks[-1])
        spawn_a, spawn_b, exit_tile = picks
    else:
        exit_tile = ordered[-1]
        spawn_a = ordered[0]
        spawn_b = ordered[1] if len(ordered) > 2 else ordered[0]

    entities = (
        AgentSpawn(id="spawn_0", pos=spawn_a, index=0),
        AgentSpawn(id="spawn_1", pos=spawn_b, index=1),
        ExitDoor(id="exit", pos=exit_tile),   # AlwaysOpen by default
    )
    graph: Graph[int] = Graph()
    graph.add_node(0)
    topology = RoomTopology(
        regions={0: Region(0, frozenset(tiles))},
        graph=graph,
        portals={},
        spawn_regions=(0, 0),
        exit_region=0,
        depths={0: 0},
    )
    return Room(
        seed=seed,
        config=GenerationConfig(),
        terrain=terrain,
        entities=entities,
        topology=topology,
        metadata={"micro": size},
    )


class CoopEnvBridge:
    obs_dim = OBS_DIM
    n_actions = N_ACTIONS

    # Bound by _begin_episode().
    room: Room
    state: EpisodeState

    def __init__(
        self,
        config=None,
        seed=None,
        max_steps=200,
        micro=None,
        shaping_gamma=0.99,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not 0.0 <= shaping_gamma <= 1.0:
            raise ValueError("shaping_gamma must be between 0 and 1")
        self.cfg = config or GenerationConfig.preset("standard")
        self.sess = EnvironmentSession(self.cfg, master_seed=seed)
        self.max_steps = max_steps
        self.shaping_gamma = float(shaping_gamma)
        self.micro = micro
        self.micro_vary = False
        self._micro_seed = 0
        self.pos: list[Vec2] = []
        self.spawns: list[Vec2] = []
        self.steps = 0
        self._ready = False
        self._exit_pos = Vec2(0, 0)
        self._span = 1.0
        self._totals = (1, 1, 1, 1)
        self._walkable_count = 1
        self._visited: set = set()
        self._regions_seen: set = set()
        self._sighted: set = set()
        self._touched: set = set()
        self._tile_region: dict = {}
        self._objectives: list = []
        self._static_blocked: dict = {}
        self._static_hazard: dict = {}
        self._static_entities: dict = {}
        self._reset_tiles: set[Vec2] = set()
        self._bridge_tiles: set[Vec2] = set()
        self._nav_signature: tuple | None = None
        self._nav_distance: dict[Vec2, int] = {}
        self._nav_target: dict[Vec2, tuple[Any, str]] = {}
        self._exit_distance: dict[Vec2, int] = {}
        self._task_target_cache: list[tuple[Any, str]] = []
        self._used_switches: set[str] = set()
        self._phi = [0.0] * N_AGENTS
        self.progress_scale = 1.0

        self.rewarded_keys: set[str] = set()
        self.rewarded_doors: set[str] = set()
        self.rewarded_switches: set[str] = set()
        self.rewarded_checkpoints: set[str] = set()
        self.rewarded_crate_switches: set[str] = set()
        self.exit_open_rewarded = False

        self.episode_metrics: dict[str, Any] = {}
        self.metrics_history: list[dict[str, Any]] = []
        self._terminated = False
        self._metrics_recorded = False


    def reset(self, seed=None):
        if self.micro is not None:
            if self.micro_vary:
                self._micro_seed += 1
                self.sess.load(micro_room(self.micro, self._micro_seed))
            else:
                self.sess.load(micro_room(self.micro))
        elif seed is not None:
            self.sess.reset(seed=seed)
        else:
            self.sess.reset()
        return self._begin_episode()

    def load_room(self, room):
        self.sess.load(room)
        return self._begin_episode()

    def _begin_episode(self):
        # Cache episode data used on every step.
        self.room = self.sess.room
        self.state = self.sess.state
        self._exit_pos = self.room.exit.pos
        self._span = float(max(self.room.width, self.room.height))
        self._totals = (
            max(1, len(self.room.keys)),
            max(1, len(self.room.doors)),
            max(1, len(self.room.switches)),
            max(1, len(self.room.checkpoints)),
        )
        self._build_view_cache()
        self.pos = [s.pos for s in self.room.spawns]
        self.spawns = list(self.pos)
        self._ready = True
        self.steps = 0
        self._holds()
        self._terminated = False
        self._metrics_recorded = False

        # Do not reward state that starts active.
        self.rewarded_keys = set(self.state.keys_collected)
        self.rewarded_doors = {
            door.id for door in self.room.doors if self.state.is_door_open(door.id)
        }
        self.rewarded_switches = {
            switch.id
            for switch in self.room.switches
            if self.state.is_switch_active(switch.id)
        }
        self.rewarded_checkpoints = set(self.state.checkpoints_reached)
        self.rewarded_crate_switches = self._crate_weighted_switches()
        self.exit_open_rewarded = self.state.exit_open
        self.progress_scale = self._calculate_progress_scale()

        self._nav_signature = None
        self._phi = [self._potential(i) for i in range(N_AGENTS)]
        self._visited = set(self.pos)
        self._regions_seen = {
            self._tile_region.get(p) for p in self.pos
        } - {None}
        self._sighted = set()
        self._touched = set()

        self.episode_metrics = {
            "seed": self.room.seed,
            "steps": 0,
            "completed": False,
            "timed_out": False,
            "exit_opened": bool(self.state.exit_open),
            "tiles_visited": 0,
            "regions_seen": 0,
            "objectives_sighted": 0,
            "objectives_touched": 0,
            "explored_fraction": 0.0,
            "hazards": 0,
            "reset_zones": 0,
            "blocked_moves": 0,
            "invalid_actions": 0,
            "keys_collected": 0,
            "keys_rewarded": 0,
            "doors_opened": 0,
            "cooperative_doors_solved": 0,
            "checkpoints_reached": 0,
            "switches_activated": 0,
            "crate_switches_solved": 0,
            "progress_scale": self.progress_scale,
            "raw_progress_reward": 0.0,
            "scaled_progress_reward": 0.0,
            "agent_returns": [0.0] * N_AGENTS,
        }
        return [self._obs(i) for i in range(N_AGENTS)]

    def step(self, actions):
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        if self._terminated:
            raise RuntimeError("episode has ended; call reset() before stepping again")

        action_list = list(actions)
        if len(action_list) != N_AGENTS:
            raise ValueError(f"expected {N_AGENTS} actions, got {len(action_list)}")

        before = self._reward_snapshot()
        useful_before = self._unfinished_requirement_ids()
        action_events: list[str] = []
        self._used_switches.clear()

        own = [R_STEP] * N_AGENTS
        for i, action in enumerate(action_list):
            penalty, event = self._apply_action(i, action)
            own[i] += penalty
            if event is not None:
                action_events.append(f"agent_{i}:{event}")
                self._count_action_event(event)

        self._holds()
        self.state.advance(1)
        for i, (penalty, event) in enumerate(self._unstick()):
            own[i] += penalty
            if event is not None:
                action_events.append(f"agent_{i}:{event}")
                self._count_action_event(event)

        self.steps += 1
        done = self._won()
        cut = not done and self.steps >= self.max_steps
        terminal = done or cut

        for i in range(N_AGENTS):
            next_phi = 0.0 if terminal else self._potential(i)
            own[i] += self.shaping_gamma * next_phi - self._phi[i]
            self._phi[i] = next_phi
            here = self.pos[i]
            if here not in self._visited:
                self._visited.add(here)

            region = self._tile_region.get(here)
            if region is not None and region not in self._regions_seen:
                self._regions_seen.add(region)

            for obj in self._objectives:
                if self._is_done(obj):
                    continue
                gap = here.chebyshev(obj.pos)
                if gap <= RADIUS and obj.id not in self._sighted:
                    self._sighted.add(obj.id)
                if gap <= 1 and obj.id not in self._touched:
                    self._touched.add(obj.id)

        self.episode_metrics["tiles_visited"] = len(self._visited)
        self.episode_metrics["regions_seen"] = len(self._regions_seen)
        self.episode_metrics["objectives_sighted"] = len(self._sighted)
        self.episode_metrics["objectives_touched"] = len(self._touched)
        self.episode_metrics["explored_fraction"] = (
            len(self._visited) / self._walkable_count
        )

        raw_progress, progress_events = self._progress_reward(before, useful_before)
        scaled_progress = raw_progress * self.progress_scale
        team = scaled_progress

        completion_reward = 0.0
        speed_reward = 0.0
        timeout_reward = 0.0

        if done:
            completion_reward = R_COMPLETE
            remaining = max(0, self.max_steps - self.steps)
            speed_reward = R_SPEED_MAX * remaining / self.max_steps
            team += completion_reward + speed_reward
            self.episode_metrics["completed"] = True
        elif cut:
            timeout_reward = R_TIMEOUT
            team += timeout_reward
            self.episode_metrics["timed_out"] = True

        rewards = [team + own[i] for i in range(N_AGENTS)]
        self.episode_metrics["steps"] = self.steps
        self.episode_metrics["keys_collected"] = len(self.state.keys_collected)
        self.episode_metrics["raw_progress_reward"] += raw_progress
        self.episode_metrics["scaled_progress_reward"] += scaled_progress
        for i, reward in enumerate(rewards):
            self.episode_metrics["agent_returns"][i] += reward

        self._terminated = done or cut
        if self._terminated:
            self._record_episode_metrics()

        info = {
            "events": action_events + [event["event"] for event in progress_events],
            "progress_events": progress_events,
            "reward": {
                "progress_raw": raw_progress,
                "progress_scale": self.progress_scale,
                "progress_scaled": scaled_progress,
                "completion": completion_reward,
                "speed": speed_reward,
                "timeout": timeout_reward,
                "individual": list(own),
                "shared": team,
            },
            "episode": deepcopy(self.episode_metrics) if self._terminated
                       else self.episode_metrics,
        }
        observations = [self._obs(i) for i in range(N_AGENTS)]
        return observations, rewards, done, cut, info

    # actions

    def _apply_action(self, i, action):
        try:
            action_index = operator.index(action)
        except TypeError:
            return R_INVALID_ACTION, "invalid_action"
        if not 0 <= action_index < N_ACTIONS:
            return R_INVALID_ACTION, "invalid_action"
        if action_index == INTERACT:
            self._use(i)
            return 0.0, None
        return self._move(i, action_index)

    def _move(self, i, action):
        d = DIRS[action % 4]
        moved = 0
        for _ in range(1):
            dest = self.pos[i] + d
            self._push(dest, d)

            if self.state.is_hazardous(dest):
                # Hazards block movement.
                return R_HAZARD, "hazard"
            if not self._walkable(dest):
                break

            self.pos[i] = dest
            moved += 1
            zone = self._reset_zone_at(dest)
            if zone is not None:
                self.pos[i] = self._reset_destination(zone, dest)
                return R_RESET_ZONE, "reset_zone"

        return (0.0, None) if moved else (R_BLOCKED, "blocked")

    def _use(self, i):
        for e in self.room.entities_at(self.pos[i]):
            if isinstance(e, Key) and not self.state.is_key_collected(e.id):
                self.state.collect_key(e.id)
            elif isinstance(e, Switch) and e.mode is not SwitchMode.HOLD:
                if e.id in self._used_switches:
                    continue
                self._used_switches.add(e.id)
                self.state.set_switch(e.id, not self.state.is_switch_active(e.id))
            elif isinstance(e, Checkpoint) and not self.state.is_checkpoint_reached(e.id):
                self.state.reach_checkpoint(e.id)
        return 0.0

    def _push(self, dest, d):
        crate = self.state.blocking_entity_at(dest)
        block = self.room.find(crate) if crate is not None else None
        if not isinstance(block, PushableBlock):
            return
        past = dest + d
        if self._walkable(past) and not is_hazard(self.room.terrain_at(past)):
            if past not in self.pos:
                self.state.place_block(block.id, past)

    def _walkable(self, p):
        return self.state.is_walkable(p)

    def _holds(self):
        here = set(self.pos)
        for s in self.room.switches:
            if s.mode is SwitchMode.HOLD:
                self.state.set_switch(s.id, s.pos in here)

    def _unstick(self):
        results: list[tuple[float, str | None]] = []
        for i, p in enumerate(self.pos):
            if self.state.is_hazardous(p):
                self.pos[i] = self.room.spawns[i].pos
                results.append((R_HAZARD, "hazard"))
            elif not self._walkable(p):
                self.pos[i] = self.room.spawns[i].pos
                results.append((0.0, "unstick"))
            else:
                results.append((0.0, None))
        return results

    def _reset_zone_at(self, pos):
        return next((zone for zone in self.room.reset_zones if zone.rect.contains(pos)), None)

    def _reset_destination(self, zone: ResetZone, entered_at: Vec2) -> Vec2:
        if zone.returns_to:
            target = self.room.find(zone.returns_to)
            if target is not None:
                return target.pos
        return min(
            (spawn.pos for spawn in self.room.spawns),
            key=lambda pos: pos.manhattan(entered_at),
        )

    # progress

    def _progress(self):
        return (
            len(self.state.keys_collected),
            sum(self.state.doors_open.values()),
            len(self.state.checkpoints_reached),
        )

    def _reward_snapshot(self) -> dict[str, Any]:
        return {
            "keys": frozenset(self.state.keys_collected),
            "checkpoints": frozenset(self.state.checkpoints_reached),
            "switches": frozenset(
                switch.id
                for switch in self.room.switches
                if self.state.is_switch_active(switch.id)
            ),
            "doors": frozenset(
                door.id for door in self.room.doors if self.state.is_door_open(door.id)
            ),
            "crate_switches": frozenset(self._crate_weighted_switches()),
            "exit_open": self.state.exit_open,
        }

    def _unfinished_requirement_ids(self) -> set[str]:
        useful: set[str] = set()
        for door in self.room.doors:
            if not self.state.is_door_open(door.id):
                useful.update(door.requirement.referenced_ids())
        if not self.state.exit_open:
            useful.update(self.room.exit.requirement.referenced_ids())
        return useful

    def _crate_weighted_switches(self) -> set[str]:
        block_positions = set(self.state.block_positions.values())
        return {
            switch.id
            for switch in self.room.switches
            if switch.mode is SwitchMode.HOLD and switch.pos in block_positions
        }

    def _door_reward(self, door: LockedDoor) -> tuple[float, str]:
        if not door.latching or door.requirement.needs_simultaneity():
            return R_COOP_DOOR, "cooperative_door"
        if door.timer:
            return R_TIMED_DOOR, "timed_door"
        return R_DOOR, "door"

    def _calculate_progress_scale(self) -> float:
        useful = self._unfinished_requirement_ids()
        raw_total = R_EXIT_OPEN if not self.state.exit_open else 0.0

        for door in self.room.doors:
            if not self.state.is_door_open(door.id):
                raw_total += self._door_reward(door)[0]

        useful_hold_switches = {
            switch.id
            for switch in self.room.switches
            if switch.id in useful and switch.mode is SwitchMode.HOLD
        }
        if self.room.blocks:
            raw_total += len(useful_hold_switches) * R_CRATE_SWITCH

        for entity_id in useful:
            entity = self.room.find(entity_id)
            if isinstance(entity, Key):
                raw_total += R_KEY
            elif isinstance(entity, Checkpoint):
                raw_total += R_CHECKPOINT
            elif isinstance(entity, Switch):
                raw_total += R_SWITCH

        return min(1.0, MAX_PROGRESS_REWARD / max(1.0, raw_total))

    def _progress_reward(self, before, useful_before):
        after = self._reward_snapshot()
        events: list[dict[str, Any]] = []

        def add(kind: str, entity_id: str, raw: float) -> None:
            events.append(
                {
                    "event": f"{kind}:{entity_id}",
                    "kind": kind,
                    "id": entity_id,
                    "raw": raw,
                    "scaled": raw * self.progress_scale,
                }
            )

        new_keys = after["keys"] - before["keys"]
        for key_id in sorted(new_keys & useful_before):
            if key_id not in self.rewarded_keys:
                self.rewarded_keys.add(key_id)
                self.episode_metrics["keys_rewarded"] += 1
                add("key", key_id, R_KEY)

        new_checkpoints = after["checkpoints"] - before["checkpoints"]
        for checkpoint_id in sorted(new_checkpoints & useful_before):
            if checkpoint_id not in self.rewarded_checkpoints:
                self.rewarded_checkpoints.add(checkpoint_id)
                self.episode_metrics["checkpoints_reached"] += 1
                add("checkpoint", checkpoint_id, R_CHECKPOINT)

        new_switches = after["switches"] - before["switches"]
        for switch_id in sorted(new_switches & useful_before):
            if switch_id not in self.rewarded_switches:
                self.rewarded_switches.add(switch_id)
                self.episode_metrics["switches_activated"] += 1
                add("switch", switch_id, R_SWITCH)

        new_crate_switches = after["crate_switches"] - before["crate_switches"]
        for switch_id in sorted(new_crate_switches & useful_before):
            if switch_id not in self.rewarded_crate_switches:
                self.rewarded_crate_switches.add(switch_id)
                self.episode_metrics["crate_switches_solved"] += 1
                add("crate_switch", switch_id, R_CRATE_SWITCH)

        new_doors = after["doors"] - before["doors"]
        for door_id in sorted(new_doors):
            if door_id in self.rewarded_doors:
                continue
            door = self.room.find(door_id)
            if not isinstance(door, LockedDoor):
                continue
            self.rewarded_doors.add(door_id)
            raw, kind = self._door_reward(door)
            self.episode_metrics["doors_opened"] += 1
            if kind == "cooperative_door":
                self.episode_metrics["cooperative_doors_solved"] += 1
            add(kind, door_id, raw)

        if after["exit_open"] and not before["exit_open"] and not self.exit_open_rewarded:
            self.exit_open_rewarded = True
            self.episode_metrics["exit_opened"] = True
            add("exit_open", self.room.exit.id, R_EXIT_OPEN)

        return sum(event["raw"] for event in events), events

    def _count_action_event(self, event: str) -> None:
        metric = {
            "hazard": "hazards",
            "reset_zone": "reset_zones",
            "blocked": "blocked_moves",
            "invalid_action": "invalid_actions",
        }.get(event)
        if metric is not None:
            self.episode_metrics[metric] += 1

    def _record_episode_metrics(self) -> None:
        if self._metrics_recorded:
            return
        self.metrics_history.append(deepcopy(self.episode_metrics))
        self._metrics_recorded = True

    def metrics_summary(self) -> dict[str, float | int]:
        episodes = len(self.metrics_history)
        completed = [m for m in self.metrics_history if m["completed"]]
        return {
            "episodes": episodes,
            "completion_rate": len(completed) / episodes if episodes else 0.0,
            "mean_completion_steps": (
                sum(m["steps"] for m in completed) / len(completed) if completed else 0.0
            ),
            "hazards": sum(m["hazards"] for m in self.metrics_history),
            "timeouts": sum(bool(m["timed_out"]) for m in self.metrics_history),
            "keys_collected": sum(m["keys_collected"] for m in self.metrics_history),
            "doors_opened": sum(m["doors_opened"] for m in self.metrics_history),
            "cooperative_doors_solved": sum(
                m["cooperative_doors_solved"] for m in self.metrics_history
            ),
            "exit_open_not_reached": sum(
                bool(m["exit_opened"]) and not bool(m["completed"])
                for m in self.metrics_history
            ),
        }

    def _won(self):
        if not self.state.exit_open:
            return False
        at_exit = [p == self._exit_pos for p in self.pos]
        return all(at_exit) if self.room.config.exit_requires_both_agents else any(at_exit)

    def _task_targets(self):
        targets = {}
        for entity_id in self._unfinished_requirement_ids():
            entity = self.room.find(entity_id)
            if entity is None or self._is_done(entity):
                continue
            if isinstance(entity, Key):
                kind = "key"
            elif isinstance(entity, Switch):
                kind = "switch"
            elif isinstance(entity, Checkpoint):
                kind = "checkpoint"
            else:
                continue
            targets[entity.id] = (entity, kind)
        return [targets[key] for key in sorted(targets)]

    def _ensure_navigation(self):
        targets = self._task_targets()
        open_doors = tuple(sorted(k for k, value in self.state.doors_open.items() if value))
        blocks = tuple(
            sorted((key, pos[0], pos[1]) for key, pos in self.state.block_positions.items())
        )
        bridges = tuple(sorted((p[0], p[1]) for p in self.state.solid_bridge_tiles()))
        signature = (
            tuple((entity.id, kind) for entity, kind in targets),
            self.state.exit_open,
            open_doors,
            blocks,
            bridges,
        )
        if signature == self._nav_signature:
            return

        distance: dict[Vec2, int] = {}
        owner: dict[Vec2, tuple[Any, str]] = {}
        pending = deque()
        for target in targets:
            pos = target[0].pos
            if pos in distance:
                continue
            distance[pos] = 0
            owner[pos] = target
            pending.append(pos)

        while pending:
            pos = pending.popleft()
            for direction in DIRS:
                neighbor = pos + direction
                if neighbor in distance or not self.state.is_walkable(neighbor):
                    continue
                distance[neighbor] = distance[pos] + 1
                owner[neighbor] = owner[pos]
                pending.append(neighbor)

        exit_distance: dict[Vec2, int] = {}
        if self.state.exit_open:
            exit_distance[self._exit_pos] = 0
            exit_pending = deque((self._exit_pos,))
            while exit_pending:
                pos = exit_pending.popleft()
                for direction in DIRS:
                    neighbor = pos + direction
                    if (
                        neighbor in exit_distance
                        or not self.state.is_walkable(neighbor)
                    ):
                        continue
                    exit_distance[neighbor] = exit_distance[pos] + 1
                    exit_pending.append(neighbor)

        self._nav_signature = signature
        self._nav_distance = distance
        self._nav_target = owner
        self._exit_distance = exit_distance
        self._task_target_cache = targets

    def _goal_info(self, i):
        self._ensure_navigation()
        me = self.pos[i]
        if me in self._exit_distance:
            distance = self._exit_distance[me]
            route = Vec2(0, 0)
            if distance:
                for direction in DIRS:
                    if self._exit_distance.get(me + direction) == distance - 1:
                        route = direction
                        break
            return (
                self.room.exit,
                "exit",
                self._exit_pos - me,
                route,
                distance,
                True,
            )

        reachable = me in self._nav_distance
        if reachable:
            target, kind = self._nav_target[me]
            distance = self._nav_distance[me]
        elif self._task_target_cache:
            target, kind = min(
                self._task_target_cache,
                key=lambda item: (item[0].pos.manhattan(me), item[0].id),
            )
            distance = target.pos.manhattan(me)
        else:
            target, kind = self.room.exit, "exit"
            distance = target.pos.manhattan(me)

        route = Vec2(0, 0)
        if reachable and distance:
            for direction in DIRS:
                neighbor = me + direction
                if self._nav_distance.get(neighbor) != distance - 1:
                    continue
                if self._nav_target.get(neighbor, (None, ""))[0] == target:
                    route = direction
                    break
            if route == Vec2(0, 0):
                for direction in DIRS:
                    if self._nav_distance.get(me + direction) == distance - 1:
                        route = direction
                        break

        delta = target.pos - me
        return target, kind, delta, route, distance, reachable

    def _potential(self, i):
        _, kind, _, _, distance, reachable = self._goal_info(i)
        if not reachable:
            return 0.0
        weight = R_TOWARD_EXIT if kind == "exit" else R_TOWARD_OBJECTIVE
        return -weight * distance / self._span

    def _is_done(self, entity):
        if isinstance(entity, Key):
            return self.state.is_key_collected(entity.id)
        if isinstance(entity, Switch):
            return self.state.is_switch_active(entity.id)
        if isinstance(entity, Checkpoint):
            return self.state.is_checkpoint_reached(entity.id)
        return True

    def _build_view_cache(self):
        """Cache fixed terrain and entity positions."""
        blocked, hazard = {}, {}
        walkable = 0
        for pos in self.room.terrain.positions():
            tile = self.room.terrain_at(pos)
            if tile in SOLID:
                blocked[pos] = True
            elif is_hazard(tile):
                hazard[pos] = True
            else:
                walkable += 1
        self._walkable_count = walkable

        self._tile_region = {}
        for rid, region in self.room.topology.regions.items():
            for tile in region.tiles:
                self._tile_region[tile] = rid

        self._objectives = [
            e for e in self.room.entities
            if isinstance(e, (Key, Switch, Checkpoint))
        ]
        self._static_blocked = blocked
        self._static_hazard = hazard

        ents: dict = {}
        for e in self.room.entities:
            if isinstance(e, Key):
                ch = KEY
            elif isinstance(e, LockedDoor):
                ch = DOOR_CLOSED
            elif isinstance(e, Switch):
                ch = SWITCH_OFF
            elif isinstance(e, Checkpoint):
                ch = CHECKPOINT
            else:
                continue
            ents.setdefault(e.pos, []).append((ch, e.id))
        self._static_entities = ents
        self._reset_tiles = {
            tile for zone in self.room.reset_zones for tile in zone.footprint()
        }
        self._bridge_tiles = {
            tile for bridge in self.room.bridges for tile in bridge.footprint()
        }

    def _obs(self, i):
        view = [0.0] * (VIEW * VIEW * CHANNELS)
        crates = set(self.state.block_positions.values())
        solid_bridges = self.state.solid_bridge_tiles()
        state = self.state
        blocked, hazard, ents = (
            self._static_blocked, self._static_hazard, self._static_entities
        )
        in_bounds = self.room.terrain.in_bounds
        exit_pos = self._exit_pos
        ox, oy = self.pos[i]

        cell = 0
        for dy in range(-RADIUS, RADIUS + 1):
            for dx in range(-RADIUS, RADIUS + 1):
                p = Vec2(ox + dx, oy + dy)
                b = cell * CHANNELS
                cell += 1
                if not in_bounds(p):
                    view[b + BLOCKED] = 1.0
                    continue
                if p in blocked:
                    view[b + BLOCKED] = 1.0
                elif p in hazard and state.is_hazardous(p):
                    view[b + HAZARD] = 1.0
                if p in self._bridge_tiles and p in solid_bridges:
                    view[b + BRIDGE] = 1.0
                if p in self._reset_tiles:
                    view[b + RESET] = 1.0
                if p in crates:
                    view[b + CRATE] = 1.0
                    view[b + BLOCKED] = 1.0
                if p == exit_pos:
                    view[b + EXIT] = 1.0
                for ch, eid in ents.get(p, ()):
                    if ch == KEY:
                        if not state.is_key_collected(eid):
                            view[b + KEY] = 1.0
                    elif ch == DOOR_CLOSED:
                        if state.is_door_open(eid):
                            view[b + DOOR_OPEN] = 1.0
                        else:
                            view[b + DOOR_CLOSED] = 1.0
                            view[b + BLOCKED] = 1.0
                    elif ch == SWITCH_OFF:
                        target = SWITCH_ON if state.is_switch_active(eid) else SWITCH_OFF
                        view[b + target] = 1.0
                    elif not state.is_checkpoint_reached(eid):
                        view[b + CHECKPOINT] = 1.0

        return view + self._extras(i)

    def _can_interact(self, i, target):
        for entity in self.room.entities_at(self.pos[i]):
            if entity.id != target.id:
                continue
            if isinstance(entity, Key) and not self.state.is_key_collected(entity.id):
                return True
            if isinstance(entity, Checkpoint) and not self.state.is_checkpoint_reached(entity.id):
                return True
            if isinstance(entity, Switch) and entity.mode is not SwitchMode.HOLD:
                return True
        return False

    def _extras(self, i):
        me, mate = self.pos[i], self.pos[1 - i]
        width = float(max(1, self.room.width - 1))
        height = float(max(1, self.room.height - 1))
        keys, doors, checks = self._progress()
        target, kind, delta, route, distance, reachable = self._goal_info(i)
        switch_mode = target.mode if isinstance(target, Switch) else None
        switches = sum(self.state.switches_active.values())

        return [
            (self._exit_pos[0] - me[0]) / width,
            (self._exit_pos[1] - me[1]) / height,
            (mate[0] - me[0]) / width,
            (mate[1] - me[1]) / height,
            delta[0] / width,
            delta[1] / height,
            float(route[0]),
            float(route[1]),
            min(1.0, distance / max(1, self.max_steps)),
            float(reachable),
            float(kind == "key"),
            float(kind == "switch"),
            float(kind == "checkpoint"),
            float(kind == "exit"),
            float(switch_mode is SwitchMode.TOGGLE),
            float(switch_mode is SwitchMode.HOLD),
            float(switch_mode is SwitchMode.ONESHOT),
            float(self._can_interact(i, target)),
            keys / max(1, len(self.room.keys)),
            doors / max(1, len(self.room.doors)),
            switches / max(1, len(self.room.switches)),
            checks / max(1, len(self.room.checkpoints)),
            1.0 if self.state.exit_open else 0.0,
            1.0 - self.steps / self.max_steps,
            float(i),
            float(self.room.config.exit_requires_both_agents),
            float(delta == Vec2(0, 0)),
            self.progress_scale,
            min(1.0, self.room.width / 64.0),
            min(1.0, self.room.height / 64.0),
        ]


if __name__ == "__main__":
    import random

    random.seed(0)
    env = CoopEnvBridge(GenerationConfig.preset("easy"), seed=1)

    solved = bad = 0
    for _ in range(100):
        obs = env.reset()
        done = cut = False
        while not (done or cut):
            assert all(len(o) == OBS_DIM for o in obs)
            bad += sum(1 for o in obs for v in o if v != v)
            obs, rewards, done, cut, _ = env.step(
                [random.randrange(N_ACTIONS) for _ in range(N_AGENTS)]
            )
        solved += done

    print(f"obs_dim={OBS_DIM} n_actions={N_ACTIONS}")
    print(f"random policy solved {solved}/100, NaNs {bad}")

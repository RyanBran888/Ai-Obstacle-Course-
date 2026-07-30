from __future__ import annotations

import operator
import random as _random
import sys
from collections import OrderedDict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "Architecture"))

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
    TemporaryBridge,
    WipeoutBall,
    WipeoutBallSize,
)
from coop_env.room import Room
from coop_env.rng import normalize_seed
from coop_env.state import EpisodeState
from coop_env.tiles import Tile, is_hazard

from DQN.DQN_model import CHANNELS, GLOBALS, N_ACTIONS, OBS_DIM, VIEW

N_AGENTS = 2
RADIUS = VIEW // 2
INTERACT = 4
WAIT = 5
DIRS = (Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0))  # N E S W

# View channels
(
    BLOCKED,
    HAZARD,
    OWN_KEY,
    TEAMMATE_KEY,
    OWN_DOOR_CLOSED,
    TEAMMATE_DOOR_CLOSED,
    DOOR_OPEN,
    TIMED_DOOR,
    TIMED_DOOR_REMAINING,
    TIMED_DOOR_SPENT,
    TIMED_DOOR_DURATION,
    SWITCH_OFF,
    SWITCH_ON,
    CRATE,
    CHECKPOINT,
    RESET,
    BRIDGE_TILE,
    BRIDGE_SOLID,
    BRIDGE_TICKS_TO_CHANGE,
    BRIDGE_ON_TICKS,
    BRIDGE_OFF_TICKS,
    EXIT,
    NORMAL_BALL_NOW,
    NORMAL_BALL_NEXT,
    BIG_BALL_NOW,
    BIG_BALL_NEXT,
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
R_HAZARD = -0.5
R_RESET_ZONE = -0.5
R_INVALID_ACTION = -0.1
R_WRONG_KEY = -0.05
R_WIPEOUT = -10.0
R_TIMEOUT = 0.0

SOLID = (Tile.VOID, Tile.WALL, Tile.OBSTACLE)
TIME_SCALE = 64.0


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
        record_metrics=True,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not 0.0 <= shaping_gamma <= 1.0:
            raise ValueError("shaping_gamma must be between 0 and 1")
        self.cfg = config or GenerationConfig.preset("standard")
        self.sess = EnvironmentSession(self.cfg, master_seed=seed)
        self.max_steps = max_steps
        self.shaping_gamma = float(shaping_gamma)
        self.record_metrics = bool(record_metrics)
        self.micro = micro
        self.micro_vary = False
        self._micro_seed = 0
        self._room_cache: OrderedDict[int, Room] = OrderedDict()
        self._room_cache_limit = 64
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
        self._bridges_by_tile: dict[Vec2, TemporaryBridge] = {}
        self._nav_signature: tuple | None = None
        self._nav_distance: list[dict[Vec2, int]] = [{}, {}]
        self._nav_target: list[dict[Vec2, tuple[Any, str]]] = [{}, {}]
        self._exit_distance: dict[Vec2, int] = {}
        self._task_target_cache: list[list[tuple[Any, str]]] = [[], []]
        self._crate_roles: dict[str, int] = {}
        self._switch_roles: dict[str, int] = {}
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
            room_seed = normalize_seed(seed)
            room = self._room_cache.get(room_seed)
            if room is None:
                self.sess.reset(seed=room_seed)
                self._room_cache[room_seed] = self.sess.room
                while len(self._room_cache) > self._room_cache_limit:
                    self._room_cache.popitem(last=False)
            else:
                self._room_cache.move_to_end(room_seed)
                self.sess.load(room)
        else:
            self.sess.reset()
        return self._begin_episode()

    def set_room_cache_limit(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("room cache limit must be positive")
        self._room_cache_limit = limit
        while len(self._room_cache) > limit:
            self._room_cache.popitem(last=False)

    def cache_rooms(self, rooms) -> None:
        for room in rooms:
            seed = normalize_seed(room.seed)
            self._room_cache[seed] = room
            self._room_cache.move_to_end(seed)
        while len(self._room_cache) > self._room_cache_limit:
            self._room_cache.popitem(last=False)

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
        self._crate_roles = self._assign_crate_roles()
        self._switch_roles = self._assign_switch_roles()
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
            "failed": False,
            "failure_reason": None,
            "exit_opened": bool(self.state.exit_open),
            "tiles_visited": 0,
            "regions_seen": 0,
            "objectives_sighted": 0,
            "objectives_touched": 0,
            "explored_fraction": 0.0,
            "hazards": 0,
            "bridge_falls": 0,
            "bridge_falls_by_agent": [0] * N_AGENTS,
            "wipeout_deaths": 0,
            "wipeout_deaths_by_agent": [0] * N_AGENTS,
            "normal_ball_deaths": 0,
            "big_ball_deaths": 0,
            "wrong_key_interactions": 0,
            "wrong_key_interactions_by_agent": [0] * N_AGENTS,
            "reset_zones": 0,
            "blocked_moves": 0,
            "invalid_actions": 0,
            "keys_collected": 0,
            "keys_collected_by_agent": [0] * N_AGENTS,
            "keys_rewarded": 0,
            "doors_opened": 0,
            "timed_doors_opened": 0,
            "timed_doors_expired": 0,
            "timed_doors_rearmed": 0,
            "key_doors_opened_by_agent": [0] * N_AGENTS,
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
        timed_before = self._timed_door_snapshot()
        useful_before = self._unfinished_requirement_ids()
        action_events: list[str] = []
        self._used_switches.clear()

        own = [R_STEP] * N_AGENTS
        fatal = False
        fatal_agents: set[int] = set()
        for i, action in enumerate(action_list):
            penalty, event = self._apply_action(i, action)
            own[i] += penalty
            if event is not None:
                action_events.append(f"agent_{i}:{event}")
                self._count_action_event(event, i)
                fatal = fatal or event.startswith("wipeout_")
                if event.startswith("wipeout_"):
                    fatal_agents.add(i)

        self._holds()
        self.state.advance(1)
        for i, (penalty, event) in enumerate(self._wipeout_hits(fatal_agents)):
            own[i] += penalty
            if event is not None:
                action_events.append(f"agent_{i}:{event}")
                self._count_action_event(event, i)
                fatal = True
        for i, (penalty, event) in enumerate(self._unstick()):
            own[i] += penalty
            if event is not None:
                action_events.append(f"agent_{i}:{event}")
                self._count_action_event(event, i)
        self._holds()
        action_events.extend(self._timed_door_events(timed_before))

        self.steps += 1
        completed = not fatal and self._won()
        done = fatal or completed
        cut = not done and self.steps >= self.max_steps
        terminal = done or cut

        for i in range(N_AGENTS):
            next_phi = 0.0 if terminal else self._potential(i)
            if not fatal:
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

        if fatal:
            raw_progress, progress_events = 0.0, []
        else:
            raw_progress, progress_events = self._progress_reward(
                before, useful_before
            )
        scaled_progress = raw_progress * self.progress_scale
        team = scaled_progress

        completion_reward = 0.0
        speed_reward = 0.0
        timeout_reward = 0.0

        if completed:
            completion_reward = R_COMPLETE
            remaining = max(0, self.max_steps - self.steps)
            speed_reward = R_SPEED_MAX * remaining / self.max_steps
            team += completion_reward + speed_reward
            self.episode_metrics["completed"] = True
        elif fatal:
            self.episode_metrics["failed"] = True
            self.episode_metrics["failure_reason"] = "wipeout"
        elif cut:
            timeout_reward = R_TIMEOUT
            team += timeout_reward
            self.episode_metrics["timed_out"] = True

        rewards = [team + own[i] for i in range(N_AGENTS)]
        self.episode_metrics["steps"] = self.steps
        self.episode_metrics["keys_collected"] = len(self.state.keys_collected)
        self.episode_metrics["keys_collected_by_agent"] = [
            sum(
                key.agent_index == i and self.state.is_key_collected(key.id)
                for key in self.room.keys
            )
            for i in range(N_AGENTS)
        ]
        self.episode_metrics["key_doors_opened_by_agent"] = [
            sum(
                self._door_agent(door) == i and self.state.is_door_open(door.id)
                for door in self.room.doors
            )
            for i in range(N_AGENTS)
        ]
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
            return self._use(i)
        if action_index == WAIT:
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
            ball = self._wipeout_at(dest, self.state.tick)
            if ball is not None:
                return R_WIPEOUT, f"wipeout_{ball.size.value}"
            zone = self._reset_zone_at(dest)
            if zone is not None:
                self.pos[i] = self._reset_destination(zone, dest)
                return R_RESET_ZONE, "reset_zone"

        return (0.0, None) if moved else (R_BLOCKED, "blocked")

    def _use(self, i):
        for e in self.room.entities_at(self.pos[i]):
            if isinstance(e, Key) and not self.state.is_key_collected(e.id):
                if not self.state.collect_key(e.id, agent_index=i):
                    return R_WRONG_KEY, "wrong_key"
                return 0.0, None
            elif isinstance(e, Switch) and e.mode is not SwitchMode.HOLD:
                if e.id in self._used_switches:
                    continue
                self._used_switches.add(e.id)
                self.state.set_switch(e.id, not self.state.is_switch_active(e.id))
            elif isinstance(e, Checkpoint) and not self.state.is_checkpoint_reached(e.id):
                self.state.reach_checkpoint(e.id)
        return 0.0, None

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

    def _navigation_walkable(self, p):
        return (
            self.state.is_walkable(p) or p in self._bridge_tiles
        ) and p not in self._reset_tiles

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
                event = "bridge_fall" if p in self._bridge_tiles else "hazard"
                results.append((R_HAZARD, event))
            elif not self._walkable(p):
                self.pos[i] = self.room.spawns[i].pos
                results.append((0.0, "unstick"))
            else:
                results.append((0.0, None))
        return results

    def _wipeout_at(self, pos, tick):
        return next(
            (
                ball
                for ball in self.room.wipeout_balls
                if pos in ball.collision_tiles_at(tick)
            ),
            None,
        )

    def _wipeout_hits(self, excluded: set[int] | None = None):
        excluded = excluded or set()
        results: list[tuple[float, str | None]] = []
        for i, pos in enumerate(self.pos):
            if i in excluded:
                results.append((0.0, None))
                continue
            ball = self._wipeout_at(pos, self.state.tick)
            if ball is None:
                results.append((0.0, None))
                continue
            results.append((R_WIPEOUT, f"wipeout_{ball.size.value}"))
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

    def _door_timer_remaining(self, door_id: str) -> int:
        return self.state.door_timer_remaining(door_id)

    def _door_needs_rearm(self, door_id: str) -> bool:
        return self.state.door_needs_rearm(door_id)

    def _bridge_is_solid(self, bridge: TemporaryBridge) -> bool:
        return self.state.bridge_is_solid(bridge.id)

    def _bridge_ticks_until_change(self, bridge: TemporaryBridge) -> int:
        return self.state.bridge_ticks_until_change(bridge.id)

    def _timed_door_snapshot(self) -> dict[str, tuple[bool, int, bool]]:
        return {
            door.id: (
                self.state.is_door_open(door.id),
                self._door_timer_remaining(door.id),
                self._door_needs_rearm(door.id),
            )
            for door in self.room.doors
            if door.timer
        }

    def _timed_door_events(
        self,
        before: dict[str, tuple[bool, int, bool]],
    ) -> list[str]:
        events: list[str] = []
        after = self._timed_door_snapshot()
        for door_id in sorted(after):
            was_open, _, needed_rearm = before[door_id]
            is_open, _, needs_rearm = after[door_id]
            if not was_open and is_open:
                self.episode_metrics["timed_doors_opened"] += 1
                events.append(f"timed_door_opened:{door_id}")
            if was_open and not is_open and needs_rearm:
                self.episode_metrics["timed_doors_expired"] += 1
                events.append(f"timed_door_expired:{door_id}")
            if needed_rearm and not needs_rearm:
                self.episode_metrics["timed_doors_rearmed"] += 1
                events.append(f"timed_door_rearmed:{door_id}")
        return events

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
            if not door.latching or not self.state.is_door_open(door.id):
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

    def _count_action_event(self, event: str, agent_index: int | None = None) -> None:
        metric = {
            "hazard": "hazards",
            "bridge_fall": "bridge_falls",
            "reset_zone": "reset_zones",
            "blocked": "blocked_moves",
            "invalid_action": "invalid_actions",
            "wrong_key": "wrong_key_interactions",
        }.get(event)
        if metric is not None:
            self.episode_metrics[metric] += 1
        if event == "bridge_fall" and agent_index is not None:
            self.episode_metrics["bridge_falls_by_agent"][agent_index] += 1
        if event == "wrong_key" and agent_index is not None:
            self.episode_metrics["wrong_key_interactions_by_agent"][agent_index] += 1
        if event.startswith("wipeout_"):
            self.episode_metrics["wipeout_deaths"] += 1
            if agent_index is not None:
                self.episode_metrics["wipeout_deaths_by_agent"][agent_index] += 1
            size = event.removeprefix("wipeout_")
            self.episode_metrics[f"{size}_ball_deaths"] += 1

    def _record_episode_metrics(self) -> None:
        if self._metrics_recorded:
            return
        if self.record_metrics:
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
            "bridge_falls": sum(
                m["bridge_falls"] for m in self.metrics_history
            ),
            "wipeout_deaths": sum(
                m["wipeout_deaths"] for m in self.metrics_history
            ),
            "wrong_key_interactions": sum(
                m["wrong_key_interactions"] for m in self.metrics_history
            ),
            "timeouts": sum(bool(m["timed_out"]) for m in self.metrics_history),
            "keys_collected": sum(m["keys_collected"] for m in self.metrics_history),
            "doors_opened": sum(m["doors_opened"] for m in self.metrics_history),
            "timed_doors_opened": sum(
                m["timed_doors_opened"] for m in self.metrics_history
            ),
            "timed_doors_expired": sum(
                m["timed_doors_expired"] for m in self.metrics_history
            ),
            "timed_doors_rearmed": sum(
                m["timed_doors_rearmed"] for m in self.metrics_history
            ),
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

    def _task_targets(self, i):
        targets = {}
        useful = self._unfinished_requirement_ids()
        for entity_id in useful:
            entity = self.room.find(entity_id)
            if entity is None:
                continue
            if isinstance(entity, Key):
                if entity.agent_index not in (None, i):
                    continue
                kind = "key"
            elif isinstance(entity, Switch):
                planned = self._planned_crate_for_switch(entity.id)
                if planned is not None and not self._is_done(planned):
                    continue
                if self._switch_roles.get(entity.id, i) != i:
                    continue
                if self._is_done(entity):
                    if entity.mode is not SwitchMode.HOLD or self.pos[i] != entity.pos:
                        continue
                kind = "switch"
            elif isinstance(entity, Checkpoint):
                if self._is_done(entity):
                    continue
                kind = "checkpoint"
            else:
                continue
            if not isinstance(entity, Switch) and self._is_done(entity):
                continue
            targets[entity.id] = (entity, kind)

        for block in self.room.blocks:
            if (
                block.target_switch_id in useful
                and block.push_from is not None
                and self._crate_roles.get(block.id) == i
                and not self._is_done(block)
            ):
                targets[block.id] = (block, "crate")
        return [targets[key] for key in sorted(targets)]

    def _assign_crate_roles(self) -> dict[str, int]:
        roles: dict[str, int] = {}
        available = set(range(N_AGENTS))
        planned = sorted(
            (
                block
                for block in self.room.blocks
                if block.target_switch_id is not None and block.push_from is not None
            ),
            key=lambda block: block.id,
        )
        for block in planned:
            push_from = block.push_from
            if push_from is None:
                continue
            candidates = available or set(range(N_AGENTS))
            ranked = [
                (self.spawns[index].manhattan(push_from), index)
                for index in candidates
            ]
            agent = min(ranked)[1]
            roles[block.id] = agent
            available.discard(agent)
        return roles

    def _assign_switch_roles(self) -> dict[str, int]:
        groups: dict[str, list[Switch]] = {}
        for switch in self.room.switches:
            if (
                switch.mode is SwitchMode.HOLD
                and switch.group.startswith("pair_")
            ):
                groups.setdefault(switch.group, []).append(switch)

        roles: dict[str, int] = {}
        for switches in groups.values():
            if len(switches) != N_AGENTS:
                continue
            first, second = sorted(switches, key=lambda switch: switch.id)
            crate = next(
                (
                    block
                    for block in self.room.blocks
                    if block.target_switch_id in {first.id, second.id}
                    and block.id in self._crate_roles
                ),
                None,
            )
            if crate is not None:
                crate_agent = self._crate_roles[crate.id]
                crate_switch = (
                    first if crate.target_switch_id == first.id else second
                )
                other_switch = second if crate_switch is first else first
                roles[crate_switch.id] = crate_agent
                roles[other_switch.id] = 1 - crate_agent
                continue
            direct = (
                self.spawns[0].manhattan(first.pos)
                + self.spawns[1].manhattan(second.pos)
            )
            crossed = (
                self.spawns[0].manhattan(second.pos)
                + self.spawns[1].manhattan(first.pos)
            )
            if direct <= crossed:
                roles[first.id], roles[second.id] = 0, 1
            else:
                roles[first.id], roles[second.id] = 1, 0
        return roles

    def _planned_crate_for_switch(
        self,
        switch_id: str,
    ) -> PushableBlock | None:
        return next(
            (
                block
                for block in self.room.blocks
                if block.target_switch_id == switch_id
                and block.push_from is not None
            ),
            None,
        )

    def _target_pos(self, target) -> Vec2:
        if isinstance(target, PushableBlock) and target.push_from is not None:
            return target.push_from
        return target.pos

    def _ensure_navigation(self):
        targets_by_agent = [self._task_targets(i) for i in range(N_AGENTS)]
        open_doors = tuple(sorted(k for k, value in self.state.doors_open.items() if value))
        blocks = tuple(
            sorted((key, pos[0], pos[1]) for key, pos in self.state.block_positions.items())
        )
        bridges = tuple(sorted((p[0], p[1]) for p in self.state.solid_bridge_tiles()))
        signature = (
            tuple(
                tuple((entity.id, kind) for entity, kind in targets)
                for targets in targets_by_agent
            ),
            self.state.exit_open,
            open_doors,
            blocks,
            bridges,
        )
        if signature == self._nav_signature:
            return

        distances: list[dict[Vec2, int]] = []
        owners: list[dict[Vec2, tuple[Any, str]]] = []
        for targets in targets_by_agent:
            distance: dict[Vec2, int] = {}
            owner: dict[Vec2, tuple[Any, str]] = {}
            pending = deque()
            for target in targets:
                pos = self._target_pos(target[0])
                if pos in distance:
                    continue
                distance[pos] = 0
                owner[pos] = target
                pending.append(pos)

            while pending:
                pos = pending.popleft()
                for direction in DIRS:
                    neighbor = pos + direction
                    if neighbor in distance or not self._navigation_walkable(neighbor):
                        continue
                    distance[neighbor] = distance[pos] + 1
                    owner[neighbor] = owner[pos]
                    pending.append(neighbor)
            distances.append(distance)
            owners.append(owner)

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
                        or not self._navigation_walkable(neighbor)
                    ):
                        continue
                    exit_distance[neighbor] = exit_distance[pos] + 1
                    exit_pending.append(neighbor)

        self._nav_signature = signature
        self._nav_distance = distances
        self._nav_target = owners
        self._exit_distance = exit_distance
        self._task_target_cache = targets_by_agent

    def _goal_info(self, i):
        self._ensure_navigation()
        me = self.pos[i]
        force_task = any(
            isinstance(target, PushableBlock)
            for target, _ in self._task_target_cache[i]
        )
        for target, kind in self._task_target_cache[i]:
            if (
                isinstance(target, Switch)
                and target.mode is SwitchMode.HOLD
                and target.pos == me
                and self.state.is_switch_active(target.id)
            ):
                return target, kind, Vec2(0, 0), Vec2(0, 0), 0, True, True
        if me in self._exit_distance and not force_task:
            distance = self._exit_distance[me]
            candidates = [
                direction
                for direction in DIRS
                if self._exit_distance.get(me + direction) == distance - 1
            ]
            route, route_wait = self._safe_route(i, candidates)
            return (
                self.room.exit,
                "exit",
                self._exit_pos - me,
                route,
                distance,
                True,
                route_wait,
            )

        nav_distance = self._nav_distance[i]
        nav_target = self._nav_target[i]
        target_cache = self._task_target_cache[i]
        reachable = me in nav_distance
        if reachable:
            target, kind = nav_target[me]
            distance = nav_distance[me]
        elif target_cache:
            target, kind = min(
                target_cache,
                key=lambda item: (
                    self._target_pos(item[0]).manhattan(me),
                    item[0].id,
                ),
            )
            distance = self._target_pos(target).manhattan(me)
        else:
            target, kind = self.room.exit, "exit"
            distance = target.pos.manhattan(me)

        candidates: list[Vec2] = []
        if reachable and distance:
            for direction in DIRS:
                neighbor = me + direction
                if nav_distance.get(neighbor) != distance - 1:
                    continue
                if nav_target.get(neighbor, (None, ""))[0] == target:
                    candidates.append(direction)
            if not candidates:
                for direction in DIRS:
                    if nav_distance.get(me + direction) == distance - 1:
                        candidates.append(direction)

        route, route_wait = self._safe_route(i, candidates)
        if distance == 0 and isinstance(target, PushableBlock):
            block_pos = self.state.block_positions.get(target.id)
            if block_pos is not None:
                push = block_pos - me
                if push.manhattan(Vec2(0, 0)) == 1:
                    route, route_wait = self._safe_route(i, [push])
        if (
            distance == 0
            and isinstance(target, Switch)
            and target.mode is SwitchMode.HOLD
        ):
            route_wait = True

        delta = self._target_pos(target) - me
        return target, kind, delta, route, distance, reachable, route_wait

    def _safe_route(self, i, candidates):
        me = self.pos[i]
        danger = self.state.lethal_wipeout_tiles()
        next_danger = self.state.lethal_wipeout_tiles(self.state.tick + 1)
        unsafe = danger | next_danger
        for bridge in self.room.bridges:
            if (
                not bridge.is_solid_at(self.state.tick)
                or not bridge.is_solid_at(self.state.tick + 1)
            ):
                unsafe.update(bridge.footprint())
        if not candidates and me not in unsafe:
            return Vec2(0, 0), False
        for direction in candidates:
            if me + direction not in unsafe and me + direction not in self._reset_tiles:
                return direction, False
        if candidates and me not in unsafe:
            return Vec2(0, 0), True
        for direction in DIRS:
            destination = me + direction
            if (
                self._navigation_walkable(destination)
                and destination not in unsafe
            ):
                return direction, False
        return Vec2(0, 0), True

    def _potential(self, i):
        _, kind, _, _, distance, reachable, _ = self._goal_info(i)
        if not reachable:
            return 0.0
        weight = R_TOWARD_EXIT if kind == "exit" else R_TOWARD_OBJECTIVE
        return -weight * distance / self._span

    def _is_done(self, entity):
        if isinstance(entity, Key):
            return self.state.is_key_collected(entity.id)
        if isinstance(entity, Switch):
            return (
                self.state.is_switch_active(entity.id)
                and not self._switch_needs_rearm(entity)
            )
        if isinstance(entity, Checkpoint):
            return self.state.is_checkpoint_reached(entity.id)
        if isinstance(entity, PushableBlock):
            target = self.room.find(entity.target_switch_id or "")
            return (
                isinstance(target, Switch)
                and self.state.block_positions.get(entity.id) == target.pos
            )
        return True

    def _switch_needs_rearm(self, switch: Switch) -> bool:
        return any(
            self._door_needs_rearm(door_id)
            for door_id in switch.controls
            if isinstance(self.room.find(door_id), LockedDoor)
        )

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
            or (
                isinstance(e, PushableBlock)
                and e.target_switch_id is not None
            )
        ]
        self._static_blocked = blocked
        self._static_hazard = hazard

        ents: dict = {}
        for e in self.room.entities:
            if isinstance(e, Key):
                kind = "key"
            elif isinstance(e, LockedDoor):
                kind = "door"
            elif isinstance(e, Switch):
                kind = "switch"
            elif isinstance(e, Checkpoint):
                kind = "checkpoint"
            else:
                continue
            ents.setdefault(e.pos, []).append((kind, e.id))
        self._static_entities = ents
        self._reset_tiles = {
            tile for zone in self.room.reset_zones for tile in zone.footprint()
        }
        self._bridge_tiles = {
            tile for bridge in self.room.bridges for tile in bridge.footprint()
        }
        self._bridges_by_tile = {
            tile: bridge
            for bridge in self.room.bridges
            for tile in bridge.footprint()
        }

    def _obs(self, i):
        view = [0.0] * (VIEW * VIEW * CHANNELS)
        crates = set(self.state.block_positions.values())
        bridge_states = {
            bridge.id: (
                self._bridge_is_solid(bridge),
                self._bridge_ticks_until_change(bridge),
            )
            for bridge in self.room.bridges
        }
        timed_states = {
            door.id: (
                self._door_timer_remaining(door.id),
                self._door_needs_rearm(door.id),
            )
            for door in self.room.doors
            if door.timer
        }
        normal_now: set[Vec2] = set()
        normal_next: set[Vec2] = set()
        big_now: set[Vec2] = set()
        big_next: set[Vec2] = set()
        for ball in self.room.wipeout_balls:
            current = set(ball.collision_tiles_at(self.state.tick))
            upcoming = set(ball.collision_tiles_at(self.state.tick + 1))
            if ball.size is WipeoutBallSize.NORMAL:
                normal_now.update(current)
                normal_next.update(upcoming)
            else:
                big_now.update(current)
                big_next.update(upcoming)
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
                bridge = self._bridges_by_tile.get(p)
                if bridge is not None:
                    solid, ticks = bridge_states[bridge.id]
                    view[b + BRIDGE_TILE] = 1.0
                    view[b + BRIDGE_SOLID] = float(solid)
                    view[b + BRIDGE_TICKS_TO_CHANGE] = ticks / TIME_SCALE
                    view[b + BRIDGE_ON_TICKS] = bridge.on_ticks / TIME_SCALE
                    view[b + BRIDGE_OFF_TICKS] = (
                        bridge.period - bridge.on_ticks
                    ) / TIME_SCALE
                if p in self._reset_tiles:
                    view[b + RESET] = 1.0
                if p in crates:
                    view[b + CRATE] = 1.0
                    view[b + BLOCKED] = 1.0
                if p == exit_pos:
                    view[b + EXIT] = 1.0
                if p in normal_now:
                    view[b + NORMAL_BALL_NOW] = 1.0
                if p in normal_next:
                    view[b + NORMAL_BALL_NEXT] = 1.0
                if p in big_now:
                    view[b + BIG_BALL_NOW] = 1.0
                if p in big_next:
                    view[b + BIG_BALL_NEXT] = 1.0
                for kind, eid in ents.get(p, ()):
                    entity = self.room.find(eid)
                    if kind == "key" and isinstance(entity, Key):
                        if not state.is_key_collected(eid):
                            channel = (
                                OWN_KEY
                                if entity.agent_index in (None, i)
                                else TEAMMATE_KEY
                            )
                            view[b + channel] = 1.0
                    elif kind == "door" and isinstance(entity, LockedDoor):
                        if state.is_door_open(eid):
                            view[b + DOOR_OPEN] = 1.0
                        else:
                            owner = self._door_agent(entity)
                            channel = (
                                OWN_DOOR_CLOSED
                                if owner in (None, i)
                                else TEAMMATE_DOOR_CLOSED
                            )
                            view[b + channel] = 1.0
                            view[b + BLOCKED] = 1.0
                        if entity.timer:
                            remaining, spent = timed_states[eid]
                            view[b + TIMED_DOOR] = 1.0
                            view[b + TIMED_DOOR_REMAINING] = (
                                remaining / TIME_SCALE
                            )
                            view[b + TIMED_DOOR_SPENT] = float(spent)
                            view[b + TIMED_DOOR_DURATION] = (
                                entity.timer / TIME_SCALE
                            )
                    elif kind == "switch":
                        target = SWITCH_ON if state.is_switch_active(eid) else SWITCH_OFF
                        view[b + target] = 1.0
                    elif kind == "checkpoint" and not state.is_checkpoint_reached(eid):
                        view[b + CHECKPOINT] = 1.0

        return view + self._extras(i)

    def _can_interact(self, i, target):
        for entity in self.room.entities_at(self.pos[i]):
            if entity.id != target.id:
                continue
            if isinstance(entity, Key) and not self.state.is_key_collected(entity.id):
                return entity.agent_index in (None, i)
            if isinstance(entity, Checkpoint) and not self.state.is_checkpoint_reached(entity.id):
                return True
            if isinstance(entity, Switch) and entity.mode is not SwitchMode.HOLD:
                return True
        return False

    def _can_push(self, i, target) -> bool:
        if not isinstance(target, PushableBlock):
            return False
        block_pos = self.state.block_positions.get(target.id)
        switch = self.room.find(target.target_switch_id or "")
        if block_pos is None or not isinstance(switch, Switch):
            return False
        direction = block_pos - self.pos[i]
        return (
            direction.manhattan(Vec2(0, 0)) == 1
            and block_pos + direction == switch.pos
        )

    def _relevant_timed_door(self, i) -> LockedDoor | None:
        me = self.pos[i]
        timed = [door for door in self.room.doors if door.timer]
        if not timed:
            return None

        def priority(door: LockedDoor):
            if self.state.is_door_open(door.id):
                state_rank = 0
            elif self._door_needs_rearm(door.id):
                state_rank = 1
            else:
                state_rank = 2
            return (
                state_rank,
                self._door_timer_remaining(door.id),
                door.pos.manhattan(me),
                door.id,
            )

        return min(timed, key=priority)

    def _relevant_bridge(
        self,
        i,
    ) -> tuple[TemporaryBridge, Vec2] | None:
        required_id = self.room.metadata.get("required_bridge_id")
        required = self.room.find(required_id) if isinstance(required_id, str) else None
        bridges = (
            [required]
            if isinstance(required, TemporaryBridge)
            else list(self.room.bridges)
        )
        if not bridges:
            return None

        me = self.pos[i]
        choices = [
            (tile.manhattan(me), bridge.id, tile, bridge)
            for bridge in bridges
            for tile in bridge.footprint()
        ]
        _, _, tile, bridge = min(choices)
        return bridge, tile

    def _extras(self, i):
        me, mate = self.pos[i], self.pos[1 - i]
        width = float(max(1, self.room.width - 1))
        height = float(max(1, self.room.height - 1))
        keys, doors, checks = self._progress()
        target, kind, delta, route, distance, reachable, route_wait = self._goal_info(i)
        switch_mode = target.mode if isinstance(target, Switch) else None
        switches = sum(self.state.switches_active.values())
        own_keys = [key for key in self.room.keys if key.agent_index == i]
        teammate_keys = [
            key for key in self.room.keys if key.agent_index == 1 - i
        ]
        own_doors = [
            door for door in self.room.doors if self._door_agent(door) == i
        ]
        teammate_doors = [
            door for door in self.room.doors if self._door_agent(door) == 1 - i
        ]
        timed_door = self._relevant_timed_door(i)
        bridge_info = self._relevant_bridge(i)

        if timed_door is None:
            timed_delta = Vec2(0, 0)
            timed_open = False
            timed_remaining = 0
            timed_duration = 0
            timed_spent = False
        else:
            timed_delta = timed_door.pos - me
            timed_open = self.state.is_door_open(timed_door.id)
            timed_remaining = self._door_timer_remaining(timed_door.id)
            timed_duration = timed_door.timer or 0
            timed_spent = self._door_needs_rearm(timed_door.id)

        if bridge_info is None:
            bridge_delta = Vec2(0, 0)
            bridge_solid = False
            bridge_ticks = 0
            bridge_on_ticks = 0
            bridge_off_ticks = 0
        else:
            bridge, bridge_tile = bridge_info
            bridge_delta = bridge_tile - me
            bridge_solid = self._bridge_is_solid(bridge)
            bridge_ticks = self._bridge_ticks_until_change(bridge)
            bridge_on_ticks = bridge.on_ticks
            bridge_off_ticks = bridge.period - bridge.on_ticks

        def completed_keys(keys_for_agent):
            if not keys_for_agent:
                return 1.0
            return sum(
                self.state.is_key_collected(key.id) for key in keys_for_agent
            ) / len(keys_for_agent)

        def opened_doors(doors_for_agent):
            if not doors_for_agent:
                return 1.0
            return sum(
                self.state.is_door_open(door.id) for door in doors_for_agent
            ) / len(doors_for_agent)

        extras = [
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
            float(kind == "crate"),
            float(switch_mode is SwitchMode.TOGGLE),
            float(switch_mode is SwitchMode.HOLD),
            float(switch_mode is SwitchMode.ONESHOT),
            float(self._can_interact(i, target)),
            float(self._can_push(i, target)),
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
            completed_keys(own_keys),
            completed_keys(teammate_keys),
            opened_doors(own_doors),
            opened_doors(teammate_doors),
            float(route_wait),
            timed_delta[0] / width,
            timed_delta[1] / height,
            float(timed_door is not None),
            float(timed_open),
            timed_remaining / TIME_SCALE,
            timed_duration / TIME_SCALE,
            float(timed_spent),
            bridge_delta[0] / width,
            bridge_delta[1] / height,
            float(bridge_info is not None),
            float(bridge_solid),
            bridge_ticks / TIME_SCALE,
            bridge_on_ticks / TIME_SCALE,
            bridge_off_ticks / TIME_SCALE,
        ]
        if len(extras) != GLOBALS:
            raise RuntimeError(f"expected {GLOBALS} global values, got {len(extras)}")
        return extras

    def _door_agent(self, door):
        owners = set()
        has_shared_key = False
        for entity_id in door.requirement.referenced_ids():
            key = self.room.find(entity_id)
            if not isinstance(key, Key):
                continue
            if key.agent_index is None:
                has_shared_key = True
            else:
                owners.add(key.agent_index)
        if has_shared_key or len(owners) != 1:
            return None
        return next(iter(owners))


if __name__ == "__main__":
    import random

    random.seed(0)
    env = CoopEnvBridge(GenerationConfig.preset("easy"), seed=1)

    solved = bad = 0
    for _ in range(100):
        obs = env.reset()
        done = cut = False
        completed = False
        while not (done or cut):
            assert all(len(o) == OBS_DIM for o in obs)
            bad += sum(1 for o in obs for v in o if v != v)
            obs, rewards, done, cut, info = env.step(
                [random.randrange(N_ACTIONS) for _ in range(N_AGENTS)]
            )
            completed = bool(info["episode"]["completed"])
        solved += completed

    print(f"obs_dim={OBS_DIM} n_actions={N_ACTIONS}")
    print(f"random policy solved {solved}/100, NaNs {bad}")

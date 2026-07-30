"""
Self-contained reimplementation of the pieces of env_bridge.py / coop_env
needed to run the pretrained agents against a level exported from Godot.

The real training package (coop_env) is not available, so this module
rebuilds only what your Godot editor can actually produce: walls, lava,
red/blue keys+doors, small/big wipeout balls, red/blue start, and a single
goal. It reproduces the exact channel layout, global-feature formulas, and
action semantics documented in env_bridge.py so the pretrained networks see
observations in the format they were trained on.

ASSUMPTIONS (see chat for context — flag if any of these are wrong):
  - Red = agent index 0, Blue = agent index 1.
  - Keys are agent-specific: only the matching-color agent can collect a key.
  - Doors open permanently (latching) once their key is collected.
  - The exit has no lock condition; it's open from the start.
  - Goal requires BOTH agents standing on it to complete the course.
  - Wipeout balls (Small/Big) ping-pong back and forth across the width you
    placed them at, one cell per step. SmallBall footprint = 1 cell,
    BigBall footprint = 3 cells. This is a guess (the real WipeoutBall
    movement code wasn't recoverable) — if agents seem to dodge balls oddly,
    this is the first place to revisit.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ---- must match networks.py exactly ----
VIEW = 7
CHANNELS = 26
GLOBALS = 51
OBS_DIM = VIEW * VIEW * CHANNELS + GLOBALS
N_ACTIONS = 6
RADIUS = VIEW // 2

N_AGENTS = 2
RED, BLUE = 0, 1

# actions
NORTH, EAST, SOUTH, WEST, INTERACT, WAIT = range(6)
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # N E S W, matches env_bridge.py

# channel indices — order matches env_bridge.py's unpacking of `range(CHANNELS)`
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

TIME_SCALE = 64.0  # kept for global-vector formula parity, unused here

# Godot TileType enum values (must match Node2d.cs)
T_EMPTY, T_WALL, T_RDOOR, T_BDOOR, T_RKEY, T_BKEY, T_SBALL, T_BBALL, \
    T_LAVA, T_GOAL, T_RSTART, T_BSTART = range(12)


@dataclass
class Ball:
    color: str  # "small" or "big"
    footprint: int  # tiles occupied at once
    lo: int  # min x of track (row y fixed)
    hi: int  # max x of track
    y: int
    phase: int = 0  # 0 = moving toward hi, 1 = moving toward lo
    head: int = 0  # leading-edge x offset from lo, 0-based

    def cells_at(self, step: int) -> set[tuple[int, int]]:
        span = max(1, self.hi - self.lo + 1)
        period = max(1, 2 * (span - 1)) if span > 1 else 1
        t = step % period
        pos = t if t <= span - 1 else period - t
        start = self.lo + max(0, pos - self.footprint + 1)
        cells = set()
        for i in range(self.footprint):
            x = min(self.hi, start + i)
            if self.lo <= x <= self.hi:
                cells.add((x, self.y))
        return cells

    def center_at(self, step: int) -> tuple[float, float]:
        """Single-point position (footprint centroid) for animating a
        moving sprite, rather than the full occupied-cell set."""
        cells = self.cells_at(step)
        if not cells:
            return (float(self.lo), float(self.y))
        xs = [c[0] for c in cells]
        return (sum(xs) / len(xs), float(self.y))


@dataclass
class Door:
    color: str  # "red" or "blue"
    pos: tuple[int, int]
    open: bool = False


@dataclass
class Key:
    color: str
    pos: tuple[int, int]
    collected: bool = False


@dataclass
class Level:
    width: int
    height: int
    blocked: set = field(default_factory=set)   # walls (static, excl. doors)
    hazard: set = field(default_factory=set)    # lava
    doors: list = field(default_factory=list)
    keys: list = field(default_factory=list)
    balls: list = field(default_factory=list)
    spawns: list = field(default_factory=list)  # [red_pos, blue_pos]
    goal: tuple = (0, 0)

    @staticmethod
    def from_json(path: str) -> "Level":
        with open(path, "r") as f:
            data = json.load(f)

        width, height = data["width"], data["height"]
        blocked, hazard, doors, keys, balls = set(), set(), [], [], []
        spawns = [None, None]
        goal = None

        for obj in data["objects"]:
            t = obj["type"]
            cx, cy = obj["x"], obj["y"]
            w, h = obj.get("width", 1), obj.get("height", 1)
            half_w = (w - 1) // 2
            half_h = (h - 1) // 2

            if t == "Wall":
                for dx in range(-half_w, half_w + 1):
                    for dy in range(-half_h, half_h + 1):
                        blocked.add((cx + dx, cy + dy))
            elif t == "Lava":
                for dx in range(-half_w, half_w + 1):
                    for dy in range(-half_h, half_h + 1):
                        hazard.add((cx + dx, cy + dy))
            elif t == "RedDoor":
                doors.append(Door("red", (cx, cy)))
            elif t == "BlueDoor":
                doors.append(Door("blue", (cx, cy)))
            elif t == "RedKey":
                keys.append(Key("red", (cx, cy)))
            elif t == "BlueKey":
                keys.append(Key("blue", (cx, cy)))
            elif t == "SmallBall":
                balls.append(Ball("small", 1, cx - half_w, cx + half_w, cy))
            elif t == "BigBall":
                balls.append(Ball("big", 3, cx - half_w, cx + half_w, cy))
            elif t == "Goal":
                goal = (cx, cy)
            elif t == "RedStart":
                spawns[RED] = (cx, cy)
            elif t == "BlueStart":
                spawns[BLUE] = (cx, cy)

        if goal is None or spawns[RED] is None or spawns[BLUE] is None:
            raise ValueError(
                "level.json is missing RedStart, BlueStart, or Goal — "
                "the course must place all three before it can be run."
            )

        return Level(width, height, blocked, hazard, doors, keys, balls,
                     spawns, goal)


class RoomSim:
    """Minimal stand-in for CoopEnvBridge, covering only what Godot can build."""

    obs_dim = OBS_DIM
    n_actions = N_ACTIONS

    def __init__(self, level: Level, max_steps: int = 300):
        self.level = level
        self.max_steps = max_steps
        self.pos = [None, None]
        self.steps = 0
        self.done = False
        self.win = False
        self.fail_reason: Optional[str] = None
        self._doors_by_pos = {}
        self._keys_by_pos = {}

    # ---- lifecycle ----

    def reset(self):
        lvl = self.level
        self.pos = [lvl.spawns[RED], lvl.spawns[BLUE]]
        self.steps = 0
        self.done = False
        self.win = False
        self.fail_reason = None
        for d in lvl.doors:
            d.open = False
        for k in lvl.keys:
            k.collected = False
        self._doors_by_pos = {d.pos: d for d in lvl.doors}
        self._keys_by_pos = {k.pos: k for k in lvl.keys}
        return [self._obs(i) for i in range(N_AGENTS)]

    def _in_bounds(self, p):
        return 0 <= p[0] < self.level.width and 0 <= p[1] < self.level.height

    def _walkable(self, p):
        if not self._in_bounds(p):
            return False
        if p in self.level.blocked:
            return False
        door = self._doors_by_pos.get(p)
        if door is not None and not door.open:
            return False
        return True

    def _hazardous(self, p):
        return p in self.level.hazard

    def _ball_cells(self, step: int):
        cells = set()
        for b in self.level.balls:
            cells |= b.cells_at(step)
        return cells

    # ---- policy-layer helpers (mirror the checkpoint's `policy`/
    # `action_safety` metadata) ----

    def can_interact(self, i):
        """True if agent i is standing on its own uncollected key."""
        p = self.pos[i]
        key = self._keys_by_pos.get(p)
        if key is None or key.collected:
            return False
        color = "red" if i == RED else "blue"
        return key.color == color

    def route_action(self, i):
        """Action index (0-3) matching the BFS-suggested direction, or
        None if already at the target (no direction to bias toward)."""
        _, _, route, _, _ = self._goal_info(i)
        if route == (0, 0):
            return None
        return DIRS.index(route)

    def safe_actions(self, i, horizon=10):
        """
        For each of the 6 actions, whether taking it now still leaves a
        way to dodge every wipeout ball for `horizon` steps into the
        future (balls move on a fixed schedule independent of the
        agents, so this is a straightforward forward search — no need
        to model the other agent or pushable blocks, since neither
        exists in this reduced level format).
        Falls back to "everything safe" if no escape exists at all
        (matches env_bridge.py's behavior of shrinking the horizon and
        eventually giving up rather than paralyzing the agent).
        """
        start_pos = self.pos[i]
        start_tick = self.steps

        def make_survives():
            memo = {}

            def survives(pos, tick, remaining):
                if remaining == 0:
                    return True
                key = (pos, tick, remaining)
                if key in memo:
                    return memo[key]
                ok = False
                for dx, dy in DIRS + ((0, 0),):
                    dest = (pos[0] + dx, pos[1] + dy)
                    if dest != pos:
                        if not self._walkable(dest) or self._hazardous(dest):
                            continue
                    next_tick = tick + 1
                    if dest in self._ball_cells(next_tick):
                        continue
                    if survives(dest, next_tick, remaining - 1):
                        ok = True
                        break
                memo[key] = ok
                return ok

            return survives

        def try_horizon(h):
            survives = make_survives()
            results = {}
            for a in range(N_ACTIONS):
                if a in (INTERACT, WAIT):
                    dest = start_pos
                else:
                    dx, dy = DIRS[a]
                    cand = (start_pos[0] + dx, start_pos[1] + dy)
                    if not self._walkable(cand) or self._hazardous(cand):
                        results[a] = False
                        continue
                    dest = cand
                next_tick = start_tick + 1
                if dest in self._ball_cells(next_tick):
                    results[a] = False
                    continue
                results[a] = survives(dest, next_tick, h - 1)
            return results

        for h in range(horizon, 0, -1):
            results = try_horizon(h)
            if any(results.values()):
                return results
        # nothing survives any lookahead — don't paralyze the agent
        return {a: True for a in range(N_ACTIONS)}

    def snapshot(self):
        """Key/door/ball state for the CURRENT self.steps tick, for
        Godot to animate (hide collected keys, hide opened doors,
        move ball sprites)."""
        return {
            "keys": [
                {"color": k.color, "pos": list(k.pos), "collected": k.collected}
                for k in self.level.keys
            ],
            "doors": [
                {"color": d.color, "pos": list(d.pos), "open": d.open}
                for d in self.level.doors
            ],
            "balls": [
                {"color": b.color, "center": list(b.center_at(self.steps))}
                for b in self.level.balls
            ],
        }

    # ---- stepping ----

    def step(self, actions):
        if self.done:
            raise RuntimeError("episode already finished; call reset() first")

        events = []
        for i, action in enumerate(actions):
            self._apply_action(i, action, events)

        self.steps += 1

        # wipeout balls: fatal if an agent is standing on a ball cell now
        ball_cells = self._ball_cells(self.steps)
        for i, p in enumerate(self.pos):
            if p in ball_cells:
                self.done = True
                self.fail_reason = "wipeout"
                events.append(f"agent_{i}:wipeout")

        if not self.done:
            self.win = all(p == self.level.goal for p in self.pos)
            if self.win:
                self.done = True

        timed_out = False
        if not self.done and self.steps >= self.max_steps:
            self.done = True
            timed_out = True
            self.fail_reason = "timeout"

        obs = [self._obs(i) for i in range(N_AGENTS)]
        info = {"events": events, "timed_out": timed_out}
        return obs, self.done, info

    def _apply_action(self, i, action, events):
        if action == INTERACT:
            p = self.pos[i]
            key = self._keys_by_pos.get(p)
            if key is not None and not key.collected:
                agent_color = "red" if i == RED else "blue"
                if key.color != agent_color:
                    events.append(f"agent_{i}:wrong_key")
                    return
                key.collected = True
                for d in self.level.doors:
                    if d.color == key.color:
                        d.open = True
                events.append(f"agent_{i}:key")
            return

        if action == WAIT:
            return

        if action in (NORTH, EAST, SOUTH, WEST):
            dx, dy = DIRS[action]
            dest = (self.pos[i][0] + dx, self.pos[i][1] + dy)
            if self._hazardous(dest):
                events.append(f"agent_{i}:hazard")
                return
            if not self._walkable(dest):
                events.append(f"agent_{i}:blocked")
                return
            self.pos[i] = dest

    # ---- BFS navigation (drives the goal-direction global features) ----

    def _bfs(self, targets: set):
        dist = {}
        pending = deque()
        for t in targets:
            if t not in dist:
                dist[t] = 0
                pending.append(t)
        while pending:
            p = pending.popleft()
            for dx, dy in DIRS:
                np_ = (p[0] + dx, p[1] + dy)
                if np_ in dist or not self._walkable(np_):
                    continue
                dist[np_] = dist[p] + 1
                pending.append(np_)
        return dist

    def _goal_info(self, i):
        color = "red" if i == RED else "blue"
        own_key = next((k for k in self.level.keys
                         if k.color == color and not k.collected), None)
        useful_key = False
        if own_key is not None:
            useful_key = any(
                d.color == color and not d.open for d in self.level.doors
            )

        me = self.pos[i]
        if own_key is not None and useful_key:
            target_pos = own_key.pos
            kind = "key"
        else:
            target_pos = self.level.goal
            kind = "exit"

        dist_map = self._bfs({target_pos})
        distance = dist_map.get(me)
        reachable = distance is not None
        if not reachable:
            distance = abs(me[0] - target_pos[0]) + abs(me[1] - target_pos[1])

        route = (0, 0)
        if reachable and distance > 0:
            for dx, dy in DIRS:
                np_ = (me[0] + dx, me[1] + dy)
                if dist_map.get(np_) == distance - 1:
                    route = (dx, dy)
                    break

        delta = (target_pos[0] - me[0], target_pos[1] - me[1])
        return kind, delta, route, distance, reachable

    # ---- observation ----

    def _obs(self, i):
        view = [0.0] * (VIEW * VIEW * CHANNELS)
        ox, oy = self.pos[i]
        ball_now = self._ball_cells(self.steps)
        ball_next = self._ball_cells(self.steps + 1)
        small_now = {c for b in self.level.balls if b.color == "small"
                     for c in b.cells_at(self.steps)}
        small_next = {c for b in self.level.balls if b.color == "small"
                      for c in b.cells_at(self.steps + 1)}
        big_now = ball_now - small_now
        big_next = ball_next - small_next

        cell = 0
        for dy in range(-RADIUS, RADIUS + 1):
            for dx in range(-RADIUS, RADIUS + 1):
                p = (ox + dx, oy + dy)
                b = cell * CHANNELS
                cell += 1

                if not self._in_bounds(p):
                    view[b + BLOCKED] = 1.0
                    continue
                if p in self.level.blocked:
                    view[b + BLOCKED] = 1.0
                elif p in self.level.hazard:
                    view[b + HAZARD] = 1.0

                if p == self.level.goal:
                    view[b + EXIT] = 1.0

                if p in small_now:
                    view[b + NORMAL_BALL_NOW] = 1.0
                if p in small_next:
                    view[b + NORMAL_BALL_NEXT] = 1.0
                if p in big_now:
                    view[b + BIG_BALL_NOW] = 1.0
                if p in big_next:
                    view[b + BIG_BALL_NEXT] = 1.0

                key = self._keys_by_pos.get(p)
                if key is not None and not key.collected:
                    agent_color = "red" if i == RED else "blue"
                    channel = OWN_KEY if key.color == agent_color else TEAMMATE_KEY
                    view[b + channel] = 1.0

                door = self._doors_by_pos.get(p)
                if door is not None:
                    if door.open:
                        view[b + DOOR_OPEN] = 1.0
                    else:
                        agent_color = "red" if i == RED else "blue"
                        channel = (
                            OWN_DOOR_CLOSED if door.color == agent_color
                            else TEAMMATE_DOOR_CLOSED
                        )
                        view[b + channel] = 1.0
                        view[b + BLOCKED] = 1.0

        return view + self._extras(i)

    def _extras(self, i):
        me, mate = self.pos[i], self.pos[1 - i]
        width = float(max(1, self.level.width - 1))
        height = float(max(1, self.level.height - 1))

        kind, delta, route, distance, reachable = self._goal_info(i)

        n_keys = max(1, len(self.level.keys))
        n_doors = max(1, len(self.level.doors))
        keys_done = sum(k.collected for k in self.level.keys)
        doors_done = sum(d.open for d in self.level.doors)

        color = "red" if i == RED else "blue"
        mate_color = "blue" if i == RED else "red"
        own_keys = [k for k in self.level.keys if k.color == color]
        mate_keys = [k for k in self.level.keys if k.color == mate_color]
        own_doors = [d for d in self.level.doors if d.color == color]
        mate_doors = [d for d in self.level.doors if d.color == mate_color]

        def frac_done(items, attr):
            if not items:
                return 1.0
            return sum(getattr(x, attr) for x in items) / len(items)

        # progress_scale: mirrors _calculate_progress_scale() with only
        # doors+their required key contributing (exit has no lock here).
        raw_total = 0.0
        for d in self.level.doors:
            if not d.open:
                raw_total += 2.0  # R_DOOR
                raw_total += 1.0  # R_KEY for its required key
        progress_scale = min(1.0, 5.0 / max(1.0, raw_total)) if raw_total else 1.0

        extras = [
            (self.level.goal[0] - me[0]) / width,
            (self.level.goal[1] - me[1]) / height,
            (mate[0] - me[0]) / width,
            (mate[1] - me[1]) / height,
            delta[0] / width,
            delta[1] / height,
            float(route[0]),
            float(route[1]),
            min(1.0, distance / max(1, self.max_steps)),
            float(reachable),
            float(kind == "key"),
            0.0,  # switch
            0.0,  # checkpoint
            float(kind == "exit"),
            0.0,  # crate
            0.0,  # switch mode TOGGLE
            0.0,  # switch mode HOLD
            0.0,  # switch mode ONESHOT
            float(self.can_interact(i)),
            0.0,  # can_push (no crates)
            keys_done / n_keys,
            doors_done / n_doors,
            0.0,  # switches activated / total
            0.0,  # checkpoints / total
            1.0,  # exit_open (no lock in this level format)
            1.0 - self.steps / self.max_steps,
            float(i),
            1.0,  # exit_requires_both_agents (fixed True)
            float(delta == (0, 0)),
            progress_scale,
            min(1.0, self.level.width / 64.0),
            min(1.0, self.level.height / 64.0),
            frac_done(own_keys, "collected"),
            frac_done(mate_keys, "collected"),
            frac_done(own_doors, "open"),
            frac_done(mate_doors, "open"),
            0.0,  # route_wait (no HOLD switches)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # timed door block (unused)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # bridge block (unused)
        ]
        if len(extras) != GLOBALS:
            raise RuntimeError(f"expected {GLOBALS} globals, got {len(extras)}")
        return extras

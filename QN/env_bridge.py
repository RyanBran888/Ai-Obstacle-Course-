from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Architecture"))

from coop_env import EnvironmentSession, GenerationConfig, Vec2 
from coop_env.entities import Checkpoint, Key, LockedDoor, Switch, SwitchMode 
from coop_env.tiles import Tile, is_hazard 

from QN_model import CHANNELS, N_ACTIONS, OBS_DIM, VIEW

N_AGENTS = 2
RADIUS = VIEW // 2
INTERACT = 8
DIRS = (Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0))  # N E S W

# view channels
BLOCKED, HAZARD, KEY, DOOR, SWITCH, CRATE, EXIT = range(CHANNELS)

# rewards
R_EXIT, R_KEY, R_DOOR, R_CHECK = 10.0, 1.0, 2.0, 0.5
R_STEP, R_HAZARD, R_STUCK = -0.01, -1.0, -0.02

SOLID = (Tile.VOID, Tile.WALL, Tile.OBSTACLE)


class CoopEnvBridge:
    obs_dim = OBS_DIM
    n_actions = N_ACTIONS

    def __init__(self, config=None, seed=None, max_steps=200):
        self.cfg = config or GenerationConfig.preset("standard")
        self.sess = EnvironmentSession(self.cfg, master_seed=seed)
        self.max_steps = max_steps
        self.pos: list[Vec2] = []
        self.steps = 0

    @property
    def room(self):
        return self.sess.room

    @property
    def state(self):
        return self.sess.state

    def reset(self, seed=None):
        self.sess.reset(seed=seed) if seed is not None else self.sess.reset()
        self.pos = [s.pos for s in self.room.spawns]
        self.steps = 0
        self._holds()
        return [self._obs(i) for i in range(N_AGENTS)]

    def step(self, actions):
        before = self._progress()

        own = [R_STEP] * N_AGENTS
        for i, a in enumerate(actions):
            own[i] += self._use(i) if a == INTERACT else self._move(i, int(a))
        self._holds()
        self.state.advance(1)
        self._unstick()
        self.steps += 1

        keys, doors, checks = (a - b for a, b in zip(self._progress(), before))
        team = keys * R_KEY + doors * R_DOOR + checks * R_CHECK
        done = self._won()
        cut = not done and self.steps >= self.max_steps
        if done:
            team += R_EXIT

        rewards = [team + own[i] for i in range(N_AGENTS)]
        return [self._obs(i) for i in range(N_AGENTS)], rewards, done, cut, {}

    # actions

    def _move(self, i, action):
        d = DIRS[action % 4]
        far = 2 if action >= 4 else 1
        moved = 0
        for _ in range(far):
            dest = self.pos[i] + d
            self._push(dest, d)
            if not self._walkable(dest):
                break
            self.pos[i] = dest
            moved += 1
            if is_hazard(self.room.terrain_at(dest)):
                self.pos[i] = self.room.spawns[i].pos
                return R_HAZARD
        return 0.0 if moved else R_STUCK

    def _use(self, i):
        #Pick up / flip whatever is underfoot
        for e in self.room.entities_at(self.pos[i]):
            if isinstance(e, Key) and not self.state.is_key_collected(e.id):
                self.state.collect_key(e.id)
            elif isinstance(e, Switch) and e.mode is not SwitchMode.HOLD:
                self.state.set_switch(e.id, not self.state.is_switch_active(e.id))
            elif isinstance(e, Checkpoint) and not self.state.is_checkpoint_reached(e.id):
                self.state.reach_checkpoint(e.id)
        return 0.0

    def _push(self, dest, d):
        # Shove a crate one tile
        crate = self.state.blocking_entity_at(dest)
        if crate is None or not crate.startswith("block"):
            return
        past = dest + d
        if self._walkable(past) and not is_hazard(self.room.terrain_at(past)):
            if past not in self.pos:
                self.state.place_block(crate, past)

    def _walkable(self, p):
        return (
            self.room.terrain.in_bounds(p)
            and self.room.terrain_at(p) not in SOLID
            and self.state.blocking_entity_at(p) is None
        )

    def _holds(self):
        here = set(self.pos)
        for s in self.room.switches:
            if s.mode is SwitchMode.HOLD:
                self.state.set_switch(s.id, s.pos in here)

    def _unstick(self):
        for i, p in enumerate(self.pos):
            if not self._walkable(p):
                self.pos[i] = self.room.spawns[i].pos

    # progress

    def _progress(self):
        return (
            len(self.state.keys_collected),
            sum(self.state.doors_open.values()),
            len(self.state.checkpoints_reached),
        )

    def _won(self):
        if not self.state.exit_open:
            return False
        at_exit = [p == self.room.exit.pos for p in self.pos]
        return all(at_exit) if self.cfg.exit_requires_both_agents else any(at_exit)

    def _obs(self, i):
        view = [0.0] * (VIEW * VIEW * CHANNELS)
        crates = set(self.state.block_positions.values())
        ox, oy = self.pos[i]

        cell = 0
        for dy in range(-RADIUS, RADIUS + 1):
            for dx in range(-RADIUS, RADIUS + 1):
                p = Vec2(ox + dx, oy + dy)
                b = cell * CHANNELS
                cell += 1
                if not self.room.terrain.in_bounds(p):
                    view[b + BLOCKED] = 1.0
                    continue
                tile = self.room.terrain_at(p)
                if tile in SOLID:
                    view[b + BLOCKED] = 1.0
                if is_hazard(tile):
                    view[b + HAZARD] = 1.0
                if p in crates:
                    view[b + CRATE] = 1.0
                if p == self.room.exit.pos:
                    view[b + EXIT] = 1.0
                for e in self.room.entities_at(p):
                    if isinstance(e, Key) and not self.state.is_key_collected(e.id):
                        view[b + KEY] = 1.0
                    elif isinstance(e, LockedDoor) and not self.state.is_door_open(e.id):
                        view[b + DOOR] = 1.0
                    elif isinstance(e, Switch):
                        view[b + SWITCH] = 1.0

        return view + self._extras(i)

    def _extras(self, i):
        me, mate = self.pos[i], self.pos[1 - i]
        span = float(max(self.room.width, self.room.height))
        keys, doors, checks = self._progress()

        left = [k for k in self.room.keys if not self.state.is_key_collected(k.id)]
        near = min(left, key=lambda k: k.pos.manhattan(me), default=None)
        kx, ky = ((near.pos - me) if near else Vec2(0, 0))

        return [
            (self.room.exit.pos[0] - me[0]) / span,
            (self.room.exit.pos[1] - me[1]) / span,
            (mate[0] - me[0]) / span,
            (mate[1] - me[1]) / span,
            kx / span,
            ky / span,
            keys / max(1, len(self.room.keys)),
            doors / max(1, len(self.room.doors)),
            1.0 if self.state.exit_open else 0.0,
            1.0 - self.steps / self.max_steps,
            me[0] / span,
            me[1] / span,
            1.0 if is_hazard(self.room.terrain_at(me)) else 0.0,
            float(i),
            sum(self.state.switches_active.values()) / max(1, len(self.room.switches)),
            checks / max(1, len(self.room.checkpoints)),
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

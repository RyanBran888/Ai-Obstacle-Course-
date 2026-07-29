"""Environment adapter for the tabular agents.

This is a deliberate sibling of `QN/env_bridge.py`, not a subclass of it. Two
reasons for the duplication:

1.  `env_bridge` imports `QN_model`, which imports torch. The tabular stack has
    no use for torch, and keeping it out means this whole package is pure
    Python and runs anywhere.
2.  The Q-table bots are meant to be a *separate* experiment. Sharing a base
    class would make it easy to accidentally change both at once.

What is NOT different, and must stay that way for the comparison to mean
anything: the action set, the movement/push/hazard rules, and the reward
constants are copied verbatim from `env_bridge.py`. If you change one, change
the other. The only thing that differs is what `reset`/`step` hand back --
`env_bridge` returns a 191-float vector, this returns a hashable state key from
whatever encoder you plug in.
"""

from __future__ import annotations

from typing import Any, Sequence

import QT_paths  # noqa: F401  -- puts Architecture/ on sys.path

from coop_env import EnvironmentSession, GenerationConfig, Vec2  # noqa: E402
from coop_env.entities import Checkpoint, Key, Switch, SwitchMode  # noqa: E402
from coop_env.tiles import Tile, is_hazard  # noqa: E402

from QT_encoder import FeatureEncoder, StateEncoder  # noqa: E402

N_AGENTS = 2

#: Action layout, identical to QN_model.ACTIONS.
#:   0-3  step one tile   north / east / south / west
#:   4-7  step two tiles  north / east / south / west
#:   8    interact with whatever is underfoot
ACTIONS: tuple[str, ...] = (
    "north",
    "east",
    "south",
    "west",
    "north_long",
    "east_long",
    "south_long",
    "west_long",
    "interact",
)
N_ACTIONS = len(ACTIONS)
INTERACT = 8

DIRS = (Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0))  # N E S W
SOLID = (Tile.VOID, Tile.WALL, Tile.OBSTACLE)

# Reward constants -- copied from env_bridge.py so both agent families are
# scored on exactly the same scale.
R_EXIT, R_KEY, R_DOOR, R_CHECK = 10.0, 1.0, 2.0, 0.5
R_STEP, R_HAZARD, R_STUCK = -0.01, -1.0, -0.02


class TabularCoopEnv:
    """Two-agent room environment that emits discrete, hashable states.

    Parameters
    ----------
    config
        Generation config. Defaults to the "standard" preset.
    seed
        Master seed for the episode stream, for reproducible runs.
    max_steps
        Episode cutoff.
    encoder
        Turns (room, state, positions, agent) into a hashable key. See
        `QT_encoder`. Defaults to `FeatureEncoder`, which generalises across
        rooms; use `PositionEncoder` when training on a fixed room.
    room_seeds
        Optional fixed pool of room seeds to cycle through. Tabular learning
        needs to see the same room many times before its table means anything,
        so a small pool (or a single seed) is the usual starting point. `None`
        draws a brand new room every episode, which is the hard setting.
    """

    n_actions = N_ACTIONS
    n_agents = N_AGENTS

    def __init__(
        self,
        config: GenerationConfig | None = None,
        seed: int | None = None,
        max_steps: int = 200,
        encoder: StateEncoder | None = None,
        room_seeds: Sequence[int] | None = None,
    ) -> None:
        self.cfg = config or GenerationConfig.preset("standard")
        self.sess = EnvironmentSession(self.cfg, master_seed=seed)
        self.max_steps = max_steps
        self.encoder = encoder or FeatureEncoder()
        self.room_seeds = list(room_seeds) if room_seeds else None
        self._episode = -1

        self.pos: list[Vec2] = []
        self.steps = 0
        self.last_action: list[int | None] = [None] * N_AGENTS
        """Previous action per agent. Read by `FeatureEncoder` to break loops."""

    # -- passthroughs ------------------------------------------------------

    @property
    def room(self):
        return self.sess.room

    @property
    def state(self):
        return self.sess.state

    # -- episode lifecycle -------------------------------------------------

    def reset(self, seed: int | None = None) -> list[Any]:
        self._episode += 1

        if seed is not None:
            self.sess.reset(seed=seed)
        elif self.room_seeds:
            self.sess.reset(seed=self.room_seeds[self._episode % len(self.room_seeds)])
        else:
            self.sess.reset()

        self.pos = [s.pos for s in self.room.spawns]
        self.steps = 0
        self.last_action = [None] * N_AGENTS
        self._holds()
        self.encoder.on_reset(self)
        return [self.encode(i) for i in range(N_AGENTS)]

    def encode(self, agent: int) -> Any:
        return self.encoder.encode(self, agent)

    def step(self, actions: Sequence[int]):
        """Apply one action per agent. Mirrors `env_bridge.CoopEnvBridge.step`."""
        before = self._progress()

        own = [R_STEP] * N_AGENTS
        for i, a in enumerate(actions):
            own[i] += self._use(i) if a == INTERACT else self._move(i, int(a))
        self.last_action = [int(a) for a in actions]
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
        obs = [self.encode(i) for i in range(N_AGENTS)]
        return obs, rewards, done, cut, {}

    # -- actions -----------------------------------------------------------

    def _move(self, i: int, action: int) -> float:
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

    def _use(self, i: int) -> float:
        for e in self.room.entities_at(self.pos[i]):
            if isinstance(e, Key) and not self.state.is_key_collected(e.id):
                self.state.collect_key(e.id)
            elif isinstance(e, Switch) and e.mode is not SwitchMode.HOLD:
                self.state.set_switch(e.id, not self.state.is_switch_active(e.id))
            elif isinstance(e, Checkpoint) and not self.state.is_checkpoint_reached(e.id):
                self.state.reach_checkpoint(e.id)
        return 0.0

    def _push(self, dest: Vec2, d: Vec2) -> None:
        crate = self.state.blocking_entity_at(dest)
        if crate is None or not crate.startswith("block"):
            return
        past = dest + d
        if self._walkable(past) and not is_hazard(self.room.terrain_at(past)):
            if past not in self.pos:
                self.state.place_block(crate, past)

    def _walkable(self, p: Vec2) -> bool:
        return (
            self.room.terrain.in_bounds(p)
            and self.room.terrain_at(p) not in SOLID
            and self.state.blocking_entity_at(p) is None
        )

    def _holds(self) -> None:
        here = set(self.pos)
        for s in self.room.switches:
            if s.mode is SwitchMode.HOLD:
                self.state.set_switch(s.id, s.pos in here)

    def _unstick(self) -> None:
        for i, p in enumerate(self.pos):
            if not self._walkable(p):
                self.pos[i] = self.room.spawns[i].pos

    # -- progress ----------------------------------------------------------

    def _progress(self) -> tuple[int, int, int]:
        return (
            len(self.state.keys_collected),
            sum(self.state.doors_open.values()),
            len(self.state.checkpoints_reached),
        )

    def _won(self) -> bool:
        if not self.state.exit_open:
            return False
        at_exit = [p == self.room.exit.pos for p in self.pos]
        return all(at_exit) if self.cfg.exit_requires_both_agents else any(at_exit)


if __name__ == "__main__":
    import random

    from QT_encoder import PositionEncoder

    random.seed(0)
    for name, enc in (("feature", FeatureEncoder()), ("position", PositionEncoder())):
        env = TabularCoopEnv(
            GenerationConfig.preset("easy"), seed=1, encoder=enc, room_seeds=[7]
        )
        seen: set = set()
        solved = 0
        for _ in range(50):
            obs = env.reset()
            done = cut = False
            while not (done or cut):
                seen.update(obs)
                obs, rewards, done, cut, _ = env.step(
                    [random.randrange(N_ACTIONS) for _ in range(N_AGENTS)]
                )
            solved += done
        print(f"{name:9s} distinct states {len(seen):6d}   random solved {solved}/50")

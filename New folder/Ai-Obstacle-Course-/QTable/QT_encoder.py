"""Turning a room into a Q-table key.

This is the piece the neural agents do not need. `QN/env_bridge.py` hands its
network a 191-float vector; a network happily interpolates over that, a table
cannot index it. A table needs a small set of discrete states, and choosing that
set *is* the design of a tabular agent -- get it wrong and no amount of training
helps.

Two encoders ship here, at opposite ends of the trade-off:

`PositionEncoder`
    Exact: (x, y, which keys are gone, which switches are thrown, which
    checkpoints are reached, is the exit open). Markov and complete, so tabular
    Q-learning provably converges to the optimal policy -- but only for the room
    it was trained on. The table is literally indexed by coordinates, so it
    transfers to a new seed about as well as a street map of one city transfers
    to another. Pair it with `room_seeds=[n]` for clean convergence on a fixed
    layout.

`FeatureEncoder`
    Relative and local: what is adjacent, roughly which way the current
    objective lies, how far, what am I standing on, where is my partner, am I
    holding a door open. Nothing absolute, so the same key means the same thing
    in every room and the table transfers. The cost is aliasing -- genuinely
    different situations collapse onto one key, which makes the problem
    non-Markov and means the agent can get stuck in a corner it cannot tell
    apart from another corner.

Both are plain `StateEncoder`s: give them the env, get back something hashable.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

import QT_paths  # noqa: F401  -- puts Architecture/ on sys.path

from coop_env import Vec2
from coop_env.entities import Checkpoint, Key, LockedDoor, Switch, SwitchMode
from coop_env.tiles import Tile, is_hazard

SOLID = (Tile.VOID, Tile.WALL, Tile.OBSTACLE)
DIRS = (Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0))  # N E S W


class StateEncoder(Protocol):
    """Anything that can turn the live environment into a hashable key."""

    def encode(self, env: Any, agent: int) -> Any: ...

    def on_reset(self, env: Any) -> None: ...

    def potential(self, env: Any, agent: int) -> float: ...


# ---------------------------------------------------------------------------
# reachability
# ---------------------------------------------------------------------------


class Reachability:
    """Connected components of the currently-walkable, hazard-free floor.

    Needed because "where should I go" is meaningless without "can I get
    there". Without this the agent walks at an objective sitting behind a
    locked door and never works out that the key is the thing to fetch. Adding
    it took a scripted best-case policy on the tutorial preset from 22/60 to
    60/60 -- the difference between an unsolvable task and a solved one.

    Recomputed only when the world changes shape (a door opens, a crate moves)
    rather than every step: a handful of floods per episode instead of four
    hundred.
    """

    __slots__ = ("_signature", "_component")

    def __init__(self) -> None:
        self._signature: Any = None
        self._component: dict[Vec2, int] = {}

    def reset(self) -> None:
        self._signature = None
        self._component = {}

    def component(self, env: Any) -> dict[Vec2, int]:
        signature = (
            tuple(sorted(env.state.doors_open.items())),
            tuple(sorted(env.state.block_positions.items())),
        )
        if signature != self._signature:
            self._component = self._flood(env)
            self._signature = signature
        return self._component

    def same_region(self, env: Any, a: Vec2, b: Vec2) -> bool:
        component = self.component(env)
        here = component.get(a)
        return here is not None and component.get(b) == here

    @staticmethod
    def _flood(env: Any) -> dict[Vec2, int]:
        room = env.room
        component: dict[Vec2, int] = {}
        next_id = 0

        def open_tile(p: Vec2) -> bool:
            return env._walkable(p) and not is_hazard(room.terrain_at(p))

        for start in room.terrain.positions():
            if start in component or not open_tile(start):
                continue
            component[start] = next_id
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                for d in DIRS:
                    nxt = cur + d
                    if nxt in component or not open_tile(nxt):
                        continue
                    component[nxt] = next_id
                    queue.append(nxt)
            next_id += 1

        return component


# ---------------------------------------------------------------------------
# what should this agent be heading for?
# ---------------------------------------------------------------------------

#: Priority order used by `current_target`, and the vocabulary of goal kinds.
GOAL_KINDS = ("key", "switch", "checkpoint", "exit", "hold", "other")


def current_target(env: Any, agent: int, reach: "Reachability | None" = None):
    """Where this agent should be heading, and what kind of thing it is.

    A *pointer*, not a policy. It says "the nearest useful thing you can
    actually walk to is over there"; the agent still has to learn how to get
    there, what to do on arrival, and whether going there first is the right
    call at all. Without something like it the state carries no sense of task
    phase, and a table cannot tell "walking to the key" from "walking to the
    exit" -- the two look identical from four adjacent tiles.

    Order of preference:

    1.  Already holding a door open -- stay put. This is the two-agent lock:
        one bot stands on a HOLD switch while the other walks through. Stepping
        off re-locks the door, so "stay" has to be a target the state can
        express, or the pair oscillates forever.
    2.  A reachable exit prerequisite.
    3.  If none is reachable, whatever unlocks the way: an uncollected key, an
        unthrown switch, an unreached checkpoint in this component.
    4.  Failing all that, the primary objective anyway, reachable or not.
    """
    room, state = env.room, env.state
    me = env.pos[agent]

    if _holding_open_door(env, me) is not None:
        return me, "hold"

    reach = reach or Reachability()
    component = reach.component(env)
    mine = component.get(me)

    primary: list[tuple[Vec2, str]] = []
    if state.exit_open:
        primary.append((room.exit.pos, "exit"))
    else:
        for entity_id in state.objectives_remaining():
            entity = room.find(entity_id)
            if entity is not None:
                primary.append((entity.pos, _kind_name(entity)))

    usable = [c for c in primary if mine is not None and component.get(c[0]) == mine]
    if usable:
        return min(usable, key=lambda c: me.manhattan(c[0]))

    unlocks: list[tuple[Vec2, str]] = []
    for key in room.keys:
        if not state.is_key_collected(key.id):
            unlocks.append((key.pos, "key"))
    for switch in room.switches:
        if not state.is_switch_active(switch.id):
            unlocks.append((switch.pos, "switch"))
    for check in room.checkpoints:
        if not state.is_checkpoint_reached(check.id):
            unlocks.append((check.pos, "checkpoint"))

    usable = [c for c in unlocks if mine is not None and component.get(c[0]) == mine]
    if usable:
        return min(usable, key=lambda c: me.manhattan(c[0]))

    if primary:
        return min(primary, key=lambda c: me.manhattan(c[0]))
    return room.exit.pos, "exit"


def _holding_open_door(env: Any, pos: Vec2) -> str | None:
    """Id of a HOLD switch under `pos` that is currently keeping a door open."""
    state = env.state
    for entity in env.room.entities_at(pos):
        if not isinstance(entity, Switch) or entity.mode is not SwitchMode.HOLD:
            continue
        if any(state.is_door_open(door_id) for door_id in entity.controls):
            return entity.id
    return None


def _kind_name(entity: Any) -> str:
    if isinstance(entity, Key):
        return "key"
    if isinstance(entity, Switch):
        return "switch"
    if isinstance(entity, Checkpoint):
        return "checkpoint"
    return "other"


def _mask(flags: list[bool]) -> int:
    out = 0
    for i, flag in enumerate(flags):
        if flag:
            out |= 1 << i
    return out


# ---------------------------------------------------------------------------
# exact encoder -- one room at a time
# ---------------------------------------------------------------------------


class PositionEncoder:
    """(position, progress) -- complete for a fixed room, useless across rooms.

    State count is roughly width * height * 2^keys * 2^switches * 2^checkpoints
    * 2, of which only the reachable few thousand ever get stored.
    """

    def __init__(self) -> None:
        self.reach = Reachability()
        self._key_ids: tuple[str, ...] = ()
        self._switch_ids: tuple[str, ...] = ()
        self._check_ids: tuple[str, ...] = ()

    def on_reset(self, env: Any) -> None:
        room = env.room
        self.reach.reset()
        self._key_ids = tuple(sorted(k.id for k in room.keys))
        self._switch_ids = tuple(sorted(s.id for s in room.switches))
        self._check_ids = tuple(sorted(c.id for c in room.checkpoints))

    def encode(self, env: Any, agent: int) -> Any:
        state = env.state
        x, y = env.pos[agent]
        return (
            x,
            y,
            _mask([state.is_key_collected(k) for k in self._key_ids]),
            _mask([state.is_switch_active(s) for s in self._switch_ids]),
            _mask([state.is_checkpoint_reached(c) for c in self._check_ids]),
            state.exit_open,
        )

    def potential(self, env: Any, agent: int) -> float:
        return _distance_potential(env, agent, self.reach)


# ---------------------------------------------------------------------------
# relative encoder -- transfers between rooms
# ---------------------------------------------------------------------------

#: Coarse distance bands. Fine detail near the target, none far away, because
#: "the key is 14 tiles north" and "17 tiles north" call for the same move.
DIST_BANDS = (0, 1, 3, 6, 12)

#: What the agent is standing on.
UNDERFOOT = ("nothing", "key", "switch", "checkpoint", "door", "exit", "hazard")


class FeatureEncoder:
    """Local, relative, and small enough to actually fill in.

    The key is a tuple of:

        neighbours    3^4 = 81   each of N/E/S/W as free / blocked / hazard
        goal_sector   8          compass direction of the current target
        goal_band     5          how far away it is, in coarse bands
        underfoot     7          what is on this tile
        goal_kind     6          key / switch / checkpoint / exit / hold / other
        mate          3          partner far / near / on the same tile
        duty          3          nobody holding / I am holding / partner is
        last_move     5          direction of the previous step, or none

    Nominally ~1.2M combinations; the reachable set is far smaller, since most
    are geometrically impossible. Expect low tens of thousands after a long run.

    Deliberate omission: the two-tile "long" actions (4-7) reach past what the
    neighbour feature can see, so the agent is partly blind about them. A second
    ring would multiply the table by 81 for a marginal gain. It learns them
    anyway, from outcome rather than from sight.
    """

    def __init__(self) -> None:
        self.reach = Reachability()

    def on_reset(self, env: Any) -> None:
        self.reach.reset()

    def encode(self, env: Any, agent: int) -> Any:
        room, state = env.room, env.state
        me = env.pos[agent]

        neighbours = 0
        for i, d in enumerate(DIRS):
            neighbours += _tile_class(env, me + d) * (3**i)

        target, kind = current_target(env, agent, self.reach)
        if target is None:
            sector, band = 0, 0
        else:
            delta = target - me
            sector = _sector(delta)
            band = _band(abs(delta.x) + abs(delta.y))

        mate = env.pos[1 - agent]
        gap = me.manhattan(mate)
        mate_code = 2 if gap == 0 else (1 if gap <= 3 else 0)

        if _holding_open_door(env, me) is not None:
            duty = 1
        elif _holding_open_door(env, mate) is not None:
            duty = 2
        else:
            duty = 0

        previous = env.last_action[agent]
        last_move = 4 if previous is None or previous >= 8 else previous % 4

        return (
            neighbours,
            sector,
            band,
            _underfoot(room, state, me),
            GOAL_KINDS.index(kind) if kind in GOAL_KINDS else 5,
            mate_code,
            duty,
            last_move,
        )

    def potential(self, env: Any, agent: int) -> float:
        return _distance_potential(env, agent, self.reach)


def _tile_class(env: Any, p: Vec2) -> int:
    """0 free, 1 blocked, 2 hazard."""
    room, state = env.room, env.state
    if not room.terrain.in_bounds(p):
        return 1
    tile = room.terrain_at(p)
    if tile in SOLID:
        return 1
    if state.blocking_entity_at(p) is not None:
        return 1
    if is_hazard(tile):
        return 2
    return 0


def _sector(delta: Vec2) -> int:
    """Eight compass sectors. 0 = due north, going clockwise."""
    dx, dy = delta
    if dx == 0 and dy == 0:
        return 0
    ax, ay = abs(dx), abs(dy)
    diagonal = ax > 0 and ay > 0 and 0.4 <= ax / (ax + ay) <= 0.6
    if diagonal:
        return {(1, -1): 1, (1, 1): 3, (-1, 1): 5, (-1, -1): 7}[
            (1 if dx > 0 else -1, 1 if dy > 0 else -1)
        ]
    if ax >= ay:
        return 2 if dx > 0 else 6
    return 4 if dy > 0 else 0


def _band(distance: int) -> int:
    band = 0
    for i, edge in enumerate(DIST_BANDS):
        if distance >= edge:
            band = i
    return band


def _underfoot(room: Any, state: Any, p: Vec2) -> int:
    if p == room.exit.pos:
        return UNDERFOOT.index("exit")
    if is_hazard(room.terrain_at(p)):
        return UNDERFOOT.index("hazard")
    for e in room.entities_at(p):
        if isinstance(e, Key) and not state.is_key_collected(e.id):
            return UNDERFOOT.index("key")
        if isinstance(e, Switch):
            return UNDERFOOT.index("switch")
        if isinstance(e, Checkpoint) and not state.is_checkpoint_reached(e.id):
            return UNDERFOOT.index("checkpoint")
        if isinstance(e, LockedDoor):
            return UNDERFOOT.index("door")
    return 0


def _distance_potential(env: Any, agent: int, reach: "Reachability | None" = None) -> float:
    """Shaping potential: closer to the target and fewer objectives left is better.

    Used as F = gamma * phi(s') - phi(s). Ng et al. showed that form leaves the
    optimal policy unchanged *for a fixed potential function*. Ours is not quite
    fixed -- it re-points when an objective completes -- so the guarantee is not
    airtight. The objective-count term keeps those switches from producing a
    misleading spike: the step down in remaining objectives more than pays for
    the jump in distance when the target moves to something further away. Set
    `Config.shaping = 0.0` for the unshaped, strictly comparable run.
    """
    room, state = env.room, env.state
    me = env.pos[agent]
    target, _ = current_target(env, agent, reach)
    span = float(max(1, max(room.width, room.height)))

    remaining = len(state.objectives_remaining()) + (0 if state.exit_open else 1)
    distance = me.manhattan(target) / span if target is not None else 1.0
    return -(distance + remaining)

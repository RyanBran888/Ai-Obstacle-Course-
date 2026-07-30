"""Turning the bridge's observation into a Q-table key.

The DQN reads 1325 floats: a 7x7x26 local view plus 51 globals. A network
interpolates over that. A table has to index it, so the continuous parts have to
be bucketed and the redundant parts dropped.

The good news is that `_extras()` has already done the hard part. Those 51
globals include `route_dx`/`route_dy`, `goal_distance`, `goal_reachable`,
`goal_key`/`switch`/`checkpoint`/`exit`/`crate`, `can_interact`, `can_push` and
`route_wait` -- which is a strictly better version of the hand-rolled feature
set an earlier version of this package used, and it comes free. This module is
mostly arithmetic on values the bridge already computed.

**The route hint.** `route_dx`/`route_dy` is a BFS-derived next step toward the
current goal, and it is in the key because the DQN gets it too -- `DQN_model`
adds `ROUTE_Q_BIAS = 0.5` to the matching action. Worth being clear about what
that means: with the route in the state, the table is not learning to navigate
from raw terrain. It is learning *when to follow the route, when to deviate for
a hazard, and what to do on arrival*. That is a real learning problem and it is
the same one the DQN faces, which is the point -- but it is not the harder
"navigate from local terrain alone" problem, and a good score here should not be
read as solving that.

Key layout, in index order:

    0  route          5   none / N / E / S / W        (from route_dx, route_dy)
    1  route_tile     3   that tile: free / blocked / hazard
    2  hazards_adj    3   0 / 1 / 2+ hazardous neighbours
    3  route_wait     2   bridge says hold position this tick
    4  goal_kind      6   none / key / switch / checkpoint / exit / crate
    5  reachable      2   goal_reachable
    6  dist           5   coarse distance band to goal
    7  can_interact   2   action 4 is legal here
    8  can_push       2   a crate can be pushed from here
    9  switch_mode    4   none / toggle / hold / oneshot
   10  mech           5   timed door and bridge status, collapsed
   11  teammate       3   far / near / same tile
   12  ball           2   a wipeout ball threatens an adjacent tile
   13  duty           3   nobody holding / I am holding / partner is holding

Nominal product is about 2.6M, but the reachable set is far smaller since most
combinations are geometrically or logically impossible. Sparse storage means
only visited keys cost anything.

Note the local terrain is summarised in two fields (`route_tile`,
`hazards_adj`) rather than the full 3^4 = 81 neighbourhood the earlier version
used. With the route direction supplied, the remaining job of the terrain
features is hazard avoidance, and two fields cover that at a thirteenth of the
size.
"""

from __future__ import annotations

from typing import Any

import QT_paths  # noqa: F401  -- puts the repo root and Architecture on sys.path

from coop_env import Vec2  # noqa: E402
from coop_env.entities import Switch, SwitchMode  # noqa: E402
from coop_env.tiles import Tile, is_hazard  # noqa: E402

from DQN.DQN_model import GLOBAL_NAMES, N_ACTIONS, ROUTE_Q_BIAS  # noqa: E402

DIRS = (Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0))  # N E S W
SOLID = (Tile.VOID, Tile.WALL, Tile.OBSTACLE)

#: Name -> position within the 51 globals, so the code reads by name rather
#: than by magic index. If the schema changes, this breaks loudly at import
#: instead of quietly reading the wrong feature.
G = {name: i for i, name in enumerate(GLOBAL_NAMES)}

#: Field positions inside the key. `act()` needs route and can_interact to
#: reproduce the DQN's masking and route bias, so they are named here rather
#: than counted by hand at the call site.
K_ROUTE = 0
K_ROUTE_WAIT = 3
K_REACHABLE = 5
K_CAN_INTERACT = 7

NO_ROUTE = 4
INTERACT = 4
WAIT = 5

GOAL_FLAGS = ("goal_key", "goal_switch", "goal_checkpoint", "goal_exit", "goal_crate")
SWITCH_FLAGS = ("switch_toggle", "switch_hold", "switch_oneshot")

#: Coarse distance bands, applied to the bridge's already-normalised
#: `goal_distance`. Fine detail near the goal, none far away.
DIST_EDGES = (0.02, 0.06, 0.15, 0.35)


class StateEncoder:
    """Buckets the bridge's globals into a hashable tuple."""

    def encode(self, bridge: Any, agent: int) -> tuple:
        g = bridge._extras(agent)

        route = _route_index(g)
        me = bridge.pos[agent]

        if route == NO_ROUTE:
            route_tile = 0
        else:
            route_tile = _tile_class(bridge, me + DIRS[route])

        hazards = sum(
            1 for d in DIRS if _tile_class(bridge, me + d) == 2
        )
        hazards_adj = 2 if hazards >= 2 else hazards

        mate = bridge.pos[1 - agent]
        gap = me.manhattan(mate)
        teammate = 2 if gap == 0 else (1 if gap <= 3 else 0)

        return (
            route,
            route_tile,
            hazards_adj,
            _bit(g[G["route_wait"]]),
            _goal_kind(g),
            _bit(g[G["goal_reachable"]]),
            _band(g[G["goal_distance"]]),
            _bit(g[G["can_interact"]]),
            _bit(g[G["can_push"]]),
            _switch_mode(g),
            _mechanism(g),
            teammate,
            _ball_threat(bridge, agent),
            _hold_duty(bridge, agent),
        )


# ---------------------------------------------------------------------------
# policy helpers -- these mirror DQN_model exactly
# ---------------------------------------------------------------------------


def valid_actions(key: tuple) -> list[int]:
    """Legal actions in this state.

    `DQN_model.action_mask` masks action 4 whenever `can_interact` is false, and
    `POLICY_CONTRACT` records that as part of the policy. The table honours the
    same contract, or the two agents would be choosing from different sets and
    any comparison between them would be measuring that instead of the learning
    rule.
    """
    actions = list(range(N_ACTIONS))
    if not key[K_CAN_INTERACT]:
        actions.remove(INTERACT)
    return actions


def route_action(key: tuple) -> int:
    """The action the BFS route recommends, or -1 if there is no usable route.

    Eligibility matches `DQN_model.route_actions`: a route exists, interaction
    is not available, and the bridge is not asking us to wait.
    """
    if key[K_ROUTE] == NO_ROUTE:
        return -1
    if not key[K_REACHABLE] or key[K_CAN_INTERACT] or key[K_ROUTE_WAIT]:
        return -1
    return key[K_ROUTE]


def policy_scores(values: list[float], key: tuple) -> list[float]:
    """Q-values plus the DQN's route bias, with illegal actions removed.

    `ROUTE_Q_BIAS` is imported rather than redefined so the two agent families
    cannot drift apart on it.
    """
    scores = list(values)
    route = route_action(key)
    if route >= 0:
        scores[route] += ROUTE_Q_BIAS

    legal = set(valid_actions(key))
    return [s if a in legal else -float("inf") for a, s in enumerate(scores)]


# ---------------------------------------------------------------------------
# field helpers
# ---------------------------------------------------------------------------


def _bit(value: float) -> int:
    return 1 if value > 0.5 else 0


def _route_index(g: list[float]) -> int:
    """Same thresholds and precedence as `DQN_model.route_actions`."""
    dx, dy = g[G["route_dx"]], g[G["route_dy"]]
    route = NO_ROUTE
    if dy < -0.5:
        route = 0
    if dx > 0.5:
        route = 1
    if dy > 0.5:
        route = 2
    if dx < -0.5:
        route = 3
    return route


def _goal_kind(g: list[float]) -> int:
    for i, flag in enumerate(GOAL_FLAGS):
        if g[G[flag]] > 0.5:
            return i + 1
    return 0


def _switch_mode(g: list[float]) -> int:
    for i, flag in enumerate(SWITCH_FLAGS):
        if g[G[flag]] > 0.5:
            return i + 1
    return 0


def _mechanism(g: list[float]) -> int:
    """Timed door and bridge status, collapsed into one field.

    Both are time-dependent obstacles the agent has to wait on, and they never
    co-occur as the *relevant* mechanism for a given agent, so one field of five
    values covers them at a fifth of the cost of two separate fields.
    """
    if g[G["timed_door_present"]] > 0.5:
        return 1 if g[G["timed_door_open"]] > 0.5 else 2
    if g[G["bridge_present"]] > 0.5:
        return 3 if g[G["bridge_solid"]] > 0.5 else 4
    return 0


def _band(distance: float) -> int:
    band = 0
    for edge in DIST_EDGES:
        if distance > edge:
            band += 1
    return band


def _tile_class(bridge: Any, p: Vec2) -> int:
    """0 free, 1 blocked, 2 hazard.

    Reset zones count as hazards. They do not kill, but they undo progress,
    which from a policy's point of view is the same instruction: do not step
    there.
    """
    room = bridge.room
    if not room.terrain.in_bounds(p):
        return 1
    if not bridge._walkable(p):
        return 1
    if is_hazard(room.terrain_at(p)):
        return 2
    if bridge._reset_zone_at(p) is not None:
        return 2
    return 0


def _hold_duty(bridge: Any, agent: int) -> int:
    """0 nobody, 1 I am holding a door open, 2 my partner is.

    Added after the `hold_switch_door` curriculum stage scored 15% against an
    84% threshold while every neighbouring stage passed. That stage is the
    two-agent lock: one bot stands on a HOLD switch, the door it controls opens,
    the other bot walks through, and stepping off re-locks it.

    Nothing in the bridge's 51 globals expresses "you are currently holding a
    door open for someone else". `switch_hold` says the *goal* is a hold switch
    and `route_wait` says hold position this tick, but neither distinguishes
    standing usefully on a switch from standing pointlessly on one -- so both
    situations collapsed onto the same key and the table could not learn that
    the first is worth staying in. Note `crate_hold_switch` passed at 95%
    without this, because a crate can weigh the switch down and no coordination
    is needed; the failure was specific to cases where a *bot* has to wait.
    """
    state = bridge.state
    for index in (agent, 1 - agent):
        for entity in bridge.room.entities_at(bridge.pos[index]):
            if not isinstance(entity, Switch) or entity.mode is not SwitchMode.HOLD:
                continue
            if any(state.is_door_open(door) for door in entity.controls):
                return 1 if index == agent else 2
    return 0


def _ball_threat(bridge, agent, route):
    """0 none · 1 clear · 2 route hit now · 3 route hit next tick
       4 my tile hit next tick · 5 ball near, off-route"""
    if not bridge._wipeout_balls:
        return 0
    tick, me = bridge.state.tick, bridge.pos[agent]

    if bridge._wipeout_at(me, tick + 1) is not None:
        return 4
    if route != NO_ROUTE:
        ahead = me + DIRS[route]
        if bridge._wipeout_at(ahead, tick) is not None:
            return 2
        if bridge._wipeout_at(ahead, tick + 1) is not None:
            return 3
    for d in DIRS:
        if bridge._wipeout_at(me + d, tick) is not None:
            return 5
    return 1

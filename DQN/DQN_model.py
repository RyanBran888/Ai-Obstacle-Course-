"""Dueling Deep Q-network for the cooperative room environment."""
from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torch.nn as nn

VIEW = 7
CHANNELS = 26
GLOBALS = 51
OBS_DIM = VIEW * VIEW * CHANNELS + GLOBALS
OBSERVATION_SCHEMA = 3

CHANNEL_NAMES: tuple[str, ...] = (
    "blocked",
    "hazard",
    "own_key",
    "teammate_key",
    "own_door_closed",
    "teammate_door_closed",
    "door_open",
    "timed_door",
    "timed_door_remaining",
    "timed_door_spent",
    "timed_door_duration",
    "switch_off",
    "switch_on",
    "crate",
    "checkpoint",
    "reset",
    "bridge_tile",
    "bridge_solid",
    "bridge_ticks_to_change",
    "bridge_on_ticks",
    "bridge_off_ticks",
    "exit",
    "normal_ball_now",
    "normal_ball_next",
    "big_ball_now",
    "big_ball_next",
)

GLOBAL_NAMES: tuple[str, ...] = (
    "exit_dx",
    "exit_dy",
    "teammate_dx",
    "teammate_dy",
    "goal_dx",
    "goal_dy",
    "route_dx",
    "route_dy",
    "goal_distance",
    "goal_reachable",
    "goal_key",
    "goal_switch",
    "goal_checkpoint",
    "goal_exit",
    "goal_crate",
    "switch_toggle",
    "switch_hold",
    "switch_oneshot",
    "can_interact",
    "can_push",
    "keys_collected",
    "doors_open",
    "switches_active",
    "checkpoints_reached",
    "exit_open",
    "time_remaining",
    "agent_index",
    "exit_requires_both",
    "on_goal",
    "progress_scale",
    "room_width",
    "room_height",
    "own_keys_collected",
    "teammate_keys_collected",
    "own_key_doors_open",
    "teammate_key_doors_open",
    "route_wait",
    "timed_door_dx",
    "timed_door_dy",
    "timed_door_present",
    "timed_door_open",
    "timed_door_remaining",
    "timed_door_duration",
    "timed_door_spent",
    "bridge_dx",
    "bridge_dy",
    "bridge_present",
    "bridge_solid",
    "bridge_ticks_to_change",
    "bridge_on_ticks",
    "bridge_off_ticks",
)

#: Action layout, in index order:
#:   0-3  step one tile   north / east / south / west
#:   4    interact with whatever is underfoot
#:   5    wait while the world advances
ACTIONS: tuple[str, ...] = (
    "north",
    "east",
    "south",
    "west",
    "interact",
    "wait",
)
N_ACTIONS = len(ACTIONS)

HIDDEN: tuple[int, ...] = (256, 128, 64)
ROUTE_Q_BIAS = 0.5
POLICY_CONTRACT = {
    "version": 1,
    "route_q_bias": ROUTE_Q_BIAS,
    "mask_invalid_interact": True,
}

_GLOBAL_BASE = OBS_DIM - len(GLOBAL_NAMES)
_ROUTE_DX = _GLOBAL_BASE + GLOBAL_NAMES.index("route_dx")
_ROUTE_DY = _GLOBAL_BASE + GLOBAL_NAMES.index("route_dy")
_GOAL_REACHABLE = _GLOBAL_BASE + GLOBAL_NAMES.index("goal_reachable")
_CAN_INTERACT = _GLOBAL_BASE + GLOBAL_NAMES.index("can_interact")
_ROUTE_WAIT = _GLOBAL_BASE + GLOBAL_NAMES.index("route_wait")


def action_mask(observations: torch.Tensor) -> torch.Tensor:
    single = observations.ndim == 1
    batch = observations.unsqueeze(0) if single else observations
    mask = torch.ones(
        (batch.shape[0], N_ACTIONS),
        dtype=torch.bool,
        device=batch.device,
    )
    mask[:, 4] = batch[:, _CAN_INTERACT] > 0.5
    return mask.squeeze(0) if single else mask


def route_actions(observations: torch.Tensor) -> torch.Tensor:
    single = observations.ndim == 1
    batch = observations.unsqueeze(0) if single else observations
    actions = torch.full(
        (batch.shape[0],),
        -1,
        dtype=torch.int64,
        device=batch.device,
    )
    dx = batch[:, _ROUTE_DX]
    dy = batch[:, _ROUTE_DY]
    actions[dy < -0.5] = 0
    actions[dx > 0.5] = 1
    actions[dy > 0.5] = 2
    actions[dx < -0.5] = 3
    eligible = (
        (batch[:, _GOAL_REACHABLE] > 0.5)
        & (batch[:, _CAN_INTERACT] < 0.5)
        & (batch[:, _ROUTE_WAIT] < 0.5)
    )
    actions[~eligible] = -1
    return actions.squeeze(0) if single else actions


def policy_scores(
    q_values: torch.Tensor,
    observations: torch.Tensor,
) -> torch.Tensor:
    single = observations.ndim == 1
    batch = observations.unsqueeze(0) if single else observations
    scores = q_values.unsqueeze(0).clone() if single else q_values.clone()
    routes = route_actions(batch)
    rows = (routes >= 0).nonzero().flatten()
    if len(rows):
        scores[rows, routes[rows]] += ROUTE_Q_BIAS
    scores = scores.masked_fill(~action_mask(batch), -torch.inf)
    return scores.squeeze(0) if single else scores


class QNetwork(nn.Module):

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden: Sequence[int] = HIDDEN,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)

        layers: list[nn.Module] = []
        size = obs_dim
        for width in self.hidden:
            layers.append(nn.Linear(size, width))
            layers.append(nn.ReLU())
            size = width
        self.trunk = nn.Sequential(*layers)

        # Separate state value from action advantage.
        self.value = nn.Linear(size, 1)
        self.advantage = nn.Linear(size, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        value = self.value(h)
        advantage = self.advantage(h)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.forward(obs).squeeze(0)

    @torch.no_grad()
    def best(self, obs: torch.Tensor) -> int:
        return int(policy_scores(self.q_values(obs), obs).argmax().item())

    def act(self, obs: torch.Tensor, eps: float) -> int:
        if random.random() < eps:
            valid = action_mask(obs).nonzero().flatten().tolist()
            return random.choice(valid)
        return self.best(obs)

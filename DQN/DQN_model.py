"""Dueling Deep Q-network for the cooperative room environment."""
from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torch.nn as nn

VIEW = 7
CHANNELS = 12
GLOBALS = 30
OBS_DIM = VIEW * VIEW * CHANNELS + GLOBALS

#: Action layout, in index order:
#:   0-3  step one tile   north / east / south / west
#:   4    interact with whatever is underfoot
ACTIONS: tuple[str, ...] = (
    "north",
    "east",
    "south",
    "west",
    "interact",
)
N_ACTIONS = len(ACTIONS)

HIDDEN: tuple[int, ...] = (256, 128, 64)


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

        layers: list[nn.Module] = []
        size = obs_dim
        for width in hidden:
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
        return int(self.q_values(obs).argmax().item())

    def act(self, obs: torch.Tensor, eps: float) -> int:
        if eps <= 0.0:
            return self.best(obs)
        if random.random() < eps:
            return random.randrange(self.n_actions)
        return self.best(obs)

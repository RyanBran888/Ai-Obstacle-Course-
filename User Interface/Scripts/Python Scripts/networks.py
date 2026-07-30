import random
import torch
import torch.nn as nn

# These match the trained agents
VIEW = 7
CHANNELS = 26
GLOBALS = 51

OBS_DIM = VIEW * VIEW * CHANNELS + GLOBALS   # 1325
N_ACTIONS = 6
HIDDEN = (256, 128, 64)


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden=HIDDEN,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = tuple(hidden)

        layers = []
        size = obs_dim

        for width in self.hidden:
            layers.append(nn.Linear(size, width))
            layers.append(nn.ReLU())
            size = width

        self.trunk = nn.Sequential(*layers)

        # Dueling DQN heads
        self.value = nn.Linear(size, 1)
        self.advantage = nn.Linear(size, n_actions)

    def forward(self, x):
        h = self.trunk(x)

        value = self.value(h)
        advantage = self.advantage(h)

        return value + advantage - advantage.mean(
            dim=-1,
            keepdim=True,
        )

    @torch.no_grad()
    def q_values(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.forward(obs).squeeze(0)

    @torch.no_grad()
    def best(self, obs):
        return int(self.q_values(obs).argmax().item())

    def act(self, obs, eps=0.0):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        return self.best(obs)
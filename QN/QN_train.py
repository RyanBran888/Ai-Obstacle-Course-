from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from QN_model import HIDDEN, N_ACTIONS, OBS_DIM, QNetwork
from QN_rewards import plot_rewards

N_AGENTS = 2


@dataclass(slots=True)
class Config:
    episodes: int = 5_000
    max_steps: int = 200

    lr: float = 1e-3
    gamma: float = 0.99

    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: int = 2_000

    clip: float = 10.0
    sync_every: int = 200
    seed: int = 0


def eps_at(episode: int, cfg: Config) -> float:
    if episode >= cfg.eps_decay:
        return cfg.eps_min
    t = episode / max(1, cfg.eps_decay)
    return cfg.eps_start + t * (cfg.eps_min - cfg.eps_start)


class Agent:
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden=HIDDEN,
        lr: float = 1e-3,
        gamma: float = 0.99,
        clip: float = 10.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.clip = clip

        self.net = QNetwork(obs_dim, n_actions, hidden).to(device)
        self.target = QNetwork(obs_dim, n_actions, hidden).to(device)
        self.sync()

        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def sync(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    def act(self, obs, eps: float) -> int:
        return self.net.act(self._t(obs), eps)

    def learn(self, obs, action: int, reward: float, next_obs, done: bool) -> float:
        #Q(s,a) <- r + gamma * max_a' Q(s',a')
        q = self.net(self._t(obs).unsqueeze(0))[0, action]

        with torch.no_grad():
            next_q = self.target(self._t(next_obs).unsqueeze(0)).max()
            target = reward + self.gamma * next_q * (0.0 if done else 1.0)

        loss = self.loss_fn(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
        self.opt.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.sync()
        self.opt.load_state_dict(ckpt["opt"])

    def _t(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.to(dtype=torch.float32, device=self.device)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)


def build_agents(n: int = N_AGENTS, **kwargs) -> list[Agent]:
    return [Agent(**kwargs) for _ in range(n)]


def train(env, cfg: Config | None = None):
    cfg = cfg or Config()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    agents = build_agents(
        N_AGENTS,
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        lr=cfg.lr,
        gamma=cfg.gamma,
        clip=cfg.clip,
    )

    history: list[float] = []
    steps = 0

    for episode in range(cfg.episodes):
        obs = env.reset()
        eps = eps_at(episode, cfg)
        total = 0.0

        for _ in range(cfg.max_steps):
            actions = [a.act(obs[i], eps) for i, a in enumerate(agents)]
            next_obs, rewards, done, cut, _ = env.step(actions)

            for i, a in enumerate(agents):
                a.learn(obs[i], actions[i], rewards[i], next_obs[i], done)

            obs = next_obs
            total += rewards[0]
            steps += 1

            if steps % cfg.sync_every == 0:
                for a in agents:
                    a.sync()

            if done or cut:
                break

        history.append(total)

    return agents, history


# Temporary training target
# A small grid the two agents share

MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0))


class ToyGrid:
    obs_dim = 4
    n_actions = N_ACTIONS

    def __init__(self, size: int = 8, max_steps: int = 200) -> None:
        self.size = size
        self.max_steps = max_steps
        self.goal = (size // 2, size // 2)
        self.pos: list[tuple[int, int]] = []
        self.steps = 0

    def reset(self) -> list[list[float]]:
        self.pos = [
            (random.randrange(self.size), random.randrange(self.size))
            for _ in range(N_AGENTS)
        ]
        self.steps = 0
        return [self._obs(i) for i in range(N_AGENTS)]

    def step(self, actions: list[int]):
        for i, action in enumerate(actions):
            if action >= 8:
                continue
            dx, dy = MOVES[action % 4]
            far = 2 if action >= 4 else 1
            x, y = self.pos[i]
            x = max(0, min(self.size - 1, x + dx * far))
            y = max(0, min(self.size - 1, y + dy * far))
            self.pos[i] = (x, y)

        self.steps += 1
        done = any(p == self.goal for p in self.pos)
        cut = not done and self.steps >= self.max_steps
        reward = 1.0 if done else -0.01
        return [self._obs(i) for i in range(N_AGENTS)], [reward] * N_AGENTS, done, cut, {}

    def _obs(self, i: int) -> list[float]:
        x, y = self.pos[i]
        return [
            (self.goal[0] - x) / self.size,
            (self.goal[1] - y) / self.size,
            x / self.size,
            y / self.size,
        ]


if __name__ == "__main__":
    cfg = Config(episodes=1500, eps_decay=600)
    agents, history = train(ToyGrid(), cfg)
    plot_rewards(history, cfg)

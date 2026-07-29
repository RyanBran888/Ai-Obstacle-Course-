from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from env_bridge import CoopEnvBridge, GenerationConfig
from QN_model import HIDDEN, N_ACTIONS, OBS_DIM, QNetwork
from QN_rewards import LivePlot, plot_rewards

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


def train(env, cfg: Config | None = None, live: bool = False):
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
    plot = LivePlot(cfg) if live else None

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
        if plot is not None:
            plot.update(history)

    if plot is not None:
        plot.close()
    return agents, history


if __name__ == "__main__":
    cfg = Config(episodes=2000, eps_decay=800, max_steps=200)
    env = CoopEnvBridge(
        GenerationConfig.preset("easy"), seed=0, max_steps=cfg.max_steps
    )
    agents, history = train(env, cfg, live=True)

    for i, a in enumerate(agents):
        a.save(f"agent{i}.pt")
    plot_rewards(history, cfg)

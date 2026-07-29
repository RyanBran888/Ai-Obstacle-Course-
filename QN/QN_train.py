from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
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
    device: str = "auto"

    lr: float = 1e-3
    # 1/(1-gamma) is the effective horizon. 0.99 gives 100 steps for tasks
    # whose median solution is 18-40, so credit takes far longer to
    # propagate than needed. 0.98 -> 50 steps, still clear of the longest.
    gamma: float = 0.98

    eps_start: float = 1.0
    eps_min: float = 0.20
    eps_decay: int = 2_000

    clip: float = 10.0
    sync_every: int = 200
    # Large batches are near-free here (dispatch-bound, not compute-bound), but
    # 256 every 8 steps replays each transition ~16x -- double the DQN norm.
    # 128 keeps the same update frequency at the standard replay ratio of 8.
    batch_size: int = 128
    replay_capacity: int = 50_000
    replay_warmup: int = 1_000
    train_every: int = 8
    # One network for both agents. The observation already carries the agent
    # index, so a shared net can still tell them apart -- and it halves the
    # gradient cost while doubling the data each update sees.
    shared_net: bool = True
    seed: int = 0


def eps_at(episode: int, cfg: Config) -> float:
    if episode >= cfg.eps_decay:
        return cfg.eps_min
    t = episode / max(1, cfg.eps_decay)
    return cfg.eps_start + t * (cfg.eps_min - cfg.eps_start)


def resolve_device(requested: str = "auto") -> torch.device:
    name = requested.strip().lower()
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if name not in {"cpu", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', or 'mps'")
    return torch.device(name)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be at least 1")
        self.capacity = capacity
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.done = np.empty(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_obs[self.index] = next_obs
        self.done[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("batch size must be between 1 and the buffer size")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.done[indices],
        )


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
        replay_capacity: int = 20_000,
        replay_seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip = clip
        self.replay = ReplayBuffer(replay_capacity, obs_dim, replay_seed)

        self.net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.sync()

        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        # Huber: rewards span a wide range, and squaring a large TD error
        # produces a gradient big enough to wreck the network
        self.loss_fn = nn.SmoothL1Loss()

    def sync(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    def act(self, obs, eps: float) -> int:
        return self.net.act(self._t(obs), eps)

    def remember(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        self.replay.add(obs, action, reward, next_obs, done)

    def learn_batch(self, batch_size: int) -> float:
        obs, actions, rewards, next_obs, done = self.replay.sample(batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        q = self.net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online net picks the action, target net scores it.
            # Using the target net for both overestimates systematically.
            best = self.net(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs_t).gather(1, best).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1.0 - done_t)

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
    replay_seed = int(kwargs.pop("replay_seed", 0))
    return [Agent(**kwargs, replay_seed=replay_seed + i) for i in range(n)]


def train(env, cfg: Config | None = None, live: bool = False):
    cfg = cfg or Config()
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if cfg.replay_capacity < cfg.batch_size:
        raise ValueError("replay_capacity must be at least batch_size")
    if cfg.replay_warmup < 0:
        raise ValueError("replay_warmup cannot be negative")
    if cfg.train_every < 1 or cfg.sync_every < 1:
        raise ValueError("train_every and sync_every must be at least 1")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    def make_agent() -> Agent:
        return Agent(
            obs_dim=env.obs_dim,
            n_actions=env.n_actions,
            lr=cfg.lr,
            gamma=cfg.gamma,
            clip=cfg.clip,
            device=device,
            replay_capacity=cfg.replay_capacity,
            replay_seed=cfg.seed,
        )

    if cfg.shared_net:
        shared = make_agent()
        agents = [shared] * N_AGENTS   # both slots drive the same network
        learners = [shared]            # ...so it is only updated once per step
    else:
        agents = [make_agent() for _ in range(N_AGENTS)]
        learners = agents

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
                # `done` only, never `cut`: a step-limit timeout is not a real
                # terminal state, and masking the bootstrap there teaches the
                # network the world ends with zero future value
                a.remember(obs[i], actions[i], rewards[i], next_obs[i], done)

            obs = next_obs
            total += rewards[0]
            steps += 1

            ready = max(cfg.batch_size, cfg.replay_warmup)
            if steps % cfg.train_every == 0 and len(agents[0].replay) >= ready:
                for a in learners:
                    a.learn_batch(cfg.batch_size)

            if steps % cfg.sync_every == 0:
                for a in learners:
                    a.sync()

            if plot is not None and steps % 250 == 0:
                plot.pump()

            if done or cut:
                break

        history.append(total)
        if plot is not None:
            plot.update(history)

    if plot is not None:
        plot.close()
    return agents, history


if __name__ == "__main__":
    cfg = Config(episodes=2000, eps_decay=800, max_steps=300)
    env = CoopEnvBridge(
        GenerationConfig.preset("easy"), seed=0, max_steps=cfg.max_steps
    )
    agents, history = train(env, cfg, live=True)

    for i, a in enumerate(agents):
        a.save(f"agent{i}.pt")
    plot_rewards(history, cfg)

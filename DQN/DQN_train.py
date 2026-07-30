from __future__ import annotations

import copy
import random
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from env_bridge import CoopEnvBridge, GenerationConfig
from DQN.DQN_model import (
    ACTIONS,
    CHANNEL_NAMES,
    GLOBAL_NAMES,
    HIDDEN,
    N_ACTIONS,
    OBS_DIM,
    OBSERVATION_SCHEMA,
    QNetwork,
)

N_AGENTS = 2


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


@dataclass(slots=True)
class Config:
    episodes: int = 5_000
    max_steps: int = 200
    device: str = "auto"

    lr: float = 5e-4
    gamma: float = 0.99
    n_step: int = 3

    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay_steps: int = 50_000

    clip: float = 10.0
    target_sync_updates: int = 1_000
    batch_size: int = 128
    replay_capacity: int = 50_000
    replay_warmup: int = 1_000
    train_every: int = 8
    important_fraction: float = 0.25
    shared_net: bool = True
    seed: int = 0


def eps_at(step: int, cfg: Config) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_min
    t = step / max(1, cfg.eps_decay_steps)
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
        self.terminal = np.empty(capacity, dtype=np.float32)
        self.discount = np.empty(capacity, dtype=np.float32)
        self.important = np.empty(capacity, dtype=np.bool_)
        self.index = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        self.index = 0
        self.size = 0

    def add(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        terminal: bool,
        discount: float,
        important: bool,
    ) -> None:
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_obs[self.index] = next_obs
        self.terminal[self.index] = float(terminal)
        self.discount[self.index] = discount
        self.important[self.index] = important
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, important_fraction: float):
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("batch size must be between 1 and the buffer size")

        wanted = int(round(batch_size * important_fraction))
        important_indices = np.flatnonzero(self.important[: self.size])
        n_important = min(wanted, len(important_indices))
        selected = (
            self.rng.choice(important_indices, size=n_important, replace=False)
            if n_important
            else np.empty(0, dtype=np.int64)
        )
        rest = self.rng.choice(
            self.size, size=batch_size - n_important, replace=False
        )
        indices = np.concatenate((selected, rest))
        self.rng.shuffle(indices)

        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.terminal[indices],
            self.discount[indices],
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
        important_fraction: float = 0.25,
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip = clip
        self.important_fraction = important_fraction
        self.replay = ReplayBuffer(replay_capacity, obs_dim, replay_seed)

        self.net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.sync()

        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def sync(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    def act(self, obs, eps: float) -> int:
        return self.net.act(self._tensor(obs), eps)

    def best_actions(self, observations) -> list[int]:
        batch = self._tensor(np.asarray(observations, dtype=np.float32))
        with torch.no_grad():
            actions = self.net(batch).argmax(dim=1).tolist()
        return [int(action) for action in actions]

    def remember(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        terminal: bool,
        discount: float,
        important: bool,
    ) -> None:
        self.replay.add(
            obs, action, reward, next_obs, terminal, discount, important
        )

    def learn_batch(self, batch_size: int) -> float:
        batch = self.replay.sample(batch_size, self.important_fraction)
        obs, actions, rewards, next_obs, terminal, discount = batch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        terminal_t = torch.as_tensor(terminal, dtype=torch.float32, device=self.device)
        discount_t = torch.as_tensor(discount, dtype=torch.float32, device=self.device)

        q = self.net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.net(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs_t).gather(1, best).squeeze(1)
            target = rewards_t + discount_t * next_q * (1.0 - terminal_t)

        loss = self.loss_fn(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
        self.opt.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "schema": OBSERVATION_SCHEMA,
                "obs_dim": self.net.obs_dim,
                "n_actions": self.net.n_actions,
                "hidden": self.net.hidden,
                "actions": ACTIONS,
                "channels": CHANNEL_NAMES,
                "globals": GLOBAL_NAMES,
                "net": self.net.state_dict(),
                "target": self.target.state_dict(),
                "opt": self.opt.state_dict(),
            },
            path,
        )

    def learning_state(self) -> dict[str, Any]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "obs_dim": self.net.obs_dim,
            "n_actions": self.net.n_actions,
            "hidden": self.net.hidden,
            "actions": ACTIONS,
            "channels": CHANNEL_NAMES,
            "globals": GLOBAL_NAMES,
            "net": _cpu_copy(self.net.state_dict()),
            "target": _cpu_copy(self.target.state_dict()),
            "opt": _cpu_copy(self.opt.state_dict()),
        }

    def load_learning_state(self, checkpoint: dict[str, Any]) -> None:
        if (
            checkpoint.get("schema") != OBSERVATION_SCHEMA
            or checkpoint.get("obs_dim") != self.net.obs_dim
            or checkpoint.get("n_actions") != self.net.n_actions
            or tuple(checkpoint.get("hidden", ())) != self.net.hidden
            or tuple(checkpoint.get("actions", ())) != ACTIONS
            or tuple(checkpoint.get("channels", ())) != CHANNEL_NAMES
            or tuple(checkpoint.get("globals", ())) != GLOBAL_NAMES
        ):
            raise ValueError(
                "checkpoint network, observation, or action contract does not match"
            )
        self.net.load_state_dict(checkpoint["net"])
        self.target.load_state_dict(checkpoint["target"])
        self.opt.load_state_dict(checkpoint["opt"])
        _optimizer_to(self.opt, self.device)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if (
            checkpoint.get("schema") != OBSERVATION_SCHEMA
            or checkpoint.get("obs_dim") != self.net.obs_dim
            or checkpoint.get("n_actions") != self.net.n_actions
            or tuple(checkpoint.get("hidden", ())) != self.net.hidden
            or tuple(checkpoint.get("actions", ())) != ACTIONS
            or tuple(checkpoint.get("channels", ())) != CHANNEL_NAMES
            or tuple(checkpoint.get("globals", ())) != GLOBAL_NAMES
        ):
            raise ValueError(
                "checkpoint network, observation, or action contract does not match"
            )
        self.net.load_state_dict(checkpoint["net"])
        if "target" in checkpoint:
            self.target.load_state_dict(checkpoint["target"])
        else:
            self.sync()
        if "opt" in checkpoint:
            self.opt.load_state_dict(checkpoint["opt"])

    def _tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.to(dtype=torch.float32, device=self.device)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)


def select_actions(
    agents: Sequence[Agent],
    observations,
    eps: float,
) -> list[int]:
    if len(agents) > 1 and all(agent is agents[0] for agent in agents[1:]):
        actions = [0] * len(agents)
        greedy: list[int] = []
        for index in range(len(agents)):
            if eps > 0.0 and random.random() < eps:
                actions[index] = random.randrange(agents[0].net.n_actions)
            else:
                greedy.append(index)
        if greedy:
            choices = agents[0].best_actions(
                [observations[index] for index in greedy]
            )
            for index, action in zip(greedy, choices, strict=True):
                actions[index] = action
        return actions
    return [
        agent.act(observations[index], eps)
        for index, agent in enumerate(agents)
    ]


def build_agents(n: int = N_AGENTS, **kwargs) -> list[Agent]:
    replay_seed = int(kwargs.pop("replay_seed", 0))
    return [Agent(**kwargs, replay_seed=replay_seed + i) for i in range(n)]


@dataclass(slots=True)
class EpisodeResult:
    reward: float
    completed: bool
    timed_out: bool
    steps: int
    metrics: dict[str, Any]


@dataclass(slots=True)
class Evaluation:
    episodes: int
    completed: int
    timeouts: int
    mean_return: float
    mean_steps: float
    mean_episode_steps: float
    mean_keys: float
    mean_doors: float
    mean_switches: float
    mean_checkpoints: float
    exit_open_rate: float
    exit_open_not_reached: int
    mean_wipeout_deaths: float
    wipeout_death_rate: float
    mean_hazard_entries: float
    hazard_entry_rate: float
    normal_ball_death_rate: float
    big_ball_death_rate: float
    mean_bridge_falls: float
    bridge_fall_rate: float
    mean_timed_doors_opened: float
    mean_timed_doors_expired: float
    mean_timed_doors_rearmed: float
    timed_door_open_rate: float
    timed_door_expiry_rate: float
    timed_door_rearm_rate: float
    mean_crate_switches: float
    crate_switch_rate: float
    mean_reset_entries: float
    reset_entry_rate: float
    mean_wrong_key_interactions: float

    @property
    def success_rate(self) -> float:
        return self.completed / self.episodes if self.episodes else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "mean_return": self.mean_return,
            "mean_steps": self.mean_steps,
            "mean_episode_steps": self.mean_episode_steps,
            "mean_keys": self.mean_keys,
            "mean_doors": self.mean_doors,
            "mean_switches": self.mean_switches,
            "mean_checkpoints": self.mean_checkpoints,
            "exit_open_rate": self.exit_open_rate,
            "timeouts": self.timeouts,
            "exit_open_not_reached": self.exit_open_not_reached,
            "mean_wipeout_deaths": self.mean_wipeout_deaths,
            "wipeout_death_rate": self.wipeout_death_rate,
            "mean_hazard_entries": self.mean_hazard_entries,
            "hazard_entry_rate": self.hazard_entry_rate,
            "normal_ball_death_rate": self.normal_ball_death_rate,
            "big_ball_death_rate": self.big_ball_death_rate,
            "mean_bridge_falls": self.mean_bridge_falls,
            "bridge_fall_rate": self.bridge_fall_rate,
            "mean_timed_doors_opened": self.mean_timed_doors_opened,
            "mean_timed_doors_expired": self.mean_timed_doors_expired,
            "mean_timed_doors_rearmed": self.mean_timed_doors_rearmed,
            "timed_door_open_rate": self.timed_door_open_rate,
            "timed_door_expiry_rate": self.timed_door_expiry_rate,
            "timed_door_rearm_rate": self.timed_door_rearm_rate,
            "mean_crate_switches": self.mean_crate_switches,
            "crate_switch_rate": self.crate_switch_rate,
            "mean_reset_entries": self.mean_reset_entries,
            "reset_entry_rate": self.reset_entry_rate,
            "mean_wrong_key_interactions": self.mean_wrong_key_interactions,
        }


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    seed: int
    completed: bool
    timed_out: bool
    reward: float
    steps: int
    keys: int
    doors: int
    switches: int
    checkpoints: int
    exit_open: bool
    wipeout_deaths: int
    hazard_entries: int
    normal_ball_deaths: int
    big_ball_deaths: int
    bridge_falls: int
    timed_doors_opened: int
    timed_doors_expired: int
    timed_doors_rearmed: int
    crate_switches: int
    reset_entries: int
    wrong_key_interactions: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "seed": self.seed,
            "completed": self.completed,
            "timed_out": self.timed_out,
            "reward": self.reward,
            "steps": self.steps,
            "keys": self.keys,
            "doors": self.doors,
            "switches": self.switches,
            "checkpoints": self.checkpoints,
            "exit_open": self.exit_open,
            "wipeout_deaths": self.wipeout_deaths,
            "hazard_entries": self.hazard_entries,
            "normal_ball_deaths": self.normal_ball_deaths,
            "big_ball_deaths": self.big_ball_deaths,
            "bridge_falls": self.bridge_falls,
            "timed_doors_opened": self.timed_doors_opened,
            "timed_doors_expired": self.timed_doors_expired,
            "timed_doors_rearmed": self.timed_doors_rearmed,
            "crate_switches": self.crate_switches,
            "reset_entries": self.reset_entries,
            "wrong_key_interactions": self.wrong_key_interactions,
        }


Transition = tuple[Any, int, float, Any, bool, bool]


class Trainer:
    def __init__(self, env, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._validate(self.cfg)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        self.device = resolve_device(self.cfg.device)
        self.env = env
        self._configure_env(env)

        def make_agent(replay_seed: int) -> Agent:
            return Agent(
                obs_dim=env.obs_dim,
                n_actions=env.n_actions,
                lr=self.cfg.lr,
                gamma=self.cfg.gamma,
                clip=self.cfg.clip,
                device=self.device,
                replay_capacity=self.cfg.replay_capacity,
                replay_seed=replay_seed,
                important_fraction=self.cfg.important_fraction,
            )

        if self.cfg.shared_net:
            shared = make_agent(self.cfg.seed)
            self.agents = [shared] * N_AGENTS
            self.learners = [shared]
        else:
            self.agents = [
                make_agent(self.cfg.seed + index) for index in range(N_AGENTS)
            ]
            self.learners = self.agents

        self.history: list[float] = []
        self.env_steps = 0
        self.updates = 0
        self.episodes = 0
        self._pending = [deque() for _ in range(N_AGENTS)]
        self._reheat_step = 0
        self._reheat_from = 0.0
        self._reheat_steps = 0

    @staticmethod
    def _validate(cfg: Config) -> None:
        if cfg.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if cfg.replay_capacity < cfg.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if cfg.replay_warmup < 0:
            raise ValueError("replay_warmup cannot be negative")
        if cfg.train_every < 1 or cfg.target_sync_updates < 1:
            raise ValueError("training intervals must be at least 1")
        if cfg.n_step < 1:
            raise ValueError("n_step must be at least 1")
        if not 0.0 <= cfg.important_fraction <= 1.0:
            raise ValueError("important_fraction must be between 0 and 1")

    def _configure_env(self, env) -> None:
        if getattr(env, "max_steps", self.cfg.max_steps) != self.cfg.max_steps:
            raise ValueError("trainer and environment max_steps must match")
        if hasattr(env, "shaping_gamma"):
            env.shaping_gamma = self.cfg.gamma

    def set_env(self, env, clear_replay: bool = False) -> None:
        if env.obs_dim != self.env.obs_dim or env.n_actions != self.env.n_actions:
            raise ValueError("new environment has a different observation or action size")
        self._configure_env(env)
        self.env = env
        if clear_replay:
            self.clear_replay()

    def clear_replay(self) -> None:
        for learner in self.learners:
            learner.replay.clear()
        for pending in self._pending:
            pending.clear()

    def learner_state(self) -> list[dict[str, Any]]:
        return [learner.learning_state() for learner in self.learners]

    def load_learner_state(self, states: Sequence[dict[str, Any]]) -> None:
        if len(states) != len(self.learners):
            raise ValueError("recovery learner count does not match")
        for learner, state in zip(self.learners, states, strict=True):
            learner.load_learning_state(state)

    def recovery_state(self) -> dict[str, Any]:
        if any(self._pending):
            raise RuntimeError("recovery checkpoints require an episode boundary")
        if self.device.type == "mps":
            torch.mps.synchronize()
        mps_rng = None
        if self.device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
            mps_rng = torch.mps.get_rng_state().cpu()
        return {
            "schema_version": 1,
            "learners": self.learner_state(),
            "env_steps": self.env_steps,
            "updates": self.updates,
            "episodes": self.episodes,
            "reheat_step": self._reheat_step,
            "reheat_from": self._reheat_from,
            "reheat_steps": self._reheat_steps,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state().cpu(),
            "mps_rng": mps_rng,
            "replay_rngs": [
                copy.deepcopy(learner.replay.rng.bit_generator.state)
                for learner in self.learners
            ],
            "replay_included": False,
        }

    def load_recovery_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("recovery trainer schema does not match")
        if state.get("replay_included") is not False:
            raise ValueError("unsupported recovery replay format")
        self.load_learner_state(state["learners"])
        self.env_steps = int(state["env_steps"])
        self.updates = int(state["updates"])
        self.episodes = int(state["episodes"])
        self._reheat_step = int(state["reheat_step"])
        self._reheat_from = float(state["reheat_from"])
        self._reheat_steps = int(state["reheat_steps"])
        self.clear_replay()
        replay_rngs = state["replay_rngs"]
        if len(replay_rngs) != len(self.learners):
            raise ValueError("recovery replay RNG count does not match")
        for learner, rng_state in zip(
            self.learners, replay_rngs, strict=True
        ):
            learner.replay.rng.bit_generator.state = copy.deepcopy(rng_state)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        mps_rng = state.get("mps_rng")
        if (
            mps_rng is not None
            and self.device.type == "mps"
            and hasattr(torch.mps, "set_rng_state")
        ):
            torch.mps.set_rng_state(mps_rng.cpu())

    def reheat_exploration(self, start: float = 0.30, steps: int = 20_000) -> None:
        self._reheat_step = self.env_steps
        self._reheat_from = max(self.cfg.eps_min, min(1.0, start))
        self._reheat_steps = max(0, steps)

    def epsilon(self) -> float:
        base = eps_at(self.env_steps, self.cfg)
        elapsed = self.env_steps - self._reheat_step
        if self._reheat_steps <= 0 or elapsed >= self._reheat_steps:
            return base
        t = elapsed / self._reheat_steps
        reheated = self._reheat_from + t * (self.cfg.eps_min - self._reheat_from)
        return max(base, reheated)

    def _emit_transition(self, agent_index: int) -> None:
        pending = self._pending[agent_index]
        count = min(self.cfg.n_step, len(pending))
        reward = 0.0
        terminal = False
        important = False
        last_next_obs = pending[0][3]
        used = 0
        for used, transition in enumerate(list(pending)[:count], start=1):
            _, _, step_reward, next_obs, step_terminal, step_important = transition
            reward += (self.cfg.gamma ** (used - 1)) * step_reward
            last_next_obs = next_obs
            terminal = terminal or step_terminal
            important = important or step_important
            if step_terminal:
                break

        first_obs, first_action = pending[0][0], pending[0][1]
        self.agents[agent_index].remember(
            first_obs,
            first_action,
            reward,
            last_next_obs,
            terminal,
            self.cfg.gamma ** used,
            important,
        )
        pending.popleft()

    def _remember(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        terminal: bool,
        important: bool,
    ) -> None:
        for index in range(N_AGENTS):
            self._pending[index].append(
                (
                    obs[index],
                    actions[index],
                    rewards[index],
                    next_obs[index],
                    terminal,
                    important,
                )
            )
            if len(self._pending[index]) >= self.cfg.n_step:
                self._emit_transition(index)
            if terminal:
                while self._pending[index]:
                    self._emit_transition(index)

    def run_episode(
        self,
        *,
        seed: int | None = None,
        learn: bool = True,
        epsilon: float | None = None,
        env=None,
    ) -> EpisodeResult:
        active_env = env or self.env
        self._configure_env(active_env)
        obs = active_env.reset(seed=seed) if seed is not None else active_env.reset()
        total = 0.0
        final_info: dict[str, Any] = {}

        for _ in range(self.cfg.max_steps):
            eps = self.epsilon() if epsilon is None else epsilon
            actions = select_actions(self.agents, obs, eps)
            next_obs, rewards, done, cut, info = active_env.step(actions)
            terminal = done or cut

            if learn:
                important = (
                    bool(info["progress_events"])
                    or terminal
                    or any(
                        "wipeout_" in event
                        or "bridge_fall" in event
                        or event.endswith(":hazard")
                        or event.startswith("timed_door_")
                        for event in info["events"]
                    )
                )
                self._remember(
                    obs, actions, rewards, next_obs, terminal, important
                )
                self.env_steps += 1
                ready = max(self.cfg.batch_size, self.cfg.replay_warmup)
                if (
                    self.env_steps % self.cfg.train_every == 0
                    and len(self.learners[0].replay) >= ready
                ):
                    for learner in self.learners:
                        learner.learn_batch(self.cfg.batch_size)
                    self.updates += 1
                    if self.updates % self.cfg.target_sync_updates == 0:
                        for learner in self.learners:
                            learner.sync()

            obs = next_obs
            total += sum(float(reward) for reward in rewards) / len(rewards)
            final_info = info
            if terminal:
                break
        else:
            raise RuntimeError("environment did not terminate at max_steps")

        metrics = dict(final_info["episode"])
        if learn:
            self.history.append(total)
            self.episodes += 1
        return EpisodeResult(
            reward=total,
            completed=bool(metrics["completed"]),
            timed_out=bool(metrics["timed_out"]),
            steps=int(metrics["steps"]),
            metrics=metrics,
        )

    def fit(
        self,
        episodes: int | None = None,
        *,
        seed_sampler: Callable[[int], int | None] | None = None,
        live: bool = False,
    ) -> list[float]:
        count = self.cfg.episodes if episodes is None else episodes
        if live:
            from DQN.DQN_rewards import LivePlot

            plot = LivePlot(self.cfg)
        else:
            plot = None
        for _ in range(count):
            seed = seed_sampler(self.episodes) if seed_sampler else None
            self.run_episode(seed=seed)
            if plot is not None:
                plot.update(self.history)
                if self.env_steps % 250 == 0:
                    plot.pump()
        if plot is not None:
            plot.close()
        return self.history


def evaluate_detailed(
    agents: Sequence[Agent],
    env,
    seeds: Sequence[int],
) -> tuple[Evaluation, tuple[EvaluationEpisode, ...]]:
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be unique")

    episodes: list[EvaluationEpisode] = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        episode_return = 0.0
        for _ in range(env.max_steps):
            actions = select_actions(agents, obs, 0.0)
            obs, rewards, done, cut, info = env.step(actions)
            episode_return += (
                sum(float(reward) for reward in rewards) / len(rewards)
            )
            if done or cut:
                metrics = info["episode"]
                episodes.append(
                    EvaluationEpisode(
                        seed=int(seed),
                        completed=bool(metrics["completed"]),
                        timed_out=bool(metrics["timed_out"]),
                        reward=episode_return,
                        steps=int(metrics["steps"]),
                        keys=int(metrics["keys_collected"]),
                        doors=int(metrics["doors_opened"]),
                        switches=int(metrics["switches_activated"]),
                        checkpoints=int(metrics["checkpoints_reached"]),
                        exit_open=bool(metrics["exit_opened"]),
                        wipeout_deaths=int(metrics["wipeout_deaths"]),
                        hazard_entries=int(metrics.get("hazards", 0)),
                        normal_ball_deaths=int(
                            metrics.get("normal_ball_deaths", 0)
                        ),
                        big_ball_deaths=int(
                            metrics.get("big_ball_deaths", 0)
                        ),
                        bridge_falls=int(metrics.get("bridge_falls", 0)),
                        timed_doors_opened=int(
                            metrics.get("timed_doors_opened", 0)
                        ),
                        timed_doors_expired=int(
                            metrics.get("timed_doors_expired", 0)
                        ),
                        timed_doors_rearmed=int(
                            metrics.get("timed_doors_rearmed", 0)
                        ),
                        crate_switches=int(
                            metrics.get("crate_switches_solved", 0)
                        ),
                        reset_entries=int(metrics.get("reset_zones", 0)),
                        wrong_key_interactions=int(
                            metrics["wrong_key_interactions"]
                        ),
                    )
                )
                break
        else:
            raise RuntimeError(f"evaluation seed {seed} did not terminate")

    count = len(episodes)
    successful_steps = [
        episode.steps for episode in episodes if episode.completed
    ]
    evaluation = Evaluation(
        episodes=count,
        completed=sum(episode.completed for episode in episodes),
        timeouts=sum(episode.timed_out for episode in episodes),
        mean_return=(
            sum(episode.reward for episode in episodes) / count if count else 0.0
        ),
        mean_steps=(
            sum(successful_steps) / len(successful_steps)
            if successful_steps
            else 0.0
        ),
        mean_episode_steps=(
            sum(episode.steps for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_keys=(
            sum(episode.keys for episode in episodes) / count if count else 0.0
        ),
        mean_doors=(
            sum(episode.doors for episode in episodes) / count if count else 0.0
        ),
        mean_switches=(
            sum(episode.switches for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_checkpoints=(
            sum(episode.checkpoints for episode in episodes) / count
            if count
            else 0.0
        ),
        exit_open_rate=(
            sum(episode.exit_open for episode in episodes) / count
            if count
            else 0.0
        ),
        exit_open_not_reached=sum(
            episode.exit_open and not episode.completed for episode in episodes
        ),
        mean_wipeout_deaths=(
            sum(episode.wipeout_deaths for episode in episodes) / count
            if count
            else 0.0
        ),
        wipeout_death_rate=(
            sum(episode.wipeout_deaths > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_hazard_entries=(
            sum(episode.hazard_entries for episode in episodes) / count
            if count
            else 0.0
        ),
        hazard_entry_rate=(
            sum(episode.hazard_entries > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        normal_ball_death_rate=(
            sum(episode.normal_ball_deaths > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        big_ball_death_rate=(
            sum(episode.big_ball_deaths > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_bridge_falls=(
            sum(episode.bridge_falls for episode in episodes) / count
            if count
            else 0.0
        ),
        bridge_fall_rate=(
            sum(episode.bridge_falls > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_timed_doors_opened=(
            sum(episode.timed_doors_opened for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_timed_doors_expired=(
            sum(episode.timed_doors_expired for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_timed_doors_rearmed=(
            sum(episode.timed_doors_rearmed for episode in episodes) / count
            if count
            else 0.0
        ),
        timed_door_open_rate=(
            sum(episode.timed_doors_opened > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        timed_door_expiry_rate=(
            sum(episode.timed_doors_expired > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        timed_door_rearm_rate=(
            sum(episode.timed_doors_rearmed > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_crate_switches=(
            sum(episode.crate_switches for episode in episodes) / count
            if count
            else 0.0
        ),
        crate_switch_rate=(
            sum(episode.crate_switches > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_reset_entries=(
            sum(episode.reset_entries for episode in episodes) / count
            if count
            else 0.0
        ),
        reset_entry_rate=(
            sum(episode.reset_entries > 0 for episode in episodes) / count
            if count
            else 0.0
        ),
        mean_wrong_key_interactions=(
            sum(episode.wrong_key_interactions for episode in episodes) / count
            if count
            else 0.0
        ),
    )
    return evaluation, tuple(episodes)


def evaluate(agents: Sequence[Agent], env, seeds: Sequence[int]) -> Evaluation:
    evaluation, _ = evaluate_detailed(agents, env, seeds)
    return evaluation


def train(env, cfg: Config | None = None, live: bool = False):
    trainer = Trainer(env, cfg)
    trainer.fit(live=live)
    return trainer.agents, trainer.history


if __name__ == "__main__":
    from DQN.DQN_rewards import plot_rewards

    cfg = Config(episodes=2_000, max_steps=300, eps_decay_steps=120_000)
    env = CoopEnvBridge(
        GenerationConfig.preset("easy"),
        seed=0,
        max_steps=cfg.max_steps,
        shaping_gamma=cfg.gamma,
    )
    agents, history = train(env, cfg, live=True)

    for index, agent in enumerate(dict.fromkeys(agents)):
        agent.save(f"agent{index}.pt")
    plot_rewards(history, cfg)

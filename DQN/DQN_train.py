from __future__ import annotations

import copy
import random
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from env_bridge import CoopEnvBridge, GenerationConfig
from DQN.DQN_model import (
    ACTIONS,
    ACTION_SAFETY_CONTRACT,
    ASSISTED_POLICY_CONTRACT,
    ASSISTED_POLICY_MODE,
    CHANNEL_NAMES,
    GLOBAL_NAMES,
    HIDDEN,
    LEARNED_POLICY_CONTRACT,
    LEARNED_POLICY_MODE,
    LEGACY_LEARNED_POLICY_CONTRACT,
    N_ACTIONS,
    OBS_DIM,
    OBSERVATION_SCHEMA,
    LEGACY_POLICY_CONTRACT,
    POLICY_MODES,
    QNetwork,
    action_scores,
    action_mask,
    route_actions,
)

N_AGENTS = 2
_AUTO_DEVICE: torch.device | None = None
_AUTO_CPU_THREADS: int | None = None
_ROUTE_AUX_WEIGHT = 0.05
_ROUTE_AUX_MARGIN = 1.0
#: Public alias so callers can show the default without reaching for a private
#: name. See Config.route_aux_weight for what it buys.
DEFAULT_ROUTE_AUX_WEIGHT = _ROUTE_AUX_WEIGHT
_WAIT_ACTION = ACTIONS.index("wait")


def _combined_action_mask(
    observation,
    safety_mask=None,
) -> np.ndarray:
    values = torch.as_tensor(observation, dtype=torch.float32)
    semantic = np.asarray(
        action_mask(values).detach().cpu(),
        dtype=np.bool_,
    )
    if semantic.shape != (N_ACTIONS,):
        raise ValueError("observation action mask has the wrong shape")

    combined = semantic.copy()
    if safety_mask is not None:
        safety = np.asarray(safety_mask, dtype=np.bool_)
        if safety.shape != (N_ACTIONS,):
            raise ValueError("environment action mask has the wrong shape")
        combined &= safety
        if not bool(combined.any()):
            combined = semantic.copy()
    if not bool(combined.any()):
        combined[_WAIT_ACTION] = True
    return combined


def _action_masks(
    observations,
    safety_masks=None,
) -> np.ndarray:
    if safety_masks is None:
        safety_masks = [None] * len(observations)
    elif len(safety_masks) != len(observations):
        raise ValueError("action mask count does not match observations")
    return np.stack(
        [
            _combined_action_mask(observation, safety_mask)
            for observation, safety_mask in zip(
                observations,
                safety_masks,
                strict=True,
            )
        ]
    )


def _environment_action_masks(env, observations) -> np.ndarray:
    return _environment_action_masks_for_modes(
        env,
        observations,
        [ASSISTED_POLICY_MODE] * len(observations),
    )


def _environment_route_labels(env) -> tuple[int, ...]:
    """The planner's next step per agent, or all -1 when unavailable.

    Environments without the accessor (stubs, older bridges) simply train on
    TD error alone rather than failing, so an absent teacher degrades to the
    previous behavior instead of crashing the run.
    """
    accessor = getattr(env, "route_action_labels", None)
    if not callable(accessor):
        return (-1,) * N_AGENTS
    reported = cast(Sequence[int], accessor())
    labels = tuple(int(value) for value in reported)
    if len(labels) != N_AGENTS:
        raise ValueError(
            f"expected {N_AGENTS} route labels, got {len(labels)}"
        )
    return labels


def _environment_action_masks_for_modes(
    env,
    observations,
    policy_modes: Sequence[str],
) -> np.ndarray:
    if len(policy_modes) != len(observations):
        raise ValueError("policy mode count does not match observations")
    use_safety = [
        mode == ASSISTED_POLICY_MODE
        for mode in policy_modes
    ]
    provider = getattr(env, "wipeout_action_masks", None)
    provided: Any = (
        provider()
        if any(use_safety) and callable(provider)
        else None
    )
    safety_masks = (
        [
            provided[index] if use_safety[index] else None
            for index in range(len(observations))
        ]
        if provided is not None
        else None
    )
    return _action_masks(observations, safety_masks)


def _masked_policy_scores(
    q_values: torch.Tensor,
    observations: torch.Tensor,
    valid_masks=None,
    policy_mode: str = LEARNED_POLICY_MODE,
) -> torch.Tensor:
    scores = action_scores(q_values, observations, policy_mode)
    if valid_masks is None:
        return scores
    masks = torch.as_tensor(
        valid_masks,
        dtype=torch.bool,
        device=q_values.device,
    )
    if masks.shape != q_values.shape:
        raise ValueError("action masks do not match Q-value shape")
    semantic = action_mask(observations)
    combined = masks & semantic
    empty = ~combined.any(dim=1)
    if bool(empty.any()):
        combined[empty] = semantic[empty]
    empty = ~combined.any(dim=1)
    if bool(empty.any()):
        combined[empty, _WAIT_ACTION] = True
    return scores.masked_fill(~combined, -torch.inf)


def _valid_actions(observation, safety_mask=None) -> list[int]:
    mask = _combined_action_mask(observation, safety_mask)
    return [int(index) for index in np.flatnonzero(mask)]


def _route_auxiliary_loss(
    q_values: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Rank the planner's step above every other legal action by a margin.

    ``labels`` comes from the environment, not from the observation: learned
    mode zeroes the route features an observation-derived label would need, so
    deriving them here would silently produce no labels and no gradient.
    """
    valid = labels >= 0
    if not bool(valid.any()):
        return q_values.new_zeros(())
    rows = valid.nonzero().flatten()
    route_labels = labels[rows]
    route_valid = valid_mask[rows].gather(
        1,
        route_labels.unsqueeze(1),
    ).squeeze(1)
    rows = rows[route_valid]
    route_labels = route_labels[route_valid]
    if not len(rows):
        return q_values.new_zeros(())
    selected = q_values[rows]
    route_q = selected.gather(
        1,
        route_labels.unsqueeze(1),
    ).squeeze(1)
    other_q = selected.masked_fill(~valid_mask[rows], -torch.inf)
    other_q = other_q.scatter(
        1,
        route_labels.unsqueeze(1),
        -torch.inf,
    )
    competitor = other_q.max(dim=1).values
    finite = torch.isfinite(competitor)
    if not bool(finite.any()):
        return q_values.new_zeros(())
    return torch.relu(
        _ROUTE_AUX_MARGIN
        + competitor[finite]
        - route_q[finite]
    ).mean()


def _normalize_policy_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in POLICY_MODES:
        raise ValueError(
            f"policy_mode must be one of {POLICY_MODES}, got {value!r}"
        )
    return mode


def _policy_contract(mode: str) -> dict[str, Any]:
    normalized = _normalize_policy_mode(mode)
    if normalized == LEARNED_POLICY_MODE:
        return dict(LEARNED_POLICY_CONTRACT)
    return dict(ASSISTED_POLICY_CONTRACT)


def _checkpoint_policy_mode(checkpoint: Mapping[str, Any]) -> str:
    """Resolve explicit new checkpoints and implicit historical checkpoints.

    The learned-v3 contract is accepted alongside learned-v4. The two differ
    only in whether the route auxiliary loss ran during training, which changes
    the weights that come out but not how an action is chosen from them, so a
    v3 checkpoint loads and acts correctly under v4 code.
    """
    explicit = checkpoint.get("policy_mode")
    contract = checkpoint.get("policy")
    if explicit is not None:
        mode = _normalize_policy_mode(str(explicit))
        accepted = (
            (LEARNED_POLICY_CONTRACT, LEGACY_LEARNED_POLICY_CONTRACT)
            if mode == LEARNED_POLICY_MODE
            else (ASSISTED_POLICY_CONTRACT,)
        )
        if contract not in accepted:
            raise ValueError(
                "checkpoint policy mode and policy contract do not match"
            )
        return mode

    if contract in (LEARNED_POLICY_CONTRACT, LEGACY_LEARNED_POLICY_CONTRACT):
        return LEARNED_POLICY_MODE
    if contract in (
        None,
        ASSISTED_POLICY_CONTRACT,
        LEGACY_POLICY_CONTRACT,
    ):
        # Before policy_mode was serialized, every action path used the
        # planner-assisted scorer. Treat absent metadata the same way.
        return ASSISTED_POLICY_MODE
    raise ValueError("checkpoint action policy does not match")


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
    policy_mode: str = LEARNED_POLICY_MODE
    #: Weight on the route ranking loss. This is what teaches the network to
    #: read route_dx/route_dy instead of memorizing individual rooms, so it is
    #: the difference between a policy that transfers to unseen rooms and one
    #: that does not. Set to 0.0 to train on TD error alone.
    route_aux_weight: float = _ROUTE_AUX_WEIGHT


def eps_at(step: int, cfg: Config) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_min
    t = step / max(1, cfg.eps_decay_steps)
    return cfg.eps_start + t * (cfg.eps_min - cfg.eps_start)


def _device_benchmark(
    device: torch.device,
    cpu_threads: int,
    net_template: QNetwork,
    target_template: QNetwork,
    action_obs_cpu: torch.Tensor,
    obs_cpu: torch.Tensor,
    next_obs_cpu: torch.Tensor,
    actions_cpu: torch.Tensor,
    rewards_cpu: torch.Tensor,
    terminal_cpu: torch.Tensor,
    discount_cpu: torch.Tensor,
) -> float:
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)

    net = copy.deepcopy(net_template).to(device)
    target = copy.deepcopy(target_template).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    loss_fn = nn.SmoothL1Loss()

    def synchronize() -> None:
        if device.type == "mps":
            torch.mps.synchronize()

    def cycle() -> None:
        for _ in range(8):
            action_obs = action_obs_cpu.to(device)
            with torch.inference_mode():
                net(action_obs).argmax(dim=1).tolist()

        obs = obs_cpu.to(device)
        next_obs = next_obs_cpu.to(device)
        actions = actions_cpu.to(device)
        rewards = rewards_cpu.to(device)
        terminal = terminal_cpu.to(device)
        discount = discount_cpu.to(device)
        q = net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = net(next_obs).argmax(dim=1, keepdim=True)
            next_q = target(next_obs).gather(1, best).squeeze(1)
            expected = rewards + discount * next_q * (1.0 - terminal)
        loss = loss_fn(q, expected)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()

    for _ in range(3):
        cycle()
    synchronize()

    samples: list[float] = []
    for _ in range(7):
        start = time.perf_counter()
        cycle()
        synchronize()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _resolve_auto_device() -> torch.device:
    global _AUTO_DEVICE, _AUTO_CPU_THREADS
    if _AUTO_DEVICE is not None:
        if _AUTO_DEVICE.type == "cpu" and _AUTO_CPU_THREADS is not None:
            torch.set_num_threads(_AUTO_CPU_THREADS)
        return _AUTO_DEVICE

    if not torch.backends.mps.is_available():
        _AUTO_DEVICE = torch.device("cpu")
        return _AUTO_DEVICE

    original_threads = torch.get_num_threads()
    cpu_rng = torch.get_rng_state()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2_907)

    try:
        torch.set_rng_state(generator.get_state())
        net_template = QNetwork(OBS_DIM, N_ACTIONS, HIDDEN)
        target_template = copy.deepcopy(net_template)
        action_obs = torch.randn((2, OBS_DIM), generator=generator)
        obs = torch.randn((128, OBS_DIM), generator=generator)
        next_obs = torch.randn((128, OBS_DIM), generator=generator)
        actions = torch.randint(
            N_ACTIONS, (128,), generator=generator, dtype=torch.int64
        )
        rewards = torch.randn(128, generator=generator)
        terminal = torch.randint(
            2, (128,), generator=generator, dtype=torch.int64
        ).to(torch.float32)
        discount = torch.full((128,), 0.99, dtype=torch.float32)

        cpu_timings: list[tuple[float, int]] = [
            (
                _device_benchmark(
                    torch.device("cpu"),
                    original_threads,
                    net_template,
                    target_template,
                    action_obs,
                    obs,
                    next_obs,
                    actions,
                    rewards,
                    terminal,
                    discount,
                ),
                original_threads,
            )
        ]
        if original_threads != 1:
            cpu_timings.append(
                (
                    _device_benchmark(
                        torch.device("cpu"),
                        1,
                        net_template,
                        target_template,
                        action_obs,
                        obs,
                        next_obs,
                        actions,
                        rewards,
                        terminal,
                        discount,
                    ),
                    1,
                )
            )
        cpu_time, cpu_threads = min(cpu_timings)
        torch.set_num_threads(original_threads)

        try:
            mps_time = _device_benchmark(
                torch.device("mps"),
                original_threads,
                net_template,
                target_template,
                action_obs,
                obs,
                next_obs,
                actions,
                rewards,
                terminal,
                discount,
            )
        except Exception as exc:
            mps_time = float("inf")
            print(
                "Auto device benchmark: Metal failed "
                f"({type(exc).__name__}); using CPU.",
                flush=True,
            )

        if cpu_time < mps_time * 0.95:
            _AUTO_DEVICE = torch.device("cpu")
            _AUTO_CPU_THREADS = cpu_threads
            torch.set_num_threads(cpu_threads)
        else:
            _AUTO_DEVICE = torch.device("mps")
            _AUTO_CPU_THREADS = None
            torch.set_num_threads(original_threads)

        mps_label = (
            f"{mps_time * 1_000:.2f} ms"
            if np.isfinite(mps_time)
            else "failed"
        )
        print(
            "Auto device benchmark: "
            f"CPU {cpu_time * 1_000:.2f} ms "
            f"({cpu_threads} thread{'s' if cpu_threads != 1 else ''}), "
            f"Metal {mps_label}; selected {_AUTO_DEVICE.type}.",
            flush=True,
        )
        return _AUTO_DEVICE
    finally:
        torch.set_rng_state(cpu_rng)
        if _AUTO_DEVICE is None or _AUTO_DEVICE.type != "cpu":
            torch.set_num_threads(original_threads)


def pin_auto_device(name: str, cpu_threads: int | None = None) -> None:
    global _AUTO_DEVICE, _AUTO_CPU_THREADS
    device = torch.device(name)
    if device.type not in {"cpu", "mps"}:
        raise ValueError(f"unsupported saved device {name!r}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("the saved run requires MPS, but MPS is unavailable")
    if device.type == "cpu":
        threads = torch.get_num_threads() if cpu_threads is None else cpu_threads
        if threads < 1:
            raise ValueError("saved CPU thread count must be positive")
        _AUTO_CPU_THREADS = threads
        torch.set_num_threads(threads)
    else:
        _AUTO_CPU_THREADS = None
    _AUTO_DEVICE = device


def resolve_device(requested: str = "auto") -> torch.device:
    name = requested.strip().lower()
    if name == "auto":
        return _resolve_auto_device()
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if name not in {"cpu", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', or 'mps'")
    return torch.device(name)


def _replay_group(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("replay_group must be an integer")
    group = int(value)
    if group < 0:
        raise ValueError("replay_group cannot be negative")
    return group


def _replay_group_weights(
    weights: Mapping[int, float] | None,
) -> dict[int, float] | None:
    if not weights:
        return None
    normalized: dict[int, float] = {}
    for raw_group, raw_weight in weights.items():
        group = _replay_group(raw_group)
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("replay group weights must be finite and nonnegative")
        normalized[group] = weight
    if not any(weight > 0.0 for weight in normalized.values()):
        raise ValueError("at least one replay group weight must be positive")
    return normalized


class _DenseReplayIndex:
    def __init__(self) -> None:
        self.values = np.empty(16, dtype=np.int64)
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, slot: int) -> int:
        if self.size == len(self.values):
            expanded = np.empty(len(self.values) * 2, dtype=np.int64)
            expanded[: self.size] = self.values
            self.values = expanded
        position = self.size
        self.values[position] = slot
        self.size += 1
        return position

    def remove(self, position: int) -> int | None:
        if position < 0 or position >= self.size:
            raise RuntimeError("replay group index is inconsistent")
        self.size -= 1
        if position == self.size:
            return None
        replacement = int(self.values[self.size])
        self.values[position] = replacement
        return replacement

    def active(self) -> np.ndarray:
        return self.values[: self.size]


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
        self.action_masks = np.empty(
            (capacity, N_ACTIONS),
            dtype=np.bool_,
        )
        self.next_action_masks = np.empty(
            (capacity, N_ACTIONS),
            dtype=np.bool_,
        )
        self.important = np.empty(capacity, dtype=np.bool_)
        #: The planner's step for this observation, or -1 where it has none.
        #: Supplied by the environment rather than read back out of the
        #: observation, because learned mode zeroes the route features there.
        self.route_labels = np.full(capacity, -1, dtype=np.int64)
        self.replay_groups = np.empty(capacity, dtype=np.int64)
        self._important_indices = np.empty(capacity, dtype=np.int64)
        self._important_positions = np.full(capacity, -1, dtype=np.int64)
        self._important_count = 0
        self._group_positions = np.full(capacity, -1, dtype=np.int64)
        self._group_indices: dict[
            tuple[int, bool], _DenseReplayIndex
        ] = {}
        self.index = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        active = self._important_indices[: self._important_count]
        self._important_positions[active] = -1
        self._important_count = 0
        self._group_positions.fill(-1)
        self._group_indices.clear()
        self.index = 0
        self.size = 0

    def group_counts(self) -> dict[int, tuple[int, int]]:
        counts: dict[int, list[int]] = {}
        for (group, important), index in self._group_indices.items():
            values = counts.setdefault(group, [0, 0])
            values[int(important)] = len(index)
        return {
            group: (values[0], values[1])
            for group, values in sorted(counts.items())
        }

    def _remove_group_slot(self, slot: int) -> None:
        position = int(self._group_positions[slot])
        if position < 0:
            return
        key = (int(self.replay_groups[slot]), bool(self.important[slot]))
        index = self._group_indices[key]
        replacement = index.remove(position)
        if replacement is not None:
            self._group_positions[replacement] = position
        self._group_positions[slot] = -1
        if not index:
            del self._group_indices[key]

    def _add_group_slot(
        self,
        slot: int,
        replay_group: int,
        important: bool,
    ) -> None:
        key = (replay_group, important)
        index = self._group_indices.get(key)
        if index is None:
            index = _DenseReplayIndex()
            self._group_indices[key] = index
        self._group_positions[slot] = index.add(slot)

    def add(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        terminal: bool,
        discount: float,
        important: bool,
        replay_group: int = 0,
        current_mask=None,
        next_mask=None,
        route_label: int = -1,
    ) -> None:
        group = _replay_group(replay_group)
        slot = self.index
        self._remove_group_slot(slot)
        position = int(self._important_positions[slot])
        if position >= 0:
            self._important_count -= 1
            replacement = int(
                self._important_indices[self._important_count]
            )
            self._important_indices[position] = replacement
            self._important_positions[replacement] = position
            self._important_positions[slot] = -1
        if important:
            self._important_indices[self._important_count] = slot
            self._important_positions[slot] = self._important_count
            self._important_count += 1

        self._add_group_slot(slot, group, bool(important))
        self.obs[slot] = obs
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.next_obs[slot] = next_obs
        self.terminal[slot] = float(terminal)
        self.discount[slot] = discount
        self.action_masks[slot] = _combined_action_mask(obs, current_mask)
        self.next_action_masks[slot] = _combined_action_mask(
            next_obs,
            next_mask,
        )
        self.important[slot] = important
        self.route_labels[slot] = route_label
        self.replay_groups[slot] = group
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _weighted_group_quotas(
        self,
        batch_size: int,
        group_weights: Mapping[int, float],
    ) -> dict[int, int]:
        weights: dict[int, float] = {}
        capacities: dict[int, int] = {}
        for group, weight in sorted(group_weights.items()):
            if weight <= 0.0:
                continue
            important = self._group_indices.get((group, True))
            ordinary = self._group_indices.get((group, False))
            size = (
                (len(important) if important is not None else 0)
                + (len(ordinary) if ordinary is not None else 0)
            )
            if size:
                weights[group] = weight
                capacities[group] = size

        if not weights:
            raise RuntimeError(
                "no replay transitions match a positive group weight"
            )
        if sum(capacities.values()) < batch_size:
            raise RuntimeError(
                "weighted replay groups contain fewer transitions than the batch"
            )

        original_weights = dict(weights)
        quotas = {group: 0 for group in weights}
        if batch_size >= len(quotas):
            for group in quotas:
                quotas[group] = 1
            total_weight = sum(original_weights.values())
            weights = {
                group: max(
                    0.0,
                    (
                        batch_size
                        * original_weights[group]
                        / total_weight
                        - quotas[group]
                    ),
                )
                for group in quotas
            }
        remaining = batch_size - sum(quotas.values())
        while remaining:
            open_groups = [
                group
                for group in weights
                if quotas[group] < capacities[group]
            ]
            total_weight = sum(weights[group] for group in open_groups)
            if not open_groups:
                raise RuntimeError("weighted replay quota allocation failed")
            allocation_weights = weights
            if total_weight <= 0.0:
                allocation_weights = original_weights
                total_weight = sum(
                    allocation_weights[group] for group in open_groups
                )

            desired = {
                group: remaining * allocation_weights[group] / total_weight
                for group in open_groups
            }
            progress = 0
            for group in open_groups:
                count = min(
                    capacities[group] - quotas[group],
                    int(desired[group]),
                )
                quotas[group] += count
                remaining -= count
                progress += count
            if not remaining:
                break

            candidates = [
                group
                for group in open_groups
                if quotas[group] < capacities[group]
            ]
            tie_breakers = {
                group: float(self.rng.random())
                for group in candidates
            }
            order = sorted(
                candidates,
                key=lambda group: (
                    -(desired[group] - int(desired[group])),
                    tie_breakers[group],
                ),
            )
            for group in order:
                if not remaining:
                    break
                quotas[group] += 1
                remaining -= 1
                progress += 1
            if not progress:
                raise RuntimeError("weighted replay quota allocation stalled")
        return quotas

    def _sample_weighted(
        self,
        batch_size: int,
        important_fraction: float,
        group_weights: Mapping[int, float],
    ) -> np.ndarray:
        quotas = self._weighted_group_quotas(batch_size, group_weights)
        selected: list[np.ndarray] = []
        for group, quota in quotas.items():
            if not quota:
                continue
            important = self._group_indices.get((group, True))
            ordinary = self._group_indices.get((group, False))
            important_size = len(important) if important is not None else 0
            ordinary_size = len(ordinary) if ordinary is not None else 0
            wanted_important = quota * important_fraction
            n_important = int(wanted_important)
            if self.rng.random() < wanted_important - n_important:
                n_important += 1
            n_important = min(n_important, important_size)
            n_ordinary = quota - n_important
            if n_ordinary > ordinary_size:
                n_important += n_ordinary - ordinary_size
                n_ordinary = ordinary_size

            if n_important:
                if important is None:
                    raise RuntimeError("important replay index is missing")
                selected.append(
                    self.rng.choice(
                        important.active(),
                        size=n_important,
                        replace=False,
                    )
                )
            if n_ordinary:
                if ordinary is None:
                    raise RuntimeError("ordinary replay index is missing")
                selected.append(
                    self.rng.choice(
                        ordinary.active(),
                        size=n_ordinary,
                        replace=False,
                    )
                )

        indices = np.concatenate(selected)
        if len(indices) != batch_size:
            raise RuntimeError("weighted replay batch size is inconsistent")
        self.rng.shuffle(indices)
        return indices

    def sample(
        self,
        batch_size: int,
        important_fraction: float,
        group_weights: Mapping[int, float] | None = None,
    ):
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("batch size must be between 1 and the buffer size")

        if group_weights is None:
            wanted = int(round(batch_size * important_fraction))
            important_indices = self._important_indices[: self._important_count]
            n_important = min(wanted, self._important_count)
            selected = (
                self.rng.choice(
                    important_indices,
                    size=n_important,
                    replace=False,
                )
                if n_important
                else np.empty(0, dtype=np.int64)
            )
            rest = self.rng.choice(
                self.size, size=batch_size - n_important, replace=False
            )
            indices = np.concatenate((selected, rest))
            self.rng.shuffle(indices)
        else:
            indices = self._sample_weighted(
                batch_size,
                important_fraction,
                group_weights,
            )

        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.terminal[indices],
            self.discount[indices],
            self.action_masks[indices],
            self.next_action_masks[indices],
            self.route_labels[indices],
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
        policy_mode: str = LEARNED_POLICY_MODE,
        route_aux_weight: float = _ROUTE_AUX_WEIGHT,
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip = clip
        self.important_fraction = important_fraction
        self.policy_mode = _normalize_policy_mode(policy_mode)
        if route_aux_weight < 0.0:
            raise ValueError("route_aux_weight cannot be negative")
        self.route_aux_weight = float(route_aux_weight)
        self.require_action_mask = False
        self.latest_learning_metrics: dict[str, float] = {}
        self.replay_group_weights: dict[int, float] | None = None
        self.replay = ReplayBuffer(replay_capacity, obs_dim, replay_seed)

        self.net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.sync()

        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def sync(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    def act(self, obs, eps: float, action_mask=None) -> int:
        if self.require_action_mask and action_mask is None:
            raise ValueError(
                "this checkpoint requires an environment action mask"
            )
        valid = _valid_actions(obs, action_mask)
        if eps > 0.0 and random.random() < eps:
            return random.choice(valid)
        return self.best_actions([obs], [action_mask])[0]

    def best_actions(self, observations, action_masks=None) -> list[int]:
        if self.require_action_mask and action_masks is None:
            raise ValueError(
                "this checkpoint requires environment action masks"
            )
        batch = self._tensor(np.asarray(observations, dtype=np.float32))
        masks = _action_masks(observations, action_masks)
        with torch.inference_mode():
            q_values = self.net(batch)
            actions = _masked_policy_scores(
                q_values,
                batch,
                masks,
                self.policy_mode,
            ).argmax(dim=1).tolist()
        return [int(action) for action in actions]

    def set_replay_group_weights(
        self,
        weights: Mapping[int, float] | None,
    ) -> None:
        self.replay_group_weights = _replay_group_weights(weights)

    def remember(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        terminal: bool,
        discount: float,
        important: bool,
        replay_group: int = 0,
        current_mask=None,
        next_mask=None,
        route_label: int = -1,
    ) -> None:
        self.replay.add(
            obs,
            action,
            reward,
            next_obs,
            terminal,
            discount,
            important,
            replay_group,
            current_mask,
            next_mask,
            route_label,
        )

    def learn_batch(self, batch_size: int) -> dict[str, float]:
        batch = self.replay.sample(
            batch_size,
            self.important_fraction,
            self.replay_group_weights,
        )
        (
            obs,
            actions,
            rewards,
            next_obs,
            terminal,
            discount,
            current_masks,
            next_masks,
            route_labels,
        ) = batch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        route_labels_t = torch.as_tensor(
            route_labels,
            dtype=torch.int64,
            device=self.device,
        )
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        terminal_t = torch.as_tensor(terminal, dtype=torch.float32, device=self.device)
        discount_t = torch.as_tensor(discount, dtype=torch.float32, device=self.device)
        current_masks_t = torch.as_tensor(
            current_masks,
            dtype=torch.bool,
            device=self.device,
        )
        next_masks_t = torch.as_tensor(
            next_masks,
            dtype=torch.bool,
            device=self.device,
        )

        q_values = self.net(obs_t)
        q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = self.net(next_obs_t)
            best = _masked_policy_scores(
                next_online,
                next_obs_t,
                next_masks_t,
                self.policy_mode,
            ).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs_t).gather(1, best).squeeze(1)
            target = rewards_t + discount_t * next_q * (1.0 - terminal_t)

        td_loss = self.loss_fn(q, target)
        # Applies in both policy modes. Assisted mode also biases scores toward
        # the route action at selection time; learned mode does not, so this
        # ranking loss is the network's only route signal there -- without it
        # the cheapest way to cut TD error is to memorize the training pool.
        # The hinge stops producing gradient once the route action leads by the
        # margin, so a network that already routes correctly trains on TD error
        # alone.
        route_loss = (
            _route_auxiliary_loss(
                q_values,
                route_labels_t,
                current_masks_t,
            )
            if self.route_aux_weight > 0.0
            else q_values.new_zeros(())
        )
        loss = td_loss + self.route_aux_weight * route_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.net.parameters(),
            self.clip,
        )
        self.opt.step()
        self.latest_learning_metrics = {
            "total_loss": float(loss.detach().item()),
            "td_loss": float(td_loss.detach().item()),
            "route_aux_loss": float(route_loss.detach().item()),
            "grad_norm": float(
                grad_norm.detach().item()
                if isinstance(grad_norm, torch.Tensor)
                else grad_norm
            ),
            "q_mean": float(q.detach().mean().item()),
            "q_abs_mean": float(q.detach().abs().mean().item()),
        }
        return dict(self.latest_learning_metrics)

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
                "policy_mode": self.policy_mode,
                "policy": _policy_contract(self.policy_mode),
                "action_safety": (
                    dict(ACTION_SAFETY_CONTRACT)
                    if self.policy_mode == ASSISTED_POLICY_MODE
                    else None
                ),
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
            "policy_mode": self.policy_mode,
            "policy": _policy_contract(self.policy_mode),
            "action_safety": (
                dict(ACTION_SAFETY_CONTRACT)
                if self.policy_mode == ASSISTED_POLICY_MODE
                else None
            ),
            "net": _cpu_copy(self.net.state_dict()),
            "target": _cpu_copy(self.target.state_dict()),
            "opt": _cpu_copy(self.opt.state_dict()),
        }

    def load_learning_state(self, checkpoint: dict[str, Any]) -> None:
        policy_mode = _checkpoint_policy_mode(checkpoint)
        action_safety = checkpoint.get("action_safety")
        if (
            checkpoint.get("schema") != OBSERVATION_SCHEMA
            or checkpoint.get("obs_dim") != self.net.obs_dim
            or checkpoint.get("n_actions") != self.net.n_actions
            or tuple(checkpoint.get("hidden", ())) != self.net.hidden
            or tuple(checkpoint.get("actions", ())) != ACTIONS
            or tuple(checkpoint.get("channels", ())) != CHANNEL_NAMES
            or tuple(checkpoint.get("globals", ())) != GLOBAL_NAMES
            or (
                policy_mode == ASSISTED_POLICY_MODE
                and action_safety != ACTION_SAFETY_CONTRACT
            )
            or (
                policy_mode == LEARNED_POLICY_MODE
                and action_safety is not None
            )
        ):
            raise ValueError(
                "checkpoint network, observation, or action contract does not match"
            )
        self.net.load_state_dict(checkpoint["net"])
        self.target.load_state_dict(checkpoint["target"])
        self.opt.load_state_dict(checkpoint["opt"])
        _optimizer_to(self.opt, self.device)
        self.policy_mode = policy_mode
        self.require_action_mask = action_safety is not None
        self.latest_learning_metrics = {}

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        policy_mode = _checkpoint_policy_mode(checkpoint)
        action_safety = checkpoint.get("action_safety")
        if (
            checkpoint.get("schema") != OBSERVATION_SCHEMA
            or checkpoint.get("obs_dim") != self.net.obs_dim
            or checkpoint.get("n_actions") != self.net.n_actions
            or tuple(checkpoint.get("hidden", ())) != self.net.hidden
            or tuple(checkpoint.get("actions", ())) != ACTIONS
            or tuple(checkpoint.get("channels", ())) != CHANNEL_NAMES
            or tuple(checkpoint.get("globals", ())) != GLOBAL_NAMES
            or action_safety not in (None, ACTION_SAFETY_CONTRACT)
            or (
                policy_mode == LEARNED_POLICY_MODE
                and action_safety is not None
            )
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
            _optimizer_to(self.opt, self.device)
        self.policy_mode = policy_mode
        self.require_action_mask = action_safety is not None
        self.latest_learning_metrics = {}

    def _tensor(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.to(dtype=torch.float32, device=self.device)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)


def select_actions(
    agents: Sequence[Agent],
    observations,
    eps: float,
    valid_masks=None,
) -> list[int]:
    if valid_masks is None and any(
        agent.require_action_mask for agent in agents
    ):
        raise ValueError(
            "these agents require environment action masks"
        )
    masks = _action_masks(observations, valid_masks)
    if len(agents) > 1 and all(agent is agents[0] for agent in agents[1:]):
        actions = [0] * len(agents)
        greedy: list[int] = []
        for index in range(len(agents)):
            if eps > 0.0 and random.random() < eps:
                actions[index] = random.choice(
                    _valid_actions(observations[index], masks[index])
                )
            else:
                greedy.append(index)
        if greedy:
            choices = agents[0].best_actions(
                [observations[index] for index in greedy],
                [masks[index] for index in greedy],
            )
            for index, action in zip(greedy, choices, strict=True):
                actions[index] = action
        return actions
    return [
        agent.act(observations[index], eps, masks[index])
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


Transition = tuple[Any, int, float, Any, bool, bool, int, Any, Any]
EVALUATION_BATCH_SIZE = 64


class Trainer:
    def __init__(self, env, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._validate(self.cfg)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        self.device = resolve_device(self.cfg.device)
        self.policy_mode = _normalize_policy_mode(self.cfg.policy_mode)
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
                policy_mode=self.policy_mode,
                route_aux_weight=self.cfg.route_aux_weight,
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
        self.replay_group_weights: dict[int, float] | None = None
        self.latest_learning_metrics: dict[str, float] = {}
        self._learning_metrics_window: deque[dict[str, float]] = deque(
            maxlen=100
        )

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
        _normalize_policy_mode(cfg.policy_mode)

    @property
    def rolling_learning_metrics(self) -> dict[str, float]:
        if not self._learning_metrics_window:
            return {}
        keys = self._learning_metrics_window[0]
        return {
            key: float(
                np.mean(
                    [
                        metrics[key]
                        for metrics in self._learning_metrics_window
                    ]
                )
            )
            for key in keys
        }

    def _configure_env(self, env) -> None:
        if getattr(env, "max_steps", self.cfg.max_steps) != self.cfg.max_steps:
            raise ValueError("trainer and environment max_steps must match")
        set_policy_mode = getattr(env, "set_policy_mode", None)
        if callable(set_policy_mode):
            set_policy_mode(self.policy_mode)
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

    def set_replay_group_weights(
        self,
        weights: Mapping[int, float] | None,
    ) -> None:
        normalized = _replay_group_weights(weights)
        self.replay_group_weights = normalized
        for learner in self.learners:
            learner.set_replay_group_weights(normalized)

    def set_learning_rate(self, value: float) -> None:
        learning_rate = float(value)
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning rate must be finite and positive")
        for learner in self.learners:
            for group in learner.opt.param_groups:
                group["lr"] = learning_rate

    def synchronize(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()

    def learner_state(self) -> list[dict[str, Any]]:
        return [learner.learning_state() for learner in self.learners]

    def load_learner_state(
        self,
        states: Sequence[dict[str, Any]],
        *,
        clear_learning_metrics: bool = True,
    ) -> None:
        if len(states) != len(self.learners):
            raise ValueError("recovery learner count does not match")
        for learner, state in zip(self.learners, states, strict=True):
            learner.load_learning_state(state)
        modes = {learner.policy_mode for learner in self.learners}
        if len(modes) != 1:
            raise ValueError("recovery learners use different policy modes")
        self.policy_mode = modes.pop()
        for agent in self.agents:
            if agent.policy_mode != self.policy_mode:
                raise ValueError("recovery agents use different policy modes")
        self.cfg.policy_mode = self.policy_mode
        self._configure_env(self.env)
        if clear_learning_metrics:
            self.latest_learning_metrics = {}
            self._learning_metrics_window.clear()

    def recovery_state(self) -> dict[str, Any]:
        if any(self._pending):
            raise RuntimeError("recovery checkpoints require an episode boundary")
        self.synchronize()
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
            "replay_group_weights": (
                dict(self.replay_group_weights)
                if self.replay_group_weights is not None
                else None
            ),
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
        self.set_replay_group_weights(state.get("replay_group_weights"))
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
        first_mask = pending[0][7]
        last_next_mask = pending[0][8]
        replay_group = pending[0][6]
        used = 0
        for used, transition in enumerate(list(pending)[:count], start=1):
            (
                _,
                _,
                step_reward,
                next_obs,
                step_terminal,
                step_important,
                step_group,
                _,
                next_mask,
                _,
            ) = transition
            if step_group != replay_group:
                raise RuntimeError("n-step transition crossed replay groups")
            reward += (self.cfg.gamma ** (used - 1)) * step_reward
            last_next_obs = next_obs
            last_next_mask = next_mask
            terminal = terminal or step_terminal
            important = important or step_important
            if step_terminal:
                break

        first_obs, first_action = pending[0][0], pending[0][1]
        # The auxiliary loss scores the head of the n-step window, so it takes
        # that step's label rather than the one the window ends on.
        first_route_label = pending[0][9]
        self.agents[agent_index].remember(
            first_obs,
            first_action,
            reward,
            last_next_obs,
            terminal,
            self.cfg.gamma ** used,
            important,
            replay_group,
            first_mask,
            last_next_mask,
            first_route_label,
        )
        pending.popleft()

    def _remember(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        current_masks,
        next_masks,
        terminal: bool,
        important: bool,
        replay_group: int = 0,
        route_labels: Sequence[int] | None = None,
    ) -> None:
        group = _replay_group(replay_group)
        labels = route_labels or (-1,) * N_AGENTS
        for index in range(N_AGENTS):
            self._pending[index].append(
                (
                    obs[index],
                    actions[index],
                    rewards[index],
                    next_obs[index],
                    terminal,
                    important,
                    group,
                    current_masks[index],
                    next_masks[index],
                    int(labels[index]),
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
        replay_group: int = 0,
        optimize: bool = True,
    ) -> EpisodeResult:
        group = _replay_group(replay_group)
        active_env = env or self.env
        self._configure_env(active_env)
        obs = active_env.reset(seed=seed) if seed is not None else active_env.reset()
        current_masks = _environment_action_masks_for_modes(
            active_env,
            obs,
            [agent.policy_mode for agent in self.agents],
        )
        total = 0.0
        final_info: dict[str, Any] = {}

        for _ in range(self.cfg.max_steps):
            eps = self.epsilon() if epsilon is None else epsilon
            actions = select_actions(
                self.agents,
                obs,
                eps,
                current_masks,
            )
            # Read before stepping: the label describes the state the agents
            # are acting from, which is the observation the loss scores.
            route_labels = _environment_route_labels(active_env)
            next_obs, rewards, done, cut, info = active_env.step(actions)
            terminal = done or cut
            next_masks = (
                _action_masks(next_obs)
                if terminal
                else _environment_action_masks_for_modes(
                    active_env,
                    next_obs,
                    [agent.policy_mode for agent in self.agents],
                )
            )

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
                    obs,
                    actions,
                    rewards,
                    next_obs,
                    current_masks,
                    next_masks,
                    terminal,
                    important,
                    group,
                    route_labels,
                )
                self.env_steps += 1
                ready = max(self.cfg.batch_size, self.cfg.replay_warmup)
                if (
                    optimize
                    and self.env_steps % self.cfg.train_every == 0
                    and len(self.learners[0].replay) >= ready
                ):
                    learning_metrics = [
                        learner.learn_batch(self.cfg.batch_size)
                        for learner in self.learners
                    ]
                    self.latest_learning_metrics = {
                        key: float(
                            np.mean(
                                [
                                    metrics[key]
                                    for metrics in learning_metrics
                                ]
                            )
                        )
                        for key in learning_metrics[0]
                    }
                    self._learning_metrics_window.append(
                        dict(self.latest_learning_metrics)
                    )
                    self.updates += 1
                    if self.updates % self.cfg.target_sync_updates == 0:
                        for learner in self.learners:
                            learner.sync()

            obs = next_obs
            current_masks = next_masks
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


def _clone_evaluation_env(env: CoopEnvBridge) -> CoopEnvBridge:
    clone = CoopEnvBridge(
        env.cfg,
        seed=env.sess.master_seed,
        max_steps=env.max_steps,
        micro=env.micro,
        shaping_gamma=env.shaping_gamma,
        record_metrics=env.record_metrics,
    )
    clone.micro_vary = env.micro_vary
    clone._micro_seed = env._micro_seed
    clone.set_room_cache_limit(env._room_cache_limit)
    clone.cache_rooms(env._room_cache.values())
    return clone


def _evaluation_envs(env, count: int) -> list[Any]:
    if count < 1:
        return []
    if type(env) is not CoopEnvBridge:
        return [env]
    if env.record_metrics:
        return [env]
    if env.micro is not None and env.micro_vary:
        return [env]

    lanes = [env]
    for _ in range(1, count):
        try:
            lanes.append(_clone_evaluation_env(env))
        except Exception:
            return [env]
    return lanes


def _batched_greedy_actions(
    agents: Sequence[Agent],
    observations: Sequence[Sequence[Any]],
    mask_batches: Sequence[Any] | None = None,
) -> list[list[int]]:
    if not observations:
        return []
    if mask_batches is None and any(
        agent.require_action_mask for agent in agents
    ):
        raise ValueError(
            "these agents require environment action masks"
        )
    if mask_batches is not None and len(mask_batches) != len(observations):
        raise ValueError("action mask batch count does not match observations")
    if len(agents) > 1 and all(agent is agents[0] for agent in agents[1:]):
        flat = [
            observation[agent_index]
            for observation in observations
            for agent_index in range(len(agents))
        ]
        flat_masks = (
            [
                masks[agent_index]
                for masks in mask_batches
                for agent_index in range(len(agents))
            ]
            if mask_batches is not None
            else None
        )
        actions = agents[0].best_actions(flat, flat_masks)
        width = len(agents)
        return [
            actions[index : index + width]
            for index in range(0, len(actions), width)
        ]

    by_agent = [
        agent.best_actions(
            [observation[index] for observation in observations],
            (
                [masks[index] for masks in mask_batches]
                if mask_batches is not None
                else None
            ),
        )
        for index, agent in enumerate(agents)
    ]
    return [
        [by_agent[agent_index][episode_index] for agent_index in range(len(agents))]
        for episode_index in range(len(observations))
    ]


def _evaluation_episode(
    seed: int,
    episode_return: float,
    metrics: dict[str, Any],
) -> EvaluationEpisode:
    return EvaluationEpisode(
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
        normal_ball_deaths=int(metrics.get("normal_ball_deaths", 0)),
        big_ball_deaths=int(metrics.get("big_ball_deaths", 0)),
        bridge_falls=int(metrics.get("bridge_falls", 0)),
        timed_doors_opened=int(metrics.get("timed_doors_opened", 0)),
        timed_doors_expired=int(metrics.get("timed_doors_expired", 0)),
        timed_doors_rearmed=int(metrics.get("timed_doors_rearmed", 0)),
        crate_switches=int(metrics.get("crate_switches_solved", 0)),
        reset_entries=int(metrics.get("reset_zones", 0)),
        wrong_key_interactions=int(metrics["wrong_key_interactions"]),
    )


def _evaluate_episodes(
    agents: Sequence[Agent],
    env,
    seeds: Sequence[int],
    batch_size: int,
) -> list[EvaluationEpisode]:
    if batch_size < 1:
        raise ValueError("evaluation batch size must be at least 1")
    if not seeds:
        return []

    policy_modes = {agent.policy_mode for agent in agents}
    if len(policy_modes) != 1:
        raise ValueError("evaluation agents use different policy modes")
    policy_mode = policy_modes.pop()
    lanes = _evaluation_envs(env, min(batch_size, len(seeds)))
    for lane in lanes:
        set_policy_mode = getattr(lane, "set_policy_mode", None)
        if callable(set_policy_mode):
            set_policy_mode(policy_mode)
    lane_count = len(lanes)
    episodes: list[EvaluationEpisode] = []
    for start in range(0, len(seeds), lane_count):
        group = seeds[start : start + lane_count]
        observations = [
            lanes[index].reset(seed=seed)
            for index, seed in enumerate(group)
        ]
        action_masks_by_lane = [
            _environment_action_masks_for_modes(
                lanes[index],
                observation,
                [agent.policy_mode for agent in agents],
            )
            for index, observation in enumerate(observations)
        ]
        returns = [0.0] * len(group)
        active = list(range(len(group)))

        for _ in range(env.max_steps):
            active_observations = [observations[index] for index in active]
            active_masks = [
                action_masks_by_lane[index] for index in active
            ]
            action_batches = _batched_greedy_actions(
                agents,
                active_observations,
                active_masks,
            )
            next_active: list[int] = []
            for active_index, actions in zip(
                active, action_batches, strict=True
            ):
                obs, rewards, done, cut, info = lanes[active_index].step(actions)
                observations[active_index] = obs
                returns[active_index] += (
                    sum(float(reward) for reward in rewards) / len(rewards)
                )
                if done or cut:
                    episodes.append(
                        _evaluation_episode(
                            int(group[active_index]),
                            returns[active_index],
                            info["episode"],
                        )
                    )
                else:
                    action_masks_by_lane[active_index] = (
                        _environment_action_masks_for_modes(
                            lanes[active_index],
                            obs,
                            [agent.policy_mode for agent in agents],
                        )
                    )
                    next_active.append(active_index)
            active = next_active
            if not active:
                break
        else:
            seed = group[active[0]]
            raise RuntimeError(f"evaluation seed {seed} did not terminate")

    order = {int(seed): index for index, seed in enumerate(seeds)}
    episodes.sort(key=lambda episode: order[episode.seed])
    return episodes


def evaluate_detailed(
    agents: Sequence[Agent],
    env,
    seeds: Sequence[int],
    *,
    batch_size: int = EVALUATION_BATCH_SIZE,
) -> tuple[Evaluation, tuple[EvaluationEpisode, ...]]:
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be unique")

    episodes = _evaluate_episodes(agents, env, seeds, batch_size)

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


def evaluate(
    agents: Sequence[Agent],
    env,
    seeds: Sequence[int],
    *,
    batch_size: int = EVALUATION_BATCH_SIZE,
) -> Evaluation:
    evaluation, _ = evaluate_detailed(
        agents, env, seeds, batch_size=batch_size
    )
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

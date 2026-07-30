"""Load a trained agent from a .pt checkpoint in one call, from any script.

    from DQN.load_model import configure_environment, load_agent

    agent = load_agent("agent_0.pt")       # net + target + optimizer restored
    configure_environment(agent, env)      # important for legacy agent-7 files
    observations = env.reset(seed=42)
    masks = env.wipeout_action_masks()
    action = agent.act(                     # greedy action for agent 0
        observations[0],
        eps=0.0,
        action_mask=masks[0],
    )

``load_agent`` returns a fully restored ``DQN_train.Agent``, so every method on
it works: ``act``, ``remember``, ``learn_batch``, ``sync``, ``save``. Pass
``train=True`` to resume training instead of running inference.

``load_policy`` is the lightweight alternative when all you need is
observations in, Q-values out -- it skips the replay buffer, the optimizer, and
the whole env_bridge/coop_env import chain.

Checkpoints written by ``Agent.save`` hold "net", "target", and "opt". A
net-only checkpoint or a bare state_dict also loads: the target is copied from
the policy net and the optimizer is left fresh.

From the shell, to check a checkpoint loads and runs:

    python DQN/load_model.py agent0.pt
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

# Importable no matter how the calling script was launched: DQN_train reaches
# for env_bridge (a sibling) and DQN.DQN_model (via the repo root), so both
# directories have to be on the path.
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE.parent), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from DQN.DQN_model import (
    ACTION_SAFETY_CONTRACT,
    ASSISTED_POLICY_CONTRACT,
    ASSISTED_POLICY_MODE,
    HIDDEN,
    LEARNED_POLICY_CONTRACT,
    LEARNED_POLICY_MODE,
    LEGACY_POLICY_CONTRACT,
    N_ACTIONS,
    OBS_DIM,
    QNetwork,
    action_scores,
)

if TYPE_CHECKING:
    from DQN.DQN_train import Agent

ArrayLike = torch.Tensor | np.ndarray | list | tuple


def configure_environment(policy: Any, env: Any) -> Any:
    """Match a bridge to a loaded learned or legacy-assisted policy.

    ``Trainer`` and the built-in evaluation helpers do this automatically.
    Call this helper for manual inference, especially with agent 7: a legacy
    assisted checkpoint needs the bridge to expose its historical route and
    safety fields.
    """
    mode = getattr(policy, "policy_mode", None)
    configure = getattr(env, "set_policy_mode", None)
    if mode not in (LEARNED_POLICY_MODE, ASSISTED_POLICY_MODE):
        raise ValueError("loaded policy has no recognized policy mode")
    if not callable(configure):
        raise TypeError("environment does not support policy-mode selection")
    configure(mode)
    return env


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Match DQN_train.resolve_device for "auto"; honour anything explicit."""
    if isinstance(requested, torch.device):
        return requested
    name = requested.strip().lower()
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def _net_state(checkpoint: Any) -> dict:
    """Pull the policy-net weights out of whatever shape the .pt file is."""
    if isinstance(checkpoint, dict):
        for key in ("net", "state_dict", "model"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _checkpoint_policy_mode(checkpoint: Any) -> str:
    """Return the action contract, treating historical files as assisted."""
    if not isinstance(checkpoint, dict):
        return ASSISTED_POLICY_MODE

    explicit = checkpoint.get("policy_mode")
    contract = checkpoint.get("policy")
    if explicit is not None:
        mode = str(explicit)
        expected = (
            LEARNED_POLICY_CONTRACT
            if mode == LEARNED_POLICY_MODE
            else (
                ASSISTED_POLICY_CONTRACT
                if mode == ASSISTED_POLICY_MODE
                else None
            )
        )
        if expected is None or contract != expected:
            raise ValueError("checkpoint action policy does not match")
        return mode

    if contract == LEARNED_POLICY_CONTRACT:
        return LEARNED_POLICY_MODE
    if contract in (
        None,
        ASSISTED_POLICY_CONTRACT,
        LEGACY_POLICY_CONTRACT,
    ):
        return ASSISTED_POLICY_MODE
    raise ValueError("checkpoint action policy does not match")


def _validate_action_safety(checkpoint: Any) -> None:
    if (
        isinstance(checkpoint, dict)
        and checkpoint.get("action_safety") is not None
        and checkpoint.get("action_safety") != ACTION_SAFETY_CONTRACT
    ):
        raise ValueError("checkpoint action safety does not match")

    if (
        isinstance(checkpoint, dict)
        and _checkpoint_policy_mode(checkpoint) == LEARNED_POLICY_MODE
        and checkpoint.get("action_safety") is not None
    ):
        raise ValueError(
            "learned checkpoints cannot require the assisted safety oracle"
        )


def load_agent(
    path: str | Path,
    device: str | torch.device = "auto",
    train: bool = False,
    **agent_kwargs,
) -> Agent:
    """Rebuild a complete Agent from a checkpoint.

    Extra keyword arguments (``lr``, ``obs_dim``, ``hidden``,
    ``replay_capacity``, ...) go straight to ``Agent.__init__``, for
    checkpoints trained with non-default shapes or hyperparameters.
    """
    from DQN.DQN_train import Agent

    resolved = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved)
    policy_mode = _checkpoint_policy_mode(checkpoint)
    _validate_action_safety(checkpoint)

    requested_mode = agent_kwargs.pop("policy_mode", policy_mode)
    if requested_mode != policy_mode:
        raise ValueError(
            f"checkpoint requires policy_mode={policy_mode!r}, "
            f"not {requested_mode!r}"
        )
    agent = Agent(
        device=resolved,
        policy_mode=policy_mode,
        **agent_kwargs,
    )
    agent.net.load_state_dict(_net_state(checkpoint))

    if isinstance(checkpoint, dict) and checkpoint.get("target"):
        agent.target.load_state_dict(checkpoint["target"])
    else:
        agent.sync()

    if isinstance(checkpoint, dict) and checkpoint.get("opt"):
        agent.opt.load_state_dict(checkpoint["opt"])
        # Restoring optimizer state also restores the checkpoint's lr, so an
        # explicitly requested one has to be put back afterwards.
        if "lr" in agent_kwargs:
            for group in agent.opt.param_groups:
                group["lr"] = agent_kwargs["lr"]

    agent.require_action_mask = (
        isinstance(checkpoint, dict)
        and checkpoint.get("action_safety") is not None
    )
    agent.policy_mode = policy_mode
    agent.net.train(train)
    agent.target.train(train)
    return agent


class Policy:
    """Callable inference wrapper: observations in, Q-values out as numpy."""

    def __init__(
        self,
        net: QNetwork,
        device: torch.device,
        policy_mode: str,
        action_safety: dict[str, Any] | None = None,
    ) -> None:
        self.net = net
        self.device = device
        self.policy_mode = policy_mode
        self.action_safety = action_safety

    @torch.no_grad()
    def __call__(self, obs: ArrayLike) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        single = t.dim() == 1
        out = self.net(t.unsqueeze(0) if single else t)
        return (out.squeeze(0) if single else out).cpu().numpy()

    def q_values(self, obs: ArrayLike) -> np.ndarray:
        return self(obs)

    def configure_environment(self, env: Any) -> Any:
        """Configure a bridge for this checkpoint's action contract."""
        return configure_environment(self, env)

    def act(
        self,
        obs: ArrayLike,
        action_mask: ArrayLike | None = None,
    ) -> int:
        if self.action_safety is not None and action_mask is None:
            raise ValueError(
                "this checkpoint requires an environment action mask"
            )
        t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            raw = self.net(t.unsqueeze(0) if t.dim() == 1 else t)
            scores = action_scores(
                raw,
                t.unsqueeze(0) if t.dim() == 1 else t,
                self.policy_mode,
            )
        scores = scores.squeeze(0)
        if action_mask is not None:
            mask = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            )
            if mask.shape != scores.shape:
                raise ValueError(
                    f"expected action mask shape {tuple(scores.shape)}, "
                    f"got {tuple(mask.shape)}"
                )
            safe_scores = scores.masked_fill(~mask, -torch.inf)
            if torch.isfinite(safe_scores).any():
                scores = safe_scores
        return int(scores.argmax().item())


def load_policy(
    path: str | Path,
    obs_dim: int = OBS_DIM,
    n_actions: int = N_ACTIONS,
    hidden=HIDDEN,
    device: str | torch.device = "auto",
) -> Policy:
    """Load just the policy network, for greedy inference."""
    resolved = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved)
    policy_mode = _checkpoint_policy_mode(checkpoint)
    _validate_action_safety(checkpoint)

    net = QNetwork(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden)
    net.load_state_dict(_net_state(checkpoint))
    net.to(resolved).eval()
    action_safety = (
        dict(checkpoint["action_safety"])
        if isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("action_safety"), dict)
        else None
    )
    return Policy(net, resolved, policy_mode, action_safety)


def _main(argv: list[str]) -> int:
    import argparse

    description = (__doc__ or "Load a trained DQN checkpoint.").splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("checkpoint", help="path to a .pt file")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    policy = load_policy(args.checkpoint, device=args.device)
    obs = np.zeros(policy.net.obs_dim, dtype=np.float32)
    action_mask = (
        np.ones(policy.net.n_actions, dtype=np.bool_)
        if policy.action_safety is not None
        else None
    )
    print(f"loaded {args.checkpoint} on {policy.device}")
    print(
        f"  obs_dim={policy.net.obs_dim} "
        f"n_actions={policy.net.n_actions} "
        f"policy_mode={policy.policy_mode}"
    )
    print(
        f"  q(zeros)={np.round(policy.q_values(obs), 4)} "
        f"-> act={policy.act(obs, action_mask)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

"""Load a trained agent from a .pt checkpoint in one call, from any script.

    from DQN.load_model import load_agent

    agent = load_agent("agent0.pt")     # net + target + optimizer restored
    action = agent.act(obs, eps=0.0)    # greedy action, int

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
    HIDDEN,
    LEGACY_POLICY_CONTRACT,
    N_ACTIONS,
    OBS_DIM,
    POLICY_CONTRACT,
    QNetwork,
    policy_scores,
)

if TYPE_CHECKING:
    from DQN.DQN_train import Agent

ArrayLike = torch.Tensor | np.ndarray | list | tuple


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


def _validate_policy(checkpoint: Any) -> None:
    if (
        isinstance(checkpoint, dict)
        and checkpoint.get("policy") is not None
        and checkpoint.get("policy")
        not in (POLICY_CONTRACT, LEGACY_POLICY_CONTRACT)
    ):
        raise ValueError("checkpoint action policy does not match")


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
    _validate_policy(checkpoint)

    agent = Agent(device=resolved, **agent_kwargs)
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

    agent.net.train(train)
    agent.target.train(train)
    return agent


class Policy:
    """Callable inference wrapper: observations in, Q-values out as numpy."""

    def __init__(self, net: QNetwork, device: torch.device) -> None:
        self.net = net
        self.device = device

    @torch.no_grad()
    def __call__(self, obs: ArrayLike) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        single = t.dim() == 1
        out = self.net(t.unsqueeze(0) if single else t)
        return (out.squeeze(0) if single else out).cpu().numpy()

    def q_values(self, obs: ArrayLike) -> np.ndarray:
        return self(obs)

    def act(self, obs: ArrayLike) -> int:
        t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            raw = self.net(t.unsqueeze(0) if t.dim() == 1 else t)
            scores = policy_scores(
                raw,
                t.unsqueeze(0) if t.dim() == 1 else t,
            )
        return int(scores.squeeze(0).argmax().item())


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
    _validate_policy(checkpoint)

    net = QNetwork(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden)
    net.load_state_dict(_net_state(checkpoint))
    net.to(resolved).eval()
    return Policy(net, resolved)


def _main(argv: list[str]) -> int:
    import argparse

    description = (__doc__ or "Load a trained DQN checkpoint.").splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("checkpoint", help="path to a .pt file")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    policy = load_policy(args.checkpoint, device=args.device)
    obs = np.zeros(policy.net.obs_dim, dtype=np.float32)
    print(f"loaded {args.checkpoint} on {policy.device}")
    print(f"  obs_dim={policy.net.obs_dim} n_actions={policy.net.n_actions}")
    print(f"  q(zeros)={np.round(policy.q_values(obs), 4)} -> act={policy.act(obs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

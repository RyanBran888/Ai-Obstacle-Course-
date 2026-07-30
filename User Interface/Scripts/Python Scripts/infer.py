"""
Runs the trained agents on a Godot-exported level.json and writes a
trajectory JSON that Godot can read back and animate.

This reproduces the checkpoint's inference-time policy wrapper, not just
raw argmax(Q) — confirmed from the checkpoint's `policy` / `action_safety`
metadata:
  - `route_q_bias`: nudge Q toward the BFS-suggested direction
  - `mask_invalid_interact`: never pick INTERACT unless something is
    actually there to interact with
  - `action_safety`: forbid an action if it leads to an unavoidable
    wipeout-ball death within the lookahead horizon, whenever a safe
    alternative exists

Usage:
    python infer.py --level level.json --out result.json
                     [--agent0 agent0.pt] [--agent1 agent1.pt]
                     [--max-steps 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import torch

from networks import QNetwork, OBS_DIM, N_ACTIONS
from env_lite import Level, RoomSim, N_AGENTS, INTERACT, WAIT


def load_agent(path: str):
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "net" in ckpt:
        state_dict = ckpt["net"]
        policy_cfg = ckpt.get("policy", {})
        safety_cfg = ckpt.get("action_safety", {})
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        policy_cfg, safety_cfg = {}, {}
    else:
        state_dict = ckpt
        policy_cfg, safety_cfg = {}, {}

    net = QNetwork(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    net.load_state_dict(state_dict)
    net.eval()
    return net, policy_cfg, safety_cfg


def choose_action(sim: RoomSim, agent_index: int, net: QNetwork, obs,
                   policy_cfg: dict, safety_cfg: dict) -> int:
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    q = net.q_values(obs_tensor).clone()

    horizon = safety_cfg.get("wipeout_action_mask_horizon", 10)
    safe = sim.safe_actions(agent_index, horizon=horizon)
    for a in range(N_ACTIONS):
        if not safe.get(a, True):
            q[a] = float("-inf")

    if policy_cfg.get("mask_invalid_interact", False):
        if not sim.can_interact(agent_index):
            q[INTERACT] = float("-inf")

    if torch.isneginf(q).all():
        # everything got masked out somehow — fall back to raw Q so the
        # agent still does *something* instead of crashing on argmax
        q = net.q_values(obs_tensor).clone()

    route_bias = policy_cfg.get("route_q_bias", 0.0)
    if route_bias:
        route_idx = sim.route_action(agent_index)
        if route_idx is not None and q[route_idx] != float("-inf"):
            q[route_idx] += route_bias

    return int(q.argmax().item())


def run(level_path: str, agent_paths: list[str], max_steps: int) -> dict:
    level = Level.from_json(level_path)
    sim = RoomSim(level, max_steps=max_steps)

    nets, policies, safeties = [], [], []
    for p in agent_paths:
        net, policy_cfg, safety_cfg = load_agent(p)
        nets.append(net)
        policies.append(policy_cfg)
        safeties.append(safety_cfg)

    obs_list = sim.reset()
    trajectory = [{
        "step": 0,
        "positions": [list(p) for p in sim.pos],
        "actions": None,
        "events": [],
        **sim.snapshot(),
    }]

    while not sim.done:
        actions = [
            choose_action(sim, i, nets[i], obs_list[i], policies[i], safeties[i])
            for i in range(N_AGENTS)
        ]

        obs_list, done, info = sim.step(actions)
        trajectory.append({
            "step": sim.steps,
            "positions": [list(p) for p in sim.pos],
            "actions": actions,
            "events": info["events"],
            **sim.snapshot(),
        })

    return {
        "success": bool(sim.win),
        "steps": sim.steps,
        "failure_reason": sim.fail_reason,
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--agent0", default="agent0.pt")
    parser.add_argument("--agent1", default="agent1.pt")
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    try:
        result = run(args.level, [args.agent0, args.agent1], args.max_steps)
    except Exception as exc:  # noqa: BLE001 - report any failure to Godot
        result = {"error": str(exc), "traceback": traceback.format_exc()}
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"infer.py failed: {exc}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result written to {args.out} "
          f"(success={result['success']}, steps={result['steps']})")


if __name__ == "__main__":
    main()

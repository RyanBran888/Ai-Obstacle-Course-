"""Train a minimal open-room navigation task.

Runs in a few seconds:

    python3 QN/mvp_2x2.py
"""

from __future__ import annotations

import random
import statistics

import torch

import DQN.DQN_train as T
from env_bridge import CoopEnvBridge, micro_room
from DQN.DQN_rewards import plot_rewards

SIZE = 6
EPISODES = 600
MAX_STEPS = 30


def draw(room, positions) -> str:
    """Room as text with the two agents at their live positions."""
    from coop_env.tiles import glyph
    from coop_env.utils.geometry import Vec2

    rows = [
        [glyph(room.terrain[Vec2(x, y)]) for x in range(room.width)]
        for y in range(room.height)
    ]
    ex = room.exit.pos
    rows[ex[1]][ex[0]] = "E"
    for i, p in enumerate(positions):
        rows[p[1]][p[0]] = "A" if i == 0 else "B"
    return "\n".join("".join(r) for r in rows)


def main() -> None:
    print(f"=== {SIZE}x{SIZE} MVP ===\n")
    room = micro_room(SIZE)
    print("room (A/B = agents, E = exit, already open):")
    print(draw(room, [s.pos for s in room.spawns]))
    walkable = sum(1 for p in room.terrain.positions() if room.terrain_at(p).name == "FLOOR")
    print(f"\n{walkable} walkable tiles, exit open from step 0, nothing to unlock")

    random.seed(0)
    torch.manual_seed(0)
    env = CoopEnvBridge(micro=SIZE, max_steps=MAX_STEPS)
    cfg = T.Config(
        episodes=EPISODES,
        eps_decay_steps=int(EPISODES * MAX_STEPS * 0.35),
        max_steps=MAX_STEPS,
        replay_warmup=300,
    )

    solves: list[bool] = []
    lengths: list[int] = []
    original = env.step

    def step(actions):
        out = original(actions)
        if out[2] or out[3]:
            metrics = out[4]["episode"]
            solves.append(bool(metrics["completed"]))
            lengths.append(metrics["steps"])
        return out

    env.step = step

    print(f"\ntraining {EPISODES} episodes...")
    agents, history = T.train(env, cfg)

    n = 100
    print(f"\n{'':16}{'first ' + str(n):>12}{'last ' + str(n):>12}")
    print(f"{'solve rate':16}{sum(solves[:n]) / n:>11.0%}{sum(solves[-n:]) / n:>12.0%}")
    print(f"{'steps to solve':16}{statistics.mean(lengths[:n]):>12.1f}"
          f"{statistics.mean(lengths[-n:]):>12.1f}")
    print(f"{'return':16}{sum(history[:n]) / n:>+12.2f}{sum(history[-n:]) / n:>+12.2f}")

    # Step count is more useful than solve rate in a tiny room.
    before, after = statistics.mean(lengths[:n]), statistics.mean(lengths[-n:])
    print(f"\n-> {before / max(after, 1e-9):.1f}x fewer steps to reach the exit")

    print("\ntrained policy, acting greedily:")
    obs = env.reset()
    print(draw(env.room, env.pos))
    for t in range(MAX_STEPS):
        actions = [a.act(obs[i], 0.0) for i, a in enumerate(agents)]
        obs, _, done, cut, _ = env.step(actions)
        print(f"\nstep {t + 1}:")
        print(draw(env.room, env.pos))
        if done:
            print(f"\nsolved in {t + 1} step(s)")
            break
        if cut:
            print(f"\nout of time after {MAX_STEPS} steps")
            break

    path = plot_rewards(history, cfg, path="mvp_2x2_rewards.png", show=False)
    print(f"\nreward curve written to {path}")


if __name__ == "__main__":
    main()

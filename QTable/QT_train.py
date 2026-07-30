"""Training loop for the two Q-table bots.

Shaped to match `DQN/DQN_train.py`: same `Config` dataclass idea, same `eps_at`,
same `build_agents` / `train(env, cfg)` signature, same `(agents, history)`
return. Swapping one import should be enough to run the same experiment with the
other learner.

The loop itself is shorter than the DQN's, and the missing pieces are the point:
no replay buffer to fill, no target network to sync, no batch schedule, no
gradient step. A table updates in place, immediately, once per transition.

One structural difference worth knowing about: actions for step t+1 are chosen
*before* the update for step t, because SARSA needs to know what the agent
actually does next. The DQN, being off-policy, does not care.

For curriculum staging see `QT_curriculum.py`, which reuses the DQN's own
`default_stages()` so both families train on identical procedural stages.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import QT_paths  # noqa: F401  -- puts the repo root and Architecture on sys.path

from coop_env import GenerationConfig  # noqa: E402

from QT_encoder import StateEncoder  # noqa: E402
from QT_env import TabularBridge  # noqa: E402
from QT_model import TabularAgent, build_agents, table_stats  # noqa: E402

N_AGENTS = 2


@dataclass(slots=True)
class Config:
    episodes: int = 4_000
    max_steps: int = 200

    # -- learning ----------------------------------------------------------
    alpha: float = 0.25
    alpha_min: float = 0.02
    alpha_decay: float = 0.9995
    """Per episode. Must be < 1.0 when lambda_ > 0; see TabularAgent."""
    gamma: float = 0.99
    rule: str = "expected_sarsa"
    lambda_: float = 0.8
    init: float = 0.0
    """Above the best achievable return gives optimistic exploration."""
    shared_table: bool = False

    # -- exploration -------------------------------------------------------
    eps_start: float = 1.0
    eps_min: float = 0.10
    eps_decay: int = 2_000
    """eps_min sits above the DQN's usual floor on purpose -- see `evaluate`."""

    # -- bookkeeping -------------------------------------------------------
    seed: int = 0
    log_every: int = 250


def eps_at(episode: int, cfg: Config) -> float:
    """Linear decay, same shape as `DQN_train.eps_at`."""
    if episode >= cfg.eps_decay:
        return cfg.eps_min
    t = episode / max(1, cfg.eps_decay)
    return cfg.eps_start + t * (cfg.eps_min - cfg.eps_start)


def make_env(
    cfg: Config,
    config: GenerationConfig | None = None,
    preset: str = "easy",
) -> TabularBridge:
    """A `TabularBridge` over the given generation config.

    Rewards, dynamics and rooms all come from `CoopEnvBridge`; nothing here
    changes them. `shaping_gamma` is left at the bridge default so the built-in
    potential shaping matches the DQN's exactly.
    """
    return TabularBridge(
        config or GenerationConfig.preset(preset),
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        encoder=StateEncoder(),
    )


def train(
    env: TabularBridge,
    cfg: Config | None = None,
    agents: list[TabularAgent] | None = None,
    rooms: Sequence[Any] | None = None,
    room_source: Any = None,
    verbose: bool = True,
) -> tuple[list[TabularAgent], list[float]]:
    """Run `cfg.episodes` episodes. Returns (agents, history).

    `agents` carries tables forward from an earlier stage instead of starting
    cold -- the main advantage tabular has in a curriculum, since entries for
    "follow the route, avoid the hazard" transfer straight into harder rooms.

    `room_source` is a zero-argument callable returning a freshly generated
    room -- one new room per episode. This is the default for curriculum stages:
    see `QT_curriculum.RoomSource`, which rejection-samples the generator until
    it produces a room matching the stage's lesson.

    `rooms` is an alternative fixed pool that episodes cycle through. Both being
    None falls back to `env.reset()`, which also generates a fresh room but
    without any stage filter applied.
    """
    cfg = cfg or Config()
    rng = random.Random(cfg.seed)

    if agents is None:
        agents = build_agents(
            N_AGENTS,
            shared_table=cfg.shared_table,
            n_actions=env.n_actions,
            alpha=cfg.alpha,
            alpha_min=cfg.alpha_min,
            alpha_decay=cfg.alpha_decay,
            gamma=cfg.gamma,
            rule=cfg.rule,
            init=cfg.init,
            lambda_=cfg.lambda_,
            rng=rng,
        )

    history: list[float] = []
    solves: list[int] = []
    started = time.time()

    for episode in range(cfg.episodes):
        if room_source is not None:
            obs = env.load_room(room_source())
        elif rooms:
            obs = env.load_room(rooms[episode % len(rooms)])
        else:
            obs = env.reset()

        eps = eps_at(episode, cfg)
        total = 0.0
        done = cut = False

        for agent in agents:
            agent.start_episode()

        actions = [a.act(obs[i], eps) for i, a in enumerate(agents)]

        for _ in range(cfg.max_steps):
            next_obs, rewards, done, cut, _ = env.step(actions)

            # Chosen before the update: SARSA learns from what happens next, not
            # from what would have been best.
            next_actions = [a.act(next_obs[i], eps) for i, a in enumerate(agents)]

            for i, agent in enumerate(agents):
                agent.remember(
                    obs[i],
                    actions[i],
                    rewards[i],
                    next_obs[i],
                    done,
                    next_action=next_actions[i],
                    eps=eps,
                )

            total += rewards[0]
            obs, actions = next_obs, next_actions

            if done or cut:
                break

        for agent in agents:
            agent.decay_alpha()

        history.append(total)
        solves.append(1 if done else 0)

        if verbose and cfg.log_every and (episode + 1) % cfg.log_every == 0:
            window = slice(-cfg.log_every, None)
            mean = sum(history[window]) / cfg.log_every
            rate = sum(solves[window]) / cfg.log_every
            states = sum(len(a.table) for a in agents)
            print(
                f"  ep {episode + 1:6d}  return {mean:8.3f}  solved {rate:5.1%}  "
                f"eps {eps:.3f}  alpha {agents[0].alpha:.3f}  states {states:7d}"
            )

    if verbose:
        print(f"  {cfg.episodes} episodes in {time.time() - started:.1f}s  "
              f"{table_stats(agents)}")

    return agents, history


def evaluate(
    env: TabularBridge,
    agents: list[TabularAgent],
    episodes: int = 200,
    eps: float = 0.05,
    rooms: Sequence[Any] | None = None,
    room_source: Any = None,
    verbose: bool = True,
) -> dict:
    """Rollouts with learning switched off.

    `eps` defaults to 0.05 rather than 0, which looks like cheating and is not.
    The state is aliased -- distinct situations share a key -- so a deterministic
    greedy policy can walk into a two-state cycle it has no way to perceive and
    burn the rest of the episode there. A trace of noise breaks the cycle.

    Pass `eps=0.0` for the greedy number; the gap between the two is a direct
    readout of how much aliasing the encoder carries.
    """
    wins = 0
    returns: list[float] = []
    lengths: list[int] = []

    for episode in range(episodes):
        if room_source is not None:
            obs = env.load_room(room_source())
        elif rooms:
            obs = env.load_room(rooms[episode % len(rooms)])
        else:
            obs = env.reset()

        total = 0.0
        done = cut = False
        steps = 0
        while not (done or cut) and steps < env.max_steps:
            actions = [a.act(obs[i], eps) for i, a in enumerate(agents)]
            obs, rewards, done, cut, _ = env.step(actions)
            total += rewards[0]
            steps += 1

        wins += done
        returns.append(total)
        lengths.append(steps)

    result = {
        "episodes": episodes,
        "solved": wins,
        "solve_rate": wins / episodes,
        "mean_return": sum(returns) / episodes,
        "mean_length": sum(lengths) / episodes,
    }
    if verbose:
        print(
            f"  eval(eps={eps:.2f}): solved {wins}/{episodes} "
            f"({result['solve_rate']:.1%})  return {result['mean_return']:.2f}  "
            f"steps {result['mean_length']:.1f}"
        )
    return result


def plot(history: list[float], path: str = "qtable_rewards.png", window: int = 50) -> str:
    """Reward curve. Skips quietly if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plot")
        return ""

    smooth, run, total = [], [], 0.0
    for v in history:
        run.append(v)
        total += v
        if len(run) > window:
            total -= run.pop(0)
        smooth.append(total / len(run))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(history)), history, lw=0.6, alpha=0.30, color="#4c8dff",
            label="episode return")
    ax.plot(range(len(smooth)), smooth, lw=2.0, color="#12356b",
            label=f"{window}-episode mean")
    ax.axhline(0.0, lw=0.8, color="#999", ls=":")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("Q-table training reward")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    cfg = Config(episodes=2_000, eps_decay=1_000, log_every=500)
    env = make_env(cfg, preset="easy")
    print("single-stage training on the 'easy' preset")
    agents, history = train(env, cfg)
    evaluate(env, agents, 200, eps=0.05)
    evaluate(env, agents, 200, eps=0.0)
    for i, agent in enumerate(agents):
        agent.save(f"qtable_agent{i}.pkl")
    plot(history, "qtable_easy.png")
    print("\nfor staged training use:  python QT_curriculum.py")

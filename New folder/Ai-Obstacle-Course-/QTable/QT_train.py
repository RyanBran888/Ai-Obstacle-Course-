"""Training loop for the two Q-table bots.

Shaped to match `QN/QN_train.py` on purpose: same `Config` dataclass, same
`eps_at`, same `build_agents` / `train(env, cfg)` signature, same
`(agents, history)` return. Swapping one import should be enough to run the
same experiment with the other learner.

What is different, and why:

    no replay buffer     A table has no interference between states, so
                         replaying old transitions buys nothing. It exists in
                         DQN to stop new gradients from wrecking old fits.

    no target network    Same reason. There is no shared function to
                         destabilise, so bootstrapping off the live table is
                         fine.

    alpha, not Adam      One scalar step size per entry. Decayed on a schedule
                         because convergence wants sum(alpha) infinite and
                         sum(alpha^2) finite, which a constant rate misses.

    eligibility traces   Tabular's answer to the credit-assignment problem the
                         replay buffer solves for DQN.

    reward shaping       Off by default. Turn it on for the procedural setting,
                         where the exit bonus alone is too sparse to find.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from QT_encoder import FeatureEncoder, PositionEncoder
from QT_env import GenerationConfig, N_ACTIONS, TabularCoopEnv
from QT_model import TabularAgent, build_agents, table_stats

N_AGENTS = 2


@dataclass(slots=True)
class Config:
    episodes: int = 5_000
    max_steps: int = 200

    # -- learning ----------------------------------------------------------
    alpha: float = 0.25
    alpha_min: float = 0.02
    alpha_decay: float = 0.9998
    """Applied once per episode, so 0.9998 halves alpha over ~3500 episodes."""
    gamma: float = 0.99
    rule: str = "expected_sarsa"
    lambda_: float = 0.8
    init: float = 0.0
    """Set this above the best achievable return for optimistic exploration."""
    shared_table: bool = False

    # -- exploration -------------------------------------------------------
    eps_start: float = 1.0
    eps_min: float = 0.10
    eps_decay: int = 2_000
    """eps_min stays higher than the DQN's 0.05 on purpose -- see `evaluate`."""

    # -- shaping -----------------------------------------------------------
    shaping: float = 1.0
    """Weight on potential-based shaping. Set to 0 for a run strictly
    comparable to the DQN, which trains without it. On procedurally generated
    rooms the exit bonus alone is too sparse to find, so this is on by
    default."""

    # -- bookkeeping -------------------------------------------------------
    seed: int = 0
    log_every: int = 250
    room_seeds: list[int] = field(default_factory=list)
    """Fixed room pool. Empty means a fresh room every episode."""


def eps_at(episode: int, cfg: Config) -> float:
    """Linear decay, identical in shape to QN_train.eps_at."""
    if episode >= cfg.eps_decay:
        return cfg.eps_min
    t = episode / max(1, cfg.eps_decay)
    return cfg.eps_start + t * (cfg.eps_min - cfg.eps_start)


def make_env(cfg: Config, preset: str = "easy", encoder=None) -> TabularCoopEnv:
    return TabularCoopEnv(
        GenerationConfig.preset(preset),
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        encoder=encoder or FeatureEncoder(),
        room_seeds=cfg.room_seeds or None,
    )


def train(env: TabularCoopEnv, cfg: Config | None = None, verbose: bool = True):
    """Run `cfg.episodes` episodes. Returns (agents, history).

    Both bots act, both learn from their own reward stream, and the loop is
    otherwise the same as the DQN one. The one structural difference is that
    actions for step t+1 are chosen *before* the update for step t, because
    SARSA needs to know what the agent actually does next.
    """
    cfg = cfg or Config()
    rng = random.Random(cfg.seed)

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
        obs = env.reset()
        eps = eps_at(episode, cfg)
        total = 0.0
        done = cut = False

        for agent in agents:
            agent.start_episode()

        phi = [_potential(env, i, cfg) for i in range(N_AGENTS)]
        actions = [a.act(obs[i], eps) for i, a in enumerate(agents)]

        for _ in range(cfg.max_steps):
            next_obs, rewards, done, cut, _ = env.step(actions)
            next_phi = [_potential(env, i, cfg) for i in range(N_AGENTS)]

            # Choose next actions first -- SARSA learns from what happens, not
            # from what would have been best.
            next_actions = [a.act(next_obs[i], eps) for i, a in enumerate(agents)]

            for i, agent in enumerate(agents):
                shaped = rewards[i]
                if cfg.shaping:
                    shaped += cfg.gamma * next_phi[i] - phi[i]
                agent.learn(
                    obs[i],
                    actions[i],
                    shaped,
                    next_obs[i],
                    done,
                    next_action=next_actions[i],
                    eps=eps,
                )

            total += rewards[0]
            obs, actions, phi = next_obs, next_actions, next_phi

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
                f"ep {episode + 1:6d}  return {mean:8.3f}  solved {rate:5.1%}  "
                f"eps {eps:.3f}  alpha {agents[0].alpha:.3f}  states {states:7d}"
            )

    if verbose:
        elapsed = time.time() - started
        print(f"\ntrained {cfg.episodes} episodes in {elapsed:.1f}s")
        print(f"tables: {table_stats(agents)}")

    return agents, history


def _potential(env: TabularCoopEnv, agent: int, cfg: Config) -> float:
    if not cfg.shaping:
        return 0.0
    return cfg.shaping * env.encoder.potential(env, agent)


def evaluate(
    env: TabularCoopEnv,
    agents: list[TabularAgent],
    episodes: int = 200,
    eps: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Rollouts with learning switched off.

    `eps` defaults to 0.05 rather than 0, which looks like cheating and is not.
    Under `FeatureEncoder` the state is aliased: distinct situations share a
    key, so a deterministic greedy policy can walk into a two-state cycle it has
    no way to perceive, let alone escape, and burn the rest of the episode in
    it. Measured on the tutorial preset, the same pair of tables scored 26%
    greedy and 60% at eps=0.10 -- the tables are fine, the determinism is the
    problem.

    Pass `eps=0.0` to see the greedy number; the gap between the two is a direct
    readout of how much aliasing the encoder is carrying. `PositionEncoder` has
    none, and scores the same either way.
    """
    wins = 0
    returns: list[float] = []
    lengths: list[int] = []

    for _ in range(episodes):
        obs = env.reset()
        total = 0.0
        done = cut = False
        steps = 0
        while not (done or cut):
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
            f"eval(eps={eps:.2f}): solved {wins}/{episodes} "
            f"({result['solve_rate']:.1%})  return {result['mean_return']:.2f}  "
            f"length {result['mean_length']:.1f}"
        )
    return result


def plot(history: list[float], path: str = "qtable_rewards.png", window: int = 50) -> str:
    """Reward curve. Falls back quietly if matplotlib is not installed."""
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


def run_fixed_room(episodes: int = 3_000) -> None:
    """One room, exact encoder. The setting where a table provably wins."""
    print("=" * 68)
    print("fixed room  |  PositionEncoder  |  expected SARSA(0.8)")
    print("=" * 68)
    cfg = Config(
        episodes=episodes,
        eps_decay=episodes // 2,
        eps_min=0.05,
        room_seeds=[7],
        shaping=0.0,
        alpha=0.3,
    )
    env = make_env(cfg, "easy", PositionEncoder())
    agents, history = train(env, cfg)
    evaluate(env, agents, 200, eps=0.0)
    for i, agent in enumerate(agents):
        agent.save(f"qtable_fixed_agent{i}.pkl")
    plot(history, "qtable_fixed_room.png")


def run_procedural(episodes: int = 4_000, preset: str = "tutorial") -> None:
    """A brand new room every episode. The setting where a table struggles."""
    print("=" * 68)
    print(f"procedural ({preset})  |  FeatureEncoder  |  expected SARSA(0.8)")
    print("=" * 68)
    cfg = Config(episodes=episodes, eps_decay=episodes // 2, shaping=1.0)
    env = make_env(cfg, preset, FeatureEncoder())
    agents, history = train(env, cfg)
    for eps in (0.0, 0.05, 0.10):
        evaluate(env, agents, 200, eps=eps)
    for i, agent in enumerate(agents):
        agent.save(f"qtable_{preset}_agent{i}.pkl")
    plot(history, f"qtable_{preset}.png")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    if mode == "fixed":
        run_fixed_room()
    elif mode == "procedural":
        preset = sys.argv[2] if len(sys.argv) > 2 else "tutorial"
        run_procedural(preset=preset)
    else:
        print("usage: python QT_train.py [fixed | procedural [preset]]")

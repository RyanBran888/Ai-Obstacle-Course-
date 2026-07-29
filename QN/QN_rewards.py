from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt


def running_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    recent: deque[float] = deque(maxlen=window)
    total = 0.0
    for v in values:
        if len(recent) == window:
            total -= recent[0]
        recent.append(v)
        total += v
        out.append(total / len(recent))
    return out


def plot_rewards(
    history: list[float],
    cfg=None,
    path: str = "training_rewards.png",
    window: int = 50,
    show: bool = True,
) -> str:
    episodes = range(len(history))
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(episodes, history, lw=0.6, alpha=0.30, color="#4c8dff", label="episode return")
    ax.plot(
        episodes,
        running_mean(history, window),
        lw=2.0,
        color="#12356b",
        label=f"{window}-episode mean",
    )
    ax.axhline(0.0, lw=0.8, color="#999", ls=":")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("Training reward")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if cfg is not None:
        from QN_train import eps_at

        eps_ax = ax.twinx()
        eps_ax.plot(
            episodes,
            [eps_at(e, cfg) for e in episodes],
            lw=1.2,
            ls="--",
            color="#c0392b",
            label="epsilon",
        )
        eps_ax.set_ylabel("epsilon")
        eps_ax.set_ylim(0, 1.05)
        h2, l2 = eps_ax.get_legend_handles_labels()
        handles += h2
        labels += l2

    ax.legend(handles, labels, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return path

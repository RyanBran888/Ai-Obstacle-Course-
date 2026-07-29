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


class LivePlot:
    def __init__(self, cfg=None, window: int = 50, every: int = 10) -> None:
        self.cfg = cfg
        self.window = window
        self.every = every
        self.on = plt.get_backend().lower() != "agg"
        if not self.on:
            return

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        (self.raw,) = self.ax.plot(
            [],
            [],
            lw=0.6,
            marker=".",
            markersize=4,
            alpha=0.40,
            color="#4c8dff",
            label="episode return",
        )
        (self.mean,) = self.ax.plot([], [], lw=2.0, color="#12356b",
                                    label=f"{window}-episode mean")
        self.ax.axhline(0.0, lw=0.8, color="#999", ls=":")
        self.ax.set_xlabel("episode")
        self.ax.set_ylabel("return")
        self.ax.set_title("Training reward (live)")
        self.ax.grid(alpha=0.25)
        self.ax.legend(loc="lower right", framealpha=0.9)
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def update(self, history: list[float]) -> None:
        if not self.on:
            return
        if history and len(history) % self.every == 0:
            x = range(len(history))
            self.raw.set_data(x, history)
            self.mean.set_data(x, running_mean(history, self.window))
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw_idle()
        self.pump()

    def pump(self) -> None:
        if self.on:
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def close(self) -> None:
        if self.on:
            plt.ioff()
            plt.close(self.fig)


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

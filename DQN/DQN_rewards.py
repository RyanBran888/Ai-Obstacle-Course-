from __future__ import annotations

import sys
from collections import deque


def _pyplot(interactive: bool):
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        if interactive and sys.platform == "darwin":
            try:
                matplotlib.use("TkAgg", force=True)
            except ImportError:
                pass
        elif not interactive:
            matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


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
        self.plt = _pyplot(interactive=True)
        self.cfg = cfg
        self.window = window
        self.every = every
        self.on = self.plt.get_backend().lower() != "agg"
        if not self.on:
            return

        self.plt.ion()
        self.fig, self.ax = self.plt.subplots(figsize=(9, 5))
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
        self.plt.show(block=False)
        self.plt.pause(0.1)

    @property
    def backend(self) -> str:
        return str(self.plt.get_backend())

    def set_title(self, title: str) -> None:
        if self.on:
            self.ax.set_title(title)
            self.fig.canvas.draw_idle()

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
            self.plt.pause(0.001)

    def close(self) -> None:
        if self.on:
            self.plt.ioff()
            self.plt.close(self.fig)

    def save(self, path: str) -> None:
        if self.on:
            self.fig.savefig(path, dpi=150)


class CurriculumPlot:
    def __init__(
        self,
        *,
        interactive: bool = True,
        every: int = 10,
        reward_window: int = 50,
        metric_window: int = 100,
    ) -> None:
        self.plt = _pyplot(interactive=interactive)
        self.every = every
        self.reward_window = reward_window
        self.metric_window = metric_window
        self.visible = interactive and self.plt.get_backend().lower() != "agg"

        self.fig, axes = self.plt.subplots(2, 2, figsize=(12, 8))
        self.reward_ax, self.success_ax = axes[0]
        self.steps_ax, self.eval_ax = axes[1]
        self.epsilon_ax = self.steps_ax.twinx()

        (self.reward_raw,) = self.reward_ax.plot(
            [], [], lw=0.5, alpha=0.25, color="#4c8dff", label="episode return"
        )
        (self.reward_mean,) = self.reward_ax.plot(
            [], [], lw=2.0, color="#12356b", label=f"{reward_window}-episode mean"
        )
        (self.success_mean,) = self.success_ax.plot(
            [], [], lw=2.0, color="#238636", label=f"{metric_window}-episode success"
        )
        (self.death_mean,) = self.success_ax.plot(
            [],
            [],
            lw=1.5,
            color="#d73a49",
            label=f"{metric_window}-episode wipeout rate",
        )
        (self.hazard_mean,) = self.success_ax.plot(
            [], [], lw=1.2, color="#e67e22",
            label=f"{metric_window}-episode hazard rate",
        )
        (self.bridge_fall_mean,) = self.success_ax.plot(
            [], [], lw=1.2, color="#8e44ad",
            label=f"{metric_window}-episode bridge-fall rate",
        )
        (self.crate_mean,) = self.success_ax.plot(
            [], [], lw=1.2, ls="--", color="#2980b9",
            label=f"{metric_window}-episode crate-solve rate",
        )
        (self.reset_mean,) = self.success_ax.plot(
            [], [], lw=1.2, ls="--", color="#7f8c8d",
            label=f"{metric_window}-episode reset rate",
        )
        (self.steps_mean,) = self.steps_ax.plot(
            [], [], lw=2.0, color="#8e44ad", label=f"{metric_window}-episode steps"
        )
        (self.epsilon_line,) = self.epsilon_ax.plot(
            [], [], lw=1.2, ls="--", color="#c0392b", label="epsilon"
        )
        (self.train_eval,) = self.eval_ax.plot(
            [], [], marker="o", color="#1f77b4", label="greedy train seeds"
        )
        (self.validation_eval,) = self.eval_ax.plot(
            [], [], marker="s", color="#ff7f0e", label="greedy validation seeds"
        )
        (self.retention_eval,) = self.eval_ax.plot(
            [], [], marker="^", color="#2ca02c", label="prior-stage retention"
        )

        self.eval_episodes: list[int] = []
        self.train_rates: list[float] = []
        self.validation_rates: list[float] = []
        self.retention_rates: list[float] = []
        self._format_axes()

        if self.visible:
            self.plt.ion()
            self.plt.show(block=False)
            self.plt.pause(0.1)

    @property
    def backend(self) -> str:
        return str(self.plt.get_backend())

    def _format_axes(self) -> None:
        self.reward_ax.set_title("Reward")
        self.reward_ax.set_ylabel("return")
        self.success_ax.set_title("Training outcomes")
        self.success_ax.set_ylabel("rolling rate")
        self.success_ax.set_ylim(-0.02, 1.02)
        self.steps_ax.set_title("Episode length and exploration")
        self.steps_ax.set_ylabel("steps")
        self.epsilon_ax.set_ylabel("epsilon")
        self.epsilon_ax.set_ylim(0.0, 1.02)
        self.eval_ax.set_title("Greedy evaluation")
        self.eval_ax.set_ylabel("success rate")
        self.eval_ax.set_ylim(-0.02, 1.02)
        for axis in (
            self.reward_ax,
            self.success_ax,
            self.steps_ax,
            self.eval_ax,
        ):
            axis.set_xlabel("training episode")
            axis.grid(alpha=0.2)
        self.reward_ax.legend(loc="lower right")
        self.success_ax.legend(loc="lower right")
        lines = [self.steps_mean, self.epsilon_line]
        self.steps_ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
        self.eval_ax.legend(loc="lower right")
        self.fig.tight_layout()

    def set_context(self, stage: str, pool_size: int) -> None:
        self.fig.suptitle(f"Curriculum: {stage} — {pool_size} training seeds")
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))

    def set_status(self, status: str) -> None:
        self.fig.suptitle(status)
        self.fig.canvas.draw_idle()
        self.pump()

    def pump(self) -> None:
        if self.visible:
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    def mark_stage(self, episode: int, name: str) -> None:
        for axis in (
            self.reward_ax,
            self.success_ax,
            self.steps_ax,
            self.eval_ax,
        ):
            axis.axvline(episode, lw=0.8, ls=":", color="#777", alpha=0.6)
        self.reward_ax.annotate(
            name,
            (episode, 1.0),
            xycoords=("data", "axes fraction"),
            rotation=90,
            va="top",
            fontsize=8,
        )

    def add_evaluation(
        self,
        episode: int,
        training_rate: float,
        validation_rate: float | None,
        retention_rate: float | None = None,
    ) -> None:
        self.eval_episodes.append(episode)
        self.train_rates.append(training_rate)
        self.validation_rates.append(
            float("nan") if validation_rate is None else validation_rate
        )
        self.retention_rates.append(
            float("nan") if retention_rate is None else retention_rate
        )

    def update(
        self,
        returns: list[float],
        completed: list[float],
        deaths: list[float],
        steps: list[float],
        epsilons: list[float],
        *,
        hazards: list[float] | None = None,
        bridge_falls: list[float] | None = None,
        crate_switches: list[float] | None = None,
        resets: list[float] | None = None,
        force: bool = False,
    ) -> None:
        if not returns or (not force and len(returns) % self.every):
            return
        episodes = list(range(len(returns)))
        self.reward_raw.set_data(episodes, returns)
        self.reward_mean.set_data(episodes, running_mean(returns, self.reward_window))
        self.success_mean.set_data(
            episodes, running_mean(completed, self.metric_window)
        )
        self.death_mean.set_data(
            episodes, running_mean(deaths, self.metric_window)
        )
        zeros = [0.0] * len(returns)
        self.hazard_mean.set_data(
            episodes, running_mean(hazards or zeros, self.metric_window)
        )
        self.bridge_fall_mean.set_data(
            episodes, running_mean(bridge_falls or zeros, self.metric_window)
        )
        self.crate_mean.set_data(
            episodes, running_mean(crate_switches or zeros, self.metric_window)
        )
        self.reset_mean.set_data(
            episodes, running_mean(resets or zeros, self.metric_window)
        )
        self.steps_mean.set_data(episodes, running_mean(steps, self.metric_window))
        self.epsilon_line.set_data(episodes, epsilons)
        self.train_eval.set_data(self.eval_episodes, self.train_rates)
        self.validation_eval.set_data(self.eval_episodes, self.validation_rates)
        self.retention_eval.set_data(self.eval_episodes, self.retention_rates)

        for axis in (self.reward_ax, self.steps_ax, self.eval_ax):
            axis.relim()
            axis.autoscale_view()
        self.success_ax.set_xlim(0, max(1, len(returns) - 1))
        self.steps_ax.set_xlim(0, max(1, len(returns) - 1))
        self.epsilon_ax.set_xlim(0, max(1, len(returns) - 1))
        self.fig.canvas.draw_idle()
        self.pump()

    def save(self, path: str) -> None:
        self.fig.savefig(path, dpi=150)

    def close(self) -> None:
        if self.visible:
            self.plt.ioff()
        self.plt.close(self.fig)


def plot_rewards(
    history: list[float],
    cfg=None,
    path: str = "training_rewards.png",
    window: int = 50,
    show: bool = True,
) -> str:
    plt = _pyplot(interactive=show)
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

    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return path

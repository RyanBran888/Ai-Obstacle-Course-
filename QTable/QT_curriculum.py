"""Curriculum training for the Q-table bots, on the DQN's own stages.

`DQN/curriculum.py` defines `default_stages()`: an ordered sequence of
`CurriculumStage` objects, each carrying a `GenerationConfig` and an `accepts`
predicate. Those are imported here rather than redefined, so both agent families
face the same lessons in the same order. If you add a stage for the DQN, the
table gets it too, for free.

**Generation stays procedural, one fresh room per episode.** A stage is not a
level and not a pool of levels. `stage.config` is a generation config and
`stage.accepts` is a filter, so `RoomSource` draws brand new random rooms and
discards the ones that do not teach the stage's lesson. Every episode gets a
room no agent has ever seen, and validation draws from a disjoint seed stream,
so a stage cannot be passed by memorising anything.

**Tables carry forward between stages.** This is the one place tabular has a
structural advantage over the network. Entries like "the route says east, east
is free, go east" or "the tile ahead is lava, do not" are stage-independent, so
they transfer intact into harder rooms and the later stages start warm. The DQN
has to protect the same knowledge with replay and a target net, which is what
`curriculum.py`'s regression detection and rollback machinery is for.

What this runner deliberately does *not* copy from `CurriculumRunner`: knowledge
regression detection, retention scoring, checkpoint rollback, lesson resolution,
pool expansion schedules. That is 2200 lines of protection against catastrophic
forgetting in a function approximator. A table cannot catastrophically forget --
writing entry X never touches entry Y -- so almost none of it applies. Stages
that fail here fail for a different reason, and mimicking the machinery would
imply otherwise.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Any, Sequence

import QT_paths  # noqa: F401  -- puts the repo root and Architecture on sys.path

from coop_env import EnvironmentSession  # noqa: E402

from DQN.curriculum import CurriculumStage, default_stages  # noqa: E402

from QT_encoder import StateEncoder  # noqa: E402
from QT_env import TabularBridge  # noqa: E402
from QT_model import TabularAgent, build_agents, table_stats  # noqa: E402
from QT_train import Config, evaluate, train  # noqa: E402

N_AGENTS = 2


@dataclass(slots=True)
class StageOutcome:
    name: str
    rounds: int
    train_rate: float
    validation_rate: float
    promoted: bool
    states: int
    seconds: float


class RoomSource:
    """Fresh procedurally generated rooms matching one stage's lesson.

    Calling the instance returns a new room every time -- there is no pool and
    nothing is reused, so an agent never sees the same layout twice.

    The rejection sampling is what turns a config into a *lesson*:
    `stage.config` gets the rough shape right, and `accepts` insists the room
    genuinely contains the mechanic being taught. Acceptance rates vary wildly
    between stages -- `switch_door` accepts almost everything, `full_course_mix`
    around one room in eighty -- so `accept_rate` is tracked and reported. A
    stage that suddenly slows to a crawl is a stage whose filter and config have
    drifted apart.
    """

    __slots__ = ("stage", "session", "max_draws", "draws", "accepted")

    def __init__(self, stage: CurriculumStage, seed: int, max_draws: int = 4_000) -> None:
        self.stage = stage
        self.session = EnvironmentSession(stage.config, master_seed=seed)
        self.max_draws = max_draws
        self.draws = 0
        self.accepted = 0

    def __call__(self) -> Any:
        for _ in range(self.max_draws):
            self.draws += 1
            self.session.reset()
            if self.stage.accepts(self.session.room):
                self.accepted += 1
                return self.session.room
        raise RuntimeError(
            f"stage {self.stage.name!r} rejected {self.max_draws} consecutive "
            "generated rooms -- its config and its accepts() predicate disagree"
        )

    @property
    def accept_rate(self) -> float:
        return self.accepted / max(1, self.draws)


def run_stage(
    stage: CurriculumStage,
    agents: list[TabularAgent] | None,
    cfg: Config,
    eval_episodes: int = 60,
    max_rounds: int = 3,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[list[TabularAgent], StageOutcome]:
    """Train on one stage until its thresholds are met or rounds run out.

    Two independent room sources on disjoint seed streams: one for training, one
    for validation. Since every episode draws a fresh room from the generator,
    both figures are measured on unseen rooms -- the difference is that the
    validation stream shares no seeds with training at all. Promotion needs
    `stage.train_threshold` and `stage.validation_threshold` together, the same
    two-gate rule `CurriculumRunner` applies to the DQN.
    """
    started = time.time()
    env = TabularBridge(
        stage.config,
        seed=seed,
        max_steps=cfg.max_steps,
        encoder=StateEncoder(),
    )

    train_rooms = RoomSource(stage, seed=seed)
    valid_rooms = RoomSource(stage, seed=seed + 500_000)

    if agents is None:
        agents = build_agents(
            N_AGENTS,
            shared_table=cfg.shared_table,
            alpha=cfg.alpha,
            alpha_min=cfg.alpha_min,
            alpha_decay=cfg.alpha_decay,
            gamma=cfg.gamma,
            rule=cfg.rule,
            init=cfg.init,
            lambda_=cfg.lambda_,
            rng=random.Random(seed),
        )

    train_rate = validation_rate = 0.0
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        if verbose:
            print(f"\n[{stage.name}] round {rounds}/{max_rounds}  "
                  f"fresh room per episode")

        train(env, cfg, agents=agents, room_source=train_rooms, verbose=verbose)

        train_rate = evaluate(
            env, agents, episodes=eval_episodes,
            eps=cfg.eps_min, room_source=train_rooms, verbose=False,
        )["solve_rate"]
        validation_rate = evaluate(
            env, agents, episodes=eval_episodes,
            eps=cfg.eps_min, room_source=valid_rooms, verbose=False,
        )["solve_rate"]

        if verbose:
            print(f"  train {train_rate:.1%} (need {stage.train_threshold:.0%})   "
                  f"validation {validation_rate:.1%} "
                  f"(need {stage.validation_threshold:.0%})   "
                  f"room accept rate {train_rooms.accept_rate:.1%}")

        if (train_rate >= stage.train_threshold
                and validation_rate >= stage.validation_threshold):
            break

    promoted = (train_rate >= stage.train_threshold
                and validation_rate >= stage.validation_threshold)

    outcome = StageOutcome(
        name=stage.name,
        rounds=rounds,
        train_rate=train_rate,
        validation_rate=validation_rate,
        promoted=promoted,
        states=sum(len(a.table) for a in agents),
        seconds=time.time() - started,
    )
    return agents, outcome


def run_curriculum(
    stages: Sequence[CurriculumStage] | None = None,
    cfg: Config | None = None,
    limit: int | None = None,
    eval_episodes: int = 60,
    max_rounds: int = 3,
    stop_on_fail: bool = False,
    seed: int = 0,
) -> tuple[list[TabularAgent], list[StageOutcome]]:
    """Walk the stages in order, carrying tables forward.

    `stop_on_fail=False` by default: a stage that misses its thresholds is
    reported and the run continues. A table that cannot clear an early stage
    will usually still learn something useful from a later one, and stopping
    early hides that. Set it True if you want strict gating.
    """
    stages = list(stages or default_stages())
    if limit:
        stages = stages[:limit]
    cfg = cfg or Config()

    agents: list[TabularAgent] | None = None
    outcomes: list[StageOutcome] = []

    print(f"curriculum: {len(stages)} stages, {cfg.episodes} episodes per round")
    print("=" * 78)

    for index, stage in enumerate(stages, 1):
        print(f"\n### stage {index}/{len(stages)}: {stage.name}")
        if stage.lesson:
            print(f"    lesson: {stage.lesson}")

        try:
            agents, outcome = run_stage(
                stage, agents, cfg,
                eval_episodes=eval_episodes, max_rounds=max_rounds, seed=seed,
            )
        except RuntimeError as exc:
            print(f"    SKIPPED: {exc}")
            continue

        outcomes.append(outcome)
        flag = "PROMOTED" if outcome.promoted else "not promoted"
        print(f"    {flag}: train {outcome.train_rate:.1%}, "
              f"validation {outcome.validation_rate:.1%}, "
              f"{outcome.states} states, {outcome.seconds:.0f}s")

        if stop_on_fail and not outcome.promoted:
            print("\nstopping: stage not promoted and stop_on_fail is set")
            break

    print("\n" + "=" * 78)
    print(f"{'stage':34s} {'rounds':>6s} {'train':>7s} {'valid':>7s} {'states':>7s}  ok")
    print("-" * 78)
    for outcome in outcomes:
        print(
            f"{outcome.name[:34]:34s} {outcome.rounds:6d} "
            f"{outcome.train_rate:6.1%} {outcome.validation_rate:6.1%} "
            f"{outcome.states:7d}  {'y' if outcome.promoted else 'n'}"
        )

    if agents:
        print(f"\nfinal tables: {table_stats(agents)}")
        for i, agent in enumerate(agents):
            agent.save(f"qtable_curriculum_agent{i}.pkl")
        print("saved qtable_curriculum_agent0.pkl, qtable_curriculum_agent1.pkl")

    return agents or [], outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum-train the Q-table bots.")
    parser.add_argument("--stages", type=int, default=None,
                        help="only run the first N stages")
    parser.add_argument("--episodes", type=int, default=1_500,
                        help="episodes per round")
    parser.add_argument("--eval", type=int, default=60,
                        help="episodes per threshold measurement")
    parser.add_argument("--rounds", type=int, default=3,
                        help="max rounds before moving on")
    parser.add_argument("--list", action="store_true",
                        help="print the stage list and exit")
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    stages = default_stages()

    if args.list:
        print(f"{len(stages)} stages:")
        for i, stage in enumerate(stages, 1):
            print(f"  {i:3d}. {stage.name:38s} "
                  f"train>={stage.train_threshold:.0%} "
                  f"valid>={stage.validation_threshold:.0%}")
        return

    cfg = Config(
        episodes=args.episodes,
        eps_decay=max(1, args.episodes // 2),
        log_every=max(1, args.episodes // 3),
    )
    run_curriculum(
        stages, cfg,
        limit=args.stages,
        eval_episodes=args.eval,
        max_rounds=args.rounds,
        stop_on_fail=args.stop_on_fail,
    )


if __name__ == "__main__":
    main()

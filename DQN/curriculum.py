from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from env_bridge import CoopEnvBridge, GenerationConfig
from DQN.DQN_train import (
    Config,
    Evaluation,
    EvaluationEpisode,
    Trainer,
    evaluate,
    evaluate_detailed,
)
from room_manifest import (
    CurriculumRoomManifest,
    build_manifest_suite,
    verify_manifest,
)

from coop_env import (
    AlwaysOpen,
    CheckpointRequirement,
    KeyRequirement,
    Room,
    RoomShape,
    SwitchRequirement,
)
from coop_env.rng import derive_seed


RoomCheck = Callable[[Room], bool]


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    config: GenerationConfig
    accepts: RoomCheck
    train_threshold: float = 0.90
    validation_threshold: float = 0.80


@dataclass(frozen=True, slots=True)
class StageResult:
    stage: str
    pool_size: int
    rounds: int
    training: Evaluation
    validation: Evaluation | None
    promoted: bool


@dataclass(frozen=True, slots=True)
class StageTestResult:
    stage: str
    evaluation: Evaluation
    episodes: tuple[EvaluationEpisode, ...]


def _base_config() -> GenerationConfig:
    return GenerationConfig(
        width=(10, 16),
        height=(9, 14),
        shape_weights={RoomShape.RECTANGLE: 1.0},
        region_count=(2, 3),
        min_region_span=4,
        branching_factor=0.15,
        corridor_width=(1, 1),
        obstacle_density=0.0,
        hazard_density=0.0,
        num_keys=(0, 0),
        num_locked_doors=(0, 0),
        num_switches=(0, 0),
        num_pushable_blocks=(0, 0),
        num_checkpoints=(0, 0),
        num_reset_zones=(0, 0),
        num_temporary_bridges=(0, 0),
        puzzle_chain_length=0,
        exit_objective_count=0,
        required_cooperative_actions=0,
        timed_door_probability=0.0,
        separate_spawns_probability=0.0,
    )


def default_stages() -> tuple[CurriculumStage, ...]:
    open_config = _base_config()
    key_config = _base_config().with_overrides(
        num_keys=(1, 1),
        num_locked_doors=(1, 1),
        puzzle_chain_length=1,
    )
    switch_config = _base_config().with_overrides(
        num_switches=(1, 1),
        num_locked_doors=(1, 1),
        puzzle_chain_length=1,
    )
    checkpoint_config = key_config.with_overrides(exit_objective_count=1)
    tutorial_config = GenerationConfig.preset(
        "tutorial",
        required_cooperative_actions=0,
        timed_door_probability=0.0,
        num_pushable_blocks=(0, 0),
        num_reset_zones=(0, 0),
        num_temporary_bridges=(0, 0),
    )

    def open_room(room: Room) -> bool:
        return (
            not room.keys
            and not room.doors
            and not room.switches
            and not room.checkpoints
            and isinstance(room.exit.requirement, AlwaysOpen)
        )

    def key_door(room: Room) -> bool:
        return (
            len(room.keys) == 1
            and bool(room.doors)
            and all(isinstance(door.requirement, KeyRequirement) for door in room.doors)
            and not room.switches
            and not room.checkpoints
            and isinstance(room.exit.requirement, AlwaysOpen)
        )

    def key_door_checkpoint(room: Room) -> bool:
        return (
            len(room.keys) == 1
            and bool(room.doors)
            and all(isinstance(door.requirement, KeyRequirement) for door in room.doors)
            and not room.switches
            and len(room.checkpoints) == 1
            and isinstance(room.exit.requirement, CheckpointRequirement)
        )

    def switch_door(room: Room) -> bool:
        return (
            not room.keys
            and bool(room.doors)
            and all(
                isinstance(door.requirement, SwitchRequirement)
                for door in room.doors
            )
            and len(room.switches) == 1
            and not room.checkpoints
            and isinstance(room.exit.requirement, AlwaysOpen)
        )

    return (
        CurriculumStage("open_navigation", open_config, open_room, 0.95, 0.90),
        CurriculumStage("key_door", key_config, key_door, 0.90, 0.80),
        CurriculumStage(
            "switch_door",
            switch_config,
            switch_door,
            0.90,
            0.80,
        ),
        CurriculumStage(
            "key_door_checkpoint",
            checkpoint_config,
            key_door_checkpoint,
            0.90,
            0.80,
        ),
        CurriculumStage(
            "tutorial_mix",
            tutorial_config,
            lambda room: True,
            0.85,
            0.75,
        ),
    )


class CurriculumRunner:
    def __init__(
        self,
        trainer: Trainer,
        *,
        stages: Sequence[CurriculumStage] | None = None,
        pool_sizes: Sequence[int] = (1, 4, 16, 64),
        validation_size: int = 64,
        test_size: int = 256,
        episodes_per_seed: int = 50,
        max_rounds: int = 8,
        run_seed: int = 0,
        data_seed: int = 0,
        live: bool = False,
        plot_every: int = 10,
        graph_path: str | None = "curriculum_training.png",
    ) -> None:
        if not pool_sizes or any(size < 1 for size in pool_sizes):
            raise ValueError("pool_sizes must contain positive values")
        if (
            validation_size < 1
            or test_size < 1
            or episodes_per_seed < 1
            or max_rounds < 1
        ):
            raise ValueError("curriculum sizes and rounds must be positive")
        if plot_every < 1:
            raise ValueError("plot_every must be positive")
        self.trainer = trainer
        self.stages = tuple(stages or default_stages())
        self.pool_sizes = tuple(sorted(set(pool_sizes)))
        self.validation_size = validation_size
        self.test_size = test_size
        self.episodes_per_seed = episodes_per_seed
        self.max_rounds = max_rounds
        self.run_seed = run_seed
        self.data_seed = data_seed
        self.live = live
        self.plot_every = plot_every
        self.graph_path = graph_path
        self.results: list[StageResult] = []
        self.test_results: list[StageTestResult] = []
        self.room_manifest: CurriculumRoomManifest | None = None
        self._test_started = False

    @property
    def completed(self) -> bool:
        expected = len(self.stages) * len(self.pool_sizes)
        return (
            len(self.results) == expected
            and all(result.promoted for result in self.results)
        )

    def prepare_room_manifest(self) -> CurriculumRoomManifest:
        if self.room_manifest is None:
            print(
                "Staging Architecture-generated train, validation, and test rooms...",
                flush=True,
            )
            self.room_manifest = build_manifest_suite(
                self.stages,
                data_seed=self.data_seed,
                train_size=self.pool_sizes[-1],
                validation_size=self.validation_size,
                test_size=self.test_size,
            )
            print(
                f"Room manifest ready: {self.room_manifest.sha256[:12]}",
                flush=True,
            )
        return self.room_manifest

    def run(self) -> list[StageResult]:
        from DQN.DQN_rewards import CurriculumPlot

        if self.results:
            raise RuntimeError("this curriculum runner has already trained")
        manifest = self.prepare_room_manifest()
        verify_manifest(manifest, self.stages, splits=("train", "validation"))
        plot = (
            CurriculumPlot(interactive=self.live, every=self.plot_every)
            if self.live or self.graph_path
            else None
        )
        plot_returns: list[float] = []
        plot_completed: list[float] = []
        plot_steps: list[float] = []
        plot_epsilons: list[float] = []
        if plot is not None and self.live:
            status = "opened" if plot.visible else "unavailable"
            print(f"Live dashboard {status} ({plot.backend})", flush=True)

        def finish_plot() -> None:
            if plot is None:
                return
            plot.update(
                plot_returns,
                plot_completed,
                plot_steps,
                plot_epsilons,
                force=True,
            )
            if self.graph_path:
                plot.save(self.graph_path)
                print(f"Dashboard saved to {self.graph_path}", flush=True)
            plot.close()

        largest_pool = self.pool_sizes[-1]
        rehearsal: list[tuple[CoopEnvBridge, tuple[int, ...]]] = []
        for stage_index, stage in enumerate(self.stages):
            if plot is not None:
                plot.mark_stage(len(plot_returns), stage.name)
            stage_rooms = manifest.stage(stage.name)
            stage_config = stage_rooms.config
            train_seeds = stage_rooms.seeds("train")
            validation_seeds = stage_rooms.seeds("validation")

            train_env = CoopEnvBridge(
                stage_config,
                seed=self.data_seed,
                max_steps=self.trainer.cfg.max_steps,
                shaping_gamma=self.trainer.cfg.gamma,
            )
            self.trainer.set_env(train_env, clear_replay=stage_index > 0)
            if stage_index > 0:
                self.trainer.reheat_exploration()

            for pool_size in self.pool_sizes:
                if plot is not None:
                    plot.set_context(stage.name, pool_size)
                pool = train_seeds[:pool_size]
                rng = random.Random(
                    derive_seed(self.run_seed, f"{stage.name}:pool:{pool_size}")
                )
                streak = 0
                round_index = 0
                training_eval: Evaluation | None = None
                validation_eval: Evaluation | None = None

                for round_index in range(1, self.max_rounds + 1):
                    episodes = max(50, self.episodes_per_seed * pool_size)
                    for _ in range(episodes):
                        if rehearsal and rng.random() < 0.20:
                            old_env, old_seeds = rng.choice(rehearsal)
                            outcome = self.trainer.run_episode(
                                seed=rng.choice(old_seeds),
                                env=old_env,
                            )
                        else:
                            outcome = self.trainer.run_episode(seed=rng.choice(pool))
                        plot_returns.append(outcome.reward)
                        plot_completed.append(float(outcome.completed))
                        plot_steps.append(float(outcome.steps))
                        plot_epsilons.append(self.trainer.epsilon())
                        if plot is not None:
                            plot.update(
                                plot_returns,
                                plot_completed,
                                plot_steps,
                                plot_epsilons,
                            )

                    training_eval = evaluate(
                        self.trainer.agents,
                        CoopEnvBridge(
                            stage_config,
                            seed=self.data_seed,
                            max_steps=self.trainer.cfg.max_steps,
                            shaping_gamma=self.trainer.cfg.gamma,
                        ),
                        pool,
                    )
                    validation_eval = None
                    passed = training_eval.success_rate >= stage.train_threshold
                    if pool_size == largest_pool:
                        validation_eval = evaluate(
                            self.trainer.agents,
                            CoopEnvBridge(
                                stage_config,
                                seed=self.data_seed,
                                max_steps=self.trainer.cfg.max_steps,
                                shaping_gamma=self.trainer.cfg.gamma,
                            ),
                            validation_seeds,
                        )
                        passed = (
                            passed
                            and validation_eval.success_rate
                            >= stage.validation_threshold
                        )
                    if plot is not None:
                        plot.add_evaluation(
                            len(plot_returns),
                            training_eval.success_rate,
                            (
                                validation_eval.success_rate
                                if validation_eval is not None
                                else None
                            ),
                        )
                        plot.update(
                            plot_returns,
                            plot_completed,
                            plot_steps,
                            plot_epsilons,
                            force=True,
                        )
                        if self.graph_path:
                            plot.save(self.graph_path)

                    streak = streak + 1 if passed else 0
                    self._print_round(
                        stage,
                        pool_size,
                        round_index,
                        training_eval,
                        validation_eval,
                    )
                    if streak >= 2:
                        break

                assert training_eval is not None
                promoted = streak >= 2
                result = StageResult(
                    stage=stage.name,
                    pool_size=pool_size,
                    rounds=round_index,
                    training=training_eval,
                    validation=validation_eval,
                    promoted=promoted,
                )
                self.results.append(result)
                if not promoted:
                    finish_plot()
                    return self.results
            rehearsal.append((train_env, train_seeds))

        finish_plot()
        return self.results

    def evaluate_final_test(self) -> list[StageTestResult]:
        if self._test_started:
            raise RuntimeError("the final test has already been started")
        if not self.completed:
            raise RuntimeError("the final test requires a completed curriculum")

        self._test_started = True
        manifest = self.prepare_room_manifest()
        verify_manifest(manifest, self.stages, splits=("test",))
        print("\nFinal greedy test on untouched rooms", flush=True)
        for stage in self.stages:
            stage_rooms = manifest.stage(stage.name)
            seeds = stage_rooms.seeds("test")
            evaluation, episodes = evaluate_detailed(
                self.trainer.agents,
                CoopEnvBridge(
                    stage_rooms.config,
                    seed=self.data_seed,
                    max_steps=self.trainer.cfg.max_steps,
                    shaping_gamma=self.trainer.cfg.gamma,
                ),
                seeds,
            )
            result = StageTestResult(stage.name, evaluation, episodes)
            self.test_results.append(result)
            self._print_test(result)
        return list(self.test_results)

    @staticmethod
    def _print_round(
        stage: CurriculumStage,
        pool_size: int,
        round_index: int,
        training: Evaluation,
        validation: Evaluation | None,
    ) -> None:
        message = (
            f"{stage.name:22} pool={pool_size:>2} round={round_index:>2} "
            f"train={training.success_rate:>6.1%}"
        )
        if validation is not None:
            message += f" validation={validation.success_rate:>6.1%}"
        message += (
            f" reward={training.mean_return:.2f} "
            f"keys={training.mean_keys:.2f} doors={training.mean_doors:.2f} "
            f"switches={training.mean_switches:.2f} "
            f"checkpoints={training.mean_checkpoints:.2f} "
            f"steps={training.mean_steps:.1f}"
        )
        print(message, flush=True)

    @staticmethod
    def _print_test(result: StageTestResult) -> None:
        evaluation = result.evaluation
        print(
            f"{result.stage:22} test={evaluation.success_rate:>6.1%} "
            f"success={evaluation.completed}/{evaluation.episodes} "
            f"reward={evaluation.mean_return:.2f} "
            f"timeouts={evaluation.timeouts} steps={evaluation.mean_steps:.1f}",
            flush=True,
        )


def make_runner(
    cfg: Config | None = None,
    *,
    stages: Sequence[CurriculumStage] | None = None,
    **kwargs,
) -> CurriculumRunner:
    training_config = cfg or Config()
    first_stage = tuple(stages or default_stages())[0]
    env = CoopEnvBridge(
        first_stage.config,
        seed=training_config.seed,
        max_steps=training_config.max_steps,
        shaping_gamma=training_config.gamma,
    )
    trainer = Trainer(env, training_config)
    return CurriculumRunner(trainer, stages=stages, **kwargs)

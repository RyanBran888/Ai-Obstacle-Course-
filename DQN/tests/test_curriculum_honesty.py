from __future__ import annotations

import sys
import unittest
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

from DQN.DQN_rewards import CurriculumPlot  # noqa: E402
from DQN.DQN_train import Config, Evaluation  # noqa: E402
from DQN.curriculum import (  # noqa: E402
    CurriculumRunner,
    _environment_policy_mode,
    _learning_metrics,
    _promotion_ready,
    _save_trained_best,
    _trained_round_is_best,
    _training_requirements_satisfied,
)
from DQN.run_curriculum import (  # noqa: E402
    _trainer_learning_report,
    parse_args,
)


class TestHonestCurriculumPromotion(unittest.TestCase):
    def make_runner(self, fresh_learned: bool) -> CurriculumRunner:
        runner = CurriculumRunner.__new__(CurriculumRunner)
        runner._fresh_learned_run = fresh_learned
        runner.minimum_fresh_training_rounds = 1
        runner.trainer = cast(Any, SimpleNamespace(updates=0))
        return runner

    def test_fresh_learned_policy_requires_a_real_first_round(self):
        runner = self.make_runner(True)
        self.assertEqual(
            runner._initial_training_round_requirement(0, 0),
            1,
        )
        self.assertEqual(
            runner._initial_optimizer_update_requirement(0, 0),
            1,
        )
        self.assertEqual(
            runner._initial_training_round_requirement(0, 1),
            0,
        )
        self.assertEqual(
            runner._initial_optimizer_update_requirement(0, 1),
            0,
        )
        self.assertEqual(
            runner._initial_training_round_requirement(1, 0),
            0,
        )

    def test_assisted_and_warm_started_policies_do_not_force_a_round(self):
        assisted = self.make_runner(False)
        self.assertEqual(
            assisted._initial_training_round_requirement(0, 0),
            0,
        )
        warm_started = self.make_runner(False)
        self.assertEqual(
            warm_started._initial_training_round_requirement(0, 0),
            0,
        )
        self.assertEqual(
            _environment_policy_mode(Config(policy_mode="learned")),
            "learned",
        )
        self.assertEqual(
            _environment_policy_mode(Config(policy_mode="assisted")),
            "assisted",
        )

    def test_promotion_gate_combines_pass_streak_and_training_rounds(self):
        active = {
            "streak": 1,
            "total_rounds": 0,
            "minimum_training_rounds": 1,
            "minimum_optimizer_updates": 1,
            "optimizer_updates_completed": 0,
            "best_meets_training_gate": False,
        }
        self.assertFalse(_promotion_ready(active, 1))
        active["total_rounds"] = 1
        self.assertFalse(_promotion_ready(active, 1))
        active["optimizer_updates_completed"] = 1
        self.assertTrue(_training_requirements_satisfied(active))
        self.assertFalse(_promotion_ready(active, 1))
        active["best_meets_training_gate"] = True
        self.assertTrue(_promotion_ready(active, 1))
        self.assertFalse(_promotion_ready(active, 2))

    def test_tied_or_worse_trained_state_replaces_ineligible_baseline(self):
        evaluation_values = {
            field.name: 0 for field in fields(Evaluation)
        }
        evaluation_values["episodes"] = 1
        evaluation = Evaluation(**evaluation_values)
        for trained_rank in ((10.0, 10.0), (9.0, 9.0)):
            with self.subTest(trained_rank=trained_rank):
                active = {
                    "best_rank": (10.0, 10.0),
                    "best_meets_training_gate": False,
                    "best_learner_state": [{"marker": "baseline"}],
                    "total_rounds": 1,
                    "minimum_training_rounds": 1,
                    "minimum_optimizer_updates": 1,
                    "optimizer_updates_completed": 1,
                }
                self.assertTrue(
                    _trained_round_is_best(active, trained_rank)
                )
                saved = _save_trained_best(
                    active,
                    rank=trained_rank,
                    training=evaluation,
                    validation=None,
                    retention=(),
                    failures=(),
                    retention_deficits={},
                    learner_state=lambda: [{"marker": "trained"}],
                )
                self.assertTrue(saved)
                self.assertEqual(
                    active["best_learner_state"],
                    [{"marker": "trained"}],
                )
                self.assertTrue(active["best_meets_training_gate"])
                self.assertTrue(_promotion_ready({**active, "streak": 1}, 1))

                self.assertFalse(
                    _trained_round_is_best(active, trained_rank)
                )

    def test_optimizer_and_best_gate_survive_active_progress_payload(self):
        evaluation_values = {
            field.name: 0 for field in fields(Evaluation)
        }
        evaluation_values["episodes"] = 1
        evaluation = Evaluation(**evaluation_values)
        active = {
            "stage_index": 0,
            "pool_index": 0,
            "scheduled_pool_size": 1,
            "active_pool_size": 1,
            "total_rounds": 1,
            "normal_rounds": 1,
            "recovery_rounds": 0,
            "phase": "normal",
            "phase_rounds": 1,
            "streak": 1,
            "expansions": [],
            "rng_state": ("test",),
            "best_rank": (1.0,),
            "best_round": 1,
            "best_training": evaluation,
            "best_validation": None,
            "best_retention": (),
            "best_failures": (),
            "best_retention_deficits": {},
            "latest_retention": (),
            "latest_failures": (),
            "latest_retention_deficits": {},
            "best_learner_state": [{"marker": "trained"}],
            "phase_limit": None,
            "needs_replay_refill": False,
            "needs_baseline_assessment": False,
            "minimum_training_rounds": 1,
            "minimum_optimizer_updates": 1,
            "optimizer_updates_completed": 7,
            "best_meets_training_gate": True,
            "consolidation_bad_rounds": 0,
        }

        payload = CurriculumRunner._active_payload(active)
        self.assertIsNotNone(payload)
        restored = CurriculumRunner._active_from_payload(cast(Any, payload))
        self.assertEqual(restored["minimum_optimizer_updates"], 1)
        self.assertEqual(restored["optimizer_updates_completed"], 7)
        self.assertTrue(restored["best_meets_training_gate"])


class TestLearningTelemetry(unittest.TestCase):
    def test_prefers_optional_rolling_trainer_metrics(self):
        class FakeTrainer:
            updates = 12
            rolling_learning_metrics = {
                "total_loss": 1.5,
                "td_loss": 1.25,
                "grad_norm": 0.75,
            }

        self.assertEqual(
            _learning_metrics(cast(Any, FakeTrainer())),
            {
                "total_loss": 1.5,
                "td_loss": 1.25,
                "grad_norm": 0.75,
            },
        )
        self.assertEqual(
            _trainer_learning_report(FakeTrainer())["optimizer_updates"],
            12,
        )

    def test_dashboard_renders_optimizer_and_loss_telemetry(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "training.png"
            plot = CurriculumPlot(interactive=False, every=1)
            plot.update(
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [20.0, 15.0, 10.0],
                [1.0, 0.5, 0.1],
                updates=[0.0, 4.0, 9.0],
                td_losses=[float("nan"), 1.2, 0.8],
                total_losses=[float("nan"), 1.3, 0.9],
                grad_norms=[float("nan"), 0.7, 0.5],
                force=True,
            )
            plot.save(str(output))
            plot.close()
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


class TestPolicyModeCLI(unittest.TestCase):
    def test_fresh_cli_defaults_to_learned(self):
        with patch.object(sys, "argv", ["run_curriculum.py"]):
            args = parse_args()
        self.assertEqual(args.policy_mode, "learned")
        self.assertEqual(args.minimum_fresh_training_rounds, 1)

    def test_assisted_mode_is_explicit(self):
        with patch.object(
            sys,
            "argv",
            ["run_curriculum.py", "--policy-mode", "assisted"],
        ):
            args = parse_args()
        self.assertEqual(args.policy_mode, "assisted")


if __name__ == "__main__":
    unittest.main()

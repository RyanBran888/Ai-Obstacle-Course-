from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

from DQN.curriculum import (  # noqa: E402
    LEARNED_NAV_GRADIENT_PROMOTION_SCALE,
    LEARNED_PROMOTION_SCALE,
    CurriculumRunner,
    _effective_nav_gradient,
    _resolve_promotion_scale,
    default_stages,
)


def _runner(scale: float):
    runner = CurriculumRunner.__new__(CurriculumRunner)
    runner.promotion_scale = scale
    runner.retention_margin = 0.15
    runner.stages = default_stages()
    return runner


def _stage(name: str):
    return next(s for s in default_stages() if s.name == name)


class TestScaleResolution(unittest.TestCase):
    def test_assisted_is_untouched(self):
        self.assertEqual(_resolve_promotion_scale(None, "assisted"), 1.0)

    def test_learned_is_scaled_down_by_default(self):
        self.assertEqual(
            _resolve_promotion_scale(None, "learned"), LEARNED_PROMOTION_SCALE
        )
        self.assertLess(LEARNED_PROMOTION_SCALE, 1.0)

    def test_the_flow_map_does_not_move_the_gate(self):
        # Its higher supervised ceiling argued for a stricter gate, but
        # training with it came out worse, so the gate tracks measured
        # episode success rather than the probe.
        scaled = _resolve_promotion_scale(None, "learned", nav_gradient=True)
        self.assertEqual(scaled, LEARNED_NAV_GRADIENT_PROMOTION_SCALE)
        self.assertEqual(scaled, LEARNED_PROMOTION_SCALE)

    def test_assisted_stays_at_full_gates_with_the_flow_map(self):
        self.assertEqual(
            _resolve_promotion_scale(None, "assisted", nav_gradient=True), 1.0
        )

    def test_an_explicit_value_wins_in_either_mode(self):
        for mode in ("learned", "assisted"):
            for gradient in (True, False):
                self.assertEqual(
                    _resolve_promotion_scale(0.75, mode, gradient), 0.75, mode
                )


class TestEffectiveNavGradient(unittest.TestCase):
    """The gate must be sized against what the observation actually carries."""

    def test_the_live_environment_outranks_the_request(self):
        trainer = type(
            "T",
            (),
            {
                "env": type("E", (), {"nav_gradient": True})(),
                "cfg": type("C", (), {"nav_gradient": None})(),
            },
        )()
        self.assertTrue(_effective_nav_gradient(trainer, "learned"))

    def test_the_config_answers_when_no_environment_exists(self):
        trainer = type("T", (), {"cfg": type("C", (), {"nav_gradient": False})()})()
        self.assertFalse(_effective_nav_gradient(trainer, "learned"))

    def test_it_defaults_off_in_either_mode(self):
        bare = type("T", (), {})()
        self.assertFalse(_effective_nav_gradient(bare, "learned"))
        self.assertFalse(_effective_nav_gradient(bare, "assisted"))

    def test_out_of_range_values_are_rejected(self):
        from typing import cast

        from DQN.DQN_train import Trainer

        class Stub:
            """Enough of a trainer to reach the scale validation.

            ``cfg`` is needed even though ``policy_mode`` is present, because
            the lookup passes it as an eagerly evaluated getattr default.
            """

            policy_mode = "learned"
            cfg = type("Cfg", (), {"policy_mode": "learned"})()

        for bad in (0.0, -0.5, 1.5):
            with self.assertRaisesRegex(ValueError, "promotion_scale"):
                CurriculumRunner(
                    trainer=cast(Trainer, Stub()), promotion_scale=bad
                )


class TestScaledGates(unittest.TestCase):
    def test_scale_one_reproduces_the_authored_gates(self):
        runner = _runner(1.0)
        for stage in default_stages():
            self.assertEqual(runner.train_threshold(stage), stage.train_threshold)
            self.assertEqual(
                runner.validation_threshold(stage), stage.validation_threshold
            )
            self.assertEqual(
                runner.max_wipeout_death_rate(stage),
                stage.max_wipeout_death_rate,
            )

    def test_success_gates_come_down(self):
        stage = _stage("open_navigation")
        runner = _runner(0.5)
        self.assertAlmostEqual(runner.train_threshold(stage), 0.475)
        self.assertAlmostEqual(runner.validation_threshold(stage), 0.45)

    def test_the_default_gate_clears_observed_learned_performance(self):
        # Learned runs produced 48.4-56.2% validation on open_navigation. A
        # gate above that floor is what left the run looping at stage one.
        runner = _runner(LEARNED_PROMOTION_SCALE)
        gate = runner.validation_threshold(_stage("open_navigation"))
        self.assertLess(gate, 0.484)

    def test_death_rate_ceilings_relax_rather_than_tighten(self):
        stage = _stage("owned_keys_both_wipeouts")
        self.assertLess(stage.max_wipeout_death_rate, 1.0)
        relaxed = _runner(0.5).max_wipeout_death_rate(stage)
        self.assertGreater(relaxed, stage.max_wipeout_death_rate)

    def test_a_death_rate_ceiling_never_exceeds_one(self):
        runner = _runner(0.1)
        for stage in default_stages():
            self.assertLessEqual(runner.max_wipeout_death_rate(stage), 1.0)

    def test_retention_floor_scales_with_the_gates(self):
        strict = _runner(1.0)._retention_threshold("open_navigation")
        relaxed = _runner(0.5)._retention_threshold("open_navigation")
        self.assertLess(relaxed, strict)
        # The hard 50% floor has to scale too, or retention alone re-blocks.
        worst = min(
            _runner(0.5)._retention_threshold(s.name) for s in default_stages()
        )
        self.assertLess(worst, 0.50)


if __name__ == "__main__":
    unittest.main()

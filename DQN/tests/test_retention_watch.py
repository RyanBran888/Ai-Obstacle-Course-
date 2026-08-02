from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

from DQN.curriculum import (  # noqa: E402
    RETENTION_WATCH_ROUNDS,
    CurriculumRunner,
    default_stages,
)


def _runner() -> CurriculumRunner:
    runner = CurriculumRunner.__new__(CurriculumRunner)
    runner.stages = default_stages()
    runner.promotion_scale = 0.8
    runner.retention_margin = 0.15
    return runner


def _active(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "phase": "normal",
        "phase_rounds": 0,
        "latest_retention": (),
        "best_retention": (),
        "latest_failures": (),
        "best_failures": (),
        "latest_retention_deficits": {},
        "best_retention_deficits": {},
    }
    base.update(overrides)
    return base


class TestGrindingIsWatched(unittest.TestCase):
    """Retention used to be checked only at the largest pool.

    A stage stuck on a small pool never reached that pool, so the decay it
    caused was invisible until far too late -- or forever.
    """

    def test_the_threshold_allows_one_retry_before_watching(self):
        self.assertEqual(RETENTION_WATCH_ROUNDS, 2)

    def test_a_fresh_pool_is_not_yet_grinding(self):
        self.assertLess(0, RETENTION_WATCH_ROUNDS)
        self.assertLess(1, RETENTION_WATCH_ROUNDS)

    def test_a_third_round_counts_as_grinding(self):
        self.assertGreaterEqual(2, RETENTION_WATCH_ROUNDS)


class TestSevereDecayOverridesTheGuard(unittest.TestCase):
    """A failing stage normally keeps its rounds instead of rehearsing.

    That is backwards when it is failing because the stages under it decayed:
    measured, four more rounds at 80/20 drove prior stages from 42.2% to 12.5%
    while the consolidation mix restored them to 96.9%.
    """

    def _plan(self, runner, active):
        rehearsal = [("switch_door", None, (0,), 2)]
        return runner._rehearsal_weights(  # type: ignore[attr-defined]
            current_group=1,
            rehearsal=rehearsal,
            active=active,
        )

    def test_severe_decay_triggers_consolidation_while_failing(self):
        runner = _runner()
        floor = runner._retention_threshold("switch_door")
        active = _active(
            latest_retention=(("switch_door", floor * 0.2),),
            latest_failures=("training success 6.2% < 67.2%",),
            latest_retention_deficits={"switch_door": 0.5},
        )
        weights, weak, consolidation, lr_scale = self._plan(runner, active)
        self.assertIn("switch_door", weak)
        self.assertTrue(consolidation, "severe decay must override the guard")
        # Full strength, not the gentler half-measure. The half-strength mix
        # was measured against this exact situation and failed to repair.
        self.assertEqual(lr_scale, 0.25)
        self.assertAlmostEqual(weights[1], 0.25, places=6)

    def test_a_failing_stage_without_severe_decay_keeps_the_gentle_mix(self):
        runner = _runner()
        active = _active(
            phase="recovery",
            latest_retention=(("switch_door", 0.99),),
            latest_failures=("training success 6.2% < 67.2%",),
            latest_retention_deficits={"switch_door": 0.02},
        )
        weights, _, consolidation, lr_scale = self._plan(runner, active)
        self.assertTrue(consolidation)
        self.assertEqual(lr_scale, 0.50)
        self.assertAlmostEqual(weights[1], 0.50, places=6)

    def test_a_mild_dip_still_lets_the_stage_keep_its_rounds(self):
        runner = _runner()
        floor = runner._retention_threshold("switch_door")
        active = _active(
            latest_retention=(("switch_door", floor * 0.95),),
            latest_failures=("training success 6.2% < 67.2%",),
            latest_retention_deficits={"switch_door": 0.01},
        )
        _, _, consolidation, lr_scale = self._plan(runner, active)
        self.assertFalse(consolidation)
        self.assertEqual(lr_scale, 1.0)

    def test_healthy_retention_never_consolidates(self):
        runner = _runner()
        active = _active(
            latest_retention=(("switch_door", 1.0),),
            latest_failures=("training success 6.2% < 67.2%",),
        )
        _, weak, consolidation, _ = self._plan(runner, active)
        self.assertEqual(weak, ())
        self.assertFalse(consolidation)


if __name__ == "__main__":
    unittest.main()

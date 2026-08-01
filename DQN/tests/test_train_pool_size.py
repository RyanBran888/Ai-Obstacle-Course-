from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

from DQN.curriculum import CurriculumRunner, default_stages  # noqa: E402


def _runner(train_pool_max: int, episodes_per_seed: int = 50):
    """A runner with only the fields the ladder maths touches.

    Building a real one would stage rooms and construct a network, which this
    is not about.
    """
    runner = CurriculumRunner.__new__(CurriculumRunner)
    runner.pool_sizes = (1, 4, 16, 64)
    runner.train_pool_max = train_pool_max
    runner.episodes_per_seed = episodes_per_seed
    runner.episodes_per_round_max = episodes_per_seed * 64
    return runner


class TestLadderReachesMoreRooms(unittest.TestCase):
    """The plateau this guards: stage ladders stopped at 16 or 64 rooms, the
    network memorized them, and held-out success stalled near 50% regardless
    of the training objective."""

    def test_a_sixteen_room_stage_climbs_past_sixteen(self):
        stage = next(
            s for s in default_stages() if s.pool_sizes == (1, 4, 16)
        )
        self.assertEqual(
            _runner(256).stage_pool_sizes(stage),
            (1, 4, 16, 32, 64, 128, 256),
        )

    def test_a_sixtyfour_room_stage_climbs_past_sixtyfour(self):
        stage = next(
            s
            for s in default_stages()
            if not s.pool_sizes or max(s.pool_sizes) == 64
        )
        self.assertEqual(
            _runner(256).stage_pool_sizes(stage),
            (1, 4, 16, 64, 128, 256),
        )

    def test_the_ladder_never_drops_a_stage_below_its_own_rungs(self):
        stage = next(
            s for s in default_stages() if s.pool_sizes == (1, 4, 16)
        )
        # A cap under the stage's own ladder must not silently shrink it.
        self.assertEqual(_runner(4).stage_pool_sizes(stage), (1, 4))
        self.assertEqual(_runner(16).stage_pool_sizes(stage), (1, 4, 16))

    def test_the_cap_is_honoured_exactly(self):
        stage = next(
            s for s in default_stages() if s.pool_sizes == (1, 4, 16)
        )
        for cap in (100, 256, 300):
            self.assertLessEqual(
                max(_runner(cap).stage_pool_sizes(stage)), cap, cap
            )

    def test_a_non_positive_cap_is_rejected(self):
        from typing import cast

        from DQN.DQN_train import Trainer

        with self.assertRaisesRegex(ValueError, "train_pool_max"):
            # The validation runs before the trainer is touched.
            CurriculumRunner(
                trainer=cast(Trainer, object()),
                train_pool_max=0,
            )


class TestWiderPoolsCostTheSame(unittest.TestCase):
    """A wider pool has to buy room variety, not proportionally more time."""

    def test_small_pools_keep_the_full_per_room_budget(self):
        runner = _runner(256)
        for pool in (1, 4, 16, 64):
            self.assertEqual(runner.episodes_for_pool(pool), 50 * pool)

    def test_large_pools_are_capped_at_the_old_largest_budget(self):
        runner = _runner(256)
        budget = runner.episodes_for_pool(64)
        for pool in (128, 256):
            self.assertEqual(runner.episodes_for_pool(pool), budget, pool)

    def test_a_wider_pool_means_fewer_episodes_per_room(self):
        runner = _runner(256)
        self.assertAlmostEqual(runner.episodes_for_pool(64) / 64, 50.0)
        self.assertAlmostEqual(runner.episodes_for_pool(256) / 256, 12.5)

    def test_a_tiny_pool_still_gets_a_usable_floor(self):
        self.assertGreaterEqual(_runner(256, episodes_per_seed=1)
                                .episodes_for_pool(1), 50)


class TestHistoricalCursorsAreUnaffected(unittest.TestCase):
    """Resume-upgrade validators replay a recorded run, so they must compare
    against the ladder that run used, not the currently configured one."""

    def test_recorded_ladder_ignores_the_cap(self):
        stage = next(
            s for s in default_stages() if s.pool_sizes == (1, 4, 16)
        )
        for cap in (16, 64, 256, 1024):
            self.assertEqual(
                _runner(cap)._recorded_pool_sizes(stage), (1, 4, 16), cap
            )


if __name__ == "__main__":
    unittest.main()

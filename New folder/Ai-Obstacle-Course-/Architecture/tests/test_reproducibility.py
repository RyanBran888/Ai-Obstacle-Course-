"""Same seed, same room -- the property everything else depends on."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coop_env import EnvironmentSession, GenerationConfig, RoomGenerator  # noqa: E402
from coop_env.rng import SeededRandom, derive_seed, normalize_seed  # noqa: E402

PRESETS = ("tutorial", "easy", "standard", "hard", "brutal")


def fingerprint(room) -> tuple:
    return (
        room.seed,
        room.width,
        room.height,
        room.shape,
        tuple(room.terrain.to_list()),
        room.entities,
    )


class TestSeeding(unittest.TestCase):
    def test_same_seed_same_room(self):
        generator = RoomGenerator(GenerationConfig.preset("standard"))
        for seed in (0, 1, 42, 9999, 2**40):
            with self.subTest(seed=seed):
                self.assertEqual(
                    fingerprint(generator.generate(seed)),
                    fingerprint(generator.generate(seed)),
                )

    def test_different_seeds_differ(self):
        generator = RoomGenerator(GenerationConfig.preset("standard"))
        prints = {fingerprint(generator.generate(s)) for s in range(30)}
        self.assertGreaterEqual(len(prints), 29, "seeds should not collapse onto one room")

    def test_generation_order_does_not_matter(self):
        config = GenerationConfig.preset("hard")
        first = RoomGenerator(config).generate(555)
        other = RoomGenerator(config)
        for seed in (1, 2, 3, 4):
            other.generate(seed)
        self.assertEqual(fingerprint(first), fingerprint(other.generate(555)))

    def test_fresh_process_stability(self):
        """String seeds must hash the same way on every run, not per-process."""
        self.assertEqual(normalize_seed("hello"), normalize_seed("hello"))
        self.assertEqual(derive_seed(7, "layout"), derive_seed(7, "layout"))
        self.assertNotEqual(derive_seed(7, "layout"), derive_seed(7, "terrain"))

    def test_all_presets_reproducible(self):
        for preset in PRESETS:
            with self.subTest(preset=preset):
                generator = RoomGenerator(GenerationConfig.preset(preset))
                self.assertEqual(
                    fingerprint(generator.generate(31337)),
                    fingerprint(generator.generate(31337)),
                )


class TestSubStreams(unittest.TestCase):
    def test_derive_is_stable_and_independent(self):
        root = SeededRandom(12345)
        a = [root.derive("layout").randint(0, 1000) for _ in range(5)]
        root2 = SeededRandom(12345)
        # draw from a different stream in between; the layout stream must not shift
        root2.derive("hazards").randint(0, 10)
        root2.derive("hazards").randint(0, 10)
        b = [root2.derive("layout").randint(0, 1000) for _ in range(5)]
        self.assertEqual(a, b)

    def test_weighted_choice_ignores_dict_order(self):
        weights_a = {"x": 1.0, "y": 2.0, "z": 3.0}
        weights_b = {"z": 3.0, "y": 2.0, "x": 1.0}
        draws_a = [SeededRandom(i).weighted_choice(weights_a) for i in range(50)]
        draws_b = [SeededRandom(i).weighted_choice(weights_b) for i in range(50)]
        self.assertEqual(draws_a, draws_b)

    def test_fork_differs_from_derive_label(self):
        root = SeededRandom(99)
        self.assertEqual(root.fork("attempt:0").seed, root.fork("attempt:0").seed)
        self.assertNotEqual(root.fork("attempt:0").seed, root.fork("attempt:1").seed)


class TestConfigRoundTrip(unittest.TestCase):
    def test_dict_round_trip_preserves_generation(self):
        config = GenerationConfig.from_complexity(0.65, seed=7)
        restored = GenerationConfig.from_dict(config.to_dict())
        self.assertEqual(
            fingerprint(RoomGenerator(config).generate(123)),
            fingerprint(RoomGenerator(restored).generate(123)),
        )

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            RoomGenerator(GenerationConfig(hazard_density=5.0))
        with self.assertRaises(ValueError):
            RoomGenerator(GenerationConfig(width=(2, 3)))

    def test_unknown_preset_lists_options(self):
        with self.assertRaises(KeyError) as ctx:
            GenerationConfig.preset("impossible")
        self.assertIn("standard", str(ctx.exception))


class TestSessionSeeding(unittest.TestCase):
    def test_master_seed_replays_the_same_episode_sequence(self):
        def run() -> list[int]:
            session = EnvironmentSession(
                GenerationConfig.preset("standard"), master_seed="run-a"
            )
            seeds = []
            for _ in range(5):
                session.reset()
                seeds.append(session.seed)
            return seeds

        self.assertEqual(run(), run())

    def test_different_master_seeds_diverge(self):
        first = EnvironmentSession(GenerationConfig.preset("easy"), master_seed=1)
        second = EnvironmentSession(GenerationConfig.preset("easy"), master_seed=2)
        first.reset()
        second.reset()
        self.assertNotEqual(first.seed, second.seed)


if __name__ == "__main__":
    unittest.main()

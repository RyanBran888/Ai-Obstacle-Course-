"""Generator behaviour: validity, variety, and respecting the config."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coop_env import GenerationConfig, RoomGenerator  # noqa: E402
from coop_env.config import PRESET_COMPLEXITY, RoomShape  # noqa: E402
from coop_env.entities import AgentSpawn, ExitDoor  # noqa: E402
from coop_env.generation.generator import GenerationError  # noqa: E402
from coop_env.tiles import Tile, is_walkable  # noqa: E402
from coop_env.validation import validate_room  # noqa: E402

SAMPLE = 40


class TestEveryRoomIsValid(unittest.TestCase):
    def test_all_presets_produce_valid_rooms(self):
        for preset in sorted(PRESET_COMPLEXITY):
            with self.subTest(preset=preset):
                generator = RoomGenerator(GenerationConfig.preset(preset))
                for seed in range(SAMPLE):
                    outcome = generator.generate_with_report(seed)
                    self.assertTrue(
                        outcome.report.ok,
                        f"{preset} seed {seed}: {outcome.report.summary()}",
                    )

    def test_no_fallback_rooms_under_normal_settings(self):
        for preset in sorted(PRESET_COMPLEXITY):
            with self.subTest(preset=preset):
                generator = RoomGenerator(GenerationConfig.preset(preset))
                fallbacks = sum(
                    1 for seed in range(SAMPLE)
                    if generator.generate_with_report(seed).fallback
                )
                self.assertEqual(fallbacks, 0, f"{preset} fell back {fallbacks} times")

    def test_continuous_complexity_axis(self):
        for step in range(11):
            complexity = step / 10
            with self.subTest(complexity=complexity):
                config = GenerationConfig.from_complexity(complexity)
                generator = RoomGenerator(config)
                for seed in range(6):
                    self.assertTrue(generator.generate_with_report(seed).report.ok)


class TestVariety(unittest.TestCase):
    def test_layouts_do_not_repeat(self):
        generator = RoomGenerator(GenerationConfig.preset("standard"))
        signatures = {
            tuple(generator.generate(seed).terrain.to_list()) for seed in range(60)
        }
        self.assertGreaterEqual(len(signatures), 59)

    def test_room_sizes_vary(self):
        generator = RoomGenerator(GenerationConfig.preset("standard"))
        sizes = {(r.width, r.height) for r in (generator.generate(s) for s in range(40))}
        self.assertGreater(len(sizes), 8, "room dimensions should vary between episodes")

    def test_multiple_silhouettes_appear(self):
        generator = RoomGenerator(GenerationConfig.preset("standard"))
        shapes = Counter(generator.generate(s).shape for s in range(60))
        self.assertGreaterEqual(len(shapes), 4, f"only saw {sorted(shapes)}")

    def test_shape_weights_are_honoured(self):
        config = GenerationConfig.preset(
            "standard", shape_weights={RoomShape.DONUT: 1.0}
        )
        generator = RoomGenerator(config)
        for seed in range(12):
            room = generator.generate(seed)
            # a donut can degrade to a rectangle when the ring would be too thin
            self.assertIn(room.shape, (RoomShape.DONUT,))


class TestConfigIsRespected(unittest.TestCase):
    def test_dimensions_stay_inside_the_configured_range(self):
        config = GenerationConfig(width=(18, 24), height=(14, 18))
        generator = RoomGenerator(config)
        for seed in range(30):
            room = generator.generate(seed)
            self.assertGreaterEqual(room.width, 18)
            self.assertLessEqual(room.width, 24)
            self.assertGreaterEqual(room.height, 14)
            self.assertLessEqual(room.height, 18)

    def test_zero_hazards_means_zero_hazard_tiles(self):
        config = GenerationConfig(
            hazard_density=0.0, platform_bridge_probability=0.0,
            num_moving_platforms=(0, 0), num_temporary_bridges=(0, 0),
        )
        generator = RoomGenerator(config)
        for seed in range(20):
            room = generator.generate(seed)
            hazards = sum(
                room.terrain.count(t)
                for t in (Tile.HAZARD_LAVA, Tile.HAZARD_SPIKES, Tile.HAZARD_WATER, Tile.HAZARD_PIT)
            )
            self.assertEqual(hazards, 0, f"seed {seed} produced {hazards} hazard tiles")

    def test_zero_obstacles_means_zero_obstacle_tiles(self):
        generator = RoomGenerator(GenerationConfig(obstacle_density=0.0))
        for seed in range(20):
            self.assertEqual(generator.generate(seed).terrain.count(Tile.OBSTACLE), 0)

    def test_higher_complexity_makes_bigger_busier_rooms(self):
        def measure(complexity: float) -> tuple[float, float]:
            generator = RoomGenerator(GenerationConfig.from_complexity(complexity))
            rooms = [generator.generate(s) for s in range(20)]
            area = sum(r.width * r.height for r in rooms) / len(rooms)
            mechanisms = sum(len(r.entities) for r in rooms) / len(rooms)
            return area, mechanisms

        low_area, low_count = measure(0.1)
        high_area, high_count = measure(0.9)
        self.assertGreater(high_area, low_area * 1.5)
        self.assertGreater(high_count, low_count)

    def test_cooperative_budget_produces_cooperative_gates(self):
        config = GenerationConfig.preset("standard", required_cooperative_actions=2)
        generator = RoomGenerator(config)
        with_coop = 0
        for seed in range(25):
            outcome = generator.generate_with_report(seed)
            if outcome.report.solvability.cooperative_clusters:
                with_coop += 1
        self.assertGreaterEqual(with_coop, 20, "co-op budget should almost always land")

    def test_harder_presets_build_longer_lock_chains(self):
        """`chain_length` is measured by the validator, not claimed by the generator."""

        def mean_chain(preset: str) -> float:
            generator = RoomGenerator(GenerationConfig.preset(preset))
            lengths = [
                generator.generate_with_report(s).report.solvability.chain_length
                for s in range(25)
            ]
            return sum(lengths) / len(lengths)

        self.assertGreater(mean_chain("brutal"), mean_chain("tutorial") + 1.0)

    def test_multi_step_chains_actually_occur(self):
        generator = RoomGenerator(GenerationConfig.preset("hard"))
        deep = sum(
            1
            for s in range(30)
            if generator.generate_with_report(s).report.solvability.chain_length >= 2
        )
        self.assertGreater(deep, 10, "a door behind a door should be common at 'hard'")

    def test_shared_keys_open_more_than_one_door(self):
        generator = RoomGenerator(GenerationConfig.preset("brutal"))
        shared = 0
        for seed in range(30):
            room = generator.generate(seed)
            if any(len(key.opens) > 1 for key in room.keys):
                shared += 1
        self.assertGreater(shared, 0, "no room reused a key across doors")

    def test_zero_cooperative_budget_still_valid(self):
        config = GenerationConfig.preset("hard", required_cooperative_actions=0)
        generator = RoomGenerator(config)
        for seed in range(20):
            self.assertTrue(generator.generate_with_report(seed).report.ok)


class TestStructure(unittest.TestCase):
    def test_every_room_has_two_spawns_and_one_exit(self):
        generator = RoomGenerator(GenerationConfig.preset("hard"))
        for seed in range(SAMPLE):
            room = generator.generate(seed)
            self.assertEqual(len(room.of_type(AgentSpawn)), 2)
            self.assertEqual(len(room.of_type(ExitDoor)), 1)
            self.assertNotEqual(room.spawns[0].pos, room.spawns[1].pos)

    def test_rooms_are_sealed_by_walls(self):
        """No walkable tile may touch the outside of the grid."""
        generator = RoomGenerator(GenerationConfig.preset("standard"))
        for seed in range(20):
            room = generator.generate(seed)
            for pos in room.terrain.positions():
                if not is_walkable(room.terrain[pos]):
                    continue
                on_edge = (
                    pos[0] == 0
                    or pos[1] == 0
                    or pos[0] == room.width - 1
                    or pos[1] == room.height - 1
                )
                self.assertFalse(on_edge, f"seed {seed}: open tile {tuple(pos)} on the border")

    def test_entities_never_share_a_tile(self):
        generator = RoomGenerator(GenerationConfig.preset("brutal"))
        for seed in range(20):
            room = generator.generate(seed)
            singles = [
                e.pos for e in room.entities
                if e.kind.name not in ("MOVING_PLATFORM", "RESET_ZONE", "TEMPORARY_BRIDGE")
            ]
            self.assertEqual(len(singles), len(set(singles)), f"seed {seed} has stacked objects")

    def test_generate_many_returns_distinct_rooms(self):
        generator = RoomGenerator(GenerationConfig.preset("easy"))
        rooms = generator.generate_many(10, start_seed=500)
        self.assertEqual(len(rooms), 10)
        self.assertEqual(len({r.seed for r in rooms}), 10)


class TestFallbackPath(unittest.TestCase):
    def test_fallback_room_is_itself_valid(self):
        from coop_env.generation.generator import _fallback_room

        room = _fallback_room(1, GenerationConfig())
        report = validate_room(room)
        self.assertTrue(report.ok, report.summary())

    def test_raise_on_failure_surfaces_the_error(self):
        # a single attempt with an impossible mechanism budget will not always
        # fail, so force the issue with a config that cannot place anything
        config = GenerationConfig(
            width=(8, 8), height=(8, 8), obstacle_density=0.6,
            hazard_density=0.6, max_attempts=1, raise_on_failure=True,
        )
        generator = RoomGenerator(config)
        failures = 0
        for seed in range(40):
            try:
                generator.generate(seed)
            except GenerationError:
                failures += 1
        self.assertGreater(failures, 0, "a hostile config should raise at least once")


if __name__ == "__main__":
    unittest.main()

"""Generator behaviour: validity, variety, and respecting the config."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coop_env import GenerationConfig, RoomGenerator  # noqa: E402
from coop_env.config import PRESET_COMPLEXITY, RoomShape  # noqa: E402
from coop_env.entities import (  # noqa: E402
    AgentSpawn,
    ExitDoor,
    LockedDoor,
    Switch,
    SwitchMode,
)
from coop_env.generation.generator import GenerationError  # noqa: E402
from coop_env.state import EpisodeState  # noqa: E402
from coop_env.tiles import Tile, is_walkable  # noqa: E402
from coop_env.utils.geometry import DIRECTIONS4  # noqa: E402
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
            hazard_density=0.0, num_temporary_bridges=(0, 0),
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

    def test_requested_crate_starts_one_push_from_a_hold_switch(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                switches = 2 if paired else 1
                config = GenerationConfig(
                    width=(18, 22),
                    height=(14, 18),
                    shape_weights={RoomShape.RECTANGLE: 1.0},
                    region_count=(2, 3),
                    obstacle_density=0.0,
                    hazard_density=0.0,
                    num_keys=(0, 0),
                    num_locked_doors=(1, 1),
                    num_switches=(switches, switches),
                    num_pushable_blocks=(1, 1),
                    num_checkpoints=(0, 0),
                    num_reset_zones=(0, 0),
                    num_temporary_bridges=(0, 0),
                    puzzle_chain_length=1,
                    exit_objective_count=0,
                    required_cooperative_actions=1,
                    timed_door_probability=0.0,
                    exit_requires_both_agents=paired,
                    raise_on_failure=True,
                )
                generator = RoomGenerator(config)
                for seed in range(8):
                    outcome = generator.generate_with_report(seed)
                    room = outcome.room
                    self.assertFalse(outcome.fallback)
                    self.assertTrue(outcome.report.ok, outcome.report.report_lines())
                    self.assertEqual(len(room.blocks), 1)

                    block = room.blocks[0]
                    target = room.find(block.target_switch_id or "")
                    self.assertIsInstance(target, Switch)
                    assert isinstance(target, Switch)
                    self.assertIs(target.mode, SwitchMode.HOLD)

                    direction = target.pos - block.pos
                    self.assertIn(direction, DIRECTIONS4)
                    self.assertEqual(block.push_from, block.pos - direction)
                    assert block.push_from is not None
                    self.assertEqual(room.terrain_at(block.push_from), Tile.FLOOR)
                    self.assertFalse(room.entities_at(block.push_from))

                    state = EpisodeState.from_room(room)
                    self.assertFalse(state.is_switch_active(target.id))
                    state.place_block(block.id, target.pos)
                    self.assertTrue(state.is_switch_active(target.id))
                    self.assertEqual(
                        room.metadata["crate_switch_pairs"][0]["block"],
                        block.id,
                    )

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

    def test_preset_keys_do_not_open_multiple_logical_doors(self):
        generator = RoomGenerator(GenerationConfig.preset("brutal"))
        for seed in range(30):
            room = generator.generate(seed)
            for key in room.keys:
                targets = set()
                for target_id in key.opens:
                    target = room.find(target_id)
                    if target_id == "exit":
                        targets.add(("exit",))
                    elif isinstance(target, LockedDoor):
                        targets.add(
                            (
                                min(target.region_a, target.region_b),
                                max(target.region_a, target.region_b),
                            )
                        )
                self.assertLessEqual(len(targets), 1)

    def test_owned_key_levels_have_multiple_distinct_doors(self):
        config = GenerationConfig(
            width=(16, 20),
            height=(11, 15),
            shape_weights={RoomShape.RECTANGLE: 1.0},
            region_count=(3, 4),
            obstacle_density=0.0,
            hazard_density=0.0,
            num_keys=(2, 2),
            num_locked_doors=(2, 3),
            num_switches=(0, 0),
            puzzle_chain_length=2,
            required_cooperative_actions=0,
            exit_objective_count=0,
            agent_specific_keys=True,
            allow_shared_keys=False,
            require_key_for_each_agent=True,
        )
        generator = RoomGenerator(config)
        for seed in range(10):
            room = generator.generate(seed)
            self.assertEqual({key.agent_index for key in room.keys}, {0, 1})
            self.assertGreaterEqual(len(room.doors), 2)
            self.assertTrue(all(len(key.opens) >= 1 for key in room.keys))

    def test_wipeout_levels_have_exact_tracks(self):
        config = GenerationConfig(
            width=(24, 28),
            height=(14, 18),
            shape_weights={RoomShape.RECTANGLE: 1.0},
            region_count=(1, 1),
            obstacle_density=0.0,
            hazard_density=0.0,
            num_keys=(0, 0),
            num_locked_doors=(0, 0),
            num_switches=(0, 0),
            puzzle_chain_length=0,
            required_cooperative_actions=0,
            exit_objective_count=0,
            num_normal_wipeout_balls=(1, 1),
            num_big_wipeout_balls=(1, 1),
        )
        room = RoomGenerator(config).generate(4)
        lengths = sorted(len(ball.track) for ball in room.wipeout_balls)
        self.assertEqual(lengths, [7, 11])

    def test_required_wipeout_crossings_are_forced_and_time_solvable(self):
        cases = ((9, 16, 1, 0), (15, 18, 0, 1))
        for width, height, normal, big in cases:
            config = GenerationConfig(
                width=(width, width),
                height=(height, height),
                shape_weights={RoomShape.RECTANGLE: 1.0},
                region_count=(1, 1),
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
                required_cooperative_actions=0,
                exit_objective_count=0,
                num_normal_wipeout_balls=(normal, normal),
                num_big_wipeout_balls=(big, big),
                require_wipeout_crossing=True,
                raise_on_failure=True,
            )
            generator = RoomGenerator(config)
            for seed in range(8):
                report = generator.generate_with_report(seed).report
                self.assertTrue(report.ok, report.report_lines())
                wipeout = report.wipeout
                self.assertIsNotNone(wipeout)
                assert wipeout is not None
                self.assertTrue(wipeout.structural_crossing)
                self.assertTrue(wipeout.time_solvable)
                self.assertEqual(wipeout.reachable_by, (0, 1))

    def test_timed_probability_one_always_builds_rearmable_timed_gates(self):
        config = GenerationConfig(
            width=(24, 24),
            height=(18, 18),
            shape_weights={RoomShape.RECTANGLE: 1.0},
            region_count=(3, 3),
            obstacle_density=0.0,
            hazard_density=0.0,
            num_keys=(0, 0),
            num_locked_doors=(1, 1),
            num_switches=(1, 1),
            num_pushable_blocks=(0, 0),
            num_checkpoints=(0, 0),
            num_reset_zones=(0, 0),
            num_temporary_bridges=(0, 0),
            puzzle_chain_length=1,
            required_cooperative_actions=0,
            exit_objective_count=0,
            timed_door_probability=1.0,
            raise_on_failure=True,
        )
        for seed in range(12):
            room = RoomGenerator(config).generate(seed)
            self.assertEqual(
                [gate["kind"] for gate in room.metadata["gates"]],
                ["timed_switch"],
            )
            self.assertTrue(all(door.timer is not None for door in room.doors))
            self.assertTrue(all(switch.mode.value == "toggle" for switch in room.switches))

    def test_required_bridge_course_forces_a_timed_crossing(self):
        config = GenerationConfig(
            width=(8, 8),
            height=(16, 18),
            shape_weights={RoomShape.RECTANGLE: 1.0},
            region_count=(1, 1),
            obstacle_density=0.0,
            hazard_density=0.0,
            hazard_weights={Tile.HAZARD_WATER: 1.0},
            num_keys=(0, 0),
            num_locked_doors=(0, 0),
            num_switches=(0, 0),
            num_pushable_blocks=(0, 0),
            num_checkpoints=(0, 0),
            num_reset_zones=(0, 0),
            num_temporary_bridges=(1, 1),
            num_normal_wipeout_balls=(0, 0),
            num_big_wipeout_balls=(0, 0),
            puzzle_chain_length=0,
            required_cooperative_actions=0,
            exit_objective_count=0,
            require_bridge_crossing=True,
            raise_on_failure=True,
        )
        generator = RoomGenerator(config)
        for seed in range(12):
            outcome = generator.generate_with_report(seed)
            room = outcome.room
            bridge = outcome.report.bridge
            self.assertFalse(outcome.fallback)
            self.assertEqual(len(room.bridges), 1)
            self.assertEqual(room.metadata["required_bridge_id"], room.bridges[0].id)
            self.assertIsNotNone(bridge)
            assert bridge is not None
            self.assertTrue(bridge.structural_crossing)
            self.assertTrue(bridge.time_solvable)
            self.assertEqual(bridge.reachable_by, (0, 1))

    def test_required_reset_zone_has_a_longer_safe_detour(self):
        config = GenerationConfig(
            width=(10, 10),
            height=(16, 18),
            shape_weights={RoomShape.RECTANGLE: 1.0},
            region_count=(1, 1),
            obstacle_density=0.0,
            hazard_density=0.0,
            num_keys=(0, 0),
            num_locked_doors=(0, 0),
            num_switches=(0, 0),
            num_pushable_blocks=(0, 0),
            num_checkpoints=(0, 0),
            num_reset_zones=(1, 1),
            num_temporary_bridges=(0, 0),
            num_normal_wipeout_balls=(0, 0),
            num_big_wipeout_balls=(0, 0),
            puzzle_chain_length=0,
            required_cooperative_actions=0,
            exit_objective_count=0,
            require_reset_detour=True,
            raise_on_failure=True,
        )
        generator = RoomGenerator(config)
        for seed in range(12):
            outcome = generator.generate_with_report(seed)
            room = outcome.room
            detour = outcome.report.reset_detour
            self.assertFalse(outcome.fallback)
            self.assertEqual(len(room.reset_zones), 1)
            self.assertEqual(
                room.metadata["required_reset_zone_id"],
                room.reset_zones[0].id,
            )
            self.assertIsNotNone(detour)
            assert detour is not None
            self.assertTrue(detour.shortcut_valid)
            self.assertTrue(detour.safe_detour)
            self.assertTrue(
                all(
                    safe > short
                    for short, safe in zip(
                        detour.shortest_steps_by_agent,
                        detour.safe_steps_by_agent,
                    )
                    if short is not None and safe is not None
                )
            )

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
                if e.kind.name not in ("RESET_ZONE", "TEMPORARY_BRIDGE")
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

    def test_required_bridge_fallback_keeps_the_crossing_contract(self):
        config = GenerationConfig(
            width=(24, 24),
            height=(18, 18),
            shape_weights={RoomShape.RECTANGLE: 1.0},
            region_count=(1, 1),
            obstacle_density=0.0,
            hazard_density=0.0,
            hazard_weights={Tile.HAZARD_WATER: 1.0},
            num_keys=(0, 0),
            num_locked_doors=(0, 0),
            num_switches=(0, 0),
            num_pushable_blocks=(0, 0),
            num_checkpoints=(0, 0),
            num_reset_zones=(0, 0),
            num_temporary_bridges=(1, 1),
            puzzle_chain_length=0,
            required_cooperative_actions=0,
            exit_objective_count=0,
            require_bridge_crossing=True,
            max_attempts=1,
        )
        outcome = RoomGenerator(config).generate_with_report(0)
        self.assertTrue(outcome.fallback)
        self.assertTrue(outcome.report.ok, outcome.report.report_lines())
        bridge = outcome.report.bridge
        self.assertIsNotNone(bridge)
        assert bridge is not None
        self.assertTrue(bridge.structural_crossing)
        self.assertTrue(bridge.time_solvable)

    def test_required_reset_fallback_keeps_the_detour_contract(self):
        config = GenerationConfig(
            width=(8, 8),
            height=(8, 8),
            shape_weights={RoomShape.RECTANGLE: 1.0},
            region_count=(1, 1),
            obstacle_density=0.6,
            hazard_density=0.6,
            num_keys=(0, 0),
            num_locked_doors=(0, 0),
            num_switches=(0, 0),
            num_pushable_blocks=(0, 0),
            num_checkpoints=(0, 0),
            num_reset_zones=(1, 1),
            num_temporary_bridges=(0, 0),
            puzzle_chain_length=0,
            required_cooperative_actions=0,
            exit_objective_count=0,
            require_reset_detour=True,
            max_attempts=1,
        )
        outcome = RoomGenerator(config).generate_with_report(0)
        self.assertTrue(outcome.fallback)
        self.assertTrue(outcome.report.ok, outcome.report.report_lines())
        detour = outcome.report.reset_detour
        self.assertIsNotNone(detour)
        assert detour is not None
        self.assertTrue(detour.shortcut_valid)
        self.assertTrue(detour.safe_detour)

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

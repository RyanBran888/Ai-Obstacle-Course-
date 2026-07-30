"""Combined-course generation and contract tests."""

from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coop_env import GenerationConfig, RoomGenerator, RoomShape  # noqa: E402
from coop_env.entities import LockedDoor, WipeoutBall, WipeoutBallSize  # noqa: E402
from coop_env.state import EpisodeState  # noqa: E402
from coop_env.tiles import Tile  # noqa: E402
from coop_env.utils.geometry import Vec2  # noqa: E402
from coop_env.validation import validate_room  # noqa: E402


def combined_config(**overrides) -> GenerationConfig:
    config = GenerationConfig(
        width=(37, 38),
        height=(25, 28),
        shape_weights={RoomShape.RECTANGLE: 1.0},
        region_count=(4, 5),
        obstacle_density=0.0,
        hazard_density=0.0,
        num_keys=(2, 2),
        num_locked_doors=(3, 3),
        num_switches=(1, 1),
        num_pushable_blocks=(1, 1),
        num_checkpoints=(0, 0),
        num_reset_zones=(1, 1),
        num_temporary_bridges=(1, 1),
        num_normal_wipeout_balls=(1, 1),
        num_big_wipeout_balls=(1, 1),
        puzzle_chain_length=3,
        exit_objective_count=1,
        required_cooperative_actions=1,
        exit_requires_both_agents=True,
        agent_specific_keys=True,
        allow_shared_keys=False,
        require_key_for_each_agent=True,
        require_combined_course=True,
    )
    return config.with_overrides(**overrides)


class TestCombinedCourseGeneration(unittest.TestCase):
    def test_course_is_direct_varied_and_fully_contracted(self):
        generator = RoomGenerator(combined_config())
        required_sizes: set[WipeoutBallSize] = set()
        signatures: set[tuple] = set()
        for seed in range(16):
            outcome = generator.generate_with_report(seed)
            room = outcome.room
            combined = outcome.report.combined
            self.assertTrue(outcome.report.ok, outcome.report.report_lines())
            self.assertEqual(outcome.attempts, 1)
            self.assertFalse(outcome.fallback)
            self.assertIsNotNone(combined)
            assert combined is not None
            self.assertTrue(combined.valid, combined.reasons)
            self.assertEqual(
                set(combined.gate_cut_ids),
                {"door_key_0", "door_key_1", "door_crate"},
            )
            self.assertEqual(
                combined.ball_cut_ids,
                (room.metadata["required_wipeout_ball_id"],),
            )
            required = room.find(room.metadata["required_wipeout_ball_id"])
            self.assertIsInstance(required, WipeoutBall)
            assert isinstance(required, WipeoutBall)
            required_sizes.add(required.size)
            self.assertEqual(
                {ball.size: len(ball.track) for ball in room.wipeout_balls},
                {
                    WipeoutBallSize.NORMAL: 7,
                    WipeoutBallSize.BIG: 11,
                },
            )
            budget = room.metadata["combined_course"]["route_budget"]
            self.assertLessEqual(
                budget["planned_action_steps"],
                budget["movement_budget"],
            )
            self.assertLess(budget["total"], budget["horizon"])
            solvability = outcome.report.solvability
            self.assertIsNotNone(solvability)
            assert solvability is not None
            self.assertTrue(solvability.exit_jointly_reachable)
            signatures.add(
                (
                    room.width,
                    room.height,
                    required.size,
                    room.bridges[0].phase,
                    room.reset_zones[0].pos,
                    room.keys[0].pos,
                )
            )
        self.assertEqual(
            required_sizes,
            {WipeoutBallSize.NORMAL, WipeoutBallSize.BIG},
        )
        self.assertGreaterEqual(len(signatures), 10)

    def test_invalid_combined_budget_is_rejected_before_generation(self):
        with self.assertRaises(ValueError):
            RoomGenerator(combined_config(num_switches=(2, 2)))

    def test_crate_keeps_the_non_latching_gate_open(self):
        room = RoomGenerator(combined_config()).generate(3)
        state = EpisodeState.from_room(room)
        self.assertFalse(state.is_door_open("door_crate"))
        state.place_block("crate_0", room.entity("switch_crate").pos)
        self.assertTrue(state.is_door_open("door_crate"))


class TestCombinedCourseValidation(unittest.TestCase):
    def setUp(self):
        self.room = RoomGenerator(combined_config()).generate(2)

    def test_ball_bypass_is_rejected(self):
        terrain = self.room.terrain.copy()
        bottom_top = self.room.height - 9
        for y in range(10, bottom_top):
            terrain[Vec2(22, y)] = Tile.FLOOR
        broken = replace(self.room, terrain=terrain)
        report = validate_room(broken)
        self.assertFalse(report.ok)
        self.assertTrue(
            any(
                issue.code in {"wipeout_unsolvable", "combined_course_invalid"}
                and "every spawn-to-exit route" in issue.message
                for issue in report.errors
            ),
            report.report_lines(),
        )

    def test_bridge_bypass_is_rejected(self):
        terrain = self.room.terrain.copy()
        bottom_y = self.room.height - 6
        terrain[Vec2(20, bottom_y - 3)] = Tile.FLOOR
        broken = replace(self.room, terrain=terrain)
        report = validate_room(broken)
        self.assertFalse(report.ok)
        self.assertTrue(
            any(
                issue.code in {"bridge_unsolvable", "combined_course_invalid"}
                and (
                    "every spawn-to-exit route" in issue.message
                    or "bridge has a bypass" in issue.message
                )
                for issue in report.errors
            ),
            report.report_lines(),
        )

    def test_latching_the_crate_gate_is_rejected(self):
        entities = tuple(
            replace(entity, latching=True)
            if isinstance(entity, LockedDoor) and entity.id == "door_crate"
            else entity
            for entity in self.room.entities
        )
        broken = replace(self.room, entities=entities)
        report = validate_room(broken)
        self.assertFalse(report.ok)
        self.assertTrue(
            any(
                issue.code == "combined_course_invalid"
                and "crate HOLD gate" in issue.message
                for issue in report.errors
            ),
            report.report_lines(),
        )

    def test_route_budget_tampering_is_rejected(self):
        metadata = deepcopy(self.room.metadata)
        metadata["combined_course"]["route_budget"]["total"] = 199
        broken = replace(self.room, metadata=metadata)
        report = validate_room(broken)
        self.assertFalse(report.ok)
        self.assertTrue(
            any(
                issue.code == "combined_course_invalid"
                and "route budget metadata" in issue.message
                for issue in report.errors
            ),
            report.report_lines(),
        )


if __name__ == "__main__":
    unittest.main()

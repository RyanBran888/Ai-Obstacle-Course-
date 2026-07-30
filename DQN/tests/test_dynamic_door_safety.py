from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))
sys.path.insert(0, str(ROOT / "Architecture" / "tests"))

from helpers import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    DOORWAY,
    LEFT_TILE,
    hold_switch_room,
)
from coop_env import RoomGenerator, Vec2  # noqa: E402
from coop_env.entities import PushableBlock, WipeoutBall  # noqa: E402
from DQN.curriculum import CurriculumRunner, default_stages  # noqa: E402
from DQN.env_bridge import (  # noqa: E402
    POLICY_MODE_ASSISTED,
    CoopEnvBridge,
    WAIT,
)


EAST = 1
FULL_7_FAILURE_SEED = 362173977868598773


def with_stationary_ball(room, *entities):
    """Activate the wipeout planner without putting danger near the fixture."""
    ball = WipeoutBall(
        id="far_ball",
        pos=Vec2(14, 1),
        track=(Vec2(14, 1),),
    )
    return replace(room, entities=room.entities + tuple(entities) + (ball,))


class TestProjectedHoldDoorSafety(unittest.TestCase):
    def make_env(self, room) -> CoopEnvBridge:
        env = CoopEnvBridge(
            max_steps=40,
            record_metrics=False,
            policy_mode=POLICY_MODE_ASSISTED,
        )
        env.load_room(room)
        return env

    def test_teammate_holding_switch_keeps_future_door_crossing_safe(self):
        room = with_stationary_ball(hold_switch_room())
        env = self.make_env(room)
        env.pos = [LEFT_TILE, DOORWAY + Vec2(-1, 0)]
        env._holds()

        self.assertTrue(env.state.is_door_open("door_0"))
        self.assertTrue(env.wipeout_action_masks(horizon=1)[1][EAST])
        self.assertTrue(env.wipeout_action_masks(horizon=10)[1][EAST])

        _, _, done, cut, info = env.step((WAIT, EAST))
        self.assertFalse(done)
        self.assertFalse(cut)
        self.assertEqual(env.pos[1], DOORWAY)
        self.assertTrue(env.state.is_door_open("door_0"))
        self.assertNotIn("agent_1:unstick", info["events"])

    def test_crate_holding_switch_keeps_future_door_crossing_safe(self):
        switch_room = hold_switch_room()
        crate = PushableBlock(
            id="crate_0",
            pos=LEFT_TILE,
            target_switch_id="switch_0",
            push_from=LEFT_TILE + Vec2(-1, 0),
        )
        room = with_stationary_ball(switch_room, crate)
        env = self.make_env(room)
        env.pos = [Vec2(2, 2), DOORWAY + Vec2(-1, 0)]
        env._holds()

        blocks = tuple(sorted(env.state.block_positions.items()))
        agents = (env.pos[0], env.pos[1])
        door = room.entity("door_0")
        self.assertTrue(env.state.is_door_open(door.id))
        self.assertTrue(
            env._wipeout_door_open_at(
                door,
                env.state.tick + 10,
                blocks,
                agents,
            )
        )
        self.assertTrue(env.wipeout_action_masks(horizon=10)[1][EAST])

        projected_off_switch = (("crate_0", LEFT_TILE + Vec2(1, 0)),)
        self.assertFalse(
            env._wipeout_door_open_at(
                door,
                env.state.tick + 1,
                projected_off_switch,
                agents,
            )
        )
        self.assertTrue(
            env._wipeout_door_open_at(
                door,
                env.state.tick + 1,
                projected_off_switch,
                (LEFT_TILE, env.pos[1]),
            )
        )

    def test_full_7_failure_seed_allows_the_learned_post_crate_route(self):
        stage = next(
            stage
            for stage in default_stages()
            if stage.name == "full_course_mix"
        )
        room = RoomGenerator(stage.config).generate(FULL_7_FAILURE_SEED)
        env = CoopEnvBridge(
            stage.config,
            seed=16,
            max_steps=200,
            record_metrics=False,
            policy_mode=POLICY_MODE_ASSISTED,
        )
        env.load_room(room)

        crate = room.entity("crate_0")
        switch = room.entity("switch_crate")
        door = room.entity("door_crate")
        self.assertEqual(crate.pos, Vec2(11, 5))
        self.assertEqual(switch.pos, Vec2(12, 5))
        self.assertEqual(door.pos, Vec2(14, 5))

        env.state.place_block(crate.id, switch.pos)
        env.pos = [door.pos + Vec2(-1, 0), room.spawns[1].pos]
        env._holds()

        self.assertTrue(env.state.is_door_open(door.id))
        self.assertTrue(env.wipeout_action_masks(horizon=1)[0][EAST])
        self.assertTrue(env.wipeout_action_masks(horizon=10)[0][EAST])


class TestFull7DynamicDoorUpgrade(unittest.TestCase):
    def make_runner_and_payload(self):
        runner = object.__new__(CurriculumRunner)
        runner.stages = default_stages()
        runner.pool_sizes = (1, 4, 16, 64)
        final_index = len(runner.stages) - 1
        results = [
            {
                "stage": stage.name,
                "pool_size": pool,
                "scheduled_pool_size": pool,
                "promoted": True,
            }
            for stage in runner.stages[:final_index]
            for pool in tuple(
                sorted(set(stage.pool_sizes or runner.pool_sizes))
            )
        ]
        payload = {
            "status": "training",
            "test_results": [],
            "test_model_sha256": None,
            "results": results,
            "active": {
                "stage_index": final_index,
                "pool_index": 0,
                "scheduled_pool_size": 1,
                "active_pool_size": 1,
                "total_rounds": 28,
                "normal_rounds": 28,
                "recovery_rounds": 0,
                "phase": "normal",
                "phase_rounds": 28,
            },
            "contract": {
                "external": {
                    "source_upgrade": {"kind": "wipeout_safety_v1"},
                },
            },
        }
        return runner, payload

    def test_accepts_only_the_exact_full_7_cursor(self):
        runner, payload = self.make_runner_and_payload()
        runner._validate_dynamic_door_cursor(payload)

        payload["active"]["stage_index"] -= 1
        with self.assertRaisesRegex(ValueError, "supported full_7 cursor"):
            runner._validate_dynamic_door_cursor(payload)

    def test_requires_the_prior_wipeout_safety_upgrade(self):
        runner, payload = self.make_runner_and_payload()
        payload["contract"]["external"]["source_upgrade"]["kind"] = "planner_v3"
        with self.assertRaisesRegex(ValueError, "wipeout-safety source"):
            runner._validate_dynamic_door_cursor(payload)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Architecture"))

from DQN.DQN_model import GLOBAL_NAMES, GLOBALS, N_ACTIONS, OBS_DIM  # noqa: E402
from DQN.env_bridge import (  # noqa: E402
    POLICY_MODE_ASSISTED,
    POLICY_MODE_LEARNED,
    R_BLOCKED,
    WAIT,
    CoopEnvBridge,
    micro_room,
)
from coop_env import Vec2  # noqa: E402
from coop_env.entities import WipeoutBall  # noqa: E402


GLOBAL_BASE = OBS_DIM - GLOBALS


def global_value(observation, name: str) -> float:
    return float(observation[GLOBAL_BASE + GLOBAL_NAMES.index(name)])


def room_with_stationary_ball():
    room = micro_room(3)
    ball = WipeoutBall(
        id="stationary_ball",
        pos=Vec2(1, 1),
        track=(Vec2(1, 1),),
    )
    return replace(room, entities=room.entities + (ball,))


class TestLearnedEnvironmentContract(unittest.TestCase):
    def test_fresh_environment_defaults_to_learned_mode(self):
        env = CoopEnvBridge(micro=3, record_metrics=False)

        self.assertEqual(env.policy_mode, POLICY_MODE_LEARNED)
        self.assertFalse(env.legacy_assistance_enabled)
        with self.assertRaisesRegex(ValueError, "unknown policy mode"):
            env.set_policy_mode("teacher")

    def test_learned_observation_hides_only_exact_route_teacher_fields(self):
        learned = CoopEnvBridge(
            micro=3,
            record_metrics=False,
            policy_mode=POLICY_MODE_LEARNED,
        )
        learned.reset()

        # The internal planner is still available for shaping and high-level
        # goal conditioning, but its chosen action is not in the observation.
        goal_info = learned._goal_info(0)
        internal_route = goal_info[3]
        self.assertNotEqual(internal_route, Vec2(0, 0))
        # Force the cached teacher's wait bit on so this fixture verifies that
        # all three route-only fields are gated, not merely zero by chance.
        target, kind, delta, route, distance, reachable, _ = goal_info
        learned._goal_cache[0] = (
            target,
            kind,
            delta,
            route,
            distance,
            reachable,
            True,
        )
        learned_obs = learned._obs(0)
        self.assertNotEqual(global_value(learned_obs, "goal_dx"), 0.0)
        self.assertNotEqual(global_value(learned_obs, "goal_dy"), 0.0)
        self.assertEqual(global_value(learned_obs, "route_dx"), 0.0)
        self.assertEqual(global_value(learned_obs, "route_dy"), 0.0)
        self.assertEqual(global_value(learned_obs, "route_wait"), 0.0)
        self.assertEqual(len(learned_obs), OBS_DIM)

        learned.set_policy_mode(POLICY_MODE_ASSISTED)
        assisted_obs = learned._obs(0)
        self.assertTrue(
            global_value(assisted_obs, "route_dx")
            or global_value(assisted_obs, "route_dy")
        )
        self.assertEqual(global_value(assisted_obs, "route_wait"), 1.0)
        self.assertEqual(len(assisted_obs), OBS_DIM)

    def test_learned_mode_returns_a_neutral_mask_without_survival_search(self):
        env = CoopEnvBridge(
            max_steps=30,
            record_metrics=False,
            policy_mode=POLICY_MODE_LEARNED,
        )
        env.load_room(room_with_stationary_ball())

        with patch.object(
            env,
            "_wipeout_action_survives",
            side_effect=AssertionError("recursive teacher must not run"),
        ) as survival_search:
            masks = env.wipeout_action_masks(horizon=10)

        self.assertFalse(survival_search.called)
        neutral = (True,) * N_ACTIONS
        self.assertEqual(masks, (neutral, neutral))

    def test_legacy_assisted_mode_preserves_recursive_wipeout_shield(self):
        env = CoopEnvBridge(
            max_steps=30,
            record_metrics=False,
            policy_mode=POLICY_MODE_ASSISTED,
        )
        env.load_room(room_with_stationary_ball())
        original = env._wipeout_action_survives

        with patch.object(
            env,
            "_wipeout_action_survives",
            wraps=original,
        ) as survival_search:
            masks = env.wipeout_action_masks(horizon=3)

        self.assertTrue(survival_search.called)
        self.assertTrue(any(not allowed for mask in masks for allowed in mask))

    def test_learned_wait_reward_does_not_consult_the_route_teacher(self):
        env = CoopEnvBridge(
            micro=3,
            record_metrics=False,
            policy_mode=POLICY_MODE_LEARNED,
        )
        env.reset()

        with patch.object(
            env,
            "_goal_info",
            side_effect=AssertionError("route teacher must not score WAIT"),
        ):
            self.assertEqual(env._apply_action(0, WAIT), (0.0, None))

        env.set_policy_mode(POLICY_MODE_ASSISTED)
        penalty, event = env._apply_action(0, WAIT)
        self.assertEqual(penalty, R_BLOCKED)
        self.assertEqual(event, "idle")

    def test_policy_mode_can_be_selected_after_construction(self):
        env = CoopEnvBridge(micro=3, record_metrics=False)
        learned_obs = env.reset()[0]
        env.set_policy_mode(POLICY_MODE_ASSISTED)
        assisted_obs = env._obs(0)

        self.assertEqual(global_value(learned_obs, "route_dx"), 0.0)
        self.assertNotEqual(global_value(assisted_obs, "route_dx"), 0.0)
        self.assertTrue(env.legacy_assistance_enabled)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

from DQN.DQN_model import (  # noqa: E402
    ACTION_SAFETY_CONTRACT,
    ASSISTED_POLICY_MODE,
    GLOBAL_NAMES,
    LEARNED_POLICY_CONTRACT,
    LEARNED_POLICY_MODE,
    OBS_DIM,
    POLICY_CONTRACT,
)
from DQN.DQN_train import (  # noqa: E402
    Agent,
    Config,
    _environment_action_masks_for_modes,
    evaluate_detailed,
)
from DQN.load_model import load_agent  # noqa: E402


def _global_index(name: str) -> int:
    return OBS_DIM - len(GLOBAL_NAMES) + GLOBAL_NAMES.index(name)


def _east_route_observation() -> np.ndarray:
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[_global_index("route_dx")] = 1.0
    observation[_global_index("goal_reachable")] = 1.0
    return observation


def _set_action_preferences(agent: Agent) -> None:
    with torch.no_grad():
        for parameter in agent.net.parameters():
            parameter.zero_()
        # Raw Q prefers west (3), while the historical +0.5 route bonus
        # changes the assisted policy's choice to east (1).
        agent.net.advantage.bias[3] = 0.2


class TestLearnedPolicyContract(unittest.TestCase):
    def make_agent(self, mode: str = LEARNED_POLICY_MODE) -> Agent:
        agent = Agent(
            hidden=(8,),
            replay_capacity=8,
            device="cpu",
            policy_mode=mode,
        )
        _set_action_preferences(agent)
        return agent

    def test_fresh_defaults_use_raw_learned_q_values(self):
        self.assertEqual(Config().policy_mode, LEARNED_POLICY_MODE)
        observation = _east_route_observation()
        learned = self.make_agent()
        assisted = self.make_agent(ASSISTED_POLICY_MODE)

        self.assertEqual(learned.best_actions([observation]), [3])
        self.assertEqual(assisted.best_actions([observation]), [1])

    def test_learned_mode_does_not_query_future_survival_masks(self):
        class Environment:
            def wipeout_action_masks(self):
                raise AssertionError("learned mode queried future state")

        observation = _east_route_observation()
        masks = _environment_action_masks_for_modes(
            Environment(),
            [observation, observation],
            [LEARNED_POLICY_MODE, LEARNED_POLICY_MODE],
        )
        self.assertEqual(masks.shape, (2, 6))
        self.assertTrue(masks[:, :4].all())

    def test_checkpoint_records_mode_and_restores_old_v2_as_assisted(self):
        learned = self.make_agent()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "learned.pt"
            learned.save(str(path))
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(
                checkpoint["policy_mode"],
                LEARNED_POLICY_MODE,
            )
            self.assertEqual(
                checkpoint["policy"],
                LEARNED_POLICY_CONTRACT,
            )
            self.assertIsNone(checkpoint["action_safety"])

            restored = self.make_agent(ASSISTED_POLICY_MODE)
            restored.load(str(path))
            self.assertEqual(restored.policy_mode, LEARNED_POLICY_MODE)
            self.assertFalse(restored.require_action_mask)

            old_v2 = learned.learning_state()
            old_v2.pop("policy_mode")
            old_v2["policy"] = dict(POLICY_CONTRACT)
            old_v2["action_safety"] = dict(ACTION_SAFETY_CONTRACT)
            restored.load_learning_state(old_v2)
            self.assertEqual(restored.policy_mode, ASSISTED_POLICY_MODE)
            self.assertTrue(restored.require_action_mask)
            with self.assertRaisesRegex(ValueError, "requires"):
                restored.act(_east_route_observation(), eps=0.0)

            old_path = Path(temporary) / "assisted-v2.pt"
            torch.save(old_v2, old_path)
            lightweight = load_agent(
                old_path,
                device="cpu",
                hidden=(8,),
                replay_capacity=8,
            )
            self.assertEqual(
                lightweight.policy_mode,
                ASSISTED_POLICY_MODE,
            )
            self.assertEqual(
                lightweight.best_actions(
                    [_east_route_observation()],
                    [np.ones(6, dtype=np.bool_)],
                ),
                [1],
            )

    def test_learned_batch_has_no_route_auxiliary_loss_and_reports_metrics(self):
        agent = self.make_agent()
        observation = _east_route_observation()
        agent.remember(
            observation,
            action=3,
            reward=1.0,
            next_obs=observation,
            terminal=True,
            discount=0.99,
            important=False,
        )

        metrics = agent.learn_batch(1)
        self.assertEqual(metrics["route_aux_loss"], 0.0)
        self.assertEqual(
            set(metrics),
            {
                "total_loss",
                "td_loss",
                "route_aux_loss",
                "grad_norm",
                "q_mean",
                "q_abs_mean",
            },
        )
        self.assertEqual(metrics, agent.latest_learning_metrics)
        self.assertAlmostEqual(
            metrics["total_loss"],
            metrics["td_loss"],
        )

    def test_evaluation_configures_lanes_for_old_assisted_agents(self):
        observation = _east_route_observation()

        class Environment:
            max_steps = 1
            record_metrics = True

            def __init__(self):
                self.mode_at_reset = None
                self.policy_mode = LEARNED_POLICY_MODE

            def set_policy_mode(self, mode):
                self.policy_mode = mode

            def reset(self, seed=None):
                self.mode_at_reset = self.policy_mode
                return [observation.copy(), observation.copy()]

            def wipeout_action_masks(self):
                return np.ones((2, 6), dtype=np.bool_)

            def step(self, actions):
                episode = {
                    "completed": True,
                    "timed_out": False,
                    "steps": 1,
                    "keys_collected": 0,
                    "doors_opened": 0,
                    "switches_activated": 0,
                    "checkpoints_reached": 0,
                    "exit_opened": True,
                    "wipeout_deaths": 0,
                    "wrong_key_interactions": 0,
                }
                return (
                    [observation.copy(), observation.copy()],
                    [1.0, 1.0],
                    True,
                    False,
                    {"episode": episode},
                )

        env = Environment()
        agent = self.make_agent(ASSISTED_POLICY_MODE)
        result, _ = evaluate_detailed([agent, agent], env, [17])
        self.assertEqual(result.completed, 1)
        self.assertEqual(env.mode_at_reset, ASSISTED_POLICY_MODE)


if __name__ == "__main__":
    unittest.main()

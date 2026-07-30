from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

from DQN.DQN_model import (  # noqa: E402
    LEARNED_POLICY_CONTRACT,
    LEGACY_LEARNED_POLICY_CONTRACT,
    N_ACTIONS,
    OBS_DIM,
    route_actions,
)
from DQN.DQN_train import (  # noqa: E402
    DEFAULT_ROUTE_AUX_WEIGHT,
    Agent,
    Config,
    Trainer,
    _environment_route_labels,
)
from DQN.load_model import load_agent  # noqa: E402
from env_bridge import CoopEnvBridge  # noqa: E402


def _fill(agent: Agent, count: int = 400, label: int = 1) -> None:
    rng = np.random.default_rng(0)
    for _ in range(count):
        agent.remember(
            rng.normal(size=OBS_DIM).astype(np.float32),
            int(rng.integers(N_ACTIONS)),
            float(rng.normal()),
            rng.normal(size=OBS_DIM).astype(np.float32),
            False,
            0.99,
            False,
            route_label=label,
        )


class TestTeacherLabelsSurviveLearnedMode(unittest.TestCase):
    """The bug this guards: learned mode zeroes route_dx/route_dy in the
    observation, so a label derived from the observation is always -1 and the
    auxiliary loss silently never fires. The label must come from the
    environment instead."""

    def test_observation_derived_labels_are_empty_in_learned_mode(self):
        bridge = CoopEnvBridge(seed=0, max_steps=20, policy_mode="learned")
        observations = bridge.reset(seed=0)
        derived = route_actions(
            torch.as_tensor(np.asarray(observations, dtype=np.float32))
        )
        # Not a defect in route_actions -- it reads features learned mode
        # deliberately withholds. This is why the loss takes an env label.
        self.assertTrue(bool((derived < 0).all()))

    def test_environment_supplies_labels_in_both_modes(self):
        for mode in ("learned", "assisted"):
            bridge = CoopEnvBridge(seed=0, max_steps=20, policy_mode=mode)
            bridge.reset(seed=0)
            labels = bridge.route_action_labels()
            self.assertEqual(len(labels), 2, mode)
            self.assertTrue(
                all(-1 <= value < N_ACTIONS for value in labels), mode
            )

    def test_labels_require_reset_first(self):
        bridge = CoopEnvBridge(seed=0, max_steps=20, policy_mode="learned")
        with self.assertRaisesRegex(RuntimeError, "call reset"):
            bridge.route_action_labels()

    def test_missing_accessor_degrades_instead_of_crashing(self):
        class Bare:
            pass

        self.assertEqual(_environment_route_labels(Bare()), (-1, -1))

    def test_wrong_label_count_is_rejected(self):
        class Wrong:
            def route_action_labels(self):
                return (0, 1, 2)

        with self.assertRaisesRegex(ValueError, "route labels"):
            _environment_route_labels(Wrong())


class TestAuxiliaryLossWiring(unittest.TestCase):
    def test_loss_is_active_in_learned_mode(self):
        torch.manual_seed(0)
        agent = Agent(
            device="cpu",
            replay_capacity=1000,
            policy_mode="learned",
            route_aux_weight=DEFAULT_ROUTE_AUX_WEIGHT,
        )
        _fill(agent)
        agent.learn_batch(64)
        metrics = agent.latest_learning_metrics
        self.assertGreater(metrics["route_aux_loss"], 0.0)
        self.assertNotAlmostEqual(
            metrics["total_loss"], metrics["td_loss"], places=9
        )

    def test_zero_weight_reproduces_td_only_training(self):
        torch.manual_seed(0)
        agent = Agent(
            device="cpu",
            replay_capacity=1000,
            policy_mode="learned",
            route_aux_weight=0.0,
        )
        _fill(agent)
        agent.learn_batch(64)
        metrics = agent.latest_learning_metrics
        self.assertEqual(metrics["route_aux_loss"], 0.0)
        self.assertAlmostEqual(
            metrics["total_loss"], metrics["td_loss"], places=9
        )

    def test_unlabelled_transitions_contribute_nothing(self):
        torch.manual_seed(0)
        agent = Agent(
            device="cpu",
            replay_capacity=1000,
            policy_mode="learned",
            route_aux_weight=DEFAULT_ROUTE_AUX_WEIGHT,
        )
        _fill(agent, label=-1)
        agent.learn_batch(64)
        self.assertEqual(agent.latest_learning_metrics["route_aux_loss"], 0.0)

    def test_negative_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            Agent(device="cpu", replay_capacity=1, route_aux_weight=-0.1)


class TestLabelsReachReplayFromEpisodes(unittest.TestCase):
    def test_a_learned_episode_stores_real_labels(self):
        torch.manual_seed(0)
        cfg = Config(
            max_steps=40,
            device="cpu",
            seed=0,
            policy_mode="learned",
            route_aux_weight=DEFAULT_ROUTE_AUX_WEIGHT,
        )
        env = CoopEnvBridge(seed=0, max_steps=40, policy_mode="learned")
        trainer = Trainer(env, cfg)
        for seed in range(4):
            trainer.run_episode(seed=seed)

        replay = trainer.learners[0].replay
        stored = replay.route_labels[: len(replay)]
        self.assertGreater(len(stored), 0)
        # The whole point: real episodes in learned mode produce usable labels.
        self.assertGreater(int((stored >= 0).sum()), 0)
        self.assertTrue(bool((stored < N_ACTIONS).all()))


class TestCheckpointCompatibility(unittest.TestCase):
    def test_contract_records_that_the_loss_now_runs(self):
        self.assertTrue(LEARNED_POLICY_CONTRACT["route_auxiliary_loss"])
        self.assertFalse(LEGACY_LEARNED_POLICY_CONTRACT["route_auxiliary_loss"])
        self.assertGreater(
            LEARNED_POLICY_CONTRACT["version"],
            LEGACY_LEARNED_POLICY_CONTRACT["version"],
        )

    def test_a_learned_v3_checkpoint_still_loads(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy_learned.pt"
            agent = Agent(
                hidden=(8,),
                replay_capacity=1,
                device="cpu",
                policy_mode="learned",
            )
            agent.save(str(path))
            checkpoint = torch.load(path, map_location="cpu")
            # Rewrite as the older contract, which differs only in a
            # training-time detail, not in how actions are chosen.
            checkpoint["policy"] = dict(LEGACY_LEARNED_POLICY_CONTRACT)
            torch.save(checkpoint, path)

            restored = load_agent(
                path, device="cpu", hidden=(8,), replay_capacity=1
            )
            self.assertEqual(restored.policy_mode, "learned")
            Agent(
                hidden=(8,),
                replay_capacity=1,
                device="cpu",
                policy_mode="learned",
            ).load(str(path))


if __name__ == "__main__":
    unittest.main()

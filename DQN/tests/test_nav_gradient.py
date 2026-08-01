from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

import torch  # noqa: E402

from DQN.DQN_model import (  # noqa: E402
    NAV_GRADIENT_CELLS,
    NAV_GRADIENT_UNREACHABLE,
    OBS_DIM,
    OBS_DIM_WITH_NAV_GRADIENT,
    observation_dim,
)
from DQN.DQN_train import (  # noqa: E402
    Config,
    Trainer,
    _clone_evaluation_env,
    evaluate,
)
from env_bridge import CoopEnvBridge  # noqa: E402


def _env(mode: str, **kwargs) -> CoopEnvBridge:
    return CoopEnvBridge(
        seed=0, max_steps=40, record_metrics=False, policy_mode=mode, **kwargs
    )


class TestObservationWidth(unittest.TestCase):
    def test_both_modes_leave_the_flow_map_off_by_default(self):
        # Training with it measured worse than without, and existing
        # checkpoints of both modes depend on this width.
        for mode in ("learned", "assisted"):
            env = _env(mode)
            self.assertFalse(env.nav_gradient, mode)
            self.assertEqual(env.obs_dim, OBS_DIM, mode)
            self.assertEqual(len(env.reset(seed=1)[0]), OBS_DIM, mode)

    def test_either_mode_can_opt_in(self):
        for mode in ("learned", "assisted"):
            env = _env(mode, nav_gradient=True)
            self.assertEqual(env.obs_dim, OBS_DIM_WITH_NAV_GRADIENT, mode)
            self.assertEqual(
                len(env.reset(seed=1)[0]), OBS_DIM_WITH_NAV_GRADIENT, mode
            )

    def test_the_helper_agrees_with_the_environment(self):
        for flag in (True, False):
            self.assertEqual(
                observation_dim(flag), _env("learned", nav_gradient=flag).obs_dim
            )

    def test_switching_policy_mode_never_resizes_a_live_observation(self):
        env = _env("learned", nav_gradient=True)
        env.reset(seed=1)
        before = len(env._obs(0))
        env.set_policy_mode("assisted")
        self.assertEqual(len(env._obs(0)), before)


class TestGradientValues(unittest.TestCase):
    def setUp(self):
        self.env = _env("learned", nav_gradient=True)
        self.env.reset(seed=3)

    def _field(self, agent: int = 0):
        return self.env._nav_gradient_view(agent)

    def test_it_has_one_value_per_visible_tile(self):
        self.assertEqual(len(self._field()), NAV_GRADIENT_CELLS)

    def test_the_agents_own_tile_is_flat(self):
        # Distance relative to where we stand, so the centre is zero by
        # construction; a non-zero centre means the field is misaligned.
        self.assertEqual(self._field()[NAV_GRADIENT_CELLS // 2], 0.0)

    def test_values_stay_within_the_clip(self):
        for value in self._field():
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_unreachable_tiles_read_as_maximally_uphill(self):
        field = self._field()
        self.assertIn(NAV_GRADIENT_UNREACHABLE, field)

    def test_some_direction_is_downhill_when_a_goal_is_reachable(self):
        # Otherwise the field carries no routing information at all.
        self.assertLess(min(self._field()), 0.0)

    def test_a_stranded_agent_reports_all_uphill_instead_of_guessing(self):
        self.env._ensure_navigation()
        self.env._nav_distance[0] = {}
        self.assertEqual(
            self._field(), [NAV_GRADIENT_UNREACHABLE] * NAV_GRADIENT_CELLS
        )


class TestEnvironmentsStayConsistent(unittest.TestCase):
    """A lane whose width disagrees with the agent is a silent shape bug."""

    def test_evaluation_clones_inherit_the_setting(self):
        for mode, flag in (("learned", True), ("learned", False), ("assisted", True)):
            env = _env(mode, nav_gradient=flag)
            clone = _clone_evaluation_env(env)
            label = f"{mode}/{flag}"
            self.assertEqual(clone.nav_gradient, env.nav_gradient, label)
            self.assertEqual(clone.obs_dim, env.obs_dim, label)

    def test_a_trainer_builds_a_network_matching_its_environment(self):
        for mode in ("learned", "assisted"):
            torch.manual_seed(0)
            env = _env(mode)
            trainer = Trainer(env, Config(max_steps=40, device="cpu", policy_mode=mode))
            self.assertEqual(trainer.learners[0].net.obs_dim, env.obs_dim, mode)

    def test_training_and_batched_evaluation_run_in_both_modes(self):
        for mode in ("learned", "assisted"):
            torch.manual_seed(0)
            env = _env(mode)
            trainer = Trainer(env, Config(max_steps=40, device="cpu", policy_mode=mode))
            trainer.run_episode(seed=0)
            result = evaluate(trainer.agents, env, [50, 51, 52])
            self.assertEqual(result.episodes, 3, mode)


class TestStaleCheckpoints(unittest.TestCase):
    """Resuming a pre-flow-map learned checkpoint must fail loudly."""

    def _trainer(self, nav_gradient: bool) -> Trainer:
        torch.manual_seed(0)
        env = _env("learned", nav_gradient=nav_gradient)
        return Trainer(
            env,
            Config(
                max_steps=40,
                device="cpu",
                policy_mode="learned",
                nav_gradient=nav_gradient,
            ),
        )

    def setUp(self):
        import tempfile

        self.path = str(Path(tempfile.mkdtemp()) / "agent_0.pt")
        self._trainer(False).learners[0].save(self.path)

    def test_a_narrower_checkpoint_is_rejected_by_both_load_paths(self):
        learner = self._trainer(True).learners[0]
        state = torch.load(self.path, map_location="cpu")
        for load in (
            lambda: learner.load(self.path),
            lambda: learner.load_learning_state(state),
        ):
            with self.assertRaises(ValueError) as caught:
                load()
            message = str(caught.exception)
            # The generic contract error sends people hunting; this one says
            # which field disagrees and how to proceed.
            self.assertIn(str(OBS_DIM), message)
            self.assertIn(str(OBS_DIM_WITH_NAV_GRADIENT), message)
            self.assertIn("--no-nav-gradient", message)

    def test_a_matching_checkpoint_still_loads(self):
        self._trainer(False).learners[0].load(self.path)


if __name__ == "__main__":
    unittest.main()

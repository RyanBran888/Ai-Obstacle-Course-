from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

from DQN.DQN_train import Agent  # noqa: E402
from DQN.load_model import (  # noqa: E402
    configure_environment,
    load_agent,
    load_policy,
)
from DQN.run_curriculum import _export_checkpoint_bundle  # noqa: E402


class TestCheckpointExports(unittest.TestCase):
    def make_checkpoint(self, path: Path) -> None:
        Agent(
            hidden=(8,),
            replay_capacity=1,
            device="cpu",
        ).save(str(path))

    def test_exports_three_identical_loadable_shared_policies(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "legacy_name.pt"
            self.make_checkpoint(canonical)

            exports = _export_checkpoint_bundle(canonical)
            self.assertEqual(
                set(exports),
                {"agent_0", "agent_1", "curriculum"},
            )
            self.assertEqual(
                {Path(export["path"]).name for export in exports.values()},
                {"agent_0.pt", "agent_1.pt", "curriculum.pt"},
            )
            self.assertEqual(
                {export["sha256"] for export in exports.values()},
                {exports["curriculum"]["sha256"]},
            )

            for export in exports.values():
                path = Path(export["path"])
                load_agent(
                    path,
                    device="cpu",
                    hidden=(8,),
                    replay_capacity=1,
                )
                load_policy(path, device="cpu", hidden=(8,))

            self.assertEqual(
                exports,
                _export_checkpoint_bundle(canonical),
            )

    def test_deduplicates_a_canonical_curriculum_filename(self):
        with TemporaryDirectory() as temporary:
            canonical = Path(temporary) / "curriculum.pt"
            self.make_checkpoint(canonical)

            exports = _export_checkpoint_bundle(canonical)
            self.assertEqual(
                Path(exports["curriculum"]["path"]),
                canonical,
            )
            self.assertTrue(Path(exports["agent_0"]["path"]).is_file())
            self.assertTrue(Path(exports["agent_1"]["path"]).is_file())

    def test_rejects_a_conflicting_existing_agent_export(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "curriculum.pt"
            conflicting = root / "agent_0.pt"
            self.make_checkpoint(canonical)
            conflicting.write_bytes(b"not the frozen checkpoint")

            with self.assertRaisesRegex(
                FileExistsError,
                "does not match the frozen checkpoint",
            ):
                _export_checkpoint_bundle(canonical)
            self.assertEqual(
                conflicting.read_bytes(),
                b"not the frozen checkpoint",
            )

    def test_manual_inference_helper_configures_the_environment(self):
        class Environment:
            policy_mode = None

            def set_policy_mode(self, mode):
                self.policy_mode = mode

        policy = type("Policy", (), {"policy_mode": "learned"})()
        env = Environment()
        self.assertIs(configure_environment(policy, env), env)
        self.assertEqual(env.policy_mode, "learned")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

import torch  # noqa: E402

from DQN.curriculum import CurriculumRunner, default_stages  # noqa: E402
from DQN.DQN_train import Config, Trainer  # noqa: E402
from env_bridge import CoopEnvBridge  # noqa: E402


def _runner(progress: Path, resume_from: Path | None = None) -> CurriculumRunner:
    torch.manual_seed(0)
    env = CoopEnvBridge(
        seed=0, max_steps=40, record_metrics=False, policy_mode="guided"
    )
    trainer = Trainer(env, Config(max_steps=40, device="cpu", policy_mode="guided"))
    return CurriculumRunner(
        trainer=trainer,
        stages=default_stages()[:1],
        progress_path=str(progress),
        resume_from=str(resume_from) if resume_from else None,
    )


class TestResumeRoundTrip(unittest.TestCase):
    """Resume has to work on an unmodified tree, or recovery is decorative."""

    def test_a_saved_state_resumes_under_identical_code(self):
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "run.progress.pt"
            written = _runner(progress)
            written._save_progress("training", None)
            self.assertTrue(progress.is_file())

            resumed = _runner(Path(tmp) / "run2.progress.pt", resume_from=progress)
            self.assertEqual(resumed.policy_mode, "guided")

    def test_a_changed_contract_is_rejected_with_the_real_reason(self):
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "run.progress.pt"
            written = _runner(progress)
            written._save_progress("training", None)

            payload = torch.load(progress, map_location="cpu", weights_only=False)
            saved = dict(payload["contract"])
            current = dict(saved)
            external = dict(saved.get("external") or {})
            external["source_sha256"] = {"DQN/env_bridge.py": "aaa"}
            saved["external"] = external
            current_external = dict(external)
            current_external["source_sha256"] = {"DQN/env_bridge.py": "bbb"}
            current["external"] = current_external
            current["retention_size"] = 999

            note = CurriculumRunner._route_aux_note(saved, current)
            # It must name the file that changed and the setting that differs,
            # not attribute every mismatch to the route auxiliary loss.
            self.assertIn("DQN/env_bridge.py", note)
            self.assertIn("retention_size", note)
            self.assertNotIn("route auxiliary loss", note)

    def test_matching_contracts_do_not_invent_a_cause(self):
        same = {"external": {"source_sha256": {"a.py": "x"}}, "run_seed": 0}
        note = CurriculumRunner._route_aux_note(same, dict(same))
        self.assertNotIn("Source files changed", note)


if __name__ == "__main__":
    unittest.main()

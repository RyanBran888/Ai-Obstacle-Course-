from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

import torch  # noqa: E402

from DQN.convert_qtable import (  # noqa: E402
    count_conflicts,
    dataset_from_keys,
    decode_key,
    extract_table,
    main,
    widen_rows,
)
from DQN.DQN_model import N_ACTIONS, OBS_DIM  # noqa: E402
from DQN.DQN_train import Agent  # noqa: E402
from DQN.load_model import load_agent, load_policy  # noqa: E402


def _observation_table(count: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        tuple(float(v) for v in rng.normal(size=OBS_DIM)): rng.normal(size=N_ACTIONS)
        for _ in range(count)
    }


class TestTableExtraction(unittest.TestCase):
    def test_reads_a_bare_mapping(self):
        table = extract_table({(0, 0): [1.0, 2.0]}, Path("t.pkl"))
        self.assertEqual(list(table), [(0, 0)])
        np.testing.assert_allclose(table[(0, 0)], [1.0, 2.0])

    def test_reads_a_wrapper_dict_and_action_named_rows(self):
        payload = {
            "q_table": {(1,): {"north": 0.5, "east": 0.25, "south": 0.0,
                               "west": 0.0, "interact": 0.0, "wait": 0.0}},
            "alpha": 0.1,
        }
        table = extract_table(payload, Path("t.pkl"))
        # Ordered by the action layout, not by dictionary insertion.
        np.testing.assert_allclose(table[(1,)], [0.5, 0.25, 0.0, 0.0, 0.0, 0.0])

    def test_reads_an_agent_object(self):
        class TabularAgent:
            def __init__(self):
                self.q = {(2, 3): np.array([1.0, 0.0])}

        table = extract_table(TabularAgent(), Path("t.pkl"))
        np.testing.assert_allclose(table[(2, 3)], [1.0, 0.0])

    def test_reads_a_dense_array(self):
        table = extract_table(np.zeros((3, 4, N_ACTIONS)), Path("t.pkl"))
        self.assertEqual(len(table), 12)
        self.assertIn((2, 3), table)

    def test_rejects_something_that_is_not_a_table(self):
        with self.assertRaisesRegex(ValueError, "could not find a Q-table"):
            extract_table("not a table", Path("t.pkl"))


class TestKeyDecoding(unittest.TestCase):
    def test_decodes_numeric_shapes(self):
        for key, expected in (
            ((1, 2, 3), [1.0, 2.0, 3.0]),
            ("(4, 5)", [4.0, 5.0]),
            (np.array([6.0]), [6.0]),
            (((1, 2), (3,)), [1.0, 2.0, 3.0]),  # nested keys are flattened
        ):
            decoded = decode_key(key)
            self.assertIsNotNone(decoded, key)
            assert decoded is not None
            np.testing.assert_allclose(decoded, expected)

    def test_refuses_opaque_keys(self):
        self.assertIsNone(decode_key("state-7"))
        self.assertIsNone(decode_key(object()))


class TestRowWidening(unittest.TestCase):
    def test_keeps_full_width_rows_and_drops_never_updated_entries(self):
        rows = np.array([[1.0, 0.0, 2.0, 3.0, 4.0, 5.0]])
        targets, weight, scored = widen_rows(rows)
        np.testing.assert_allclose(targets[0], rows[0])
        # The exact zero means "never updated", so it is in neither mask.
        self.assertFalse(weight[0, 1])
        self.assertFalse(scored[0, 1])

    def test_pads_narrow_rows_below_every_real_entry(self):
        rows = np.array([[1.0, 2.0, 3.0, 4.0]])
        targets, weight, scored = widen_rows(rows)
        self.assertEqual(targets.shape, (1, N_ACTIONS))
        self.assertTrue((targets[0, 4:] < rows.min()).all())
        # Padded columns are trained, so the fit cannot prefer them...
        self.assertTrue(weight[0, 4:].all())
        # ...but they are not scored, having no table opinion to agree with.
        self.assertFalse(scored[0, 4:].any())

    def test_trims_rows_wider_than_the_action_space(self):
        targets, _, _ = widen_rows(np.zeros((1, N_ACTIONS + 3)))
        self.assertEqual(targets.shape, (1, N_ACTIONS))


class TestDatasetFromKeys(unittest.TestCase):
    def test_full_observation_keys_are_used_verbatim(self):
        table = _observation_table(4)
        data = dataset_from_keys(table, allow_lossy=False, label="t")
        self.assertEqual(data.obs.shape, (4, OBS_DIM))
        self.assertFalse(data.lossy)
        self.assertIsNone(data.transform)
        np.testing.assert_allclose(data.obs[0], np.float32(list(table)[0]))

    def test_compact_keys_are_refused_without_allow_lossy(self):
        table = {(x, y): np.zeros(N_ACTIONS) + 1.0 for x in range(2) for y in range(2)}
        with self.assertRaisesRegex(ValueError, "--allow-lossy"):
            dataset_from_keys(table, allow_lossy=False, label="t")

    def test_compact_keys_are_one_hot_encoded_when_allowed(self):
        table = {(x, y): np.zeros(N_ACTIONS) + 1.0 for x in range(3) for y in range(4)}
        data = dataset_from_keys(table, allow_lossy=True, label="t")
        self.assertTrue(data.lossy)
        assert data.transform is not None
        self.assertEqual(data.transform["encoding"], "one_hot")
        # One column per level of each key dimension, each row a pair of ones.
        self.assertEqual(data.transform["width"], 3 + 4)
        np.testing.assert_allclose(data.obs.sum(axis=1), 2.0)

    def test_opaque_keys_name_the_way_out(self):
        table = {"state-0": np.ones(N_ACTIONS)}
        with self.assertRaisesRegex(ValueError, "--key-fn"):
            dataset_from_keys(table, allow_lossy=True, label="t")

    def test_mixed_key_widths_are_rejected(self):
        table = {(0, 0): np.ones(N_ACTIONS), (1, 1, 1): np.ones(N_ACTIONS)}
        with self.assertRaisesRegex(ValueError, "mixed widths"):
            dataset_from_keys(table, allow_lossy=True, label="t")


class TestConflictCounting(unittest.TestCase):
    def test_finds_identical_observations_wanting_different_actions(self):
        key = tuple(0.0 for _ in range(OBS_DIM))
        first = dataset_from_keys({key: np.array([1.0, -1.0, 0.5, 0.5, 0.5, 0.5])},
                                  allow_lossy=False, label="a")
        second = dataset_from_keys({key: np.array([-1.0, 1.0, 0.5, 0.5, 0.5, 0.5])},
                                   allow_lossy=False, label="b")
        from DQN.convert_qtable import merge

        conflicting, duplicated = count_conflicts(merge([first, second]))
        self.assertEqual((conflicting, duplicated), (1, 1))

    def test_agreeing_duplicates_are_not_conflicts(self):
        table = _observation_table(3)
        data = dataset_from_keys(table, allow_lossy=False, label="t")
        from DQN.convert_qtable import merge

        conflicting, duplicated = count_conflicts(merge([data, data]))
        self.assertEqual(conflicting, 0)
        self.assertEqual(duplicated, 3)


class TestEndToEnd(unittest.TestCase):
    """A whole conversion, checked through the loaders the project ships."""

    def _convert(self, root: Path, *extra: str) -> Path:
        source = root / "qtable_curriculum_agent0.pkl"
        with source.open("wb") as handle:
            pickle.dump(_observation_table(24), handle)
        out = root / "converted"
        code = main([
            str(source), "--out", str(out), "--device", "cpu",
            "--epochs", "120", "--batch", "16", "--quiet",
            "--min-agreement", "0", *extra,
        ])
        self.assertEqual(code, 0)
        return out

    def test_writes_all_three_names_and_they_load(self):
        with TemporaryDirectory() as temporary:
            out = self._convert(Path(temporary))
            names = ["agent_0.pt", "agent_1.pt", "curriculum.pt"]
            for name in names:
                path = out / name
                self.assertTrue(path.is_file(), name)
                # Every door the project opens checkpoints with.
                load_agent(path, device="cpu")
                load_policy(path, device="cpu")
                Agent(device="cpu", replay_capacity=1).load(str(path))
            self.assertTrue((out / "conversion_report.json").is_file())

    def test_a_single_table_produces_one_shared_policy(self):
        with TemporaryDirectory() as temporary:
            out = self._convert(Path(temporary))
            digests = {
                (out / name).read_bytes()
                for name in ("agent_0.pt", "agent_1.pt", "curriculum.pt")
            }
            self.assertEqual(len(digests), 1)

    def test_provenance_is_recorded_in_the_checkpoint(self):
        with TemporaryDirectory() as temporary:
            out = self._convert(Path(temporary))
            checkpoint = torch.load(out / "curriculum.pt", map_location="cpu")
            source = checkpoint["qtable_source"]
            self.assertFalse(checkpoint["qtable_lossy"])
            self.assertEqual(source["states"], 24)
            self.assertIn("fidelity", source)
            self.assertTrue(source["converted_from"][0].endswith(".pkl"))

    def test_a_failed_fit_writes_nothing_at_all(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "qtable_curriculum_agent0.pkl"
            with source.open("wb") as handle:
                pickle.dump(_observation_table(8), handle)
            out = root / "converted"
            with self.assertRaises(SystemExit):
                main([
                    str(source), "--out", str(out), "--device", "cpu",
                    "--epochs", "1", "--quiet",
                    "--min-agreement", "1.01",  # unreachable on purpose
                ])
            self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

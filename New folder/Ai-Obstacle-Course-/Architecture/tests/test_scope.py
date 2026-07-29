"""Guard tests for the project's stated scope: environment only, no agents.

The brief is explicit that this deliverable must not contain agents, policies,
rewards, training loops, neural networks, or pathfinding for agents. These
tests fail if any of that creeps in, which makes the boundary enforceable
rather than merely documented.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import coop_env  # noqa: E402
from coop_env import EnvironmentSession, GenerationConfig, RoomGenerator  # noqa: E402
from coop_env.interfaces import MultiAgentEnvironmentAdapter, RewardFunction  # noqa: E402

PACKAGE_ROOT = Path(coop_env.__file__).resolve().parent

#: Third-party machine-learning and simulation stacks that must not be imported.
FORBIDDEN_IMPORTS = {
    "gym", "gymnasium", "pettingzoo", "mlagents", "mlagents_envs",
    "torch", "tensorflow", "keras", "jax", "flax", "stable_baselines3",
    "sklearn", "numpy", "scipy", "pygame",
}

#: Class names that would indicate an agent implementation had been added.
FORBIDDEN_CLASS_NAMES = {
    "Agent", "Policy", "QNetwork", "ActorCritic", "Trainer", "Learner",
    "ReplayBuffer", "Rollout", "PolicyNetwork", "Brain", "Controller",
}


def source_files() -> list[Path]:
    """Real modules only -- macOS writes `._name` sidecars on non-HFS volumes."""
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if not path.name.startswith(".")
    )


class TestNoThirdPartyDependencies(unittest.TestCase):
    def test_package_imports_only_the_standard_library(self):
        offenders: list[str] = []
        for path in source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    root = name.split(".")[0]
                    if root in FORBIDDEN_IMPORTS:
                        offenders.append(f"{path.name} imports {name}")
        self.assertEqual(offenders, [], f"unexpected dependencies: {offenders}")


class TestNoAgentImplementations(unittest.TestCase):
    def test_no_agent_or_policy_classes_are_defined(self):
        offenders: list[str] = []
        for path in source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in FORBIDDEN_CLASS_NAMES:
                    offenders.append(f"{path.name}:{node.lineno} defines {node.name}")
        self.assertEqual(offenders, [], f"agent-shaped classes found: {offenders}")

    def test_the_session_offers_no_step_method(self):
        self.assertFalse(
            hasattr(EnvironmentSession, "step"),
            "stepping implies agents acting; the session only advances the clock",
        )

    def test_placeholders_stay_unimplemented(self):
        adapter = MultiAgentEnvironmentAdapter()
        for call in (
            lambda: adapter.reset(),
            lambda: adapter.step(None),
            lambda: adapter.observation_space(0),
            lambda: adapter.action_space(0),
            lambda: adapter.render(),
        ):
            with self.assertRaises(NotImplementedError):
                call()

    def test_reward_function_is_a_placeholder(self):
        room = RoomGenerator(GenerationConfig.preset("easy")).generate(1)
        session = EnvironmentSession(GenerationConfig.preset("easy"))
        session.load(room)
        with self.assertRaises(NotImplementedError):
            RewardFunction()(room, session.state, 0)

    def test_rooms_carry_no_stored_solution(self):
        """The generator must not leave a walkthrough behind in the room."""
        generator = RoomGenerator(GenerationConfig.preset("brutal"))
        banned = {"solution", "walkthrough", "path", "route", "plan", "actions", "moves"}
        for seed in range(10):
            room = generator.generate(seed)
            leaked = banned & set(room.metadata)
            self.assertEqual(leaked, set(), f"seed {seed} leaked {leaked}")

    def test_no_reward_values_are_attached_to_the_world(self):
        room = RoomGenerator(GenerationConfig.preset("hard")).generate(2)
        for entity in room.entities:
            fields = getattr(entity, "__slots__", ()) or ()
            for name in fields:
                self.assertNotIn(
                    "reward", str(name).lower(), f"{entity.id} carries a reward field"
                )


class TestPublicSurface(unittest.TestCase):
    def test_exports_resolve(self):
        for name in coop_env.__all__:
            self.assertTrue(hasattr(coop_env, name), f"missing export {name}")

    def test_agent_slot_count_is_two(self):
        self.assertEqual(coop_env.AGENT_SLOTS, 2)

    def test_spawns_are_positions_not_actors(self):
        """A spawn marks a tile; it has no state, inventory, or behaviour."""
        room = RoomGenerator(GenerationConfig.preset("standard")).generate(1)
        for spawn in room.spawns:
            self.assertEqual(
                set(type(spawn).__dataclass_fields__), {"id", "pos", "index"}
            )


if __name__ == "__main__":
    unittest.main()

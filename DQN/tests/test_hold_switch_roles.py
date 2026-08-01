from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "Architecture"))

from DQN.curriculum import default_stages  # noqa: E402
from DQN.DQN_model import GLOBAL_NAMES  # noqa: E402
from env_bridge import N_AGENTS, CoopEnvBridge, SwitchMode  # noqa: E402

WAIT = 5
INTERACT = 4

#: The room the curriculum stalled on: both agents stood on the lever and the
#: planner returned "no route" for both until the step limit.
STUCK_ROOM_SEED = 7625850530215616496


def _stage(name: str):
    return next(s for s in default_stages() if s.name == name)


def _env(name: str) -> CoopEnvBridge:
    return CoopEnvBridge(
        _stage(name).config,
        seed=0,
        max_steps=200,
        record_metrics=False,
        policy_mode="guided",
    )


def _follow_planner(env, seed: int) -> tuple[bool, int]:
    """Play a room using only the planner's own route, as guided is taught."""
    env.reset(seed=seed)
    base = env.obs_dim - len(GLOBAL_NAMES)
    stalled = 0
    for step in range(200):
        labels = env.route_action_labels()
        if all(a < 0 for a in labels):
            stalled += 1
        actions = []
        for i, a in enumerate(labels):
            if a >= 0:
                actions.append(a)
                continue
            obs = env._obs(i)
            can_interact = obs[base + GLOBAL_NAMES.index("can_interact")] > 0.5
            actions.append(INTERACT if can_interact else WAIT)
        _, _, done, cut, _ = env.step(tuple(actions))
        if done or cut:
            return bool(done), stalled
    return False, stalled


class TestHoldSwitchOwnership(unittest.TestCase):
    def test_every_hold_switch_has_exactly_one_owner(self):
        """Unowned levers are claimed by every agent, which strands them all."""
        for name in ("hold_switch_door", "crate_hold_switch", "paired_levers"):
            env = _env(name)
            for seed in range(6):
                env.reset(seed=seed)
                roles = env._switch_roles
                for switch in env.room.switches:
                    if switch.mode is not SwitchMode.HOLD:
                        continue
                    self.assertIn(switch.id, roles, f"{name} seed {seed}")
                    self.assertIn(roles[switch.id], range(N_AGENTS))

    def test_only_the_owner_is_sent_to_the_lever(self):
        env = _env("hold_switch_door")
        env.reset(seed=STUCK_ROOM_SEED)
        owner = env._switch_roles["switch_0"]
        targets = [
            {t[0].id for t in env._task_targets(i)} for i in range(N_AGENTS)
        ]
        self.assertIn("switch_0", targets[owner])
        for other in range(N_AGENTS):
            if other != owner:
                self.assertNotIn("switch_0", targets[other])


class TestPlannerFinishesHoldRooms(unittest.TestCase):
    """Guided is trained on the planner, so a stalled planner stalls training."""

    def test_the_room_the_curriculum_stalled_on_now_completes(self):
        solved, stalled = _follow_planner(_env("hold_switch_door"), STUCK_ROOM_SEED)
        self.assertTrue(solved)
        self.assertLess(stalled, 20, "planner still stalls with no route")

    def test_hold_rooms_do_not_strand_both_agents(self):
        for name in ("hold_switch_door", "crate_hold_switch"):
            env = _env(name)
            for seed in range(8):
                solved, stalled = _follow_planner(env, seed)
                # A long stretch where neither agent has a route is the exact
                # signature of the stall, whether or not the room is solved.
                self.assertLess(stalled, 50, f"{name} seed {seed} stranded")


if __name__ == "__main__":
    unittest.main()

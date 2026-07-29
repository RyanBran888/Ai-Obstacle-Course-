"""Episode lifecycle and the four reset behaviours."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import solvable_key_room  # noqa: E402

from coop_env import EnvironmentSession, GenerationConfig  # noqa: E402


class TestResetBehaviours(unittest.TestCase):
    def setUp(self):
        self.session = EnvironmentSession(
            GenerationConfig.preset("standard"), master_seed=4242
        )

    def test_reset_generates_a_new_room_each_time(self):
        seeds = []
        layouts = []
        for _ in range(6):
            self.session.reset()
            seeds.append(self.session.seed)
            layouts.append(tuple(self.session.room.terrain.to_list()))
        self.assertEqual(len(set(seeds)), 6)
        self.assertEqual(len(set(layouts)), 6)

    def test_reset_with_seed_rebuilds_the_same_room(self):
        self.session.reset(seed=6161)
        first = tuple(self.session.room.terrain.to_list())
        self.session.reset()
        self.session.reset(seed=6161)
        self.assertEqual(tuple(self.session.room.terrain.to_list()), first)

    def test_same_room_reset_keeps_the_layout(self):
        self.session.reset()
        room = self.session.room
        state = self.session.state
        for key in room.keys:
            state.collect_key(key.id)
        state.advance(20)

        self.session.reset(same_room=True)
        self.assertIs(self.session.room, room)
        self.assertEqual(self.session.state.tick, 0)
        self.assertEqual(self.session.state.keys_collected, set())

    def test_reset_state_keeps_the_episode(self):
        self.session.reset()
        index = self.session.episode_index
        self.session.state.advance(5)
        self.session.reset_state()
        self.assertEqual(self.session.state.tick, 0)
        self.assertEqual(self.session.episode_index, index)

    def test_reroll_keeps_the_seed(self):
        self.session.reset(seed=808)
        self.session.reset(reroll=True)
        self.assertEqual(self.session.seed, 808)

    def test_episode_index_advances(self):
        self.assertEqual(self.session.episode_index, -1)
        for expected in range(3):
            self.session.reset()
            self.assertEqual(self.session.episode_index, expected)


class TestSessionGuards(unittest.TestCase):
    def test_state_before_reset_raises(self):
        session = EnvironmentSession(GenerationConfig.preset("easy"))
        with self.assertRaises(RuntimeError):
            _ = session.room
        with self.assertRaises(RuntimeError):
            _ = session.state
        with self.assertRaises(RuntimeError):
            session.reset(same_room=True)

    def test_load_accepts_an_external_room(self):
        session = EnvironmentSession(GenerationConfig())
        room = solvable_key_room()
        session.load(room)
        self.assertIs(session.room, room)
        self.assertEqual(session.state.tick, 0)

    def test_history_and_stats_are_recorded(self):
        session = EnvironmentSession(GenerationConfig.preset("easy"), master_seed=1)
        for _ in range(4):
            session.reset()
        self.assertEqual(len(session.history), 4)
        stats = session.stats()
        self.assertEqual(stats["episodes"], 4)
        self.assertEqual(stats["fallbacks"], 0)
        self.assertGreaterEqual(stats["mean_attempts"], 1.0)

    def test_reset_hooks_fire(self):
        session = EnvironmentSession(GenerationConfig.preset("easy"), master_seed=2)
        seen: list[int] = []
        session.on_reset.append(lambda room, state: seen.append(room.seed))
        session.reset()
        session.reset(same_room=True)
        self.assertEqual(len(seen), 2)

    def test_advance_time_only_moves_the_clock(self):
        session = EnvironmentSession(GenerationConfig.preset("standard"), master_seed=3)
        session.reset()
        session.advance_time(10)
        self.assertEqual(session.state.tick, 10)
        self.assertEqual(session.state.keys_collected, set())


if __name__ == "__main__":
    unittest.main()

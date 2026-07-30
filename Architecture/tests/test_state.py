"""Mechanism behaviour and episode reset."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import (  # noqa: E402
    LEFT_TILE,
    LEFT_TILE_B,
    hold_switch_room,
    paired_lever_room,
    solvable_key_room,
    two_region_room,
)

from coop_env import GenerationConfig, RoomGenerator  # noqa: E402
from coop_env.entities import (  # noqa: E402
    Key,
    PushableBlock,
    Switch,
    SwitchMode,
    TemporaryBridge,
    WipeoutBall,
    WipeoutBallSize,
)
from coop_env.requirements import SwitchRequirement, TriggerMode  # noqa: E402
from coop_env.state import EpisodeState  # noqa: E402
from coop_env.tiles import Tile  # noqa: E402
from coop_env.utils.geometry import Vec2  # noqa: E402


class TestDoorLogic(unittest.TestCase):
    def test_key_door_opens_and_stays_open(self):
        state = EpisodeState.from_room(solvable_key_room())
        self.assertFalse(state.is_door_open("door_0"))
        state.collect_key("key_0")
        self.assertTrue(state.is_door_open("door_0"))
        state.advance(50)
        self.assertTrue(state.is_door_open("door_0"), "latching doors do not re-lock")

    def test_hold_door_closes_when_released(self):
        state = EpisodeState.from_room(hold_switch_room())
        state.set_switch("switch_0", True)
        self.assertTrue(state.is_door_open("door_0"))
        state.set_switch("switch_0", False)
        self.assertFalse(state.is_door_open("door_0"))

    def test_paired_levers_need_both(self):
        state = EpisodeState.from_room(paired_lever_room())
        state.set_switch("switch_0", True)
        self.assertFalse(state.is_door_open("door_0"))
        state.set_switch("switch_1", True)
        self.assertTrue(state.is_door_open("door_0"))

    def test_latched_paired_door_stays_open_after_release(self):
        state = EpisodeState.from_room(paired_lever_room())
        state.set_switch("switch_0", True)
        state.set_switch("switch_1", True)
        state.set_switch("switch_0", False)
        self.assertTrue(state.is_door_open("door_0"))

    def test_timed_door_relocks(self):
        room = two_region_room(
            door_requirement=SwitchRequirement(("switch_0",)),
            extra_entities=(
                Switch(id="switch_0", pos=LEFT_TILE, mode=SwitchMode.TOGGLE),
            ),
        )
        doors = list(room.doors)
        timed = type(doors[0])(
            id=doors[0].id, pos=doors[0].pos, requirement=doors[0].requirement,
            latching=True, timer=5, horizontal=doors[0].horizontal,
        )
        room = type(room)(
            seed=room.seed, config=room.config, terrain=room.terrain,
            entities=tuple(e for e in room.entities if e.id != "door_0") + (timed,),
            topology=room.topology,
        )
        state = EpisodeState.from_room(room)
        state.set_switch("switch_0", True)
        self.assertTrue(state.is_door_open("door_0"))
        self.assertEqual(state.door_timer_remaining("door_0"), 5)
        state.advance(4)
        self.assertTrue(state.is_door_open("door_0"))
        state.advance(1)
        self.assertFalse(state.is_door_open("door_0"), "timer should expire")
        self.assertTrue(state.door_needs_rearm("door_0"))
        state.set_switch("switch_0", False)
        self.assertFalse(state.door_needs_rearm("door_0"))
        state.set_switch("switch_0", True)
        self.assertTrue(state.is_door_open("door_0"))
        self.assertEqual(state.door_timer_remaining("door_0"), 5)

    def test_oneshot_switch_cannot_be_turned_off(self):
        room = two_region_room(
            door_requirement=SwitchRequirement(("switch_0",)),
            extra_entities=(Switch(id="switch_0", pos=LEFT_TILE, mode=SwitchMode.ONESHOT),),
        )
        state = EpisodeState.from_room(room)
        state.set_switch("switch_0", True)
        state.set_switch("switch_0", False)
        self.assertTrue(state.is_switch_active("switch_0"))

    def test_only_the_assigned_agent_collects_a_key(self):
        room = two_region_room(
            extra_entities=(
                Key(id="owned_key", pos=LEFT_TILE, agent_index=0),
            ),
        )
        state = EpisodeState.from_room(room)
        self.assertFalse(state.collect_key("owned_key", agent_index=1))
        self.assertFalse(state.is_key_collected("owned_key"))
        self.assertTrue(state.collect_key("owned_key", agent_index=0))
        self.assertTrue(state.is_key_collected("owned_key"))


class TestCrateOnLever(unittest.TestCase):
    def test_a_crate_holds_a_lever_down(self):
        """Object transportation: park a crate on a lever to free up an agent."""
        room = two_region_room(
            door_requirement=SwitchRequirement(("switch_0",), TriggerMode.ALL),
            latching=False,
            extra_entities=(
                Switch(id="switch_0", pos=LEFT_TILE, mode=SwitchMode.HOLD,
                       controls=("door_0",)),
                PushableBlock(id="block_0", pos=LEFT_TILE_B),
            ),
        )
        state = EpisodeState.from_room(room)
        self.assertFalse(state.is_switch_active("switch_0"))
        state.place_block("block_0", LEFT_TILE)
        self.assertTrue(state.is_switch_active("switch_0"))
        self.assertTrue(state.is_door_open("door_0"))

        state.place_block("block_0", LEFT_TILE_B)
        self.assertFalse(state.is_door_open("door_0"), "lever releases with the crate")


class TestBridgeTiming(unittest.TestCase):
    def test_bridge_duty_cycle(self):
        bridge = TemporaryBridge(id="b", pos=Vec2(0, 0), tiles=(Vec2(0, 0),), period=4, on_ticks=2)
        self.assertEqual([bridge.is_solid_at(t) for t in range(6)],
                         [True, True, False, False, True, True])

    def test_state_exposes_bridge_phase_and_hazard_transition(self):
        bridge = TemporaryBridge(
            id="bridge_0",
            pos=LEFT_TILE,
            tiles=(LEFT_TILE,),
            period=4,
            on_ticks=2,
        )
        room = two_region_room(extra_entities=(bridge,))
        room.terrain[LEFT_TILE] = Tile.HAZARD_WATER
        state = EpisodeState.from_room(room)
        self.assertTrue(state.bridge_is_solid("bridge_0"))
        self.assertEqual(state.bridge_ticks_until_change("bridge_0"), 2)
        self.assertTrue(state.is_walkable(LEFT_TILE))
        state.advance(2)
        self.assertFalse(state.bridge_is_solid("bridge_0"))
        self.assertTrue(state.is_hazardous(LEFT_TILE))


class TestWipeoutTiming(unittest.TestCase):
    def test_normal_ball_moves_left_to_right_and_back(self):
        track = tuple(Vec2(x, 2) for x in range(1, 8))
        ball = WipeoutBall(id="ball", pos=track[0], track=track)
        self.assertEqual(
            [ball.position_at(t).x for t in range(13)],
            [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1],
        )

    def test_big_ball_has_an_eleven_tile_track_and_large_hitbox(self):
        track = tuple(Vec2(x, 3) for x in range(1, 12))
        ball = WipeoutBall(
            id="big",
            pos=track[0],
            track=track,
            size=WipeoutBallSize.BIG,
        )
        self.assertEqual(ball.expected_track_length, 11)
        self.assertEqual(len(ball.collision_tiles_at(0)), 9)


class TestWalkability(unittest.TestCase):
    def test_closed_door_blocks_and_open_door_does_not(self):
        room = solvable_key_room()
        state = EpisodeState.from_room(room)
        door = room.doors[0]
        self.assertFalse(state.is_walkable(door.pos))
        state.collect_key("key_0")
        self.assertTrue(state.is_walkable(door.pos))

    def test_crate_blocks_its_tile(self):
        room = two_region_room(extra_entities=(PushableBlock(id="block_0", pos=LEFT_TILE),))
        state = EpisodeState.from_room(room)
        self.assertFalse(state.is_walkable(LEFT_TILE))
        state.place_block("block_0", LEFT_TILE_B)
        self.assertTrue(state.is_walkable(LEFT_TILE))


class TestReset(unittest.TestCase):
    def test_reset_restores_every_mechanism(self):
        generator = RoomGenerator(GenerationConfig.preset("brutal"))
        room = generator.generate(11)
        state = EpisodeState.from_room(room)
        before = state.snapshot()

        for key in room.keys:
            state.collect_key(key.id)
        for switch in room.switches:
            state.set_switch(switch.id, True)
        for checkpoint in room.checkpoints:
            state.reach_checkpoint(checkpoint.id)
        for block in room.blocks:
            state.place_block(block.id, Vec2(1, 1))
        state.advance(37)

        self.assertNotEqual(state.snapshot(), before)
        state.reset()
        self.assertEqual(state.snapshot(), before)
        self.assertEqual(state.tick, 0)
        self.assertFalse(state.exit_open)

    def test_snapshot_round_trip(self):
        room = RoomGenerator(GenerationConfig.preset("standard")).generate(5)
        state = EpisodeState.from_room(room)
        state.advance(9)
        for key in room.keys:
            state.collect_key(key.id)
        saved = state.snapshot()

        state.reset()
        state.restore(saved)
        self.assertEqual(state.snapshot(), saved)

    def test_unknown_mechanism_ids_are_rejected(self):
        state = EpisodeState.from_room(solvable_key_room())
        with self.assertRaises(KeyError):
            state.collect_key("nope")
        with self.assertRaises(KeyError):
            state.set_switch("nope", True)

    def test_advance_cannot_go_backwards(self):
        state = EpisodeState.from_room(solvable_key_room())
        with self.assertRaises(ValueError):
            state.advance(-1)


if __name__ == "__main__":
    unittest.main()

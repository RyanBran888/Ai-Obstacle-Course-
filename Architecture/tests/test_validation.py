"""The validator must reject rooms the generator would never produce.

A validator that only ever sees good input proves nothing, so these tests feed
it deliberately broken rooms and check it says no for the right reason.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import (  # noqa: E402
    DOORWAY,
    LEFT_TILE,
    LEFT_TILE_B,
    RIGHT_TILE,
    RIGHT_TILE_B,
    hold_switch_room,
    paired_plate_room,
    sealed_key_room,
    solvable_key_room,
    two_region_room,
)

from coop_env import GenerationConfig, RoomGenerator  # noqa: E402
from coop_env.entities import (  # noqa: E402
    Key,
    PressurePlate,
    PushableBlock,
    Switch,
    SwitchMode,
)
from coop_env.requirements import (  # noqa: E402
    CompositeRequirement,
    KeyRequirement,
    PlateRequirement,
    SwitchRequirement,
    TriggerMode,
)
from coop_env.tiles import Tile  # noqa: E402
from coop_env.utils.geometry import Vec2  # noqa: E402
from coop_env.validation import analyse, build_connectivity, validate_room  # noqa: E402


def codes(report) -> set[str]:
    return {issue.code for issue in report.errors}


class TestAcceptsGoodRooms(unittest.TestCase):
    def test_simple_key_room_is_valid(self):
        report = validate_room(solvable_key_room())
        self.assertTrue(report.ok, report.report_lines())

    def test_generated_rooms_pass(self):
        generator = RoomGenerator(GenerationConfig.preset("standard"))
        for seed in range(15):
            self.assertTrue(validate_room(generator.generate(seed)).ok)


class TestRejectsUnsolvable(unittest.TestCase):
    def test_key_sealed_behind_its_own_door(self):
        report = validate_room(sealed_key_room())
        self.assertFalse(report.ok)
        self.assertIn("unsolvable", codes(report))

    def test_exit_requires_an_unreachable_trigger(self):
        room = two_region_room(
            door_requirement=KeyRequirement(("key_0",)),
            exit_requirement=SwitchRequirement(("switch_0",)),
            extra_entities=(
                Key(id="key_0", pos=RIGHT_TILE, opens=("door_0",)),
                Switch(id="switch_0", pos=RIGHT_TILE_B, mode=SwitchMode.ONESHOT),
            ),
        )
        report = validate_room(room)
        self.assertFalse(report.ok)

    def test_requirement_pointing_at_a_missing_object(self):
        room = two_region_room(door_requirement=KeyRequirement(("ghost_key",)))
        report = validate_room(room)
        self.assertFalse(report.ok)
        self.assertIn("dangling_requirement", codes(report))


class TestRejectsBadPlacement(unittest.TestCase):
    def test_spawn_inside_a_wall(self):
        room = two_region_room(spawns=(Vec2(0, 0), Vec2(2, 8)))
        report = validate_room(room)
        self.assertFalse(report.ok)
        self.assertTrue({"spawn_terrain", "bad_terrain"} & codes(report))

    def test_spawn_on_a_hazard(self):
        room = two_region_room()
        room.terrain[room.spawns[0].pos] = Tile.HAZARD_LAVA
        report = validate_room(room)
        self.assertFalse(report.ok)
        self.assertTrue({"spawn_hazard", "spawn_terrain", "bad_terrain"} & codes(report))

    def test_both_spawns_on_one_tile(self):
        room = two_region_room(spawns=(LEFT_TILE, LEFT_TILE))
        report = validate_room(room)
        self.assertFalse(report.ok)
        self.assertTrue({"spawn_overlap", "stacked"} & codes(report))

    def test_stacked_objects_are_caught(self):
        room = two_region_room(
            extra_entities=(
                Key(id="key_0", pos=LEFT_TILE),
                Switch(id="switch_0", pos=LEFT_TILE, mode=SwitchMode.TOGGLE),
            )
        )
        report = validate_room(room)
        self.assertFalse(report.ok)
        self.assertIn("stacked", codes(report))

    def test_object_sealed_in_solid_rock(self):
        room = two_region_room(extra_entities=(Key(id="key_0", pos=Vec2(0, 5)),))
        report = validate_room(room)
        self.assertFalse(report.ok)
        self.assertTrue({"bad_terrain", "orphan_entity"} & codes(report))


class TestCooperativeAnalysis(unittest.TestCase):
    def test_paired_plates_are_flagged_as_cooperative(self):
        room = paired_plate_room()
        report = validate_room(room)
        self.assertTrue(report.ok, report.report_lines())
        self.assertEqual(len(report.solvability.cooperative_clusters), 1)

    def test_hold_lever_is_one_way_and_cooperative(self):
        room = hold_switch_room()
        report = validate_room(room)
        self.assertTrue(report.ok, report.report_lines())
        solvability = report.solvability
        self.assertEqual(len(solvability.one_way_clusters), 1)
        self.assertEqual(len(solvability.cooperative_clusters), 1)
        # either slot could be the one to go through...
        self.assertEqual(len(solvability.exit_reachable_by), 2)
        # ...but never both at once, because someone has to hold the lever
        self.assertFalse(solvability.exit_jointly_reachable)
        self.assertNotIn(1, solvability.joint_reachable)

    def test_hold_lever_room_fails_when_both_agents_must_exit(self):
        config = GenerationConfig(exit_requires_both_agents=True)
        room = hold_switch_room()
        room = type(room)(
            seed=room.seed,
            config=config,
            terrain=room.terrain,
            entities=room.entities,
            topology=room.topology,
            shape=room.shape,
        )
        report = validate_room(room)
        self.assertFalse(report.ok)

    def test_a_lone_key_door_is_not_cooperative(self):
        report = validate_room(solvable_key_room())
        self.assertEqual(len(report.solvability.cooperative_clusters), 0)
        self.assertTrue(report.solvability.exit_jointly_reachable)

    def test_paired_plate_room_still_lets_both_agents_through(self):
        """A latching co-op door stays open, so the pair is not split up."""
        report = validate_room(paired_plate_room())
        self.assertTrue(report.solvability.exit_jointly_reachable)

    def test_two_plates_out_of_reach_of_two_agents(self):
        """Three simultaneous plates cannot be covered by two agent slots."""
        room = two_region_room(
            door_requirement=PlateRequirement(
                ("plate_0", "plate_1", "plate_2"), TriggerMode.SIMULTANEOUS
            ),
            exit_requirement=KeyRequirement(("key_0",)),
            extra_entities=(
                PressurePlate(id="plate_0", pos=LEFT_TILE, group="trio"),
                PressurePlate(id="plate_1", pos=LEFT_TILE_B, group="trio"),
                PressurePlate(id="plate_2", pos=Vec2(5, 5), group="trio"),
                Key(id="key_0", pos=RIGHT_TILE, opens=("exit",)),
            ),
        )
        report = validate_room(room)
        self.assertFalse(report.ok, "three simultaneous plates need three agents")


class TestConnectivityModel(unittest.TestCase):
    def test_regions_are_split_by_doors(self):
        model = build_connectivity(solvable_key_room())
        self.assertEqual(len(model.regions), 2)
        self.assertEqual(len(model.clusters), 1)

    def test_crate_is_attributed_to_the_surrounding_region(self):
        room = two_region_room(extra_entities=(PushableBlock(id="block_0", pos=LEFT_TILE),))
        model = build_connectivity(room)
        self.assertIsNotNone(model.region_of_entity("block_0"))
        self.assertTrue(validate_room(room).ok)

    def test_touching_doors_combine_into_one_passage(self):
        room = two_region_room(door_requirement=KeyRequirement(("key_0",)))
        model = build_connectivity(room)
        cluster = next(iter(model.clusters.values()))
        self.assertIn(DOORWAY, cluster.tiles)
        self.assertEqual(set(cluster.regions), {0, 1})

    def test_composite_requirement_needs_every_part(self):
        room = two_region_room(
            exit_requirement=CompositeRequirement(
                (KeyRequirement(("key_0",)), SwitchRequirement(("switch_0",))),
                TriggerMode.ALL,
            ),
            extra_entities=(
                Key(id="key_0", pos=LEFT_TILE),
                Switch(id="switch_0", pos=LEFT_TILE_B, mode=SwitchMode.ONESHOT),
            ),
        )
        self.assertTrue(validate_room(room).ok)


class TestSolvabilityDirectly(unittest.TestCase):
    def test_analyse_reports_reachable_regions(self):
        room = solvable_key_room()
        result = analyse(room, build_connectivity(room))
        self.assertTrue(result.solvable)
        self.assertEqual(len(result.reachable), 2)
        self.assertEqual(result.reachable[0], {0, 1})

    def test_analyse_stops_at_a_sealed_door(self):
        room = sealed_key_room()
        result = analyse(room, build_connectivity(room))
        self.assertFalse(result.solvable)
        self.assertEqual(result.reachable[0], {0})
        self.assertEqual(len(result.blocked_clusters), 1)


if __name__ == "__main__":
    unittest.main()

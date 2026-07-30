"""Renderers must be correct, self-contained, and side-effect free."""

from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import solvable_key_room  # noqa: E402

from coop_env import GenerationConfig, RoomGenerator, RoomShape  # noqa: E402
from coop_env.rendering import (  # noqa: E402
    AsciiOptions,
    GalleryEntry,
    SvgOptions,
    render_ascii,
    render_gallery,
    render_mechanism_report,
    render_svg,
)
from coop_env.state import EpisodeState  # noqa: E402
from coop_env.validation import validate_room  # noqa: E402


class TestAscii(unittest.TestCase):
    def test_map_dimensions_match_the_room(self):
        room = RoomGenerator(GenerationConfig.preset("standard")).generate(3)
        text = render_ascii(room, options=AsciiOptions(show_header=False, show_legend=False))
        lines = text.split("\n")
        self.assertEqual(len(lines), room.height)
        for line in lines:
            self.assertEqual(len(line), room.width)

    def test_spawns_and_exit_appear(self):
        text = render_ascii(solvable_key_room())
        self.assertIn("1", text)
        self.assertIn("2", text)
        self.assertIn("E", text)
        self.assertIn("k", text)

    def test_live_state_hides_a_collected_key(self):
        room = solvable_key_room()
        state = EpisodeState.from_room(room)
        options = AsciiOptions(live_state=True, show_legend=False, show_header=False)
        before = render_ascii(room, state, options)
        state.collect_key("key_0")
        after = render_ascii(room, state, options)
        self.assertNotEqual(before, after)
        self.assertEqual(before.count("k"), after.count("k") + 1)

    def test_legend_only_lists_what_is_present(self):
        room = solvable_key_room()
        text = render_ascii(room)
        self.assertIn("key", text)
        self.assertNotIn("temporary bridge", text)

    def test_mechanism_report_names_the_lock(self):
        report = render_mechanism_report(solvable_key_room())
        self.assertIn("door_0", report)
        self.assertIn("key_0", report)


class TestSvg(unittest.TestCase):
    def test_output_is_parseable_xml(self):
        generator = RoomGenerator(GenerationConfig.preset("brutal"))
        for seed in range(6):
            svg = render_svg(generator.generate(seed))
            root = ElementTree.fromstring(svg)
            self.assertTrue(root.tag.endswith("svg"))

    def test_no_external_resources(self):
        svg = render_svg(RoomGenerator(GenerationConfig.preset("hard")).generate(1))
        for forbidden in ("http://", "https://", "<script", "xlink:href", "<image"):
            if forbidden == "http://":
                # the SVG namespace declaration is the one allowed occurrence
                self.assertEqual(svg.count("http://www.w3.org/2000/svg"), svg.count("http://"))
                continue
            self.assertNotIn(forbidden, svg)

    def test_dimensions_scale_with_cell_size(self):
        room = solvable_key_room()
        small = ElementTree.fromstring(render_svg(room, options=SvgOptions(cell=10)))
        large = ElementTree.fromstring(render_svg(room, options=SvgOptions(cell=20)))
        self.assertLess(int(small.get("width")), int(large.get("width")))

    def test_rendering_does_not_mutate_anything(self):
        room = RoomGenerator(GenerationConfig.preset("standard")).generate(8)
        state = EpisodeState.from_room(room)
        before_terrain = room.terrain.to_list()
        before_state = state.snapshot()
        render_svg(room, state)
        render_ascii(room, state)
        self.assertEqual(room.terrain.to_list(), before_terrain)
        self.assertEqual(state.snapshot(), before_state)

    def test_live_state_changes_the_picture(self):
        room = solvable_key_room()
        state = EpisodeState.from_room(room)
        before = render_svg(room, state)
        state.collect_key("key_0")
        self.assertNotEqual(before, render_svg(room, state))

    def test_labels_option_emits_entity_ids(self):
        room = solvable_key_room()
        self.assertIn("key_0", render_svg(room, options=SvgOptions(labels=True)))
        self.assertNotIn("key_0", render_svg(room, options=SvgOptions(labels=False)))

    def test_owned_key_legend_uses_agent_colors_without_generic_key_door(self):
        config = GenerationConfig.preset(
            "tutorial",
            num_keys=(2, 2),
            num_locked_doors=(2, 2),
            region_count=(3, 3),
            puzzle_chain_length=2,
            agent_specific_keys=True,
            allow_shared_keys=False,
            require_key_for_each_agent=True,
            num_normal_wipeout_balls=(0, 0),
            num_big_wipeout_balls=(0, 0),
        )
        svg = render_svg(RoomGenerator(config).generate(3))
        self.assertIn("agent 1 key/door", svg)
        self.assertIn("agent 2 key/door", svg)
        self.assertNotIn(">key door<", svg)

    def test_big_wipeout_ball_shows_its_three_by_three_hitbox(self):
        config = GenerationConfig(
            width=(24, 24),
            height=(14, 14),
            shape_weights={RoomShape.RECTANGLE: 1.0},
            region_count=(1, 1),
            obstacle_density=0.0,
            hazard_density=0.0,
            num_keys=(0, 0),
            num_locked_doors=(0, 0),
            num_switches=(0, 0),
            puzzle_chain_length=0,
            exit_objective_count=0,
            required_cooperative_actions=0,
            num_big_wipeout_balls=(1, 1),
        )
        room = RoomGenerator(config).generate(4)
        root = ElementTree.fromstring(
            render_svg(room, options=SvgOptions(cell=10))
        )
        hitboxes = [
            element
            for element in root.iter()
            if element.get("data-wipeout-hitbox") == "3x3"
        ]
        self.assertEqual(len(hitboxes), 1)
        self.assertEqual(float(hitboxes[0].get("width", "0")), 30.0)
        self.assertEqual(float(hitboxes[0].get("height", "0")), 30.0)


class TestGallery(unittest.TestCase):
    def test_gallery_contains_every_room(self):
        generator = RoomGenerator(GenerationConfig.preset("standard"))
        entries = []
        for seed in range(5):
            outcome = generator.generate_with_report(seed)
            entries.append(GalleryEntry(outcome.room, outcome.report))
        html = render_gallery(entries, "Test gallery")
        self.assertEqual(html.count("<svg"), 5)
        for entry in entries:
            self.assertIn(str(entry.room.seed), html)
        self.assertIn("Test gallery", html)

    def test_invalid_rooms_are_marked(self):
        from helpers import sealed_key_room

        room = sealed_key_room()
        html = render_gallery([GalleryEntry(room, validate_room(room))])
        self.assertIn("invalid", html)

    def test_gallery_accepts_bare_rooms(self):
        rooms = RoomGenerator(GenerationConfig.preset("easy")).generate_many(3, 20)
        self.assertIn("<svg", render_gallery(rooms))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Inspect a single generated room.

    python3 tools/inspect_room.py --seed 42
    python3 tools/inspect_room.py --seed 42 --preset brutal --svg output/room.svg
    python3 tools/inspect_room.py --seed 42 --tick 7 --labels

Prints the ASCII map, the lock structure, and the validation verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coop_env import GenerationConfig, RoomGenerator  # noqa: E402
from coop_env.config import PRESET_COMPLEXITY  # noqa: E402
from coop_env.rendering.ascii_renderer import (  # noqa: E402
    AsciiOptions,
    render_ascii,
    render_mechanism_report,
)
from coop_env.rendering.svg_renderer import SvgOptions, render_svg  # noqa: E402
from coop_env.state import EpisodeState  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default="standard", choices=sorted(PRESET_COMPLEXITY))
    parser.add_argument("--complexity", type=float, default=None)
    parser.add_argument("--tick", type=int, default=None, help="render live state at this tick")
    parser.add_argument("--svg", default=None, help="also write an SVG here")
    parser.add_argument("--cell", type=int, default=24)
    parser.add_argument("--labels", action="store_true", help="annotate entity ids in the SVG")
    args = parser.parse_args()

    if args.complexity is not None:
        config = GenerationConfig.from_complexity(args.complexity)
    else:
        config = GenerationConfig.preset(args.preset)

    outcome = RoomGenerator(config).generate_with_report(args.seed)
    room = outcome.room

    state: EpisodeState | None = None
    if args.tick is not None:
        state = EpisodeState.from_room(room)
        state.advance(args.tick)

    print(render_ascii(room, state, AsciiOptions(live_state=state is not None)))
    print()
    print(render_mechanism_report(room))
    print()

    report = outcome.report
    print(f"validation: {report.summary()}")
    print(f"attempts:   {outcome.attempts}{'  (fallback room)' if outcome.fallback else ''}")
    for issue in report.issues:
        print(f"  {issue}")
    if report.stats:
        print("stats:")
        for key, value in report.stats.items():
            print(f"  {key}: {value}")

    solvability = report.solvability
    if solvability:
        print("solvability:")
        print(f"  regions reachable by slot 0: {len(solvability.reachable[0])}")
        print(f"  regions reachable by slot 1: {len(solvability.reachable[1])}")
        print(f"  exit reachable by slots:     {list(solvability.exit_reachable_by)}")
        print(f"  doors needing both slots:    {sorted(solvability.cooperative_clusters)}")
        print(f"  one-way (hold) passages:     {sorted(solvability.one_way_clusters)}")

    if args.svg:
        path = Path(args.svg)
        path.parent.mkdir(parents=True, exist_ok=True)
        options = SvgOptions(cell=args.cell, labels=args.labels)
        path.write_text(render_svg(room, state, options), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

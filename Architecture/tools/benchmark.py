#!/usr/bin/env python3
"""Measure generator throughput, retry rate, and layout variety.

    python3 tools/benchmark.py                 # every preset, 50 rooms each
    python3 tools/benchmark.py --count 200
    python3 tools/benchmark.py --preset brutal --count 100

Useful as a regression check: a change that makes rooms harder to validate
shows up immediately as a jump in mean attempts or fallback count.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coop_env import GenerationConfig, RoomGenerator  # noqa: E402
from coop_env.config import PRESET_COMPLEXITY  # noqa: E402


def run(preset: str, count: int, start_seed: int) -> dict[str, object]:
    config = GenerationConfig.preset(preset)
    generator = RoomGenerator(config)

    attempts: list[int] = []
    areas: list[int] = []
    regions: list[int] = []
    shapes: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    fallbacks = 0
    invalid = 0
    coop_rooms = 0
    coop_doors = 0
    signatures: set[str] = set()

    start = time.perf_counter()
    for offset in range(count):
        outcome = generator.generate_with_report(start_seed + offset)
        room = outcome.room
        attempts.append(outcome.attempts)
        areas.append(room.width * room.height)
        regions.append(room.topology.region_count)
        shapes[room.shape.value] += 1
        for log in outcome.rejected:
            rejects[log.stage] += 1
        fallbacks += int(outcome.fallback)
        invalid += int(not outcome.report.ok)
        solvability = outcome.report.solvability
        if solvability and solvability.cooperative_clusters:
            coop_rooms += 1
            coop_doors += len(solvability.cooperative_clusters)
        signatures.add(_signature(room))
    elapsed = time.perf_counter() - start

    return {
        "preset": preset,
        "rooms": count,
        "ms_per_room": elapsed / count * 1000,
        "mean_attempts": statistics.fmean(attempts),
        "max_attempts": max(attempts),
        "fallbacks": fallbacks,
        "invalid": invalid,
        "unique_layouts": len(signatures),
        "coop_rooms": coop_rooms,
        "coop_doors_per_room": coop_doors / count,
        "mean_area": statistics.fmean(areas),
        "mean_regions": statistics.fmean(regions),
        "shapes": len(shapes),
        "rejects": dict(rejects),
    }


def _signature(room) -> str:
    """A cheap fingerprint used only to confirm layouts are not repeating."""
    return f"{room.width}x{room.height}:{''.join(str(t) for t in room.terrain.to_list())}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default=None, choices=sorted(PRESET_COMPLEXITY))
    args = parser.parse_args()

    presets = (
        [args.preset]
        if args.preset
        else sorted(PRESET_COMPLEXITY, key=lambda k: PRESET_COMPLEXITY[k])
    )

    header = (
        f"{'preset':10s} {'ms/room':>8s} {'attempts':>9s} {'max':>4s} {'fallback':>9s} "
        f"{'invalid':>8s} {'unique':>8s} {'co-op':>8s} {'regions':>8s} {'area':>7s}"
    )
    print(header)
    print("-" * len(header))
    for preset in presets:
        row = run(preset, args.count, args.seed)
        print(
            f"{row['preset']:10s} {row['ms_per_room']:8.1f} {row['mean_attempts']:9.2f} "
            f"{row['max_attempts']:4d} {row['fallbacks']:9d} {row['invalid']:8d} "
            f"{row['unique_layouts']:5d}/{row['rooms']:<2d} {row['coop_rooms']:5d}/{row['rooms']:<2d} "
            f"{row['mean_regions']:8.1f} {row['mean_area']:7.0f}"
        )
        if row["rejects"]:
            print(f"{'':10s} retries by stage: {row['rejects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

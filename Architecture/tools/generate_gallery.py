#!/usr/bin/env python3
"""Render a batch of generated rooms to one HTML page for visual inspection.

    python3 tools/generate_gallery.py --preset standard --count 24
    python3 tools/generate_gallery.py --complexity 0.85 --count 12 --cell 16
    python3 tools/generate_gallery.py --sweep            # one row per preset

The page is self-contained; open it straight from disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coop_env import GenerationConfig, RoomGenerator  # noqa: E402
from coop_env.config import PRESET_COMPLEXITY  # noqa: E402
from coop_env.rendering.gallery import GalleryEntry, save_gallery  # noqa: E402


def build_entries(config: GenerationConfig, count: int, start_seed: int) -> list[GalleryEntry]:
    generator = RoomGenerator(config)
    entries: list[GalleryEntry] = []
    for offset in range(count):
        outcome = generator.generate_with_report(start_seed + offset)
        note = ""
        if outcome.report.solvability:
            coop = len(outcome.report.solvability.cooperative_clusters)
            if coop:
                note = f"{coop} door(s) need both slots"
        entries.append(GalleryEntry(outcome.room, outcome.report, note))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="standard", choices=sorted(PRESET_COMPLEXITY))
    parser.add_argument("--complexity", type=float, default=None, help="overrides --preset (0.0-1.0)")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--cell", type=int, default=14, help="tile size in pixels")
    parser.add_argument("--out", default="output/gallery.html")
    parser.add_argument("--sweep", action="store_true", help="sample every preset instead")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        entries: list[GalleryEntry] = []
        per_preset = max(1, args.count // len(PRESET_COMPLEXITY))
        for name, level in sorted(PRESET_COMPLEXITY.items(), key=lambda kv: kv[1]):
            config = GenerationConfig.from_complexity(level)
            batch = build_entries(config, per_preset, args.seed)
            for entry in batch:
                entry.note = f"{name} ({level:.2f}) {entry.note}".strip()
            entries.extend(batch)
        title = "Difficulty sweep"
        subtitle = f"{per_preset} rooms per preset, seeds from {args.seed}"
    else:
        if args.complexity is not None:
            config = GenerationConfig.from_complexity(args.complexity)
            label = f"complexity {args.complexity:.2f}"
        else:
            config = GenerationConfig.preset(args.preset)
            label = f"preset '{args.preset}'"
        entries = build_entries(config, args.count, args.seed)
        title = f"Generated rooms - {label}"
        subtitle = f"{args.count} rooms, seeds {args.seed}-{args.seed + args.count - 1}"

    save_gallery(entries, str(out_path), title, subtitle, cell=args.cell)
    valid = sum(1 for e in entries if e.report and e.report.ok)
    print(f"wrote {out_path} ({len(entries)} rooms, {valid} valid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

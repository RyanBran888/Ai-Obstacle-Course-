"""Preview curriculum room layouts in HTML.

Two sources are supported:

    python3 DQN/preview_maps.py
        Sample rooms from the current curriculum before training.

    python3 DQN/preview_maps.py --manifest runs/tune_2/rooms.json
        Verify and export recorded train and validation rooms to
        runs/tune_2/rooms_maps/.

Other options:

    --count 12
    --stage key_door
    --split validation
    --split test --allow-test
    --cell 18
    --ascii
    --out preview.html
    --out-dir runs/tune_2/designer_maps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from html import escape
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "Architecture")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import coop_env  # noqa: E402
from coop_env import GenerationConfig, RoomGenerator, Tile  # noqa: E402
from coop_env.entities import EntityKind  # noqa: E402
from coop_env.rendering import (  # noqa: E402
    AsciiOptions,
    SvgOptions,
    render_ascii,
    render_svg,
)
from coop_env.rendering.palette import DARK  # noqa: E402

from curriculum import default_stages  # noqa: E402
from room_manifest import SELECTION_ALGORITHM, room_fingerprints  # noqa: E402

DEFAULT_OUT = "curriculum_maps.html"
SPLITS = ("train", "validation", "test")


def sample_stage(stage, count: int, data_seed: int = 0):
    """Generate a fresh preview from one curriculum stage."""
    generator = RoomGenerator(stage.config)
    rooms, seed, tried = [], data_seed, 0
    while len(rooms) < count and tried < count * 400:
        tried += 1
        seed += 1
        outcome = generator.generate_with_report(seed)
        room = outcome.room
        if (
            outcome.report.ok
            and not outcome.fallback
            and not room.metadata.get("fallback")
            and stage.accepts(room)
        ):
            rooms.append(room)
    if len(rooms) != count:
        raise ValueError(
            f"{stage.name}: found only {len(rooms)} valid rooms after {tried} attempts"
        )
    return rooms


def _json_hash(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = data.get("schema_version")
    if schema not in (3, 4):
        raise ValueError(
            f"{path}: expected room manifest schema 3 or 4, "
            f"found {schema!r}"
        )
    if schema == 4 and data.get("selection_algorithm") != SELECTION_ALGORITHM:
        raise ValueError(f"{path}: room selection algorithm does not match")
    if data.get("generator_version") != coop_env.__version__:
        raise ValueError(
            f"{path}: generator version {data.get('generator_version')!r} "
            f"does not match installed coop_env {coop_env.__version__!r}"
        )
    if not isinstance(data.get("stages"), list) or not data["stages"]:
        raise ValueError(f"{path}: manifest has no stages")
    return data


def _verify_room(
    entry: dict,
    split: str,
    record: dict,
    generator: RoomGenerator,
):
    outcome = generator.generate_with_report(record["seed"])
    room = outcome.room
    label = f"{entry['stage']} {split} seed {record['seed']}"
    if (
        not outcome.report.ok
        or outcome.fallback
        or bool(room.metadata.get("fallback"))
    ):
        raise ValueError(f"{label}: no longer generates a valid room")

    actual_hashes = room_fingerprints(room)
    saved_hashes = (
        record["geometry_sha256"],
        record["navigation_sha256"],
        record["task_sha256"],
    )
    if actual_hashes != saved_hashes:
        raise ValueError(f"{label}: layout or task fingerprint changed")

    actual_metadata = (
        outcome.attempts,
        room.width,
        room.height,
        room.shape.value,
        dict(room.counts()),
    )
    saved_metadata = (
        record["attempts"],
        record["width"],
        record["height"],
        record["shape"],
        record["counts"],
    )
    if actual_metadata != saved_metadata:
        raise ValueError(f"{label}: generated metadata changed")
    return room


def manifest_stage(entry: dict, split: str, count: int | None):
    """Rebuild and verify recorded rooms."""
    config = entry["config"]
    if _json_hash(config) != entry.get("config_sha256"):
        raise ValueError(f"{entry['stage']}: saved config hash does not match")
    generator = RoomGenerator(GenerationConfig.from_dict(config))
    records = entry["splits"].get(split)
    if not isinstance(records, list):
        raise ValueError(f"{entry['stage']}: manifest has no {split} split")
    selected = records if count is None else records[:count]
    return [_verify_room(entry, split, record, generator) for record in selected]


def selected_splits(split: str) -> tuple[str, ...]:
    if split == "development":
        return ("train", "validation")
    return SPLITS if split == "all" else (split,)


def _validate_limits(
    limits: Mapping[str, Mapping[str, int]] | None,
    data: dict,
) -> None:
    if limits is None:
        return
    stages = {entry["stage"] for entry in data["stages"]}
    for split, stage_limits in limits.items():
        if split not in SPLITS:
            raise ValueError(f"unknown room split in export limits: {split!r}")
        for stage, limit in stage_limits.items():
            if stage not in stages:
                raise ValueError(f"unknown curriculum stage in export limits: {stage!r}")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError(
                    f"{split}/{stage} export limit must be a nonnegative integer"
                )


def _room_limit(
    split: str,
    stage: str,
    count: int | None,
    limits: Mapping[str, Mapping[str, int]] | None,
) -> int | None:
    if limits is None:
        return count
    return limits.get(split, {}).get(stage, count)


def collect(args, data: dict | None = None):
    """Collect sections for terminal or single-page output."""
    out = []
    if args.manifest:
        manifest = data or load_manifest(Path(args.manifest))
        splits = selected_splits(args.split)
        for split in splits:
            for entry in manifest["stages"]:
                name = entry["stage"]
                if args.stage and name != args.stage:
                    continue
                total = len(entry["splits"].get(split, []))
                rooms = manifest_stage(entry, split, args.count)
                heading = name if len(splits) == 1 else f"{name} / {split}"
                out.append(
                    (heading, f"{split} split, {len(rooms)} of {total} rooms", rooms)
                )
    else:
        count = args.count if args.count is not None else 6
        for stage in default_stages():
            if args.stage and stage.name != args.stage:
                continue
            rooms = sample_stage(stage, count)
            out.append((stage.name, f"sampled {len(rooms)} rooms", rooms))
    return out


CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; padding:28px; background:#0a0d13; color:#e6ebf5;
       font:14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
h1 { font-size:21px; margin:0 0 4px; }
h2 { font-size:16px; margin:34px 0 4px; color:#9ecbff; }
.sub { color:#8a94a8; font-size:12px; }
a { color:#9ecbff; text-decoration:none; }
a:hover { text-decoration:underline; }
.legend { display:flex; flex-wrap:wrap; gap:8px 20px; padding:14px 18px;
          margin:18px 0 6px; background:#141926; border:1px solid #222b3d;
          border-radius:10px; }
.legend div { display:flex; align-items:center; gap:7px; font-size:11px; color:#cfd6e4; }
.sw { width:12px; height:12px; border-radius:3px; display:inline-block; }
.grid { display:flex; flex-wrap:wrap; gap:18px; align-items:flex-start; margin-top:12px; }
.card { display:grid; grid-template-columns:minmax(0,max-content);
        background:#141926; border:1px solid #222b3d; border-radius:10px; padding:10px; }
.card svg { display:block; max-width:100%; height:auto; border-radius:6px; }
.map-link { display:block; line-height:0; }
.meta { margin-top:8px; font-size:11px; color:#8a94a8; min-width:0;
        overflow-wrap:anywhere; line-height:1.7; }
.meta b { color:#cfd6e4; font-weight:600; }
table { border-collapse:collapse; margin-top:14px; font-size:12px; }
th,td { text-align:left; padding:4px 14px 4px 0; }
th { color:#8a94a8; font-weight:600; }
.index { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
         gap:14px; margin-top:20px; }
.index section { background:#141926; border:1px solid #222b3d;
                 border-radius:10px; padding:14px 16px; }
.index h2 { margin:0 0 9px; }
.index a { display:block; padding:3px 0; }
"""


def legend_html() -> str:
    items = [
        (DARK.tile_color(Tile.FLOOR), "floor"),
        (DARK.tile_color(Tile.WALL), "wall"),
        (DARK.tile_color(Tile.OBSTACLE), "obstacle"),
        (DARK.tile_color(Tile.HAZARD_LAVA), "lava"),
        (DARK.tile_color(Tile.HAZARD_SPIKES), "spikes"),
        (DARK.tile_color(Tile.HAZARD_WATER), "water"),
        (DARK.tile_color(Tile.HAZARD_PIT), "pit"),
        (DARK.entity_color(EntityKind.AGENT_SPAWN), "agent spawn"),
        (DARK.entity_color(EntityKind.EXIT_DOOR), "exit"),
        (DARK.accents.get("agent_0", "#59a5ff"), "agent 1 key / door"),
        (DARK.accents.get("agent_1", "#ff6fb1"), "agent 2 key / door"),
        (DARK.entity_color(EntityKind.KEY), "shared key"),
        (DARK.accents.get("lock_switch", "#4fd6e8"), "switch door"),
        (DARK.accents.get("lock_hold", "#ff9a3c"), "hold door"),
        (DARK.entity_color(EntityKind.SWITCH), "switch / lever"),
        (DARK.entity_color(EntityKind.CHECKPOINT), "checkpoint"),
        (DARK.entity_color(EntityKind.PUSHABLE_BLOCK), "crate"),
        (DARK.entity_color(EntityKind.RESET_ZONE), "reset zone"),
        (DARK.entity_color(EntityKind.TEMPORARY_BRIDGE), "temp bridge"),
        (DARK.accents.get("wipeout_normal", "#ff9f43"), "7x1 wipeout ball"),
        (DARK.accents.get("wipeout_big", "#ff3b4f"), "11x1 big ball"),
    ]
    cells = "".join(
        f'<div><span class="sw" style="background:{c}"></span>{escape(label)}</div>'
        for c, label in items
    )
    return f'<div class="legend">{cells}</div>'


def summary_table(sections) -> str:
    rows = [
        "<tr><th>stage</th><th>rooms</th><th>size range</th>"
        "<th>keys</th><th>doors</th><th>switches</th><th>checkpoints</th>"
        "<th>balls</th></tr>"
    ]
    for name, _, rooms in sections:
        if not rooms:
            rows.append(f"<tr><td>{escape(name)}</td><td>0</td><td colspan=6>-</td></tr>")
            continue
        sizes = sorted((r.width, r.height) for r in rooms)
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{len(rooms)}</td>"
            f"<td>{sizes[0][0]}x{sizes[0][1]} - {sizes[-1][0]}x{sizes[-1][1]}</td>"
            f"<td>{sum(len(r.keys) for r in rooms) / len(rooms):.1f}</td>"
            f"<td>{sum(len(r.doors) for r in rooms) / len(rooms):.1f}</td>"
            f"<td>{sum(len(r.switches) for r in rooms) / len(rooms):.1f}</td>"
            f"<td>{sum(len(r.checkpoints) for r in rooms) / len(rooms):.1f}</td>"
            f"<td>{sum(len(r.wipeout_balls) for r in rooms) / len(rooms):.1f}</td>"
            "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _map_svg(room, cell: int) -> str:
    return render_svg(
        room,
        options=SvgOptions(cell=cell, legend=False, header=False, grid_lines=False),
    )


def card(
    room,
    cell: int,
    asset: tuple[str, str] | None = None,
) -> str:
    svg, href = asset if asset is not None else (_map_svg(room, cell), None)
    map_html = svg
    if href is not None:
        map_html = (
            f'<a class="map-link" href="{escape(href, quote=True)}" '
            f'title="Open map-only SVG for seed {room.seed}">{svg}</a>'
        )
    counts = room.counts()
    bits = [
        f"<b>seed</b> {room.seed}",
        f"<b>{room.width}x{room.height}</b> {escape(room.shape.value)}",
        f"<b>regions</b> {room.topology.region_count}",
    ]
    for label, key in (
        ("keys", "key"), ("doors", "locked_door"), ("switches", "switch"),
        ("checkpoints", "checkpoint"), ("crates", "pushable_block"),
        ("balls", "wipeout_ball"),
    ):
        if counts.get(key):
            bits.append(f"<b>{label}</b> {counts[key]}")
    bits.append(f"<b>exit needs</b> {escape(room.exit.requirement.describe())}")
    return (
        f'<div class="card">{map_html}'
        f'<div class="meta">{" · ".join(bits)}</div></div>'
    )


def build_html(
    sections,
    cell: int,
    source: str,
    *,
    title: str = "Curriculum rooms",
    back_href: str | None = None,
    map_assets: Mapping[int, tuple[str, str]] | None = None,
) -> str:
    body = [
        f"<h1>{escape(title)}</h1>",
        f'<div class="sub">{escape(source)}</div>',
    ]
    if back_href is not None:
        body.append(f'<div class="sub"><a href="{escape(back_href)}">← all maps</a></div>')
    body.extend((legend_html(), summary_table(sections)))
    for name, note, rooms in sections:
        body.append(f"<h2>{escape(name)}</h2>")
        body.append(f'<div class="sub">{escape(note)}</div>')
        body.append('<div class="grid">')
        body.extend(
            card(
                room,
                cell,
                map_assets.get(room.seed) if map_assets is not None else None,
            )
            for room in rooms
        )
        body.append("</div>")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{CSS}</style></head><body>"
        + "".join(body)
        + "</body></html>"
    )


def _slug(value: str) -> str:
    slug = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    ).strip("_")
    return slug or "stage"


def _site_index(
    manifest_path: Path,
    data: dict,
    pages: dict[str, list[tuple[str, str, int, int]]],
) -> str:
    groups = []
    for split in SPLITS:
        if split not in pages:
            continue
        links = []
        for stage, href, shown, total in pages[split]:
            count = str(total) if shown == total else f"{shown} of {total}"
            links.append(
                f'<a href="{escape(href, quote=True)}">'
                f"{escape(stage)} <span class=\"sub\">({count})</span></a>"
            )
        groups.append(
            f"<section><h2>{escape(split)}</h2>{''.join(links)}</section>"
        )

    total_shown = sum(
        shown
        for split_pages in pages.values()
        for _, _, shown, _ in split_pages
    )
    total_saved = sum(
        total
        for split_pages in pages.values()
        for _, _, _, total in split_pages
    )
    completeness = (
        f"all {total_saved} recorded layouts"
        if total_shown == total_saved
        else f"{total_shown} of {total_saved} recorded layouts"
    )
    source = escape(str(manifest_path.resolve()))
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Curriculum map export</title><style>{CSS}</style></head><body>"
        "<h1>Curriculum map export</h1>"
        f'<div class="sub">source: {source}</div>'
        f'<div class="sub">generator {escape(data["generator_version"])} · '
        f"{completeness}</div>"
        f'<div class="index">{"".join(groups)}</div>'
        "</body></html>"
    )


def export_manifest_site(
    manifest_path: Path,
    data: dict,
    output_dir: Path,
    *,
    splits: tuple[str, ...],
    stage_name: str | None,
    count: int | None,
    cell: int,
    limits: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[int, int]:
    """Export pages, optionally capping individual split/stage pools."""
    _validate_limits(limits, data)
    output_dir.mkdir(parents=True, exist_ok=True)
    pages: dict[str, list[tuple[str, str, int, int]]] = {}
    total_rooms = 0

    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        pages[split] = []
        used_names: set[str] = set()
        for entry in data["stages"]:
            stage = entry["stage"]
            if stage_name and stage != stage_name:
                continue
            stage_slug = _slug(stage)
            filename = f"{stage_slug}.html"
            if filename in used_names:
                raise ValueError(f"stage names collide after filename cleanup: {stage}")
            used_names.add(filename)

            total = len(entry["splits"].get(split, []))
            limit = _room_limit(split, stage, count, limits)
            rooms = manifest_stage(entry, split, limit)
            shown = len(rooms)
            if shown == 0:
                continue
            note = f"{split} split, {shown} of {total} recorded rooms"

            asset_dir = split_dir / stage_slug
            asset_dir.mkdir(parents=True, exist_ok=True)
            map_assets: dict[int, tuple[str, str]] = {}
            for room in rooms:
                if room.seed in map_assets:
                    raise ValueError(
                        f"{stage} {split} contains duplicate seed {room.seed}"
                    )
                asset_name = f"seed-{room.seed}.svg"
                svg = _map_svg(room, cell)
                (asset_dir / asset_name).write_text(svg, encoding="utf-8")
                map_assets[room.seed] = (
                    svg,
                    f"{stage_slug}/{asset_name}",
                )

            page = build_html(
                [(stage, note, rooms)],
                cell,
                f"verified against {manifest_path.resolve()}",
                title=f"{stage} · {split}",
                back_href="../index.html",
                map_assets=map_assets,
            )
            page_path = split_dir / filename
            page_path.write_text(page, encoding="utf-8")
            pages[split].append(
                (stage, f"{split}/{filename}", shown, total)
            )
            total_rooms += shown

    page_count = sum(len(value) for value in pages.values())
    if page_count == 0:
        raise ValueError("no stages matched")
    (output_dir / "index.html").write_text(
        _site_index(manifest_path, data, pages),
        encoding="utf-8",
    )
    return total_rooms, page_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="limit rooms per stage (manifest default: all; sample default: 6)",
    )
    parser.add_argument("--stage", default=None, help="only this stage")
    parser.add_argument("--manifest", default=None, help="saved rooms.json")
    parser.add_argument(
        "--split",
        default="development",
        choices=(*SPLITS, "development", "all"),
        help="manifest split (default: train and validation)",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="explicitly expose held-out test layouts",
    )
    parser.add_argument("--cell", type=int, default=16)
    parser.add_argument("--ascii", action="store_true", help="print instead of HTML")
    parser.add_argument(
        "--out",
        default=None,
        help=f"single HTML file (sample default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="manifest export folder (default: <manifest stem>_maps beside manifest)",
    )
    args = parser.parse_args()

    if args.count is not None and args.count < 1:
        parser.error("--count must be positive")
    if args.cell < 4:
        parser.error("--cell must be at least 4")
    if args.out and args.out_dir:
        parser.error("choose --out or --out-dir, not both")
    if (
        args.manifest
        and "test" in selected_splits(args.split)
        and not args.allow_test
    ):
        parser.error(
            "test rooms stay hidden until final evaluation; "
            "add --allow-test only after testing"
        )
    if not args.manifest and args.split not in ("development", "all", "train"):
        parser.error("--split requires --manifest")
    if not args.manifest and args.out_dir:
        parser.error("--out-dir requires --manifest")

    try:
        data = load_manifest(Path(args.manifest)) if args.manifest else None
        print("collecting rooms...", flush=True)

        if args.manifest and not args.ascii and not args.out:
            assert data is not None
            manifest_path = Path(args.manifest)
            output_dir = (
                Path(args.out_dir)
                if args.out_dir
                else manifest_path.parent / f"{manifest_path.stem}_maps"
            )
            total, pages = export_manifest_site(
                manifest_path,
                data,
                output_dir,
                splits=selected_splits(args.split),
                stage_name=args.stage,
                count=args.count,
                cell=args.cell,
            )
            print(f"wrote {output_dir / 'index.html'}")
            print(f"verified {total} rooms across {pages} stage pages")
            return 0

        sections = collect(args, data)
        if not sections:
            print("no stages matched", file=sys.stderr)
            return 1

        if args.ascii:
            for name, note, rooms in sections:
                print(f"\n{'=' * 62}\n{name}  ({note})\n{'=' * 62}")
                for room in rooms:
                    print()
                    print(render_ascii(room, options=AsciiOptions(show_legend=False)))
                    print(f"  exit needs {room.exit.requirement.describe()}")
            return 0

        source = (
            f"verified against manifest {Path(args.manifest).resolve()}"
            if args.manifest
            else "fresh samples from the current curriculum"
        )
        path = Path(args.out or DEFAULT_OUT)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_html(sections, args.cell, source), encoding="utf-8")
        total = sum(len(rooms) for _, _, rooms in sections)
        print(f"wrote {path}  ({total} rooms across {len(sections)} sections)")
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"preview failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

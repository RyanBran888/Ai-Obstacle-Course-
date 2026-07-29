"""SVG rendering, for looking at rooms during development.

Produces a standalone SVG string: no external assets, no dependencies, opens in
any browser. Pass an `EpisodeState` to draw the live picture (platforms at their
current position, doors open, keys already taken) instead of the blueprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from ..entities import (
    AgentSpawn,
    Checkpoint,
    Entity,
    EntityKind,
    ExitDoor,
    Key,
    LockedDoor,
    MovingPlatform,
    PushableBlock,
    ResetZone,
    Switch,
    SwitchMode,
    TemporaryBridge,
)
from ..requirements import (
    CheckpointRequirement,
    KeyRequirement,
    SwitchRequirement,
)
from ..room import Room
from ..state import EpisodeState
from ..tiles import Tile, is_hazard
from ..utils.geometry import Vec2
from .palette import Theme, get_theme


@dataclass(slots=True)
class SvgOptions:
    cell: int = 22
    padding: int = 16
    header: bool = True
    legend: bool = True
    grid_lines: bool = True
    labels: bool = False
    """Draw entity ids next to each object -- useful when debugging a lock chain."""
    theme: str | Theme = "dark"
    title: str | None = None


def render_svg(
    room: Room,
    state: EpisodeState | None = None,
    options: SvgOptions | None = None,
) -> str:
    """Render one room as a standalone SVG document."""
    opts = options or SvgOptions()
    theme = get_theme(opts.theme)
    cell = opts.cell
    pad = opts.padding

    map_w = room.width * cell
    map_h = room.height * cell
    header_h = 52 if opts.header else 0
    legend_rows = _legend_entries(room)
    legend_h = _legend_height(legend_rows, map_w) if opts.legend else 0

    total_w = map_w + pad * 2
    total_h = map_h + pad * 2 + header_h + legend_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">',
        _defs(theme),
        f'<rect width="{total_w}" height="{total_h}" fill="{theme.background}"/>',
    ]

    if opts.header:
        parts.append(_header(room, state, theme, pad, total_w, opts))

    origin_y = pad + header_h
    parts.append(f'<g transform="translate({pad},{origin_y})">')
    parts.extend(_terrain_layer(room, theme, cell))
    if opts.grid_lines:
        parts.append(_grid_layer(room, theme, cell))
    parts.extend(_entity_layer(room, state, theme, cell, opts))
    parts.append("</g>")

    if opts.legend:
        parts.append(
            _legend_layer(legend_rows, theme, pad, origin_y + map_h + 14, map_w)
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


def _defs(theme: Theme) -> str:
    return (
        "<defs>"
        f'<pattern id="hatch" width="6" height="6" patternTransform="rotate(45)" '
        f'patternUnits="userSpaceOnUse">'
        f'<rect width="6" height="6" fill="none"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="{theme.entity_color(EntityKind.RESET_ZONE)}" '
        f'stroke-width="2" stroke-opacity="0.55"/>'
        "</pattern>"
        "</defs>"
    )


def _terrain_layer(room: Room, theme: Theme, cell: int) -> list[str]:
    out: list[str] = []
    for y in range(room.height):
        for x in range(room.width):
            pos = Vec2(x, y)
            tile = Tile(room.terrain[pos])
            if tile is Tile.VOID:
                continue
            color = theme.tile_color(tile)
            px, py = x * cell, y * cell
            if tile is Tile.FLOOR and (x + y) % 2 == 0:
                color = _shade(color, 1.06)
            out.append(
                f'<rect x="{px}" y="{py}" width="{cell}" height="{cell}" fill="{color}"/>'
            )
            if is_hazard(tile):
                out.append(_hazard_mark(tile, px, py, cell, theme))
            elif tile is Tile.OBSTACLE:
                out.append(
                    f'<rect x="{px + 3}" y="{py + 3}" width="{cell - 6}" height="{cell - 6}" '
                    f'fill="none" stroke="{_shade(theme.tile_color(tile), 0.7)}" stroke-width="2"/>'
                )
    return out


def _hazard_mark(tile: Tile, px: int, py: int, cell: int, theme: Theme) -> str:
    """A small motif so hazard types stay distinguishable in greyscale."""
    mid_x, mid_y = px + cell / 2, py + cell / 2
    ink = _shade(theme.tile_color(tile), 0.55)
    if tile is Tile.HAZARD_SPIKES:
        pts = f"{px + 4},{py + cell - 4} {mid_x},{py + 4} {px + cell - 4},{py + cell - 4}"
        return f'<polygon points="{pts}" fill="{ink}" fill-opacity="0.75"/>'
    if tile is Tile.HAZARD_WATER:
        return (
            f'<path d="M{px + 3} {mid_y} q {cell / 4} -4 {cell / 2} 0 t {cell / 2} 0" '
            f'fill="none" stroke="{ink}" stroke-width="2" stroke-opacity="0.8"/>'
        )
    if tile is Tile.HAZARD_LAVA:
        return (
            f'<circle cx="{mid_x}" cy="{mid_y}" r="{cell * 0.18:.1f}" '
            f'fill="{_shade(theme.tile_color(tile), 1.35)}" fill-opacity="0.9"/>'
        )
    return (
        f'<rect x="{px + 4}" y="{py + 4}" width="{cell - 8}" height="{cell - 8}" '
        f'fill="#000" fill-opacity="0.45"/>'
    )


def _grid_layer(room: Room, theme: Theme, cell: int) -> str:
    lines: list[str] = []
    for x in range(room.width + 1):
        lines.append(f"M{x * cell} 0V{room.height * cell}")
    for y in range(room.height + 1):
        lines.append(f"M0 {y * cell}H{room.width * cell}")
    return (
        f'<path d="{"".join(lines)}" stroke="{theme.grid}" stroke-width="1" '
        f'fill="none" stroke-opacity="0.5"/>'
    )


def _entity_layer(
    room: Room, state: EpisodeState | None, theme: Theme, cell: int, opts: SvgOptions
) -> list[str]:
    out: list[str] = []
    order = {
        EntityKind.RESET_ZONE: 0,
        EntityKind.TEMPORARY_BRIDGE: 1,
        EntityKind.MOVING_PLATFORM: 2,
        EntityKind.CHECKPOINT: 3,
        EntityKind.KEY: 5,
        EntityKind.SWITCH: 6,
        EntityKind.PUSHABLE_BLOCK: 7,
        EntityKind.LOCKED_DOOR: 8,
        EntityKind.EXIT_DOOR: 9,
        EntityKind.AGENT_SPAWN: 10,
    }
    for entity in sorted(room.entities, key=lambda e: (order.get(e.kind, 0), e.id)):
        out.extend(_draw_entity(entity, room, state, theme, cell))
        if opts.labels:
            out.append(_label(entity, theme, cell))
    return out


def _draw_entity(
    entity: Entity, room: Room, state: EpisodeState | None, theme: Theme, cell: int
) -> list[str]:
    color = theme.entity_color(entity.kind)

    if isinstance(entity, ResetZone):
        r = entity.rect
        return [
            f'<rect x="{r.x * cell}" y="{r.y * cell}" width="{r.w * cell}" '
            f'height="{r.h * cell}" fill="url(#hatch)" stroke="{color}" '
            f'stroke-width="1.5" stroke-dasharray="4 3"/>'
        ]

    if isinstance(entity, TemporaryBridge):
        solid = state.solid_bridge_tiles() if state else set()
        out = []
        for tile in entity.tiles:
            on = tile in solid if state else True
            out.append(
                f'<rect x="{tile[0] * cell + 2}" y="{tile[1] * cell + 2}" '
                f'width="{cell - 4}" height="{cell - 4}" fill="{color}" '
                f'fill-opacity="{0.85 if on else 0.2}" stroke="{color}" '
                f'stroke-width="1" stroke-dasharray="3 2"/>'
            )
        return out

    if isinstance(entity, MovingPlatform):
        out = []
        if len(entity.path) > 1:
            pts = " ".join(
                f"{p[0] * cell + cell / 2},{p[1] * cell + cell / 2}" for p in entity.path
            )
            out.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-opacity="0.4" stroke-dasharray="3 3"/>'
            )
        current = entity.position_at(state.tick) if state else entity.path[0] if entity.path else entity.pos
        out.append(
            f'<rect x="{current[0] * cell + 2}" y="{current[1] * cell + 2}" '
            f'width="{cell - 4}" height="{cell - 4}" rx="3" fill="{color}" '
            f'stroke="{_shade(color, 0.7)}" stroke-width="1.5"/>'
        )
        return out

    px, py = entity.pos[0] * cell, entity.pos[1] * cell
    mid_x, mid_y = px + cell / 2, py + cell / 2

    if isinstance(entity, AgentSpawn):
        return [
            f'<circle cx="{mid_x}" cy="{mid_y}" r="{cell * 0.34:.1f}" fill="{color}" '
            f'fill-opacity="0.22" stroke="{color}" stroke-width="2"/>',
            f'<text x="{mid_x}" y="{mid_y + cell * 0.17:.1f}" fill="{color}" '
            f'font-size="{cell * 0.5:.0f}" text-anchor="middle" font-weight="700">'
            f"{entity.index + 1}</text>",
        ]

    if isinstance(entity, ExitDoor):
        open_now = state.exit_open if state else False
        fill_opacity = 0.9 if open_now else 0.28
        return [
            f'<rect x="{px + 2}" y="{py + 2}" width="{cell - 4}" height="{cell - 4}" '
            f'rx="2" fill="{color}" fill-opacity="{fill_opacity}" stroke="{color}" '
            f'stroke-width="2"/>',
            f'<text x="{mid_x}" y="{mid_y + cell * 0.19:.1f}" fill="{theme.background}" '
            f'font-size="{cell * 0.55:.0f}" text-anchor="middle" font-weight="700">E</text>',
        ]

    if isinstance(entity, Key):
        if state is not None and state.is_key_collected(entity.id):
            return []
        return [
            f'<circle cx="{mid_x - cell * 0.12:.1f}" cy="{mid_y}" r="{cell * 0.17:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="2.5"/>',
            f'<path d="M{mid_x + cell * 0.02:.1f} {mid_y} h {cell * 0.28:.1f} '
            f'm -{cell * 0.1:.1f} 0 v {cell * 0.12:.1f}" stroke="{color}" '
            f'stroke-width="2.5" fill="none"/>',
        ]

    if isinstance(entity, LockedDoor):
        lock_color = _lock_color(entity, theme)
        open_now = state.is_door_open(entity.id) if state else False
        if entity.horizontal:
            bar = f'x="{px + 1}" y="{py + cell * 0.28:.1f}" width="{cell - 2}" height="{cell * 0.44:.1f}"'
        else:
            bar = f'x="{px + cell * 0.28:.1f}" y="{py + 1}" width="{cell * 0.44:.1f}" height="{cell - 2}"'
        opacity = 0.18 if open_now else 0.95
        marks = [
            f'<rect {bar} rx="2" fill="{lock_color}" fill-opacity="{opacity}" '
            f'stroke="{lock_color}" stroke-width="1.5"/>'
        ]
        if not entity.latching:
            marks.append(
                f'<circle cx="{mid_x}" cy="{mid_y}" r="{cell * 0.1:.1f}" '
                f'fill="{theme.background}"/>'
            )
        if entity.timer:
            marks.append(
                f'<circle cx="{mid_x}" cy="{mid_y}" r="{cell * 0.13:.1f}" fill="none" '
                f'stroke="{theme.background}" stroke-width="1.5"/>'
            )
        return marks

    if isinstance(entity, Switch):
        held = entity.mode is SwitchMode.HOLD
        on = state.is_switch_active(entity.id) if state else False
        base = (
            f'<rect x="{px + cell * 0.2:.1f}" y="{py + cell * 0.55:.1f}" '
            f'width="{cell * 0.6:.1f}" height="{cell * 0.22:.1f}" rx="2" fill="{color}" '
            f'fill-opacity="{0.9 if on else 0.45}"/>'
        )
        lever = (
            f'<line x1="{mid_x}" y1="{py + cell * 0.62:.1f}" '
            f'x2="{mid_x + (cell * 0.2 if on else -cell * 0.2):.1f}" '
            f'y2="{py + cell * 0.25:.1f}" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round"/>'
        )
        out = [base, lever]
        if held:
            out.append(
                f'<circle cx="{mid_x}" cy="{py + cell * 0.22:.1f}" r="{cell * 0.1:.1f}" '
                f'fill="{theme.accents["lock_hold"]}"/>'
            )
        return out

    if isinstance(entity, PushableBlock):
        pos = state.block_positions.get(entity.id, entity.pos) if state else entity.pos
        bx, by = pos[0] * cell, pos[1] * cell
        return [
            f'<rect x="{bx + 3}" y="{by + 3}" width="{cell - 6}" height="{cell - 6}" '
            f'rx="2" fill="{color}" fill-opacity="0.8" stroke="{_shade(color, 0.65)}" '
            f'stroke-width="2"/>',
            f'<path d="M{bx + 5} {by + 5} L{bx + cell - 5} {by + cell - 5} '
            f'M{bx + cell - 5} {by + 5} L{bx + 5} {by + cell - 5}" '
            f'stroke="{_shade(color, 0.65)}" stroke-width="1.5"/>',
        ]

    if isinstance(entity, Checkpoint):
        reached = state.is_checkpoint_reached(entity.id) if state else False
        return [
            f'<path d="M{px + cell * 0.3:.1f} {py + cell * 0.8:.1f} '
            f'V{py + cell * 0.2:.1f} h {cell * 0.4:.1f} l -{cell * 0.12:.1f} {cell * 0.15:.1f} '
            f'l {cell * 0.12:.1f} {cell * 0.15:.1f} h -{cell * 0.4:.1f}" '
            f'fill="{color}" fill-opacity="{0.9 if reached else 0.4}" stroke="{color}" '
            f'stroke-width="1.5" stroke-linejoin="round"/>'
        ]

    return []


def _label(entity: Entity, theme: Theme, cell: int) -> str:
    return (
        f'<text x="{entity.pos[0] * cell + cell / 2}" y="{entity.pos[1] * cell - 2}" '
        f'fill="{theme.muted}" font-size="9" text-anchor="middle">{escape(entity.id)}</text>'
    )


def _lock_color(door: LockedDoor, theme: Theme) -> str:
    if door.timer:
        return theme.accents["lock_timed"]
    if not door.latching:
        return theme.accents["lock_hold"]
    requirement = door.requirement
    if requirement.needs_simultaneity():
        return theme.accents["lock_paired"]
    if isinstance(requirement, KeyRequirement):
        return theme.accents["lock_key"]
    if isinstance(requirement, SwitchRequirement):
        return theme.accents["lock_switch"]
    if isinstance(requirement, CheckpointRequirement):
        return theme.entity_color(EntityKind.CHECKPOINT)
    return theme.entity_color(EntityKind.LOCKED_DOOR)


# ---------------------------------------------------------------------------
# chrome
# ---------------------------------------------------------------------------


def _header(
    room: Room, state: EpisodeState | None, theme: Theme, pad: int, total_w: int, opts: SvgOptions
) -> str:
    title = opts.title or f"seed {room.seed}"
    bits = [
        f"{room.width}x{room.height}",
        room.shape.value,
        f"{room.topology.region_count} regions",
        f"{len(room.doors)} doors",
        f"{len(room.keys)} keys",
    ]
    coop = room.metadata.get("cooperative_gates") or 0
    if coop:
        bits.append(f"{coop} co-op")
    if room.metadata.get("fallback"):
        bits.append("fallback")
    if state is not None:
        bits.append(f"tick {state.tick}")
    subtitle = "  ·  ".join(bits)
    return (
        f'<text x="{pad}" y="{pad + 12}" fill="{theme.text}" font-size="15" '
        f'font-weight="700">{escape(title)}</text>'
        f'<text x="{pad}" y="{pad + 30}" fill="{theme.muted}" font-size="11">'
        f"{escape(subtitle)}</text>"
    )


def _legend_entries(room: Room) -> list[tuple[str, str]]:
    """(colour, label) pairs for whatever this room actually contains."""
    theme = get_theme("dark")
    entries: list[tuple[str, str]] = [
        (theme.tile_color(Tile.FLOOR), "floor"),
        (theme.tile_color(Tile.WALL), "wall"),
    ]
    for tile, label in (
        (Tile.OBSTACLE, "obstacle"),
        (Tile.HAZARD_LAVA, "lava"),
        (Tile.HAZARD_SPIKES, "spikes"),
        (Tile.HAZARD_WATER, "water"),
        (Tile.HAZARD_PIT, "pit"),
    ):
        if room.terrain.count(tile):
            entries.append((theme.tile_color(tile), label))

    kinds = {e.kind for e in room.entities}
    entries.append((theme.entity_color(EntityKind.AGENT_SPAWN), "spawn"))
    entries.append((theme.entity_color(EntityKind.EXIT_DOOR), "exit"))
    if EntityKind.KEY in kinds:
        entries.append((theme.entity_color(EntityKind.KEY), "key"))
    if EntityKind.LOCKED_DOOR in kinds:
        if any(isinstance(d.requirement, KeyRequirement) for d in room.doors):
            entries.append((theme.accents["lock_key"], "key door"))
        if any(
            isinstance(d.requirement, SwitchRequirement)
            and d.latching
            and not d.requirement.needs_simultaneity()
            for d in room.doors
        ):
            entries.append((theme.accents["lock_switch"], "switch door"))
        if any(not d.latching for d in room.doors):
            entries.append((theme.accents["lock_hold"], "hold door"))
        if any(d.timer for d in room.doors):
            entries.append((theme.accents["lock_timed"], "timed door"))
        if any(d.requirement.needs_simultaneity() for d in room.doors):
            entries.append((theme.accents["lock_paired"], "paired-lever door"))
    if EntityKind.SWITCH in kinds:
        entries.append((theme.entity_color(EntityKind.SWITCH), "switch"))
    if EntityKind.MOVING_PLATFORM in kinds:
        entries.append((theme.entity_color(EntityKind.MOVING_PLATFORM), "platform"))
    if EntityKind.PUSHABLE_BLOCK in kinds:
        entries.append((theme.entity_color(EntityKind.PUSHABLE_BLOCK), "block"))
    if EntityKind.CHECKPOINT in kinds:
        entries.append((theme.entity_color(EntityKind.CHECKPOINT), "checkpoint"))
    if EntityKind.RESET_ZONE in kinds:
        entries.append((theme.entity_color(EntityKind.RESET_ZONE), "reset zone"))
    if EntityKind.TEMPORARY_BRIDGE in kinds:
        entries.append((theme.entity_color(EntityKind.TEMPORARY_BRIDGE), "temp bridge"))
    return entries


_LEGEND_ITEM_W = 112
_LEGEND_ROW_H = 18


def _legend_columns(map_w: int) -> int:
    return max(1, map_w // _LEGEND_ITEM_W)


def _legend_height(entries: list[tuple[str, str]], map_w: int) -> int:
    columns = _legend_columns(map_w)
    rows = (len(entries) + columns - 1) // columns
    return rows * _LEGEND_ROW_H + 12


def _legend_layer(
    entries: list[tuple[str, str]], theme: Theme, pad: int, top: int, map_w: int
) -> str:
    columns = _legend_columns(map_w)
    parts: list[str] = []
    for index, (color, label) in enumerate(entries):
        col = index % columns
        row = index // columns
        x = pad + col * _LEGEND_ITEM_W
        y = top + row * _LEGEND_ROW_H
        parts.append(
            f'<rect x="{x}" y="{y}" width="10" height="10" rx="2" fill="{color}"/>'
            f'<text x="{x + 15}" y="{y + 9}" fill="{theme.muted}" font-size="10">'
            f"{escape(label)}</text>"
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _shade(hex_color: str, factor: float) -> str:
    """Lighten (factor > 1) or darken (factor < 1) a #rrggbb colour."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return hex_color
    channels = [int(value[i : i + 2], 16) for i in (0, 2, 4)]
    scaled = [max(0, min(255, int(c * factor))) for c in channels]
    return "#" + "".join(f"{c:02x}" for c in scaled)


def save_svg(room: Room, path: str, state: EpisodeState | None = None, **kwargs) -> str:
    """Render a room and write it to `path`. Returns the path."""
    options = SvgOptions(**kwargs) if kwargs else None
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_svg(room, state, options))
    return path

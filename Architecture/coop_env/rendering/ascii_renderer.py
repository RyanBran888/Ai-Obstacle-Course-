"""Text rendering, for terminals, logs, and test fixtures.

Fast enough to call in a loop and diff-friendly, which makes it the renderer of
choice when checking that a seed still produces the same room.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..entities import AgentSpawn, EntityKind, LockedDoor, Switch, SwitchMode
from ..room import Room
from ..state import EpisodeState
from ..tiles import Tile, glyph
from ..utils.geometry import Vec2
from .palette import ENTITY_DRAW_ORDER, ENTITY_GLYPHS


@dataclass(slots=True)
class AsciiOptions:
    show_legend: bool = True
    show_header: bool = True
    show_border: bool = False
    live_state: bool = False
    """When true, draw the current state (doors open, keys taken) instead of the blueprint."""


def render_ascii(
    room: Room,
    state: EpisodeState | None = None,
    options: AsciiOptions | None = None,
) -> str:
    """Render a room as a block of text."""
    opts = options or AsciiOptions()
    canvas = [
        [glyph(room.terrain[Vec2(x, y)]) for x in range(room.width)]
        for y in range(room.height)
    ]

    for kind in ENTITY_DRAW_ORDER:
        for entity in room.entities:
            if entity.kind is not kind:
                continue
            mark = _glyph_for(entity, state if opts.live_state else None)
            if mark is None:
                continue
            for tile in _draw_tiles(entity, state if opts.live_state else None):
                if 0 <= tile[1] < room.height and 0 <= tile[0] < room.width:
                    canvas[tile[1]][tile[0]] = mark

    lines = ["".join(row) for row in canvas]
    if opts.show_border:
        width = max((len(line) for line in lines), default=0)
        lines = [f"|{line.ljust(width)}|" for line in lines]
        edge = "+" + "-" * width + "+"
        lines = [edge, *lines, edge]

    out: list[str] = []
    if opts.show_header:
        out.append(_header(room, state))
    out.extend(lines)
    if opts.show_legend:
        out.append("")
        out.append(_legend(room))
    return "\n".join(out)


def _glyph_for(entity, state: EpisodeState | None) -> str | None:
    if isinstance(entity, AgentSpawn):
        return str(entity.index + 1)
    if isinstance(entity, LockedDoor):
        if state is not None and state.is_door_open(entity.id):
            return "'"
        if entity.timer:
            return "T"
        # lowercase for hold-doors, so a door is never confused with its lever
        return "D" if entity.latching else "d"
    if isinstance(entity, Switch):
        if entity.mode is not SwitchMode.HOLD:
            return "S"
        # paired levers get their own mark: they only work if held together.
        # '&' echoes how a SIMULTANEOUS requirement prints itself.
        return "&" if entity.group.startswith("pair") else "H"
    if entity.kind is EntityKind.KEY and state is not None:
        if state.is_key_collected(entity.id):
            return None
    return ENTITY_GLYPHS.get(entity.kind)


def _draw_tiles(entity, state: EpisodeState | None) -> tuple[Vec2, ...]:
    return entity.footprint()


def _header(room: Room, state: EpisodeState | None) -> str:
    parts = [
        f"seed {room.seed}",
        f"{room.width}x{room.height}",
        room.shape.value,
        f"{room.topology.region_count} regions",
    ]
    coop = room.metadata.get("cooperative_gates")
    if coop:
        parts.append(f"{coop} co-op gate(s)")
    if room.metadata.get("fallback"):
        parts.append("FALLBACK")
    if state is not None:
        parts.append(f"tick {state.tick}")
        parts.append("exit OPEN" if state.exit_open else "exit locked")
    return " | ".join(parts)


def _legend(room: Room) -> str:
    """Only mention what the room actually contains."""
    entries: list[str] = [
        f"{glyph(Tile.FLOOR)} floor",
        f"{glyph(Tile.WALL)} wall",
    ]
    present = {t for t in Tile if room.terrain.count(t)}
    for tile, label in (
        (Tile.OBSTACLE, "obstacle"),
        (Tile.HAZARD_LAVA, "lava"),
        (Tile.HAZARD_SPIKES, "spikes"),
        (Tile.HAZARD_WATER, "water"),
        (Tile.HAZARD_PIT, "pit"),
    ):
        if tile in present:
            entries.append(f"{glyph(tile)} {label}")

    kinds = {e.kind for e in room.entities}
    labels: list[tuple[str, str]] = [("1/2", "agent spawn"), ("E", "exit")]
    if EntityKind.KEY in kinds:
        labels.append(("k", "key"))
    if any(d.latching and not d.timer for d in room.doors):
        labels.append(("D", "locked door"))
    if any(not d.latching for d in room.doors):
        labels.append(("d", "hold-door (open only while held)"))
    if any(d.timer for d in room.doors):
        labels.append(("T", "timed door"))
    if EntityKind.SWITCH in kinds:
        labels.append(("S", "switch"))
        if any(
            s.mode is SwitchMode.HOLD and not s.group.startswith("pair")
            for s in room.switches
        ):
            labels.append(("H", "hold-lever"))
        if any(s.group.startswith("pair") for s in room.switches):
            labels.append(("&", "paired lever (both held at once)"))
    if EntityKind.PUSHABLE_BLOCK in kinds:
        labels.append(("B", "pushable block"))
    if EntityKind.CHECKPOINT in kinds:
        labels.append(("C", "checkpoint"))
    if EntityKind.RESET_ZONE in kinds:
        labels.append((",", "reset zone"))
    if EntityKind.TEMPORARY_BRIDGE in kinds:
        labels.append(("-", "temporary bridge"))

    entries.extend(f"{mark} {label}" for mark, label in labels)
    return "  ".join(entries)


def render_mechanism_report(room: Room) -> str:
    """A text summary of the room's locks and what opens them."""
    lines = [room.summary(), ""]
    exit_door = room.exit
    lines.append(f"exit @ {tuple(exit_door.pos)} requires {exit_door.requirement.describe()}")
    if room.doors:
        lines.append("doors:")
        for door in sorted(room.doors, key=lambda d: d.id):
            lines.append(f"  {door.describe()}")
    triggers = [
        *((f"  key {k.id} @ {tuple(k.pos)} ({k.color})") for k in room.keys),
        *(
            f"  switch {s.id} @ {tuple(s.pos)} [{s.mode.value}]"
            for s in room.switches
        ),
    ]
    if triggers:
        lines.append("triggers:")
        lines.extend(triggers)
    return "\n".join(lines)

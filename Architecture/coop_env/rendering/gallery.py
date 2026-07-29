"""Build a single HTML page showing many generated rooms side by side.

The fastest way to tell whether a change to the generator helped: render fifty
seeds and scroll. Everything is inlined, so the page opens straight from disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable, Sequence

from ..room import Room
from ..validation.validator import ValidationReport
from .svg_renderer import SvgOptions, render_svg


@dataclass(slots=True)
class GalleryEntry:
    room: Room
    report: ValidationReport | None = None
    note: str = ""


_PAGE_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 28px;
  background: #0a0d13; color: #e6ebf5;
  font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
}
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: #8a94a8; font-size: 12px; margin-bottom: 24px; }
.summary {
  display: flex; flex-wrap: wrap; gap: 10px 24px;
  padding: 14px 18px; margin-bottom: 26px;
  background: #141926; border: 1px solid #222b3d; border-radius: 10px;
}
.summary div span { color: #8a94a8; }
.grid { display: flex; flex-wrap: wrap; gap: 22px; align-items: flex-start; }
/* Cards hug their map so relative room sizes stay comparable at a glance.
   The single-column grid pins the card width to the SVG and lets the caption
   wrap inside it instead of stretching the card. */
.card {
  display: grid; grid-template-columns: minmax(0, max-content);
  background: #141926; border: 1px solid #222b3d; border-radius: 10px;
  padding: 12px; max-width: 100%;
}
.card.invalid { border-color: #7a2c34; }
.card svg { display: block; max-width: 100%; height: auto; border-radius: 6px; }
.meta {
  margin-top: 10px; font-size: 11px; color: #8a94a8; line-height: 1.7;
  min-width: 0; overflow-wrap: anywhere;
}
.meta b { color: #cfd6e4; font-weight: 600; }
.tag {
  display: inline-block; padding: 1px 7px; border-radius: 999px;
  font-size: 10px; margin-right: 5px; border: 1px solid currentColor;
}
.tag.ok { color: #57e39b; }
.tag.bad { color: #ff8a8a; }
.tag.coop { color: #e368c8; }
.tag.fallback { color: #ffb454; }
"""


def render_gallery(
    entries: Sequence[GalleryEntry] | Iterable[Room],
    title: str = "Generated rooms",
    subtitle: str = "",
    cell: int = 14,
) -> str:
    """Render a gallery page. Accepts rooms or `GalleryEntry` objects."""
    items: list[GalleryEntry] = [
        e if isinstance(e, GalleryEntry) else GalleryEntry(room=e) for e in entries
    ]
    options = SvgOptions(cell=cell, legend=False, header=False, grid_lines=False)

    cards: list[str] = []
    for entry in items:
        room = entry.room
        valid = entry.report.ok if entry.report else True
        svg = render_svg(room, None, options)
        tags = [f'<span class="tag {"ok" if valid else "bad"}">{"valid" if valid else "invalid"}</span>']
        coop = room.metadata.get("cooperative_gates") or 0
        if coop:
            tags.append(f'<span class="tag coop">{coop} co-op</span>')
        if room.metadata.get("fallback"):
            tags.append('<span class="tag fallback">fallback</span>')

        counts = room.counts()
        meta_bits = [
            f"<b>seed</b> {room.seed}",
            f"<b>{room.width}x{room.height}</b> {escape(room.shape.value)}",
            f"<b>regions</b> {room.topology.region_count}",
            f"<b>doors</b> {counts.get('locked_door', 0)}",
            f"<b>keys</b> {counts.get('key', 0)}",
            f"<b>switches</b> {counts.get('switch', 0)}",
            f"<b>platforms</b> {counts.get('moving_platform', 0)}",
            f"<b>attempts</b> {room.metadata.get('attempts', 1)}",
        ]
        if entry.note:
            meta_bits.append(escape(entry.note))
        if entry.report and not entry.report.ok:
            meta_bits.append(escape(entry.report.summary()))

        cards.append(
            f'<div class="card{"" if valid else " invalid"}">{svg}'
            f'<div class="meta">{"".join(tags)}<br>{" · ".join(meta_bits)}</div></div>'
        )

    stats = _aggregate(items)
    summary = "".join(
        f"<div><span>{escape(label)}</span> {escape(str(value))}</div>"
        for label, value in stats
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{_PAGE_CSS}</style></head><body>"
        f"<h1>{escape(title)}</h1>"
        f"<div class='sub'>{escape(subtitle)}</div>"
        f"<div class='summary'>{summary}</div>"
        f"<div class='grid'>{''.join(cards)}</div>"
        "</body></html>"
    )


def _aggregate(entries: list[GalleryEntry]) -> list[tuple[str, object]]:
    if not entries:
        return []
    total = len(entries)
    valid = sum(1 for e in entries if (e.report.ok if e.report else True))
    coop = sum(1 for e in entries if (e.room.metadata.get("cooperative_gates") or 0) > 0)
    fallback = sum(1 for e in entries if e.room.metadata.get("fallback"))
    attempts = sum(e.room.metadata.get("attempts", 1) for e in entries)
    shapes = {}
    for entry in entries:
        shapes[entry.room.shape.value] = shapes.get(entry.room.shape.value, 0) + 1
    sizes = [e.room.width * e.room.height for e in entries]
    return [
        ("rooms", total),
        ("valid", f"{valid}/{total}"),
        ("with co-op gate", f"{coop}/{total}"),
        ("fallbacks", fallback),
        ("mean attempts", f"{attempts / total:.2f}"),
        ("mean area", f"{sum(sizes) / total:.0f} tiles"),
        ("shapes", ", ".join(f"{k}:{v}" for k, v in sorted(shapes.items()))),
    ]


def save_gallery(
    entries: Sequence[GalleryEntry] | Iterable[Room],
    path: str,
    title: str = "Generated rooms",
    subtitle: str = "",
    cell: int = 14,
) -> str:
    html = render_gallery(entries, title, subtitle, cell)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return path

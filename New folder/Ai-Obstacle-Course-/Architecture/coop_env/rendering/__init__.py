"""Renderers for visual inspection.

Nothing in the generation or validation layers imports this package -- rendering
is strictly a consumer of `Room` and `EpisodeState`.
"""

from .ascii_renderer import AsciiOptions, render_ascii, render_mechanism_report
from .gallery import GalleryEntry, render_gallery, save_gallery
from .palette import DARK, LIGHT, Theme, get_theme
from .svg_renderer import SvgOptions, render_svg, save_svg

__all__ = [
    "render_ascii",
    "AsciiOptions",
    "render_mechanism_report",
    "render_svg",
    "SvgOptions",
    "save_svg",
    "render_gallery",
    "save_gallery",
    "GalleryEntry",
    "Theme",
    "get_theme",
    "DARK",
    "LIGHT",
]

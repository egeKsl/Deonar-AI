# src/viz/draw.py
# Backwards-compatible façade that re-exports the previous draw.py API.
from ._colors import _hsv_to_bgr, _hash_color
from ._shapes import (
    _draw_filled_poly_alpha,
    _rounded_rect_pts,
    _draw_rounded_rect,
    draw_dashed_line,
    _rounded_rect,
)
from ._labels import _label_chip, put_hud, put_hud_enhanced, draw_zone_rect
from .pretty import (
    PrettyDrawConfig,
    draw_box_pretty,
    draw_centroid_pretty,
    draw_detection_pretty,
)
from .animator import CrossAnimator
from .colorer import CountColorer

# re-export commonly used helpers
__all__ = [
    "_hsv_to_bgr",
    "_hash_color",
    "_draw_filled_poly_alpha",
    "_rounded_rect_pts",
    "_draw_rounded_rect",
    "draw_dashed_line",
    "_rounded_rect",
    "_label_chip",
    "put_hud",
    "put_hud_enhanced",
    "draw_zone_rect",
    "PrettyDrawConfig",
    "draw_box_pretty",
    "draw_centroid_pretty",
    "draw_detection_pretty",
    "CrossAnimator",
    "CountColorer",
]

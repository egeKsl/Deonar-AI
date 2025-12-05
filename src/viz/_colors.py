# src/viz/_colors.py
from typing import Tuple


def _hsv_to_bgr(h, s, v) -> Tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (int(b * 255), int(g * 255), int(r * 255))


def _hash_color(tid: int) -> Tuple[int, int, int]:
    """Deterministic vivid color for a track id (BGR)."""
    h = (int(tid) * 0.6180339887) % 1.0  # golden ratio
    return _hsv_to_bgr(h, 0.75, 1.0)

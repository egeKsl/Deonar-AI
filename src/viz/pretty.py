# src/viz/pretty.py
"""HUD (heads-up display) rendering: draws the count overlay, FPS, slot status, and debug info onto frames."""
import cv2
import numpy as np
from typing import Optional, Tuple
from ._colors import _hash_color
from ._shapes import _draw_filled_poly_alpha, _draw_rounded_rect, _rounded_rect_pts
from ._labels import _label_chip

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


class PrettyDrawConfig:
    def __init__(
        self,
        mode: str = "auto",
        crowd_switch: int = 6,
        box_thick: int = 2,
        box_radius: int = 6,
        fill_alpha: float = 0.22,
        label_mode: str = "id",
        font_scale: float = 0.45,
        centroid_radius: int = 3,
    ):
        self.mode = mode
        self.crowd_switch = int(crowd_switch)
        self.box_thick = int(box_thick)
        self.box_radius = int(box_radius)
        self.fill_alpha = float(fill_alpha)
        self.label_mode = label_mode
        self.font_scale = float(font_scale)
        self.centroid_radius = int(centroid_radius)


def draw_box_pretty(
    img,
    box: Box,
    tid: int,
    color: Optional[Tuple[int, int, int]] = None,
    cfg: Optional[PrettyDrawConfig] = None,
    txt: Optional[str] = None,
):
    if cfg is None:
        cfg = PrettyDrawConfig()
    x1, y1, x2, y2 = map(int, box)
    if x2 <= x1 or y2 <= y1:
        return
    color = color or _hash_color(int(tid))
    _draw_filled_poly_alpha(
        img,
        _rounded_rect_pts(x1, y1, x2, y2, r=cfg.box_radius),
        color,
        alpha=cfg.fill_alpha,
    )
    _draw_rounded_rect(
        img, x1, y1, x2, y2, color, thick=cfg.box_thick, radius=cfg.box_radius
    )
    if cfg.label_mode == "id":
        label = f"#{int(tid)}" if txt is None else txt
        _label_chip(
            img,
            x1,
            max(0, y1 - 18),
            label,
            bg=color,
            fg=(255, 255, 255),
            scale=cfg.font_scale,
            thick=1,
        )


def _centroid_from(box, mask):
    x1, y1, x2, y2 = box
    if mask is not None and isinstance(mask, np.ndarray):
        ys, xs = np.where(mask)
        if xs.size > 0 and ys.size > 0:
            return float(xs.mean()), float(ys.mean())
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def draw_centroid_pretty(
    img,
    box,
    tid,
    mask=None,
    color: Optional[Tuple[int, int, int]] = None,
    cfg: Optional[PrettyDrawConfig] = None,
):
    if cfg is None:
        cfg = PrettyDrawConfig()
    cx, cy = _centroid_from(box, mask)
    color = color or _hash_color(int(tid))
    cv2.circle(
        img, (int(cx), int(cy)), cfg.centroid_radius, color, -1, lineType=cv2.LINE_AA
    )
    if cfg.label_mode == "id":
        _label_chip(
            img,
            int(cx) + 6,
            int(cy) - 14,
            f"#{int(tid)}",
            bg=color,
            fg=(255, 255, 255),
            scale=cfg.font_scale,
            thick=1,
        )


def draw_detection_pretty(
    img,
    box,
    tid,
    mask,
    color_if_counted,
    cfg: PrettyDrawConfig,
    num_dets_in_frame: int,
    txt: Optional[str] = None,
):
    mode = cfg.mode
    if mode == "auto" and num_dets_in_frame >= cfg.crowd_switch:
        mode = "centroid"
    color = color_if_counted
    if mode == "centroid":
        draw_centroid_pretty(img, box, tid, mask, color=color, cfg=cfg)
    else:
        draw_box_pretty(img, box, tid, color=color, cfg=cfg, txt=txt)

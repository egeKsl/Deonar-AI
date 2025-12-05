# src/viz/_shapes.py
import cv2
import numpy as np


def _draw_filled_poly_alpha(img, pts, color_bgr, alpha: float):
    if alpha <= 0:
        return
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color_bgr)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img)


def _rounded_rect_pts(x1, y1, x2, y2, r=6) -> np.ndarray:
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    w, h = x2 - x1, y2 - y1
    r = max(0.0, min(r, min(w, h) * 0.5))
    return np.array(
        [
            (x1 + r, y1),
            (x2 - r, y1),
            (x2, y1 + r),
            (x2, y2 - r),
            (x2 - r, y2),
            (x1 + r, y2),
            (x1, y2 - r),
            (x1, y1 + r),
        ],
        dtype=np.int32,
    )


def _draw_rounded_rect(img, x1, y1, x2, y2, color, thick=2, radius=6):
    pts = _rounded_rect_pts(x1, y1, x2, y2, r=radius)
    cv2.polylines(
        img,
        [pts],
        isClosed=True,
        color=color,
        thickness=int(thick),
        lineType=cv2.LINE_AA,
    )


def _rounded_rect(img, top_left, bottom_right, color, radius=12, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_dashed_line(img, p1, p2, dash=8, gap=6, color=(0, 0, 255), thickness=2):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    dist = int(np.hypot(x2 - x1, y2 - y1))
    if dist <= 1:
        return
    for i in range(0, dist, dash + gap):
        t1 = i / dist
        t2 = min(i + dash, dist) / dist
        xa = int(x1 + (x2 - x1) * t1)
        ya = int(y1 + (y2 - y1) * t1)
        xb = int(x1 + (x2 - x1) * t2)
        yb = int(y1 + (y2 - y1) * t2)
        cv2.line(img, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)

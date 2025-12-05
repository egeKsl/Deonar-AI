# src/geometry/geom.py
import cv2


def clamp_roi(rx, ry, rw, rh, W, H):
    rx = max(0, min(rx, W - 1))
    ry = max(0, min(ry, H - 1))
    rw = max(1, min(rw, W - rx))
    rh = max(1, min(rh, H - ry))
    return rx, ry, rw, rh


def line_side(px, py, ax, ay, bx, by):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def project_point_to_segment(px, py, ax, ay, bx, by):
    """returns (qx, qy) projection of P onto segment AB in same space."""
    abx, aby = (bx - ax), (by - ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-6:
        return ax, ay
    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    t = max(0.0, min(1.0, t))
    return ax + t * abx, ay + t * aby


def _prepare_geometry(cap, XR, YR, WR, HR):
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    rx, ry = int(W * XR), int(H * YR)
    rw, rh = int(W * WR), int(H * HR)
    rx, ry, rw, rh = clamp_roi(rx, ry, rw, rh, W, H)
    return fps, W, H, total, rx, ry, rw, rh


def _build_lines(count_line_roi_str, rx, ry, rw, rh):
    """Parse COUNT_LINE_ROI and return (lines_roi, lines_full)."""
    lines_roi, lines_full = [], []
    for line_str in count_line_roi_str.strip().split(";"):
        if not line_str.strip():
            continue
        ax_r, ay_r, bx_r, by_r = map(float, line_str.split(","))
        ax_roi, ay_roi = ax_r * rw, ay_r * rh
        bx_roi, by_roi = bx_r * rw, by_r * rh
        ax_full, ay_full = rx + ax_roi, ry + ay_roi
        bx_full, by_full = rx + bx_roi, ry + by_roi
        lines_roi.append((ax_roi, ay_roi, bx_roi, by_roi))
        lines_full.append((ax_full, ay_full, bx_full, by_full))
    return lines_roi, lines_full

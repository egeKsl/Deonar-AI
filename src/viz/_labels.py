# src/viz/_labels.py
"""Label drawing helpers: track ID overlays, count text, HUD text rendering."""
import cv2
import numpy as np
from ._shapes import _rounded_rect, _draw_filled_poly_alpha, _rounded_rect_pts
from typing import Tuple


def _label_chip(
    img,
    x,
    y,
    text: str,
    bg: Tuple[int, int, int],
    fg=(255, 255, 255),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.45,
    thick=1,
    pad=3,
    r=4,
    alpha=0.90,
):
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    w, h = tw + pad * 2, th + pad * 2
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    chip = _rounded_rect_pts(x1, y1, x2, y2, r=r)
    _draw_filled_poly_alpha(img, chip, bg, alpha=alpha)
    cv2.polylines(img, [chip], True, bg, 1, cv2.LINE_AA)
    tx = x1 + pad
    ty = y2 - pad - base
    b, g, r_ = bg
    luminance = 0.2126 * r_ + 0.7152 * g + 0.0722 * b
    fg_auto = (0, 0, 0) if luminance > 160 else (255, 255, 255)
    use_fg = fg if isinstance(fg, tuple) and len(fg) == 3 else fg_auto
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, scale, use_fg, thick, cv2.LINE_AA)


def put_hud(img, lines, org=(8, 24), color=(40, 255, 40)):
    x, y = org
    for line in lines:
        cv2.putText(
            img,
            line,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
        )
        y += 34


def put_hud_enhanced(
    img,
    hud,
    org=(18, 28),
    font_scale=0.65,
    thickness=2,
    line_spacing=28,
    blur_strength=9,
    radius=12,
    alpha=0.45,
):
    if img is None or not isinstance(img, np.ndarray):
        return
    H, W = img.shape[:2]
    count_line = f"COUNT:  UP {hud.get('up',0)}   DOWN {hud.get('down',0)}   TOTAL: {hud.get('total',0)}"
    infer_fps = float(hud.get("infer_fps", hud.get("fps", 0.0)) or 0.0)
    e2e_fps = float(hud.get("e2e_fps", 0.0) or 0.0)
    frame_line = f"FRAMES: {hud.get('frame_in',0)}/{hud.get('out_index',0)}   SOURCE: {hud.get('res','?')}   FPS: {infer_fps:.1f}/{e2e_fps:.1f}"
    q_line = f"PACING_Q: {hud.get('pacing_out_q_fill',0)}/{hud.get('pacing_out_q_max',0)}   RES_Q: {hud.get('res_q_fill',0)}/{hud.get('res_q_max',0)}"
    lines = [count_line, frame_line, q_line]
    base_x, base_y = org
    pad_x = 16
    pad_y = 8
    max_w = min(900, W - 2 * pad_x)
    text_sizes = [
        cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        for t in lines
    ]
    text_w = max(ts[0] for ts in text_sizes)
    box_w = min(max_w, text_w + pad_x * 2)
    box_h = int(len(lines) * line_spacing + pad_y * 2)
    x1 = max(8, base_x)
    y1 = max(8, base_y - 6)
    x2 = min(W - 8, x1 + box_w)
    y2 = min(H - 8, y1 + box_h)
    blur_margin = 8
    bx1 = max(0, x1 - blur_margin)
    by1 = max(0, y1 - blur_margin)
    bx2 = min(W, x2 + blur_margin)
    by2 = min(H, y2 + blur_margin)
    roi = img[by1:by2, bx1:bx2].copy()
    try:
        k = blur_strength
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
    except Exception:
        blurred = roi
    overlay = img.copy()
    overlay[by1:by2, bx1:bx2] = blurred
    mask = np.zeros_like(overlay, dtype=np.uint8)
    _rounded_rect(
        mask, (x1, y1), (x2, y2), (255, 255, 255), radius=radius, thickness=-1
    )
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_bool = (mask_gray > 0).astype(np.uint8)[:, :, None]
    img[by1:by2, bx1:bx2] = (
        img[by1:by2, bx1:bx2]
        * (1 - alpha * mask_bool[by1 - by1 : by2 - by1, bx1 - bx1 : bx2 - bx1])
        + overlay[by1:by2, bx1 : bx1 + (bx2 - bx1)]
        * (alpha * mask_bool[by1 - by1 : by2 - by1, bx1 - bx1 : bx2 - bx1])
    ).astype(np.uint8)
    border = img.copy()
    _rounded_rect(border, (x1, y1), (x2, y2), (40, 40, 40), radius=radius, thickness=1)
    cv2.addWeighted(border, 0.18, img, 0.82, 0, img)
    shadow = img.copy()
    sh_off = 4
    _rounded_rect(
        shadow,
        (x1 + sh_off, y1 + sh_off),
        (x2 + sh_off, y2 + sh_off),
        (10, 10, 10),
        radius=radius,
        thickness=-1,
    )
    cv2.addWeighted(shadow, 0.06, img, 0.94, 0, img)
    COLOR_COUNT = (200, 255, 200)
    COLOR_FPS = (255, 220, 120)
    COLOR_SYS = (200, 200, 210)
    SHADOW_COLOR = (10, 10, 10)
    tx = x1 + pad_x // 2
    ty = y1 + pad_y + line_spacing - 8
    for i, txt in enumerate(lines):
        cv2.putText(
            img,
            txt,
            (tx + 1, ty + 1 + i * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            SHADOW_COLOR,
            thickness + 1,
            cv2.LINE_AA,
        )
        if i == 0:
            color = COLOR_COUNT
        elif i == 1:
            color = COLOR_FPS
        else:
            color = COLOR_SYS
        cv2.putText(
            img,
            txt,
            (tx, ty + i * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    sep_y = y1 + box_h - pad_y - (line_spacing // 2)
    cv2.line(img, (x1 + 6, sep_y), (x2 - 6, sep_y), (100, 100, 100), 1, cv2.LINE_AA)
    return


def put_hud_slot_enhanced(
    img,
    rows,
    org=(18, 28),
    align="top-left",  # NEW: top-left | top-right | bottom-left | bottom-right
    font_scale=0.60,
    thickness=1,
    line_spacing=30,
    radius=14,
    alpha=0.78,
):
    """
    GOD-LEVEL slot HUD card with strong contrast, glow bullets, hierarchy, and accent strip.

    Args:
        img: target frame (BGR).
        rows: list of (key, value, accent_bgr).
        org: margin from edge (x, y).
        align: HUD alignment anchor.
    """
    if img is None or not isinstance(img, np.ndarray) or not rows:
        return

    H, W = img.shape[:2]
    margin_x, margin_y = int(org[0]), int(org[1])
    pad_x, pad_y = 16, 14
    bullet_r = 4
    bullet_gap = 12

    # ---------------------------
    # Measure text
    # ---------------------------
    key_sizes = [
        cv2.getTextSize(f"{k}:", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness + 1)[0]
        for k, _, _ in rows
    ]
    val_sizes = [
        cv2.getTextSize(v, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        for _, v, _ in rows
    ]

    key_w = max((s[0] for s in key_sizes), default=0)
    row_w = max((k[0] + v[0] for k, v in zip(key_sizes, val_sizes)), default=0)

    box_w = row_w + pad_x * 2 + bullet_r * 2 + bullet_gap + 26
    box_h = int(len(rows) * line_spacing + pad_y * 2)

    box_w = min(box_w, W - 16)
    box_h = min(box_h, H - 16)

    # ---------------------------
    # ALIGNMENT MATH (THE ONLY REAL CHANGE)
    # ---------------------------
    if align == "top-left":
        x1 = margin_x
        y1 = margin_y

    elif align == "top-right":
        x1 = W - box_w - margin_x
        y1 = margin_y

    elif align == "bottom-left":
        x1 = margin_x
        y1 = H - box_h - margin_y

    elif align == "bottom-right":
        x1 = W - box_w - margin_x
        y1 = H - box_h - margin_y

    else:
        # fallback
        x1 = margin_x
        y1 = margin_y

    x1 = max(8, x1)
    y1 = max(8, y1)
    x2, y2 = x1 + box_w, y1 + box_h

    # ---------------------------
    # Background card
    # ---------------------------
    overlay = img.copy()
    _rounded_rect(
        overlay,
        (x1, y1),
        (x2, y2),
        (10, 12, 14),
        radius=radius,
        thickness=-1,
    )
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    _rounded_rect(
        img,
        (x1, y1),
        (x2, y2),
        (90, 100, 110),
        radius=radius,
        thickness=1,
    )

    # Accent strip
    cv2.rectangle(
        img,
        (x1 + radius, y1),
        (x2 - radius, y1 + 4),
        (60, 180, 255),
        -1,
    )

    key_color = (170, 180, 190)
    value_color = (245, 248, 252)
    shadow = (0, 0, 0)

    ty = y1 + pad_y + 18

    # ---------------------------
    # Rows
    # ---------------------------
    for key, value, accent in rows:
        by = ty - 6
        bx = x1 + pad_x + bullet_r

        # Glow rings
        cv2.circle(img, (bx, by), bullet_r + 4, accent, 1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), bullet_r + 2, accent, 1, cv2.LINE_AA)

        # Bullet core
        cv2.circle(img, (bx, by), bullet_r, accent, -1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), bullet_r + 1, (0, 0, 0), 1, cv2.LINE_AA)

        key_text = f"{key}:"
        kx = x1 + pad_x + bullet_r * 2 + bullet_gap
        vx = kx + key_w + 10

        cv2.putText(
            img,
            key_text,
            (kx + 2, ty + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            shadow,
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            key_text,
            (kx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            key_color,
            thickness + 1,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            value,
            (vx + 2, ty + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            shadow,
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            value,
            (vx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            value_color,
            thickness,
            cv2.LINE_AA,
        )

        ty += line_spacing


def draw_zone_rect(
    img,
    rect_xyxy,
    color_fill=(180, 220, 255),
    alpha=0.25,
    border_color=(0, 200, 255),
    thickness=2,
):
    """Filled translucent zone rectangle with crisp border."""
    import cv2

    x1, y1, x2, y2 = map(int, rect_xyxy)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_fill, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, thickness, cv2.LINE_AA)

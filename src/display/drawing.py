# src/display/drawing.py

import cv2, traceback

import numpy as np
from numpy import where

from pathlib import Path
from datetime import datetime

from src.viz.draw import (
    put_hud,
    CrossAnimator,
    CountColorer,
    draw_zone_rect,
    PrettyDrawConfig,
    draw_detection_pretty,
)

from src.utils.logger import log

HUD_MODE_IDLE = "idle"
HUD_MODE_SLOT = "slot_active"


def _parse_bgr(s, fallback):
    try:
        b, g, r = [int(x.strip()) for x in s.split(",")]
        return (b, g, r)
    except Exception:
        return fallback


def _prepare_drawing(args):
    counted_bgr = _parse_bgr(args.count_box_color, (255, 0, 0))
    base_bgr = _parse_bgr(args.box_color, (0, 255, 0))
    colorer = CountColorer(
        mode=args.count_box_mode,
        duration_frames=args.count_box_frames,
        counted_color=counted_bgr,
        base_color=base_bgr,
    )
    animator = CrossAnimator(duration_frames=20)
    pretty_cfg = PrettyDrawConfig(
        mode=args.viz_mode,
        crowd_switch=args.viz_crowd_switch,
        box_thick=args.viz_box_thick,
        box_radius=args.viz_box_radius,
        fill_alpha=args.viz_fill_alpha,
        font_scale=args.viz_font_scale,
        centroid_radius=args.viz_centroid_radius,
    )
    return animator, colorer, pretty_cfg


def _setup_windows(show_windows, args, rw, rh, W, H):
    if show_windows and (not args.no_roi):
        cv2.namedWindow("ROI View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI View", 960, int(960 * rh / max(1, rw)))
    if show_windows and (not args.no_full):
        cv2.namedWindow("Full View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Full View", 960, int(960 * H / max(1, W)))


def _save_screenshot(full_disp, source_path, frame_idx, run_root):
    """
    Save a unique screenshot under:
      <run_root>/images/<source>_<timestamp>_f<frame>.jpg
    """
    if full_disp is None:
        raise

    # images directory inside this run
    out_dir = Path(run_root) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(source_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"{prefix}_{ts}_f{frame_idx}.jpg"

    out_path = out_dir / fname

    cv2.imwrite(str(out_path), full_disp)
    log.success("DRAWING-SCREENSHOT", f"Saved: {out_path}")


# -------------------- Extracted drawing helpers --------------------


def _draw_lines_roi(
    img, use_zone, zone_rect_roi, use_dual, lineA_roi, lineB_roi, lines_roi
):
    """
    Draw counting lines or zone rectangle on ROI image.
    Logic preserved exactly from original inner helper.
    """
    if use_zone and zone_rect_roi:
        draw_zone_rect(
            img,
            zone_rect_roi,
            color_fill=(180, 220, 255),
            alpha=0.25,
            border_color=(0, 200, 255),
            thickness=2,
        )
        return

    if use_dual and (lineA_roi is not None) and (lineB_roi is not None):
        ax, ay, bx, by = map(int, lineA_roi)
        cv2.line(img, (ax, ay), (bx, by), (0, 200, 255), 2)
        cv2.putText(
            img,
            "A",
            (ax, ay - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        ax2, ay2, bx2, by2 = map(int, lineB_roi)
        cv2.line(img, (ax2, ay2), (bx2, by2), (255, 60, 200), 2)
        cv2.putText(
            img,
            "B",
            (ax2, ay2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 60, 200),
            2,
            cv2.LINE_AA,
        )
        return

    # Default multi-line drawing
    for ax, ay, bx, by in lines_roi or []:
        try:
            cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), (0, 200, 255), 2)
        except Exception:
            # defensive: skip malformed lines
            continue


def _draw_detections_roi(img, dets, colorer, feeder, use_zone, zone, pretty_cfg):
    """
    Draw detections on the ROI image (pretty draw with masks and zone color logic).
    """
    for box, tid, conf, cl, mask in dets or []:
        try:
            box_color = colorer.color_for(tid, getattr(feeder, "out_index", 0))
        except Exception:
            box_color = getattr(colorer, "base_color", (255, 0, 0))

        # zone-specific color override logic
        if use_zone and zone is not None:
            try:
                x1, y1, x2, y2 = box
                if mask is not None:
                    ys, xs = where(mask)
                    if xs.size > 0 and ys.size > 0:
                        cx, cy = xs.mean(), ys.mean()
                    else:
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                else:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # Orange tweak (but only if NOT ghost)
                if tid not in getattr(zone, "ghost_ids", set()):
                    if zone._inside(cx, cy) and box_color == colorer.base_color:
                        box_color = (0, 165, 255)

                # Ghost override ALWAYS last
                if tid in getattr(zone, "ghost_ids", set()) and zone._inside(cx, cy):
                    box_color = (0, 0, 255)
            except Exception:
                # ignore zone-related failure for this detection
                pass

        # draw pretty detection in ROI (mask supported)
        try:
            draw_detection_pretty(
                img,
                box,
                int(tid),
                mask,
                box_color,
                pretty_cfg,
                num_dets_in_frame=len(dets or []),
            )
        except Exception:
            # defensive: if pretty draw fails, draw a simple rect
            try:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            except Exception:
                pass


def _draw_lines_full(
    img,
    rx,
    ry,
    rw,
    rh,
    use_zone,
    zone_rect_full,
    use_dual,
    lineA_full,
    lineB_full,
    lines_full,
):
    """
    Draw ROI rectangle and counting lines on full image.
    """
    # rectangle marking ROI on full view
    try:
        cv2.rectangle(
            img, (int(rx), int(ry)), (int(rx + rw), int(ry + rh)), (0, 200, 255), 2
        )
    except Exception:
        pass

    if use_zone and zone_rect_full:
        draw_zone_rect(
            img,
            zone_rect_full,
            color_fill=(180, 220, 255),
            alpha=0.25,
            border_color=(0, 200, 255),
            thickness=2,
        )
        return

    if use_dual and (lineA_full is not None) and (lineB_full is not None):
        try:
            ax, ay, bx, by = map(int, lineA_full)
            cv2.line(img, (ax, ay), (bx, by), (0, 200, 255), 2)
            cv2.putText(
                img,
                "A",
                (ax, ay - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            ax2, ay2, bx2, by2 = map(int, lineB_full)
            cv2.line(img, (ax2, ay2), (bx2, by2), (255, 60, 200), 2)
            cv2.putText(
                img,
                "B",
                (ax2, ay2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 60, 200),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            # skip malformed dual lines
            pass
        return

    for ax, ay, bx, by in lines_full or []:
        try:
            cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), (0, 200, 255), 2)
        except Exception:
            continue


def _draw_detections_full(
    img, dets, colorer, feeder, rx, ry, pretty_cfg, use_zone, zone
):
    """
    Draw detections on the full image (translating ROI coordinates into full-space).
    """
    for box, tid, conf, cl, mask in dets or []:
        try:
            box_color = colorer.color_for(tid, getattr(feeder, "out_index", 0))
        except Exception:
            box_color = getattr(colorer, "base_color", (255, 0, 0))

        # zone-specific tweak
        if use_zone and zone is not None:
            try:
                x1, y1, x2, y2 = box
                if mask is not None:
                    ys, xs = where(mask)
                    if xs.size > 0 and ys.size > 0:
                        cx, cy = xs.mean(), ys.mean()
                    else:
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                else:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if tid not in getattr(zone, "ghost_ids", set()):
                    if zone._inside(cx, cy) and box_color == colorer.base_color:
                        box_color = (0, 165, 255)
                if tid in getattr(zone, "ghost_ids", set()) and zone._inside(cx, cy):
                    box_color = (0, 0, 255)
            except Exception:
                pass

        # draw detection translated to full-space
        try:
            full_box = (
                int(box[0] + rx),
                int(box[1] + ry),
                int(box[2] + rx),
                int(box[3] + ry),
            )
            draw_detection_pretty(
                img,
                full_box,
                int(tid),
                None,
                box_color,
                pretty_cfg,
                num_dets_in_frame=len(dets or []),
            )
        except Exception:
            # fallback simple rect
            try:
                x1, y1, x2, y2 = map(
                    int, (box[0] + rx, box[1] + ry, box[2] + rx, box[3] + ry)
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            except Exception:
                pass


def _build_hud_data(feeder, state, res, threaded_streaming, args, W, H, rw, rh, total):
    """
    Build HUD data dict (prefers threaded `res` if provided).
    Mirrors original logic exactly.
    """
    try:
        hud_data = {}
        hud_data["up"] = getattr(state, "up_count", 0)
        hud_data["down"] = getattr(state, "down_count", 0)
        hud_data["total"] = hud_data["up"] + hud_data["down"]
        hud_data["frame_in"] = int(getattr(feeder, "frame_in", 0))
        hud_data["out_index"] = int(getattr(feeder, "out_index", 0))
        # try from res first (in threaded mode display gets res from infer)
        if threaded_streaming and isinstance(res, dict):
            hud_data["infer_fps"] = float(
                res.get("infer_fps", res.get("avg_fps", 0.0)) or 0.0
            )
            hud_data["e2e_fps"] = float(res.get("e2e_fps", 0.0) or 0.0)
            # Keep legacy `fps` key for backward compatibility in any custom HUD renderer.
            hud_data["fps"] = hud_data["infer_fps"]
            hud_data["pacing_out_q_fill"] = int(res.get("pacing_out_q_fill", 0) or 0)
            hud_data["res_q_fill"] = int(res.get("result_q_fill", 0) or 0)
            hud_data["pacing_out_q_max"] = int(
                res.get("pacing_out_q_max", getattr(args, "cap_qsize", 0) or 0) or 0
            )
            hud_data["res_q_max"] = int(
                res.get("result_q_max", getattr(args, "res_qsize", 0) or 0) or 0
            )
        else:
            hud_data["infer_fps"] = float(getattr(feeder, "infer_fps", 0.0) or 0.0)
            hud_data["e2e_fps"] = float(getattr(feeder, "e2e_fps", 0.0) or 0.0)
            hud_data["fps"] = (
                hud_data["infer_fps"]
                if hud_data["infer_fps"] > 0.0
                else float(getattr(feeder, "avg_fps", 0.0) or 0.0)
            )
            hud_data["pacing_out_q_fill"] = int(
                getattr(feeder, "pacing_out_qsize", 0) or 0
            )
            hud_data["res_q_fill"] = int(getattr(feeder, "result_qsize", 0) or 0)
            hud_data["pacing_out_q_max"] = int(
                getattr(args, "pacing_out_qsize", 0) or 0
            )
            hud_data["res_q_max"] = int(getattr(args, "res_qsize", 0) or 0)
        hud_data["res"] = f"{int(W)}x{int(H)}"
        return hud_data
    except Exception:
        return {
            "up": getattr(state, "up_count", 0),
            "down": getattr(state, "down_count", 0),
            "total": getattr(state, "up_count", 0) + getattr(state, "down_count", 0),
            "frame_in": int(getattr(feeder, "frame_in", 0)),
            "out_index": int(getattr(feeder, "out_index", 0)),
            "infer_fps": 0.0,
            "e2e_fps": 0.0,
            "fps": 0.0,
            "res": f"{int(W)}x{int(H)}",
            "cap_q": 0,
            "res_q": 0,
            "cap_max": int(getattr(args, "cap_qsize", 0) or 0),
            "res_max": int(getattr(args, "res_qsize", 0) or 0),
        }


# -------------------- Main compose function (keeps original signature) --------------------


def _compose_frames(
    roi_frame,
    dets,
    lines_roi,
    lines_full,
    feeder,
    rw,
    rh,
    rx,
    ry,
    W,
    H,
    total,
    state,
    animator,
    colorer,
    args,
    pretty_cfg,
    # zone extras
    use_zone=False,
    zone=None,
    zone_rect_roi=None,
    zone_rect_full=None,
    # dual-line extras
    use_dual=False,
    lineA_roi=None,
    lineB_roi=None,
    lineA_full=None,
    lineB_full=None,
    # for threading: if True, we expect a res dict passed via `res` kwarg that
    # contains queue/fps info. Otherwise, legacy display is used.
    threaded_streaming=False,
    res=None,
    hud_mode=HUD_MODE_IDLE,
    slot_hud=None,
):
    """
    Return (roi_disp, full_disp) with overlays. Draw zone OR lines.
    Refactored into smaller helper functions for clarity while preserving
    original behaviour.
    """
    # Defensive copies / fallbacks ------------------------------------------------
    try:
        roi_disp = roi_frame.copy()
    except Exception:
        # ensure we always have an image
        roi_disp = 255 * np.ones((int(rh or 240), int(rw or 320), 3), dtype=np.uint8)

    # --- ROI drawing (lines + detections) --------------------------------------
    try:
        _draw_lines_roi(
            roi_disp, use_zone, zone_rect_roi, use_dual, lineA_roi, lineB_roi, lines_roi
        )
    except Exception:
        log.warn("DRAW", "Failed drawing ROI lines: " + traceback.format_exc())

    try:
        _draw_detections_roi(
            roi_disp, dets, colorer, feeder, use_zone, zone, pretty_cfg
        )
    except Exception:
        log.warn("DRAW", "Failed drawing ROI detections: " + traceback.format_exc())

    # --- ROI HUD (legacy simple HUD on ROI) -----------------------------------
    try:
        hud_lines = [
            f"Up: {state.up_count} Down: {state.down_count}",
            f"Frame {getattr(feeder, 'frame_in', '?')}/{total if total > 0 else '?'} {rw}x{rh}",
        ]
        if use_zone:
            hud_lines.insert(0, "Zone mode")
        elif use_dual:
            hud_lines.insert(0, "Dual-line mode (A/B)")
        else:
            hud_lines.insert(0, f"Goats in ROI: {len(dets or [])}")
        put_hud(roi_disp, hud_lines)
    except Exception:
        log.debug("DRAW", "ROI HUD failed: " + traceback.format_exc())

    # --- Full view base image & overlays ---------------------------------------
    try:
        full_disp = feeder.full.copy()
    except Exception:
        # fallback safe full image
        full_disp = 255 * np.ones(
            (int(H or rh or 480), int(W or rw or 640), 3), dtype=np.uint8
        )

    try:
        _draw_lines_full(
            full_disp,
            rx,
            ry,
            rw,
            rh,
            use_zone,
            zone_rect_full,
            use_dual,
            lineA_full,
            lineB_full,
            lines_full,
        )
    except Exception:
        log.warn("DRAW", "Failed drawing full lines: " + traceback.format_exc())

    try:
        _draw_detections_full(
            full_disp, dets, colorer, feeder, rx, ry, pretty_cfg, use_zone, zone
        )
    except Exception:
        log.warn("DRAW", "Failed drawing full detections: " + traceback.format_exc())

    # --- HUD for threaded streaming or legacy full HUD ------------------------
    try:
        if hud_mode == HUD_MODE_SLOT and slot_hud:
            # ---- SLOT ACTIVE HUD (MINIMAL, 3 LINES) ----
            lines = [
                f"SLOT: {slot_hud['slot_id']}  |  START: {slot_hud['slot_start'][:19].replace('T',' ')}",
                f"COUNT: UP {slot_hud['up']}  DOWN {slot_hud['down']}  TOTAL {slot_hud['total']}",
                f"FRAME: {getattr(feeder, 'frame_in', '?')}/{getattr(feeder, 'out_index', '?')}   FPS: {float(res.get('infer_fps', res.get('avg_fps', 0.0))):.1f}/{float(res.get('e2e_fps', 0.0)):.1f}   TIME: {slot_hud['now']}",
            ]
            put_hud(full_disp, lines, org=(8, 32))
        elif threaded_streaming:
            # prefer enhanced HUD when streaming threaded results
            hud_data = _build_hud_data(
                feeder, state, res, threaded_streaming, args, W, H, rw, rh, total
            )
            try:
                # lazy import (user may have added put_hud_enhanced to viz.draw)
                from src.viz.draw import put_hud_enhanced

                put_hud_enhanced(full_disp, hud_data)
            except Exception:
                # fallback to simple put_hud text if enhanced not available
                put_hud(
                    full_disp,
                    [
                        f"Counts Up:{hud_data['up']} Down:{hud_data['down']} Total:{hud_data['total']}",
                        f"Frame {hud_data['frame']}/{total if total > 0 else '?'} Source {hud_data['res']}",
                    ],
                    org=(8, 32),
                )
        else:
            # original HUD behavior for non-threaded cases
            put_hud(
                full_disp,
                [
                    f"Counts Up:{state.up_count} Down:{state.down_count} Total:{state.up_count + state.down_count}",
                    f"Frame {getattr(feeder, 'frame_in', '?')}/{total if total > 0 else '?'} Source {W}x{H}",
                ],
                org=(8, 32),
            )
    except Exception:
        log.debug("DRAW", "Full HUD failed: " + traceback.format_exc())

    # --- animator draw (non-fatal) ---------------------------------------------
    try:
        if animator is not None:
            animator.draw(roi_disp, full_disp, rx, ry, getattr(feeder, "out_index", 0))
    except Exception:
        log.warn("ANIMATOR", "Animator draw failure:\n" + traceback.format_exc())

    return roi_disp, full_disp

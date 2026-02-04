# src/counting/counting_modes/init_counting_modes.py

from src.counting.counting_modes.dual_lines import DualLineCounter
from src.utils.logger import log


def _init_dual_lines(args, lines_roi, lines_full, rx, ry, rw, rh):
    """
    Initializes dual-line mode if enabled and only 1 ROI line is present.
    Returns: (use_dual, dual_counter, lineA_roi, lineB_roi, lineA_full, lineB_full)
    """
    use_dual = bool(args.dual_lines_enabled)
    dual = None
    lineA_roi = lineB_roi = None
    lineA_full = lineB_full = None

    if use_dual:
        if len(lines_roi) != 1:
            use_dual = False
        else:
            lineA_roi = lines_roi[0]
            lineA_full = lines_full[0]

            # --- Compute Line B (parallel, robust for vertical/horizontal/slanted) ---
            delta = int(args.dual_offset_px)
            widthB = int(args.dual_width_px)

            ax, ay, bx, by = lineA_roi
            # direction vector from A->B
            dx = bx - ax
            dy = by - ay
            # length (avoid zero division)
            import math

            L = math.hypot(dx, dy) or 1.0

            # unit tangent (along the line)
            tx = dx / L
            ty = dy / L

            # perpendicular normal vector (one of the two normals)
            nx = -dy / L
            ny = dx / L

            # Decide sign so that:
            # - if line is mostly vertical (|dy| >= |dx|) -> shift left (negative x)
            # - if line is mostly horizontal (|dx| > |dy|) -> shift up (negative y)
            if abs(dx) > abs(dy):
                # mostly horizontal -> prefer upward shift
                sign = -1.0
            else:
                # mostly vertical or diagonal -> prefer leftward shift
                sign = 1.0

            sx = nx * (delta * sign)
            sy = ny * (delta * sign)

            if widthB > 0:
                # build a line of length widthB centered on the midpoint of A-B,
                # then shift it by (sx, sy)
                cx = (ax + bx) / 2.0
                cy = (ay + by) / 2.0
                half_w = widthB / 2.0
                # endpoints along tangent direction, then shifted by normal
                x1f = cx - tx * half_w + sx
                y1f = cy - ty * half_w + sy
                x2f = cx + tx * half_w + sx
                y2f = cy + ty * half_w + sy
            else:
                # simply offset both endpoints of the original line
                x1f = ax + sx
                y1f = ay + sy
                x2f = bx + sx
                y2f = by + sy

            # Now convert to ints and clamp inside ROI bounds [0..rw],[0..rh]
            raw_x1 = int(round(x1f))
            raw_y1 = int(round(y1f))
            raw_x2 = int(round(x2f))
            raw_y2 = int(round(y2f))
            # clamp inside ROI size
            x1 = max(0, min(rw, raw_x1))
            y1 = max(0, min(rh, raw_y1))
            x2 = max(0, min(rw, raw_x2))
            y2 = max(0, min(rh, raw_y2))
            lineB_roi = (x1, y1, x2, y2)

            # full-frame coords
            lineB_full = (rx + x1, ry + y1, rx + x2, ry + y2)

            dual = DualLineCounter(
                lineA_roi=lineA_roi,
                lineB_roi=lineB_roi,
                hyst_frames=args.dual_hyst_frames,
                window_frames=args.dual_window_frames,
                id_lock_frames=args.dual_id_lock_frames,
                quiet=args.quiet,
                debug_enabled=args.debug_enabled,
                min_nonzero_abs=getattr(args, "dual_min_nonzero_abs", 3),
                min_nonzero_ratio=getattr(args, "dual_min_nonzero_ratio", 0.6),
                motion_min_frames=getattr(args, "dual_motion_min_frames", 6),
                motion_min_displacement_px=getattr(
                    args, "dual_motion_min_displacement_px", 12.0
                ),
                motion_max_lookback_frames=getattr(
                    args, "dual_motion_max_lookback_frames", 30
                ),
                motion_dir_consistency_ratio=getattr(
                    args, "dual_motion_dir_consistency_ratio", 0.7
                ),
                motion_dir_consistency_min_frames=getattr(
                    args, "dual_motion_dir_consistency_min_frames", 4
                ),
                motion_axis_min_displacement_px=getattr(
                    args, "dual_motion_axis_min_displacement_px", 10.0
                ),
            )

            if not args.quiet:
                log.info(
                    "MODES-DUAL",
                    f" Enabled mode={args.dual_mode} | "
                    f"A={tuple(map(int,lineA_roi))}  B={tuple(map(int,lineB_roi))}  "
                    f"Δ={delta}px widthB={widthB if widthB>0 else 'same as A'}",
                )

    return use_dual, dual, lineA_roi, lineB_roi, lineA_full, lineB_full


def _init_zone(args, rw, rh, rx, ry):
    """
    Initializes zone mode if COUNT_MODE=zone and ratios are valid.
    Returns: (use_zone, zone_placeholder, zone_rect_roi, zone_rect_full)
    """
    use_zone = args.count_mode == "zone" and not args.dual_lines_enabled
    zone_rect_roi = None
    zone_rect_full = None
    zone = None

    if use_zone:
        if not args.zone_rect_ratios:
            log.error(
                "MODES-ZONE",
                " ZONE_RECT missing in .env; falling back to COUNT_MODE=line.",
            )
            use_zone = False
        else:
            rx_r, ry_r, rw_r, rh_r = args.zone_rect_ratios
            ok_ratio = (
                (0.0 <= rx_r <= 1.0)
                and (0.0 <= ry_r <= 1.0)
                and (0.0 < rw_r <= 1.0)
                and (0.0 < rh_r <= 1.0)
            )
            if not ok_ratio:
                log.error(
                    "MODES-ZONE",
                    " ZONE_RECT ratios out of [0..1]. Falling back to COUNT_MODE=line.",
                )
                use_zone = False
            else:
                zx1 = int(rw * rx_r)
                zy1 = int(rh * ry_r)
                zw = int(rw * rw_r)
                zh = int(rh * rh_r)
                zx2 = zx1 + max(1, zw)
                zy2 = zy1 + max(1, zh)
                if zx1 < 0 or zy1 < 0 or zx2 > rw or zy2 > rh:
                    log.error(
                        "MODES-ZONE",
                        f" ZONE_RECT ({rx_r:.3f},{ry_r:.3f},{rw_r:.3f},{rh_r:.3f}) "
                        f"exceeds ROI bounds {rw}x{rh}. Falling back to COUNT_MODE=line.",
                    )
                    use_zone = False
                else:
                    zone_rect_roi = (float(zx1), float(zy1), float(zx2), float(zy2))
                    zone_rect_full = (
                        float(rx + zx1),
                        float(ry + zy1),
                        float(rx + zx2),
                        float(ry + zy2),
                    )
                    log.info(
                        "MODES-ZONE",
                        f" Enabled. ROI-rect px: {zone_rect_roi}  (within ROI {rw}x{rh})",
                    )

    return use_zone, zone, zone_rect_roi, zone_rect_full

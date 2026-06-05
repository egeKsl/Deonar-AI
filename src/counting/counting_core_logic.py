# src/counting/counting_core_logic.py

import math
import numpy as np
from src.utils.logger import log
from src.geometry.geom import line_side
from datetime import datetime, timezone


def _side_sign(cx, cy, ax, ay, bx, by, margin_px):
    """Return -1 / 0 / +1 based on which side of the line (with margin)."""
    raw = line_side(cx, cy, ax, ay, bx, by)
    # distance from point to line segment
    num = abs((bx - ax) * (ay - cy) - (by - ay) * (ax - cx))
    den = max(1e-6, math.hypot(bx - ax, by - ay))
    dist = num / den
    if dist < margin_px:
        return 0
    return 1 if raw > 0 else -1


def _update_counts_for_frame(
    dets, lines_roi, lines_full, feeder, fps, args, state, animator, csvs, colorer
):
    """LINE MODE: Mutates state (up/down counts, histories) and triggers animations/CSV."""
    for box, tid, conf, cl, mask in dets:
        x1, y1, x2, y2 = box

        # centroid by mask when available, else box center
        if mask is not None:
            ys, xs = np.where(mask)
            if xs.size > 0 and ys.size > 0:
                cx, cy = xs.mean(), ys.mean()
            else:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        else:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        state.age_frames[tid] += 1

        for line_idx, (ax_roi, ay_roi, bx_roi, by_roi) in enumerate(lines_roi):
            sgn = _side_sign(
                cx, cy, ax_roi, ay_roi, bx_roi, by_roi, args.line_margin_px
            )
            if sgn != 0:
                state.side_hist[tid].append(sgn)

            h = state.side_hist[tid]
            confirmed = None
            if len(h) >= args.min_side_frames and all(v == h[0] and v != 0 for v in h):
                confirmed = h[0]

            if confirmed is None:
                continue

            prev = state.last_side.get(tid, None)
            state.last_side[tid] = confirmed

            flipped = (
                prev is not None and prev != 0 and confirmed != 0 and prev != confirmed
            )
            ready_age = state.age_frames[tid] >= args.min_age
            ready_cooldown = (
                feeder.out_index - state.last_counted[tid]
            ) >= args.cooldown_frames

            if flipped and ready_age and ready_cooldown:
                direction = "up" if prev < confirmed else "down"
                ts_s = feeder.frame_in / max(1.0, fps)

                # WRITE FIRST
                wrote = False
                if csvs.ev_writer:
                    wrote = csvs.write_event(
                        ts_s=ts_s,
                        src_frame_idx=feeder.frame_in,  # ORIGINAL source frame index
                        proc_frame_idx=feeder.out_index,  # processed/ROI index
                        tid=tid,
                        direction=direction,
                        cx=cx,
                        cy=cy,
                    )

                if wrote:
                    if direction == "up":
                        state.up_count += 1
                    else:
                        state.down_count += 1
                    state.last_counted[tid] = feeder.out_index

                    if not args.quiet:
                        log.success(
                            "LINE-CROSS",
                            f" t={ts_s:.2f}s id={tid} dir={direction} "
                            f"up={state.up_count} down={state.down_count}",
                        )

                    animator.trigger(
                        cx,
                        cy,
                        (ax_roi, ay_roi, bx_roi, by_roi),
                        lines_full[line_idx],
                        direction,
                        feeder.out_index,
                    )
                    colorer.mark_counted(tid, feeder.out_index)
                else:
                    if not args.quiet:
                        log.warn(
                            "CSV-DUPLICATE",
                            f" duplicate track_id={tid} — skipped writing & count bump",
                        )


def _update_counts_dual_for_frame(
    dets,
    dual,  # DualLineCounter instance
    mode,  # "verify" | "recover"
    feeder,
    fps,
    args,
    state,  # up/down counters etc.
    csvs,
    colorer,
    animator=None,
    lineA_roi=None,
    lineA_full=None,
    lineB_roi=None,
    lineB_full=None,
):
    """
    DUAL-LINE MODE: updates counts per frame.
    - mode="verify": commit a count only if the same ID crosses A then B consistently.
    - mode="recover": Line A counts immediately; Line B may recover a miss once per ID.

    Notes:
    - We use the same CSV/logging style as single-line mode.
    - We mark counted IDs with colorer to turn them blue (persist/window).
    - We optionally trigger the +1 animation along the relevant line (A for A-counts, B for verified/recovered).
    """
    margin = args.line_margin_px

    for box, tid, conf, cl, mask in dets:
        cx, cy = dual._centroid_from(box, mask)
        tid_i = int(tid)

        # -------------------------------
        # PHASE 1: record motion (eyes)
        # -------------------------------
        dual.motion.update(
            tid=tid_i,
            frame_idx=feeder.out_index,
            cx=cx,
            cy=cy,
        )

        if mode == "verify":
            ev = dual.update_verify(tid_i, cx, cy, feeder.out_index, margin_px=margin)
            if ev and ev.get("type") == "count":
                geom_event = ev
                # -------------------------------
                # PHASE 2: motion gating (brain)
                # -------------------------------
                decision = dual.motion_analyzer.analyze(
                    motion_history=dual.motion,
                    tid=geom_event["tid"],
                    event_frame=geom_event["frame_idx"],
                    geometry_direction=geom_event["direction"],
                    line_vector=(
                        (
                            lineB_roi[2] - lineB_roi[0],
                            lineB_roi[3] - lineB_roi[1],
                        )
                        if geom_event["trigger_line"] == "B"
                        else (
                            lineA_roi[2] - lineA_roi[0],
                            lineA_roi[3] - lineA_roi[1],
                        )
                    ),
                )

                ts_s = feeder.frame_in / max(1.0, fps)

                csvs.write_decision(
                    ts_s=ts_s,
                    proc_frame_idx=feeder.out_index,
                    tid=geom_event["tid"],
                    mode=geom_event["mode"],
                    line=geom_event["trigger_line"],
                    geometry_direction=geom_event["direction"],
                    decision=decision["decision"],
                    reason=decision["reason"],
                    confidence=decision.get("confidence", 0.0),
                    dx=decision["delta"][0],
                    dy=decision["delta"][1],
                    dominant_axis=decision.get("dominant_axis"),
                )

                if decision["decision"] == "defer":
                    if not args.quiet:
                        log.debug(
                            "DUAL-MOTION",
                            f"tid={geom_event['tid']} deferred: {decision['reason']}",
                        )
                    continue

                if decision["decision"] == "reject":
                    if not args.quiet:
                        log.warn(
                            "DUAL-MOTION-REJECT",
                            f"tid={geom_event['tid']} rejected: {decision['reason']} "
                            f"Δ={decision['delta']} axis={decision['dominant_axis']}",
                        )
                    continue

                # Only ACCEPT reaches here
                # FINALIZE COMMIT (single authority)

                # 1️⃣ Lock ID
                dual.last_counted[tid_i] = feeder.out_index

                # 2️⃣ Clear pending geometry (if any)
                dual.pending_A.pop(tid_i, None)

                # 3️⃣ Clear motion history
                dual.motion.clear_tid(tid_i)

                direction = ev.get("direction", "down")
                ts_s = feeder.frame_in / max(1.0, fps)

                wrote = csvs.write_event(
                    ts_s=ts_s,
                    src_frame_idx=feeder.frame_in,
                    proc_frame_idx=feeder.out_index,
                    tid=int(tid_i),
                    direction=direction,
                    cx=cx,
                    cy=cy,
                )

                if wrote:
                    if direction == "up":
                        state.up_count += 1
                    else:
                        state.down_count += 1

                    if not args.quiet:
                        log.success(
                            "DUAL-VERIFY",
                            f" t={ts_s:.2f}s id={tid_i} dir={direction} "
                            f"Up={state.up_count} Down={state.down_count}",
                        )

                    colorer.mark_counted(tid_i, feeder.out_index)
                    if (
                        animator
                        and (lineB_roi is not None)
                        and (lineB_full is not None)
                    ):
                        animator.trigger(
                            cx, cy, lineB_roi, lineB_full, direction, feeder.out_index
                        )



                else:
                    if not args.quiet:
                        log.warn(
                            "CSV-DUPLICATE",
                            f" duplicate track_id={tid_i} — skipped writing & count bump",
                        )

        else:  # mode == "recover"

            def _on_A_count(tid_local, dir_local):
                # DO NOT write CSV
                # DO NOT update counters
                # DO NOT animate

                # Geometry intent only
                event = {
                    "tid": tid_local,
                    "frame_idx": feeder.out_index,
                    "direction": dir_local,
                    "mode": "recover",
                    "line_sequence": ["A"],
                    "trigger_line": "A",
                    "reason": "Immediate A crossing",
                    "type": "count",
                    "subsystem": "dual_recover_A",
                }
                dual.geometry_events.append(event)

            ev = dual.update_recover(
                tid_i,
                cx,
                cy,
                feeder.out_index,
                margin_px=margin,
                on_A_count_cb=_on_A_count,
            )
            # If B recovered a miss (one-shot per ID lock inside DualLineCounter)
            if ev and ev["type"] == "count":
                geom_event = ev
                # -------------------------------
                # PHASE 2: motion gating (brain)
                # -------------------------------
                decision = dual.motion_analyzer.analyze(
                    motion_history=dual.motion,
                    tid=geom_event["tid"],
                    event_frame=geom_event["frame_idx"],
                    geometry_direction=geom_event["direction"],
                    line_vector=(
                        (
                            lineB_roi[2] - lineB_roi[0],
                            lineB_roi[3] - lineB_roi[1],
                        )
                        if geom_event["trigger_line"] == "B"
                        else (
                            lineA_roi[2] - lineA_roi[0],
                            lineA_roi[3] - lineA_roi[1],
                        )
                    ),
                )

                ts_s = feeder.frame_in / max(1.0, fps)

                csvs.write_decision(
                    ts_s=ts_s,
                    proc_frame_idx=feeder.out_index,
                    tid=geom_event["tid"],
                    mode=geom_event["mode"],
                    line=geom_event["trigger_line"],
                    geometry_direction=geom_event["direction"],
                    decision=decision["decision"],
                    reason=decision["reason"],
                    confidence=decision.get("confidence", 0.0),
                    dx=decision["delta"][0],
                    dy=decision["delta"][1],
                    dominant_axis=decision.get("dominant_axis"),
                )

                if decision["decision"] == "defer":
                    if not args.quiet:
                        log.debug(
                            "DUAL-MOTION",
                            f"tid={geom_event['tid']} deferred: {decision['reason']}",
                        )
                    continue

                if decision["decision"] == "reject":
                    if not args.quiet:
                        log.warn(
                            "DUAL-MOTION-REJECT",
                            f"tid={geom_event['tid']} rejected: {decision['reason']} "
                            f"Δ={decision['delta']} axis={decision['dominant_axis']}",
                        )
                    continue

                # Only ACCEPT reaches here
                # FINALIZE COMMIT (single authority)

                # 1️⃣ Lock ID
                dual.last_counted[tid_i] = feeder.out_index

                # 2️⃣ Clear pending geometry (if any)
                dual.pending_A.pop(tid_i, None)

                # 3️⃣ Clear motion history
                dual.motion.clear_tid(tid_i)

                dirB = ev["direction"]
                ts_s = feeder.frame_in / max(1.0, fps)
                wrote = csvs.write_event(
                    ts_s,
                    feeder.frame_in,  # source/original index
                    feeder.out_index,  # processed index
                    int(tid),
                    dirB,
                    cx,
                    cy,
                )
                if wrote:
                    if dirB == "up":
                        state.up_count += 1
                    else:
                        state.down_count += 1
                    if not args.quiet:
                        log.success(
                            "DUAL-RECOVER",
                            f" t={ts_s:.2f}s id={tid} dir={dirB} (recovery) "
                            f"Up={state.up_count} Down={state.down_count}",
                        )
                    colorer.mark_counted(int(tid), feeder.out_index)
                    if (
                        animator
                        and (lineB_roi is not None)
                        and (lineB_full is not None)
                    ):
                        animator.trigger(
                            cx, cy, lineB_roi, lineB_full, dirB, feeder.out_index
                        )


                else:
                    if not args.quiet:
                        log.warn(
                            "CSV-DUPLICATE",
                            f" duplicate track_id={tid} — skipped writing & count bump",
                        )

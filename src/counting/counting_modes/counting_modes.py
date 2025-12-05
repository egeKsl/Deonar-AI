# src/counting/counting_modes/counting_modes.py

import numpy as np
from src.utils.logger import log
from src.counting.counting_core_logic import (
    _update_counts_dual_for_frame,
    _update_counts_for_frame,
)


def process_frame_zone(dets, zone, feeder, fps, args, state, csvs, colorer):
    """
    Handles zone mode updates for each frame.
    - Writes CSV events if counts happen
    - Updates up/down state counters
    - Collects back-projection trails for debug
    """
    for box, tid, conf, cl, mask in dets:
        # centroid (mask first, fallback to box center)
        x1, y1, x2, y2 = box
        if mask is not None:
            ys, xs = np.where(mask)
            if xs.size > 0 and ys.size > 0:
                cx, cy = xs.mean(), ys.mean()
            else:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        else:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if int(tid) < 0:
            continue

        ev = zone.update_for_frame(int(tid), float(cx), float(cy), feeder.out_index)

        if not ev:
            continue

        if ev["type"] == "count":
            direction = ev.get("direction", "skip")
            ts_s = feeder.frame_in / max(1.0, fps)
            method = ev.get("method", "?")
            enc = f"{direction}:{ev.get('entry_edge','?')}→{ev.get('exit_edge','?')}|{method}"

            wrote = csvs.write_event(
                ts_s=ts_s,
                src_frame_idx=feeder.frame_in,
                proc_frame_idx=feeder.out_index,
                tid=int(tid),
                direction=enc,
                cx=cx,
                cy=cy,
            )

            if wrote and direction in ("up", "down"):
                if direction == "up":
                    state.up_count += 1
                else:
                    state.down_count += 1
                colorer.mark_counted(int(tid), feeder.out_index)

            if not args.quiet:
                if direction in ("up", "down"):
                    log.success(
                        "MODES-ZONE",
                        f" t={ts_s:.2f}s id={tid} {direction} "
                        f"({ev.get('entry_edge','?')}→{ev.get('exit_edge','?')}|{method}) "
                        f"Up={state.up_count} Down={state.down_count}",
                    )
                else:
                    log.error(
                        "MODES-ZONE",
                        f" t={ts_s:.2f}s id={tid} {direction} "
                        f"({ev.get('entry_edge','?')}→{ev.get('exit_edge','?')}|{method}) "
                        f"Up={state.up_count} Down={state.down_count}",
                    )

        elif ev["type"] == "entry":
            # optional: debug logging
            if not args.quiet:
                log.debug(
                    "MODES-ZONE",
                    f" ENTRY: tid={tid} via {ev.get('edge_name','?')} "
                    f"at pt={ev.get('pt')} method={ev.get('method','?')} "
                    f"(frame={feeder.out_index})",
                )

        elif ev and ev["type"] == "exit":
            if not args.quiet:
                entry_edge = ev.get("entry_edge", "?")
                exit_edge = ev.get("edge_name", "?")
                log.warn(
                    "MODES-ZONE",
                    f" EXIT: tid={tid} {entry_edge} → {exit_edge} "
                    f"(no count, frame={feeder.out_index}, method={ev.get('method','?')})",
                )

        elif ev and ev["type"] == "entry" and ev.get("method") == "ghost_born":
            zone.ghost_ids.add(int(tid))
            log.error(
                "MODES-ZONE",
                f" Ghost-born entry: tid={ev['tid']} via {ev['edge_name']} "
                f"({ev['pt'][0]:.1f},{ev['pt'][1]:.1f})",
            )


def process_frame_dual(
    dets,
    dual,
    feeder,
    fps,
    args,
    state,
    csvs,
    colorer,
    animator,
    lineA_roi,
    lineA_full,
    lineB_roi,
    lineB_full,
):
    mode = args.dual_mode
    _update_counts_dual_for_frame(
        dets=dets,
        dual=dual,
        mode=mode,
        feeder=feeder,
        fps=fps,
        args=args,
        state=state,
        csvs=csvs,
        colorer=colorer,
        animator=animator,
        lineA_roi=lineA_roi,
        lineA_full=lineA_full,
        lineB_roi=lineB_roi,
        lineB_full=lineB_full,
    )


def process_frame_line(
    dets, lines_roi, lines_full, feeder, fps, args, state, animator, csvs, colorer
):

    _update_counts_for_frame(
        dets, lines_roi, lines_full, feeder, fps, args, state, animator, csvs, colorer
    )

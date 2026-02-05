# src/app/single_threaded.py

import cv2, time

from src.geometry.geom import _prepare_geometry, _build_lines
from src.counting.state import _init_count_state
from src.display.drawing import (
    _prepare_drawing,
    _setup_windows,
    _save_screenshot,
    _compose_frames,
)
from src.counting.counting_modes.counting_modes import (
    process_frame_zone,
    process_frame_dual,
    process_frame_line,
)
from src.counting.counting_modes.init_counting_modes import _init_dual_lines, _init_zone

from src.runtime_configs.tracker_cfg import make_tracker_yaml

from src.infer.yolo_infer import resolve_class_filter, track_once
from src.infer.loader import load_model

from src.capture.stream import ROIStream

from src.io.io import setup_output, CsvWriters

from src.counting.counting_modes.zone import ZoneCounter

from src.utils.logger import log

from src.utils.progress import init_progress, progress_update, stop_progress


def _pace_or_autoskip(args, t0, fps, feeder, start_src_frame, use_autoskip):
    """Keep real-time pace (sync) and optionally autoskip if lagging."""
    elapsed = time.perf_counter() - t0
    expected = (feeder.frame_in - start_src_frame) / max(1.0, fps)
    lag = elapsed - expected
    if lag < -0.002:
        time.sleep(min(0.05, -lag))
    elif use_autoskip and lag > args.max_lag_s:
        n_skip = int(lag * fps)
        if n_skip > 0:
            feeder.request_skip(min(n_skip, int(fps)))
        if not args.quiet:
            log.debug(
                "RUNNER-AUTOSKIP", f" lag={lag:.2f}s -> request skip {n_skip} frames"
            )


def run_original(args):
    # 1) Open source video + geometry
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")
    XR = args.roi_xr
    YR = args.roi_yr
    WR = args.roi_wr
    HR = args.roi_hr
    log.info(
        "RUNNER-INFO",
        f" Source opened: {args.source} | ROI ratios x={XR} y={YR} w={WR} h={HR}",
    )
    fps, W, H, total, rx, ry, rw, rh = _prepare_geometry(cap, XR, YR, WR, HR)
    roi_area = rw * rh

    # 2) Lines
    lines_roi, lines_full = _build_lines(args.count_line_roi, rx, ry, rw, rh)

    # 3) Dual-line init
    use_dual, dual, lineA_roi, lineB_roi, lineA_full, lineB_full = _init_dual_lines(
        args, lines_roi, lines_full, rx, ry, rw, rh
    )

    # 4) Zone init
    use_zone, zone, zone_rect_roi, zone_rect_full = _init_zone(args, rw, rh, rx, ry)

    # 5) Model
    model, device = load_model(
        args.weights, args.device, args.half, args.fuse, args.quiet
    )
    keep_class_ids = resolve_class_filter(model, args.count_classes)

    # 6) ROI streamer
    feeder = ROIStream(cap, rx, ry, rw, rh, stride=args.stride)

    # 7) CSV + video
    decisions_path = args.csv_decisions if use_dual else None
    csvs = CsvWriters(events_path=args.csv_events, ts_path=args.csv_timeseries, decisions_path=decisions_path)
    writer, show_windows, saving_enabled, out_path = setup_output(args, W, H, fps)
    _setup_windows(show_windows, args, rw, rh, W, H)

    # 8) Counters + drawing
    state = _init_count_state(args.min_side_frames)
    animator, colorer, pretty_cfg = _prepare_drawing(args)

    # 9) Timers
    use_sync, use_autoskip, paused = bool(args.sync), bool(args.autoskip), False
    t0, start_src_frame = (
        time.perf_counter(),
        int(cap.get(cv2.CAP_PROP_POS_FRAMES)) or 0,
    )
    last_full_disp = None
    last_ts_sec_holder = [-1]

    # 10) Tracker config
    tracker_yaml = make_tracker_yaml(args, fps)

    # 11) Zone object
    if use_zone and zone_rect_roi is not None:
        zone = ZoneCounter(
            rect_roi_xyxy=zone_rect_roi,
            born_inside_policy=args.zone_born_inside,
            backfill_wait_frames=args.zone_backfill_wait,
            near_border_px=args.zone_near_border_px,
            quiet=args.quiet,
        )

    if writer is not None:
        init_progress(total, prefix="[WRITE]")

    # 12) Frame loop
    for roi_frame in feeder:
        if paused:
            key = cv2.waitKey(30) & 0xFF if show_windows else 255
            if key == ord("p"):
                paused = False
            elif key == ord("q"):
                break
            elif key == ord("s"):
                _save_screenshot(last_full_disp, args.source, feeder.out_index)
            continue

        # detection + tracking
        dets = track_once(
            model, roi_frame, args, tracker_yaml, roi_area, keep_class_ids
        )

        # counting branch
        if use_zone and zone is not None:
            process_frame_zone(dets, zone, feeder, fps, args, state, csvs, colorer)
        elif use_dual and dual is not None:
            process_frame_dual(
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
            )
        else:
            process_frame_line(
                dets,
                lines_roi,
                lines_full,
                feeder,
                fps,
                args,
                state,
                animator,
                csvs,
                colorer,
            )

        # per-second timeseries
        now_sec = int(feeder.frame_in / max(1.0, fps))
        if csvs.ts_writer and now_sec != last_ts_sec_holder[0]:
            last_ts_sec_holder[0] = now_sec
            csvs.write_timeseries(now_sec, state.up_count, state.down_count)

        # compose frame overlays
        roi_disp, full_disp = _compose_frames(
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
            pretty_cfg=pretty_cfg,  # <-- add
            # zone
            use_zone=use_zone,
            zone=zone,
            zone_rect_roi=zone_rect_roi,
            zone_rect_full=zone_rect_full,
            # dual lines
            use_dual=use_dual,
            lineA_roi=lineA_roi,
            lineB_roi=lineB_roi,
            lineA_full=lineA_full,
            lineB_full=lineB_full,
        )

        last_full_disp = full_disp.copy()

        # display
        if show_windows:
            if not args.no_roi:
                cv2.imshow("ROI View", roi_disp)
            if not args.no_full:
                cv2.imshow("Full View", full_disp)

        # save
        if writer is not None:
            writer.write(full_disp)
            progress_update(
                feeder.out_index,
                total,
                t0,
                src_idx=feeder.frame_in,
                every=args.progress_every,
                eff_fps=fps * args.playback_speed,
            )

        # pacing
        if use_sync:
            _pace_or_autoskip(args, t0, fps, feeder, start_src_frame, use_autoskip)

        # keyboard
        key = (
            cv2.waitKey(1) & 0xFF
            if (show_windows and (not args.no_roi or not args.no_full))
            else 255
        )
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            _save_screenshot(last_full_disp, args.source, feeder.out_index)

    # 13) Cleanup
    cap.release()
    if show_windows and (not args.no_roi):
        cv2.destroyWindow("ROI View")
    if show_windows and (not args.no_full):
        cv2.destroyWindow("Full View")
    csvs.close()
    if writer is not None:
        writer.release()
        stop_progress()
        if not args.quiet:
            log.blank()
        if not args.quiet and out_path:
            log.success("RUNNER-SAVE", f" File written: {out_path}")
    if not args.quiet:
        log.info(
            "RUNNER-INFO",
            f" Done. Up={state.up_count}, Down={state.down_count}, Total={state.up_count + state.down_count}",
        )

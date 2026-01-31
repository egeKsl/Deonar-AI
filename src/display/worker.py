# src/display/worker.py
"""
DisplayWorker (threaded).

Responsibilities:
 - Prepare drawing context (animator, colorer, pretty_cfg), CsvWriters, windows (once)
 - Build pixel lines and initialize dual/zone objects (once)
 - For each result from result_queue:
     - run process_frame_* (counting/tracking) to keep parity with run_original
     - call _compose_frames(...) and show / optionally save screenshots
 - Keep UI stable across threading (letterbox when window enlarged) and draw HUD.

Design notes:
 - DisplayWorker expects the result dict produced by InferenceWorker to contain stable keys.
 - Accepts an 'injected' dict from runner to reuse drawing/csv objects prepared once.
 - All display operations are defensive and isolated.
"""

import threading
import queue
import time
import cv2
import traceback
from typing import Tuple, Optional

import numpy as np
from scipy import stats

from src.utils.logger import log
from src.display.drawing import (
    _prepare_drawing,
    _compose_frames,
    _setup_windows,
    _save_screenshot,
)
from src.geometry.geom import _build_lines
from src.counting.counting_modes.init_counting_modes import _init_dual_lines
from src.io.io import CsvWriters
from src.counting.counting_modes.counting_modes import (
    process_frame_zone,
    process_frame_dual,
    process_frame_line,
)
from src.counting.state import _init_count_state
import os
from src.utils.video_recorder import VideoRecorder


# ----- small helpers for robust display -----
def resize_with_aspect(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to fit (target_w, target_h) preserving aspect ratio and centering with black bars."""
    if image is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    if w == 0 or h == 0:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def get_window_size_or_default(
    win_name: str, default_w: int, default_h: int
) -> Tuple[int, int]:
    """
    Return (w,h) for window client area. If platform/driver returns 0, use defaults.
    cv2.getWindowImageRect returns (x,y,w,h) on supported platforms.
    """
    try:
        rect = cv2.getWindowImageRect(win_name)
        if isinstance(rect, tuple) and len(rect) == 4:
            _, _, w, h = rect
            if w > 0 and h > 0:
                return int(w), int(h)
    except Exception:
        pass
    return int(default_w), int(default_h)


def show_in_resizable_window(
    win_name: str,
    frame: np.ndarray,
    default_w: int,
    default_h: int,
    preserve_aspect: bool = True,
) -> np.ndarray:
    """
    Ensures the window exists and shows a resized frame that fits the window.
    Returns the displayed image (after resize) for HUD drawing.
    """
    try:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    except Exception:
        pass
    win_w, win_h = get_window_size_or_default(win_name, default_w, default_h)
    if preserve_aspect:
        disp = resize_with_aspect(frame, win_w, win_h)
    else:
        disp = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(win_name, disp)
    return disp


class DisplayWorker(threading.Thread):
    def __init__(
        self,
        result_queue: "queue.Queue",
        stop_event: threading.Event,
        args,
        injected: dict | None = None,
    ):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.args = args
        self.injected = injected or {}

        # drawing/context (can be injected from runner)
        self.animator = self.injected.get("animator")
        self.colorer = self.injected.get("colorer")
        self.pretty_cfg = self.injected.get("pretty_cfg")

        # CSV writers (can be injected or constructed)
        self.csvs = self.injected.get("csvs")
        self._owns_csvs = ("csvs" not in (injected or {})) and (self.csvs is None)
        # Note: we'll set _owns_csvs True only if we create csvs later;
        # alternatively initialize False then set True when we create them below.
        self._owns_csvs = False

        # cached geometry & lines
        self._cached_rx = None
        self._cached_ry = None
        self._cached_rw = None
        self._cached_rh = None
        self._cached_W = None
        self._cached_H = None
        self._cached_lines_roi = None
        self._cached_lines_full = None

        # dual-mode objects cached
        self._dual_inited = False
        self._use_dual = False
        self._dual_obj = None
        self._lineA_roi = None
        self._lineB_roi = None
        self._lineA_full = None
        self._lineB_full = None

        # counting state (own instance)
        self.state = _init_count_state(self.args.min_side_frames)

        # total frames hint (injected)
        self.total_frames = int(
            self.injected.get("total", getattr(self.args, "total_frames", 0))
        )

        # windows setup guard
        self._windows_setup = False

        # --- new: video recorder config ---
        record_video = bool(getattr(self.args, "record_video", False))
        self.video_recorder: Optional[VideoRecorder] = None
        if record_video:
            # Prefer config fps; otherwise use capture fps; fallback 25.0
            cap_fps = injected.get("fps", 25.0)
            fps = float(getattr(self.args, "video_fps", cap_fps) or cap_fps)
            video_path = getattr(
                self.args,
                "video_path",
                "outputs/videos/goats_output.mp4",
            )
            fourcc = getattr(self.args, "video_fourcc", "mp4v")
            overwrite = bool(getattr(self.args, "overwrite_video", False))

            self.video_recorder = VideoRecorder(
                path=video_path,
                fps=fps,
                fourcc=fourcc,
                overwrite=overwrite,
            )
            log.info(
                "DISPLAY-WORKER",
                f"Video recording enabled -> {video_path} (fps={fps}, fourcc={fourcc})",
            )
        else:
            log.debug("DISPLAY-WORKER", "Video recording disabled")

    # ---------------- drawing context ----------------
    def _ensure_drawing(self, sample_frame):
        """Prepare drawing resources & windows (idempotent)."""
        if (
            (self.animator is None)
            or (self.colorer is None)
            or (self.pretty_cfg is None)
        ):
            try:
                animator, colorer, pretty_cfg = _prepare_drawing(self.args)
                if self.animator is None:
                    self.animator = animator
                if self.colorer is None:
                    self.colorer = colorer
                if self.pretty_cfg is None:
                    self.pretty_cfg = pretty_cfg
                log.debug("DISPLAY-WORKER", "Prepared drawing context")
            except Exception:
                log.error("DISPLAY-WORKER", "Failed to prepare drawing context")
                log.debug("DISPLAY-WORKER", traceback.format_exc())
                self.animator = self.animator or None
                self.colorer = self.colorer or None
                self.pretty_cfg = self.pretty_cfg or None

        if not self._windows_setup and not self.args.no_roi:
            try:
                _setup_windows(
                    True,
                    self.args,
                    self._cached_rw or 640,
                    self._cached_rh or 480,
                    self._cached_W or 640,
                    self._cached_H or 480,
                )
            except Exception:
                log.debug("DISPLAY-WORKER", "Window setup failed (continuing)")
            self._windows_setup = True

        if self.csvs is None:
            try:
                events_path = self.args.csv_events
                ts_path = self.args.csv_timeseries
                self.csvs = CsvWriters(events_path, ts_path)
                self._owns_csvs = True
                log.debug("DISPLAY-WORKER", "Created CSV writers")
            except Exception:
                self.csvs = None
                self._owns_csvs = False
                log.error("DISPLAY-WORKER", "Failed to create CSV writers")

    # ---------------- geometry & lines ----------------
    def _derive_geometry(self, res: dict):
        """Return rx,ry,rw,rh,W,H with sensible fallbacks and cache values."""
        rx = res.get("rx")
        ry = res.get("ry")
        rw = res.get("rw")
        rh = res.get("rh")
        W = res.get("W")
        H = res.get("H")

        feeder = res.get("feeder")
        if (
            rx is None or ry is None or rw is None or rh is None
        ) and feeder is not None:
            rx = rx if rx is not None else getattr(feeder, "rx", None)
            ry = ry if ry is not None else getattr(feeder, "ry", None)
            rw = rw if rw is not None else getattr(feeder, "rw", None)
            rh = rh if rh is not None else getattr(feeder, "rh", None)
            W = W if W is not None else getattr(feeder, "W", None)
            H = H if H is not None else getattr(feeder, "H", None)

        full = res.get("frame")
        roi = res.get("roi")
        if full is not None and (W is None or H is None):
            try:
                H, W = int(full.shape[0]), int(full.shape[1])
            except Exception:
                W, H = W or None, H or None
        if roi is not None and (rw is None or rh is None):
            try:
                rh, rw = int(roi.shape[0]), int(roi.shape[1])
            except Exception:
                rw, rh = rw or None, rh or None

        if W is None or H is None:
            if full is not None:
                H, W = int(full.shape[0]), int(full.shape[1])
            else:
                W, H = (640, 480)
        if rw is None or rh is None:
            rw, rh = W, H
            rx, ry = 0, 0

        rx = 0 if rx is None else int(rx)
        ry = 0 if ry is None else int(ry)
        rw = max(1, int(rw))
        rh = max(1, int(rh))
        W = int(W)
        H = int(H)

        # cache
        self._cached_rx, self._cached_ry, self._cached_rw, self._cached_rh = (
            rx,
            ry,
            rw,
            rh,
        )
        self._cached_W, self._cached_H = W, H
        """
        log.debug(
            "DISPLAY-WORKER",
            f"Derived geometry: rx={rx}, ry={ry}, rw={rw}, rh={rh}, W={W}, H={H}",
        )
        """
        return rx, ry, rw, rh, W, H

    def _ensure_lines(self, rx: int, ry: int, rw: int, rh: int):
        if self._cached_lines_roi is None or self._cached_lines_full is None:
            try:
                lines_roi, lines_full = _build_lines(
                    self.args.count_line_roi, rx, ry, rw, rh
                )
                self._cached_lines_roi = lines_roi
                self._cached_lines_full = lines_full
                log.debug("DISPLAY-WORKER", f"Built lines_roi ({len(lines_roi)} lines)")
            except Exception:
                self._cached_lines_roi = []
                self._cached_lines_full = []
                log.debug(
                    "DISPLAY-WORKER",
                    "Failed to build lines from args.count_line_roi (using empty)",
                )
        return self._cached_lines_roi, self._cached_lines_full

    def _ensure_dual(self, rx: int, ry: int, rw: int, rh: int):
        if not self._dual_inited:
            try:
                use_dual = self.args.dual_lines_enabled
                if use_dual:
                    lines_roi, lines_full = self._ensure_lines(rx, ry, rw, rh)
                    use_dual, dual_obj, lineA_roi, lineB_roi, lineA_full, lineB_full = (
                        _init_dual_lines(
                            self.args, lines_roi, lines_full, rx, ry, rw, rh
                        )
                    )
                    if use_dual:
                        self._use_dual = True
                        self._dual_obj = dual_obj
                        self._lineA_roi = lineA_roi
                        self._lineB_roi = lineB_roi
                        self._lineA_full = lineA_full
                        self._lineB_full = lineB_full
                self._dual_inited = True
            except Exception:
                log.debug("DISPLAY-WORKER", "Dual-line init failed or skipped")
                self._dual_inited = True
        return (
            self._use_dual,
            self._dual_obj,
            self._lineA_roi,
            self._lineB_roi,
            self._lineA_full,
            self._lineB_full,
        )

    # ---------------- main run loop ----------------
    def run(self):
        log.info("DISPLAY-WORKER", "DisplayWorker started")
        last_full_disp = None
        while not self.stop_event.is_set():
            try:
                res = self.result_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # stable field extraction
                frame = res.get("frame")
                roi = res.get("roi")
                feeder = res.get("feeder")
                frame_in = res.get("frame_in_counter") or (
                    getattr(feeder, "frame_in", None) if feeder else None
                )
                out_idx = res.get("out_index_counter") or (
                    getattr(feeder, "out_index", None) if feeder else None
                )
                dets = res.get("dets") or []

                if frame is None and roi is None:
                    log.debug(
                        "DISPLAY-WORKER", "Empty result (no frame and no roi); skipping"
                    )
                    continue

                # ensure drawing/csvs/windows prepared
                sample_for_drawing = frame if frame is not None else roi
                self._ensure_drawing(sample_for_drawing)

                # derive geometry & lines
                rx, ry, rw, rh, W, H = self._derive_geometry(res)
                lines_roi, lines_full = self._ensure_lines(rx, ry, rw, rh)
                use_dual, dual_obj, lineA_roi, lineB_roi, lineA_full, lineB_full = (
                    self._ensure_dual(rx, ry, rw, rh)
                )

                # run counting/tracking logic (same as run_original)
                try:
                    fps = res.get("avg_fps", 0.0)
                    if getattr(self.args, "use_zone", False):
                        process_frame_zone(
                            dets,
                            None,
                            feeder,
                            fps,
                            self.args,
                            self.state,
                            self.csvs,
                            self.colorer,
                        )
                    elif use_dual and dual_obj is not None:
                        process_frame_dual(
                            dets,
                            dual_obj,
                            feeder,
                            fps,
                            self.args,
                            self.state,
                            self.csvs,
                            self.colorer,
                            self.animator,
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
                            self.args,
                            self.state,
                            self.animator,
                            self.csvs,
                            self.colorer,
                        )
                except Exception:
                    log.error(
                        "DISPLAY-WORKER",
                        f"process_frame_* failed: {traceback.format_exc()}",
                    )

                # compose frames
                try:
                    roi_frame = (
                        roi
                        if roi is not None
                        else (
                            frame.copy()
                            if frame is not None
                            else (255 * np.ones((rh, rw, 3), dtype=np.uint8))
                        )
                    )
                    roi_disp, full_disp = _compose_frames(
                        roi_frame=roi_frame,
                        dets=dets,
                        lines_roi=lines_roi,
                        lines_full=lines_full,
                        feeder=feeder,
                        rw=rw,
                        rh=rh,
                        rx=rx,
                        ry=ry,
                        W=W,
                        H=H,
                        total=self.total_frames,
                        state=self.state,
                        animator=self.animator,
                        colorer=self.colorer,
                        args=self.args,
                        pretty_cfg=self.pretty_cfg,
                        use_zone=getattr(self.args, "use_zone", False),
                        zone=None,
                        zone_rect_roi=None,
                        zone_rect_full=None,
                        use_dual=use_dual,
                        lineA_roi=lineA_roi,
                        lineB_roi=lineB_roi,
                        lineA_full=lineA_full,
                        lineB_full=lineB_full,
                        threaded_streaming=True,
                        res=res,
                    )
                    last_full_disp = full_disp.copy()
                except Exception:
                    log.error("DISPLAY-WORKER", "compose_frames failed")
                    log.debug("DISPLAY-WORKER", traceback.format_exc())
                    try:
                        fallback = (
                            roi if roi is not None and hasattr(roi, "copy") else None
                        )
                        if fallback is None:
                            fallback = 255 * np.ones(
                                (int(rh or 240), int(rw or 320), 3), dtype=np.uint8
                            )
                        roi_disp = fallback
                        full_disp = frame.copy() if frame is not None else fallback
                    except Exception:
                        roi_disp = 255 * np.ones((240, 320, 3), dtype=np.uint8)
                        full_disp = roi_disp

                # --- record to video first ---
                try:
                    if self.video_recorder is not None:
                        self.video_recorder.write(full_disp)
                except Exception:
                    log.debug("DISPLAY-WORKER", "Video recording failed", exc_info=True)

                # display / publish / handle UI events
                try:
                    default_roi_w = min(960, rw) if rw else 640
                    default_roi_h = (
                        int(default_roi_w * (rh / max(1, rw))) if rw else 480
                    )
                    default_full_w = min(960, W) if W else 640
                    default_full_h = int(default_full_w * (H / max(1, W))) if W else 480

                    shown_roi = None
                    shown_full = None

                    # Prefer WebRTC if available
                    webrtc_server = None
                    try:
                        webrtc_server = self.injected.get("webrtc_server")
                    except Exception:
                        webrtc_server = None

                    if webrtc_server is not None:
                        # Push frame into WebRTC pipeline
                        try:
                            webrtc_server.publish(full_disp)
                        except Exception:
                            log.debug(
                                "DISPLAY-WORKER",
                                "WebRTC publish failed: " + traceback.format_exc(),
                            )

                        # Handle control commands from WebRTC client (/control -> queue)
                        try:
                            ctrl_q = self.injected.get("webrtc_control_q")
                            if ctrl_q is not None:
                                while True:
                                    try:
                                        cmd = ctrl_q.get_nowait()
                                    except Exception:
                                        break

                                    try:
                                        if (
                                            isinstance(cmd, dict)
                                            and cmd.get("cmd") == "screenshot"
                                        ):
                                            _save_screenshot(
                                                full_disp,
                                                getattr(self.args, "source", None),
                                                getattr(feeder, "out_index", None),
                                            )
                                        elif (
                                            isinstance(cmd, dict)
                                            and cmd.get("cmd") == "quit"
                                        ):
                                            # Just signal stop; browser will see stream stop
                                            self.stop_event.set()
                                    except Exception:
                                        log.debug(
                                            "DISPLAY-WORKER",
                                            "control cmd failed: "
                                            + traceback.format_exc(),
                                        )
                        except Exception:
                            log.debug(
                                "DISPLAY-WORKER",
                                "control queue handling failed: "
                                + traceback.format_exc(),
                            )

                        # When using WebRTC, no cv2.imshow / waitKey
                        continue

                    # FALLBACK: original OpenCV windows
                    if not self.args.no_roi:
                        shown_roi = show_in_resizable_window(
                            "ROI View",
                            roi_disp,
                            default_roi_w,
                            default_roi_h,
                            preserve_aspect=True,
                        )
                    if not self.args.no_full:
                        shown_full = show_in_resizable_window(
                            "Full View",
                            full_disp,
                            default_full_w,
                            default_full_h,
                            preserve_aspect=True,
                        )

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.stop_event.set()
                        break
                    elif key == ord("s"):
                        try:
                            _save_screenshot(
                                last_full_disp,
                                self.args.source,
                                getattr(feeder, "out_index", None),
                            )
                        except Exception:
                            log.debug(
                                "DISPLAY-WORKER",
                                "screenshot failed",
                                exc_info=True,
                            )

                except Exception:
                    log.error("DISPLAY-WORKER", "imshow/publish failed")
                    log.debug("DISPLAY-WORKER", traceback.format_exc())

            except Exception:
                log.error(
                    "DISPLAY-WORKER",
                    "Error processing display item: " + traceback.format_exc(),
                )
            finally:
                # Balanced task_done: only call if queue supports it
                try:
                    if hasattr(self.result_queue, "task_done"):
                        self.result_queue.task_done()
                except Exception:
                    pass

                # close csvs if we own them
                try:
                    if self._owns_csvs and self.csvs is not None:
                        try:
                            self.csvs.close()
                        except Exception:
                            pass
                except Exception:
                    pass

                # just before showing or immediately after (prefer after imshow+waitKey to measure when UI loop processed)
                display_ts = time.monotonic()
                # mark display_shown
                try:
                    frame_idx = res.get("frame_id") or getattr(
                        feeder, "src_frame_idx", None
                    )
                    if hasattr(self, "injected") and self.injected.get("metrics"):
                        self.injected["metrics"].mark(
                            int(frame_idx or -1), "display_shown", ts=display_ts
                        )
                except Exception:
                    pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            log.debug("DISPLAY-WORKER", "cv2.destroyAllWindows failed")
            pass
        log.info("DISPLAY-WORKER", "DisplayWorker exiting")

        try:
            if self.video_recorder is not None:
                stats = self.video_recorder.get_stats()
                log.info("VIDEO", f"Recording stats: {stats}")
                self.video_recorder.close()
        except Exception:
            log.debug("DISPLAY-WORKER", "Failed to close video recorder", exc_info=True)

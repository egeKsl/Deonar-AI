# src/capture/stream_threaded.py
"""
Threaded OpenCV capture (producer).

Behavior (unchanged):
 - Opens an OpenCV VideoCapture for `source`.
 - Reads frames in a loop and pushes tuples (frame_id:int, timestamp:float, frame:np.ndarray)
   into the provided capture_queue.
 - If the capture_queue is full, it drops the oldest frame (drop-oldest policy) to keep
   latency low.
 - On read failures the thread will try to reconnect after `reconnect_delay` seconds.
 - Publishes `self.cap_info` (dict) once capture is opened successfully.

This file improves robustness, logging, and exposes a small helper API while preserving
your original logic and behavior.
"""

from __future__ import annotations

import cv2
import time
import threading
import queue
from typing import Any, Dict, Optional, Tuple

from src.utils.logger import log


class ThreadedVideoCapture(threading.Thread):
    """
    Threaded OpenCV capture.

    Args:
        source: OpenCV source (rtsp/http path, filename, or int index).
        capture_queue: queue.Queue to put (frame_id, timestamp, frame) tuples into.
        stop_event: threading.Event used to signal shutdown.
        cap_backend: optional OpenCV backend flag (cv2.CAP_FFMPEG ...).
        reconnect_delay: seconds to wait before reconnect attempts after failure.
        buffersize: optional integer passed to CAP_PROP_BUFFERSIZE (best-effort).
    """

    def __init__(
        self,
        source: Any,
        capture_queue: "queue.Queue[Tuple[int, float, Any]]",
        stop_event: threading.Event,
        cap_backend: Optional[int] = None,
        reconnect_delay: float = 3.0,
        buffersize: Optional[int] = 2,
        metrics: Optional[Any] = None,
    ) -> None:
        super().__init__(name="ThreadedVideoCapture", daemon=True)
        self.source = source
        self.capture_queue = capture_queue
        self.stop_event = stop_event
        self.reconnect_delay = float(reconnect_delay)
        self.cap_backend = cap_backend
        self.buffersize = int(buffersize) if buffersize is not None else None

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_id: int = 0

        # Published metadata (populated after successful open): fps,width,height,total,source
        self.cap_info: Dict[str, Any] = {}
        # Internal flag if open_capture successfully called
        self._opened_once = False
        self.metrics = metrics

    # ---------- public helpers ----------
    def is_opened(self) -> bool:
        """Return True if the underlying cv2.VideoCapture is open right now."""
        try:
            return (
                self.cap is not None and getattr(self.cap, "isOpened", lambda: False)()
            )
        except Exception:
            return False

    # ---------- internal helpers ----------
    def open_capture(self) -> bool:
        """
        Try to open the capture. Returns True on success.
        Populates self.cap_info on success.
        """
        try:
            src = self.source
            if isinstance(self.source, str) and self.source.isdigit():
                src = int(self.source)

            if self.cap_backend is not None:
                self.cap = cv2.VideoCapture(src, self.cap_backend)
            else:
                self.cap = cv2.VideoCapture(src)

            # Attempt to set CAP_PROP_BUFFERSIZE if provided (best-effort)
            if self.cap is not None and self.buffersize is not None:
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffersize)
                except Exception:
                    # Some backends ignore this; ignore failures
                    log.debug(
                        "CAPTURE-THREAD",
                        "Could not set CAP_PROP_BUFFERSIZE; backend may not support it",
                    )
                    pass

            if not self.cap or not getattr(self.cap, "isOpened", lambda: False)():
                log.warn("CAPTURE-THREAD", f"CAP not ready for {self.source}")
                # ensure cap is cleaned up
                try:
                    if self.cap is not None:
                        self.cap.release()
                except Exception:
                    pass
                self.cap = None
                return False

            log.info("CAPTURE-THREAD", f"Opened capture {self.source}")
            self._opened_once = True

            # read and store capture metadata (best-effort)
            try:
                fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
                W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)
                self.cap_info = {
                    "fps": fps,
                    "width": W,
                    "height": H,
                    "total": total,
                    "source": self.source,
                }
                log.debug(
                    "CAPTURE-THREAD",
                    f"Capture metadata: fps={fps} width={W} height={H} total={total}",
                )
            except Exception as e:
                log.warn("CAPTURE-THREAD", f"Could not read capture metadata: {e}")

            return True

        except Exception as e:
            log.error("CAPTURE-THREAD", f"open_capture failed: {e}")
            # ensure cap is None on failure
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None
            return False

    def _put_drop_oldest(self, item: Dict[str, Any]) -> None:
        """
        Put item(dict) into capture_queue using drop-oldest policy.
        Keep logging minimal (debug) to avoid flood.
        """
        try:
            self.capture_queue.put_nowait(item)
        except queue.Full:
            try:
                _ = self.capture_queue.get_nowait()
            except Exception:
                pass
            try:
                self.capture_queue.put_nowait(item)
            except Exception:
                # final fallback: give up on frame
                log.debug(
                    "CAPTURE-THREAD",
                    f"Dropping frame {item.get('frame_index')} (queue full after drop attempt)",
                )

    # ---------- main loop ----------
    def run(self) -> None:
        """Thread loop: open capture, read frames, push into queue, reconnect on failure."""
        log.debug("CAPTURE-THREAD", f"Thread started for source={self.source}")
        while not self.stop_event.is_set():
            # ensure capture open
            if self.cap is None or not self.is_opened():
                ok = self.open_capture()
                if not ok:
                    # wait and retry
                    time.sleep(self.reconnect_delay)
                    continue

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                log.error("CAPTURE-THREAD", f"cap.read exception: {e}")
                ret = False
                frame = None

            if not ret or frame is None:
                log.warn(
                    "CAPTURE-THREAD", f"read failed; reconnecting {self.source}"
                )
                try:
                    if self.cap is not None:
                        self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(self.reconnect_delay)
                continue

            ts = time.monotonic()
            self.frame_id += 1

            # Build item dict per agreed contract
            fps_hint = (
                float(self.cap_info.get("fps"))
                if self.cap_info and self.cap_info.get("fps")
                else None
            )
            try:
                width = int(frame.shape[1])
                height = int(frame.shape[0])
            except Exception:
                log.error("CAPTURE-THREAD", "Could not read frame dimensions")
                width = 0
                height = 0

            item = {
                "frame": frame,
                "frame_index": int(self.frame_id),
                "capture_time": ts,
                "source_time": None,  # decoder PTS not implemented here; keep None
                "fps_hint": fps_hint,
                "meta": {
                    "width": width,
                    "height": height,
                    "source": self.source,
                },
            }

            # push into capture_queue using drop-oldest policy (non-blocking)
            self._put_drop_oldest(item)
            
            # after you push item into queue, mark captured (if metrics provided)
            try:
                # assume runner injected metrics into global context? Better: pass metrics into ThreadedVideoCapture ctor.
                if hasattr(self, "metrics") and self.metrics is not None:
                    self.metrics.mark(int(self.frame_id), "captured", ts=ts)
            except Exception:
                pass    

        # cleanup on exit
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        log.info("CAPTURE-THREAD", "ThreadedVideoCapture exiting")

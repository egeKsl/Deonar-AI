# src/utils/video_recorder.py
from __future__ import annotations

import os
import time
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

from src.utils.logger import log


class VideoRecorder:
    """
    Tiny wrapper around cv2.VideoWriter with extra safety + stats.

    Usage:
        rec = VideoRecorder(path, fps=25, fourcc="mp4v")
        rec.ensure_open(frame.shape[1], frame.shape[0])
        rec.write(frame_bgr)
        stats = rec.get_stats()
        rec.close()
    """

    def __init__(
        self,
        path: str,
        fps: float,
        fourcc: str = "mp4v",
        strict_size: bool = True,
    ) -> None:
        """
        :param path: Output video path (will create parent dirs).
        :param fps: Target FPS for VideoWriter.
        :param fourcc: FourCC codec string (e.g. 'mp4v', 'XVID').
        :param strict_size:
            If True, any frame size change after first frame raises RuntimeError.
            If False, recording is stopped and error is logged instead.
        """
        self.path = path
        self.fps = float(fps)
        self.fourcc_str = fourcc or "mp4v"
        self.strict_size = bool(strict_size)

        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[Tuple[int, int]] = None  # (w, h)
        self._enabled: bool = True  # flipped to False on fatal error

        # Stats
        self._frame_count: int = 0
        self._open_time: Optional[float] = None  # monotonic
        self._close_time: Optional[float] = None  # monotonic
        self._write_times_ms: List[float] = []  # per-frame write duration
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Core open / write / close
    # ------------------------------------------------------------------ #
    def ensure_open(self, width: int, height: int) -> None:
        """Lazy-open the writer on first frame; re-use afterwards."""
        if not self._enabled:
            return
        if self._writer is not None:
            return

        try:
            parent = os.path.dirname(self.path) or "."
            os.makedirs(parent, exist_ok=True)
        except Exception as e:
            msg = f"Failed to create directory for video '{self.path}': {e}"
            log.error("VIDEO", msg)
            self._last_error = msg
            self._enabled = False
            return

        fourcc = cv2.VideoWriter_fourcc(*self.fourcc_str)
        writer = cv2.VideoWriter(
            self.path,
            fourcc,
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            msg = (
                f"Failed to open VideoWriter for '{self.path}' "
                f"(fps={self.fps}, size=({width},{height}), fourcc={self.fourcc_str})"
            )
            log.error("VIDEO", msg)
            try:
                writer.release()
            except Exception:
                pass
            self._writer = None
            self._enabled = False
            self._last_error = msg
            return

        self._writer = writer
        self._size = (width, height)
        self._open_time = time.monotonic()
        self._close_time = None
        self._frame_count = 0
        self._write_times_ms.clear()
        log.info("VIDEO", f"Video recording started: {self.path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        """Write a single BGR frame if recording is enabled."""
        if not self._enabled:
            return
        if frame_bgr is None:
            return

        h, w = frame_bgr.shape[:2]
        self.ensure_open(w, h)
        if self._writer is None:
            # open() must have failed; error already logged
            return

        # Enforce constant frame size
        if self._size is not None and self._size != (w, h):
            msg = (
                f"Frame size changed from {self._size} to {(w, h)} "
                f"for '{self.path}'. Recording requires fixed size."
            )
            log.error("VIDEO", msg)
            self._last_error = msg

            # Clean up writer
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
            self._enabled = False
            self._close_time = time.monotonic()

            if self.strict_size:
                # Hard fail so caller notices the bug.
                raise RuntimeError(msg)
            # Non-strict: just stop recording.
            return

        # Normal write with timing
        try:
            t0 = time.perf_counter()
            self._writer.write(frame_bgr)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._write_times_ms.append(dt_ms)
            self._frame_count += 1
        except Exception as e:
            msg = f"VideoWriter.write failed: {e}"
            log.debug("VIDEO", msg, exc_info=True)
            self._last_error = msg

    def close(self) -> None:
        """Close the underlying writer and freeze stats."""
        if self._writer is not None:
            try:
                self._writer.release()
                log.info("VIDEO", f"Video recording stopped: {self.path}")
            except Exception as e:
                msg = f"VideoWriter.release failed: {e}"
                log.debug("VIDEO", msg, exc_info=True)
                self._last_error = msg

        self._writer = None
        self._enabled = False
        if self._close_time is None:
            self._close_time = time.monotonic()

    # ------------------------------------------------------------------ #
    # Stats / inspection
    # ------------------------------------------------------------------ #
    def get_stats(self) -> Dict[str, Any]:
        """
        Return a dictionary with detailed stats:

        {
          "path": ...,
          "fps_config": float,
          "fourcc": "mp4v",
          "size": (w, h) or None,
          "frame_count": int,
          "duration_nominal_s": float or None,
          "duration_wall_s": float or None,
          "avg_write_ms": float or None,
          "min_write_ms": float or None,
          "max_write_ms": float or None,
          "open_time_monotonic": float or None,
          "close_time_monotonic": float or None,
          "is_open": bool,
          "enabled": bool,
          "last_error": str or None,
        }
        """
        is_open = self._writer is not None

        # Duration based on FPS (what the file "represents")
        duration_nominal_s: Optional[float] = None
        if self.fps > 0 and self._frame_count > 0:
            duration_nominal_s = self._frame_count / self.fps

        # Wall-clock duration (how long we were recording)
        duration_wall_s: Optional[float] = None
        if self._open_time is not None:
            end_t = (
                self._close_time if self._close_time is not None else time.monotonic()
            )
            duration_wall_s = max(0.0, end_t - self._open_time)

        # Write timing stats
        avg_write_ms = min_write_ms = max_write_ms = None
        if self._write_times_ms:
            min_write_ms = min(self._write_times_ms)
            max_write_ms = max(self._write_times_ms)
            avg_write_ms = sum(self._write_times_ms) / len(self._write_times_ms)

        return {
            "path": self.path,
            "fps_config": self.fps,
            "fourcc": self.fourcc_str,
            "size": self._size,
            "frame_count": self._frame_count,
            "duration_nominal_s": duration_nominal_s,
            "duration_nominal_min": (
                duration_nominal_s / 60.0 if duration_nominal_s is not None else None
            ),
            "duration_wall_s": duration_wall_s,
            "duration_wall_min": (
                duration_wall_s / 60.0 if duration_wall_s is not None else None
            ),
            "avg_write_ms": avg_write_ms,
            "min_write_ms": min_write_ms,
            "max_write_ms": max_write_ms,
            "open_time_monotonic": self._open_time,
            "close_time_monotonic": self._close_time,
            "is_open": is_open,
            "enabled": self._enabled,
            "last_error": self._last_error,
        }

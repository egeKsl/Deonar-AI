# src/counting/counting_modes/dual_lines.py
"""
Dual-line counting utilities.

Behaviour preserved from original implementation:
 - Two parallel lines (A primary, B verify/recovery).
 - Two modes supported by the pipeline:
     * verify: require A flip then B flip within window to count.
     * recover: count on A immediately; B may recover (one-shot) if A missed.
 - Uses sign-history confirmation (strict then loose majority) to prevent noise.
 - Prevents double-counting using per-tid lock frames.

Public:
 - class DualLineCounter
    - update_verify(tid, cx, cy, frame_idx, margin_px) -> Optional[event_dict]
    - update_recover(tid, cx, cy, frame_idx, margin_px, on_A_count_cb) -> Optional[event_dict]
"""

from collections import defaultdict, deque
from typing import Tuple, Optional, Dict, Any, Callable
import math
import numpy as np

from src.utils.logger import log


def _line_side(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """
    Cross product sign for point (px,py) relative to oriented line (ax,ay)->(bx,by).
    Positive => one side, negative => the other side; magnitude proportional to area.
    """
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _confirm_side(
    side_hist: deque,
    need_frames: int,
    min_ratio: float = 0.6,
    min_nonzero_abs: int = 1,
    min_nonzero_ratio: float = 0.0,
) -> Optional[int]:
    """
    Confirm a stable side (-1/+1) from a deque of recent side-sign samples.

    Strategy:
      - STRICT: if the last `need_frames` non-zero signs are all identical and count == need_frames -> return that sign.
      - LOOSE: majority among the last `need_frames` non-zero signs >= min_ratio -> return that sign.
      - Otherwise return None.

    side_hist elements are expected to be -1, 0, or +1 (0 means "near the line / ignore").
    """
    if need_frames <= 0:
        need_frames = 1

    if len(side_hist) < need_frames:
        return None

    window = list(side_hist)[-need_frames:]
    nonzeros = [s for s in window if s != 0]
    if not nonzeros:
        return None
    # Require a minimum count of non-zero samples to avoid confirming from noise.
    min_n = max(int(min_nonzero_abs or 0), int(math.ceil(need_frames * min_nonzero_ratio)))
    if len(nonzeros) < max(1, min_n):
        return None

    # STRICT: all identical and we have need_frames non-zero samples
    if len(nonzeros) == need_frames and all(s == nonzeros[0] for s in nonzeros):
        return nonzeros[0]

    # LOOSE: majority among non-zero samples
    pos = sum(1 for s in nonzeros if s > 0)
    neg = sum(1 for s in nonzeros if s < 0)
    total = len(nonzeros)
    if total > 0:
        if pos / total >= min_ratio:
            return +1
        if neg / total >= min_ratio:
            return -1

    return None


class MotionHistory:
    """
    Phase-1 motion recorder.
    Stores recent centroid positions per track_id.
    No decisions, no counting, no direction inference.
    """

    def __init__(self, max_frames: int = 60):
        self.max_frames = max_frames
        # tid -> deque[(frame_idx, cx, cy)]
        self.history = defaultdict(lambda: deque(maxlen=self.max_frames))

    def update(self, tid: int, frame_idx: int, cx: float, cy: float):
        self.history[tid].append((frame_idx, cx, cy))

    def get(self, tid: int):
        """Return list of (frame_idx, cx, cy) for this tid."""
        return list(self.history.get(tid, []))

    def clear_tid(self, tid: int):
        self.history.pop(tid, None)

    def clear_all(self):
        self.history.clear()


class MotionIntentAnalyzer:
    """
    Phase-2 brain.
    Reads motion history + geometry event and decides
    whether motion supports the geometry decision.

    This class is READ-ONLY: no counters, no side effects.
    """

    def __init__(
        self,
        min_frames: int = 6,
        min_displacement_px: float = 12.0,
        max_lookback_frames: int = 30,
        dir_consistency_ratio: float = 0.7,
        dir_consistency_min_frames: int = 4,
        axis_min_displacement_px: float = 10.0,
    ):
        self.min_frames = min_frames
        self.min_disp = min_displacement_px
        self.max_lookback = max_lookback_frames
        self.dir_consistency_ratio = dir_consistency_ratio
        self.dir_consistency_min_frames = dir_consistency_min_frames
        self.axis_min_disp = axis_min_displacement_px

    def analyze(
        self,
        motion_history,
        tid: int,
        event_frame: int,
        geometry_direction: str,
        line_vector: tuple,  # (dx_line, dy_line)
    ) -> dict:
        """
        Returns a decision dict:
        accept | reject | defer
        """

        hist = motion_history.get(tid)
        if not hist:
            return self._defer("no motion history")

        # only frames before geometry event
        past = [(f, x, y) for (f, x, y) in hist if f <= event_frame]

        if len(past) < self.min_frames:
            return self._defer("insufficient motion frames")

        # keep recent window
        past = past[-self.max_lookback :]

        f0, x0, y0 = past[0]
        f1, x1, y1 = past[-1]

        dx = x1 - x0
        dy = y1 - y0

        disp = (dx * dx + dy * dy) ** 0.5
        if disp < self.min_disp:
            return self._reject(dx, dy, "motion too small")

        # line normal (direction of crossing)
        lx, ly = line_vector
        ln = (lx * lx + ly * ly) ** 0.5 or 1.0
        nx, ny = -ly / ln, lx / ln

        projection = dx * nx + dy * ny
        motion_dir = "up" if projection > 0 else "down"

        # Axis aligned with the normal (dominant component of normal)
        dominant_axis = "x" if abs(nx) >= abs(ny) else "y"
        axis_disp = abs(dx) if dominant_axis == "x" else abs(dy)
        if axis_disp < self.axis_min_disp:
            return self._reject(dx, dy, "axis displacement too small", dominant_axis)

        # Direction consistency across recent frames (along dominant axis)
        axis_steps = []
        for i in range(1, len(past)):
            _, x_prev, y_prev = past[i - 1]
            _, x_cur, y_cur = past[i]
            step = (x_cur - x_prev) if dominant_axis == "x" else (y_cur - y_prev)
            if step != 0:
                axis_steps.append(step)

        if axis_steps:
            overall_sign = 1 if (dx if dominant_axis == "x" else dy) > 0 else -1
            consistent = sum(1 for s in axis_steps if (s > 0) == (overall_sign > 0))
            if (
                len(axis_steps) < self.dir_consistency_min_frames
                or (consistent / len(axis_steps)) < self.dir_consistency_ratio
            ):
                return self._reject(dx, dy, "direction inconsistent", dominant_axis)
        else:
            return self._defer("no directional steps")

        if motion_dir != geometry_direction:
            return self._reject(
                dx,
                dy,
                f"motion contradicts geometry ({motion_dir})",
                dominant_axis,
            )

        confidence = min(1.0, disp / (self.min_disp * 3))

        return {
            "decision": "accept",
            "confidence": confidence,
            "dominant_axis": dominant_axis,
            "delta": (dx, dy),
            "reason": "motion supports geometry",
        }

    def _reject(self, dx, dy, reason, axis=None):
        return {
            "decision": "reject",
            "confidence": 0.0,
            "dominant_axis": axis,
            "delta": (dx, dy),
            "reason": reason,
        }

    def _defer(self, reason):
        return {
            "decision": "defer",
            "confidence": 0.0,
            "dominant_axis": None,
            "delta": (0.0, 0.0),
            "reason": reason,
        }


class DualLineCounter:
    """
    Two-line counter object.

    Parameters
    ----------
    lineA_roi : (ax, ay, bx, by)  # ROI coordinate floats
    lineB_roi : (ax, ay, bx, by)
    hyst_frames : int    # frames required for strict confirmation (hysteresis)
    window_frames : int  # allowed frames between A flip and B flip for verify mode
    id_lock_frames : int # frames to lock a tid after counting (prevent duplicate)
    quiet : bool         # if True, suppress extra logging
    debug_enabled : bool # internal debug logging
    """

    def __init__(
        self,
        lineA_roi: Tuple[float, float, float, float],
        lineB_roi: Tuple[float, float, float, float],
        hyst_frames: int = 2,
        window_frames: int = 30,
        id_lock_frames: int = 60,
        quiet: bool = False,
        debug_enabled: bool = False,
        min_nonzero_abs: int = 3,
        min_nonzero_ratio: float = 0.6,
        motion_min_frames: int = 6,
        motion_min_displacement_px: float = 12.0,
        motion_max_lookback_frames: int = 30,
        motion_dir_consistency_ratio: float = 0.7,
        motion_dir_consistency_min_frames: int = 4,
        motion_axis_min_displacement_px: float = 10.0,
    ):
        self.A = tuple(lineA_roi)
        self.B = tuple(lineB_roi)
        self.hyst = max(1, int(hyst_frames))
        self.window = max(1, int(window_frames))
        self.lock_frames = max(1, int(id_lock_frames))
        self.quiet = bool(quiet)
        self.debug_enabled = bool(debug_enabled)
        self.min_nonzero_abs = max(1, int(min_nonzero_abs))
        self.min_nonzero_ratio = float(min_nonzero_ratio)

        # per-tid histories and last confirmed side
        self.side_hist_A = defaultdict(lambda: deque(maxlen=max(3, self.hyst)))
        self.side_hist_B = defaultdict(lambda: deque(maxlen=max(3, self.hyst)))
        self.last_side_A: Dict[int, int] = {}
        self.last_side_B: Dict[int, int] = {}

        # pending A flips for verify mode: tid -> {"dir":"up"|"down", "frame": idx}
        self.pending_A: Dict[int, Dict[str, Any]] = {}

        # last frame when tid was counted (for lock)
        self.last_counted: Dict[int, int] = defaultdict(lambda: -(10**9))

        # ---- Phase 1: motion history (eyes) ----
        self.motion = MotionHistory(max_frames=60)

        self.motion_analyzer = MotionIntentAnalyzer(
            min_frames=motion_min_frames,
            min_displacement_px=motion_min_displacement_px,
            max_lookback_frames=motion_max_lookback_frames,
            dir_consistency_ratio=motion_dir_consistency_ratio,
            dir_consistency_min_frames=motion_dir_consistency_min_frames,
            axis_min_displacement_px=motion_axis_min_displacement_px,
        )

        # ---- Phase 1.5: geometry evidence buffer ----
        self.geometry_events = []  # List[dict]

    def _dbg(self, msg: str) -> None:
        if self.debug_enabled:
            log.debug("DUAL-DBG", msg)

    @staticmethod
    def centroid_from(
        box: Tuple[float, float, float, float], mask
    ) -> Tuple[float, float]:
        """
        Compute centroid from box or mask.
        Returns (cx, cy).
        """
        x1, y1, x2, y2 = box
        if mask is not None:
            ys, xs = np.where(mask)
            if xs.size > 0 and ys.size > 0:
                return float(xs.mean()), float(ys.mean())
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    # Alias for backward compatibility
    _centroid_from = centroid_from

    def _side_sign(
        self,
        cx: float,
        cy: float,
        line_roi: Tuple[float, float, float, float],
        margin_px: float,
    ) -> int:
        """
        Compute side sign with distance margin.
        Returns -1, 0, or +1.
        """
        ax, ay, bx, by = line_roi
        raw = _line_side(cx, cy, ax, ay, bx, by)
        # perpendicular distance from point to infinite line
        num = abs((bx - ax) * (ay - cy) - (by - ay) * (ax - cx))
        den = max(1e-6, math.hypot(bx - ax, by - ay))
        dist = num / den
        if dist < margin_px:
            return 0
        return 1 if raw > 0 else -1

    @staticmethod
    def _direction_flip(prev: Optional[int], cur: int) -> Optional[str]:
        """
        Convert sign flip into "up" / "down" string or None.
        prev, cur expected in {-1, 0, +1}
        If prev==0 or cur==0 or prev==cur -> no flip
        """
        if prev is None or prev == 0 or cur == 0 or prev == cur:
            return None
        return "up" if prev < cur else "down"

    def reset_tid(self, tid: int) -> None:
        """Clear internal state for a tid (helpful for testing or forced resets)."""
        if tid in self.side_hist_A:
            del self.side_hist_A[tid]
        if tid in self.side_hist_B:
            del self.side_hist_B[tid]
        self.last_side_A.pop(tid, None)
        self.last_side_B.pop(tid, None)
        self.pending_A.pop(tid, None)
        self.last_counted.pop(tid, None)

    # ---------- VERIFY MODE ----------
    def update_verify(
        self, tid: int, cx: float, cy: float, frame_idx: int, margin_px: int
    ) -> Optional[Dict[str, Any]]:
        """
        VERIFY mode update:
          - Detect flip on A -> create pending entry (dir, frame)
          - If a flip on B occurs for the same tid and same direction within 'window', commit a count
            (respect lock_frames).
        Returns event dict {"type":"count", "subsystem":"dual_verify", "direction":"up|down", "tid":tid} on commit
        or None otherwise.
        """
        sA = self._side_sign(cx, cy, self.A, margin_px)
        sB = self._side_sign(cx, cy, self.B, margin_px)

        if sA != 0:
            self.side_hist_A[tid].append(sA)
        if sB != 0:
            self.side_hist_B[tid].append(sB)

        cA = _confirm_side(
            self.side_hist_A[tid],
            self.hyst,
            min_nonzero_abs=self.min_nonzero_abs,
            min_nonzero_ratio=self.min_nonzero_ratio,
        )
        cB = _confirm_side(
            self.side_hist_B[tid],
            self.hyst,
            min_nonzero_abs=self.min_nonzero_abs,
            min_nonzero_ratio=self.min_nonzero_ratio,
        )

        # Check A flip (create pending entry)
        if cA is not None:
            prevA = self.last_side_A.get(tid, None)
            self.last_side_A[tid] = cA
            flipA = None
            if prevA is not None and prevA != 0 and cA != 0 and prevA != cA:
                flipA = "up" if prevA < cA else "down"
            if flipA is not None:
                self.pending_A[tid] = {"dir": flipA, "frame": frame_idx}
                self._dbg(
                    f"tid={tid} A-flip={flipA} @ {frame_idx} (prevA={prevA} cA={cA})"
                )

        # Check B flip; only meaningful if there is a pending A
        if cB is not None and tid in self.pending_A:
            prevB = self.last_side_B.get(tid, None)
            self.last_side_B[tid] = cB
            flipB = None
            if prevB is not None and prevB != 0 and cB != 0 and prevB != cB:
                flipB = "up" if prevB < cB else "down"

            if flipB is not None:
                pend = self.pending_A.get(tid)
                if pend is None:
                    self._dbg(
                        f"tid={tid} B-flip but pending_A missing (race); ignoring."
                    )
                    return None

                same_dir = flipB == pend["dir"]
                within_win = frame_idx - pend["frame"] <= self.window
                unlocked = frame_idx - self.last_counted[tid] >= self.lock_frames

                self._dbg(
                    f"tid={tid} B-flip={flipB} @ {frame_idx} | pending dir={pend['dir']} @ {pend['frame']} | "
                    f"within_win={within_win} unlocked={unlocked}"
                )

                if same_dir and within_win and unlocked:
                    event = {
                        "tid": tid,
                        "frame_idx": frame_idx,
                        "direction": flipB,
                        "type": "count",
                        "subsystem": "dual_verify",
                        "mode": "verify",
                        "line_sequence": ["A", "B"],
                        "trigger_line": "B",
                        "reason": "A then B crossed within window",
                    }
                    self.geometry_events.append(event)
                    return event
                else:
                    # drop pending and log reason
                    reason = []
                    if not same_dir:
                        reason.append("dir_mismatch")
                    if not within_win:
                        reason.append("window_expired")
                    if not unlocked:
                        reason.append("locked")
                    self._dbg(
                        f"tid={tid} drop pending A -> {','.join(reason) or 'unknown'}"
                    )

        return None

    # ---------- RECOVER MODE ----------
    def update_recover(
        self,
        tid: int,
        cx: float,
        cy: float,
        frame_idx: int,
        margin_px: int,
        on_A_count_cb: Callable[[int, str], None],
    ) -> Optional[Dict[str, Any]]:
        """
        RECOVER mode update:
          - A: on flip, immediately call on_A_count_cb and commit a count if unlocked
          - B: if A didn't count recently for this tid, allow B to count as a one-shot recovery (if unlocked)
        on_A_count_cb is called when A produces an immediate count to allow external state
        (it mirrors the original behaviour which invoked a callback).
        Returns a dict similar to VERIFY on commit or None.
        """
        # A side / immediate counts
        sA = self._side_sign(cx, cy, self.A, margin_px)
        if sA != 0:
            self.side_hist_A[tid].append(sA)
        cA = _confirm_side(
            self.side_hist_A[tid],
            self.hyst,
            min_nonzero_abs=self.min_nonzero_abs,
            min_nonzero_ratio=self.min_nonzero_ratio,
        )
        if cA is not None:
            prevA = self.last_side_A.get(tid, None)
            self.last_side_A[tid] = cA
            flipA = self._direction_flip(prevA, cA)
            if flipA is not None:
                if (frame_idx - self.last_counted[tid]) >= self.lock_frames:
                    # call callback to let the pipeline increment counters in real-time
                    try:
                        on_A_count_cb(tid, flipA)
                    except Exception:
                        # swallow callback failure; still return event dict
                        log.debug(
                            "DUAL-RECOVER",
                            f"on_A_count_cb failed for tid={tid}",
                            exc_info=True,
                        )

                    event = {
                        "tid": tid,
                        "frame_idx": frame_idx,
                        "direction": flipA,
                        "mode": "recover",
                        "line_sequence": ["A"],
                        "trigger_line": "A",
                        "reason": "Immediate A crossing",
                        "type": "count",
                        "subsystem": "dual_recover_A",
                    }
                    self.geometry_events.append(event)
                    return event

        # B side recovery (one-shot if lock has elapsed)
        sB = self._side_sign(cx, cy, self.B, margin_px)
        if sB != 0:
            self.side_hist_B[tid].append(sB)
        cB = _confirm_side(
            self.side_hist_B[tid],
            self.hyst,
            min_nonzero_abs=self.min_nonzero_abs,
            min_nonzero_ratio=self.min_nonzero_ratio,
        )
        if cB is not None:
            prevB = self.last_side_B.get(tid, None)
            self.last_side_B[tid] = cB
            flipB = self._direction_flip(prevB, cB)
            if flipB is not None:
                if (frame_idx - self.last_counted[tid]) >= self.lock_frames:

                    event = {
                        "tid": tid,
                        "frame_idx": frame_idx,
                        "direction": flipB,
                        "mode": "recover",
                        "line_sequence": ["B"],
                        "trigger_line": "B",
                        "reason": "Recovered at B (A missed)",
                        "type": "count",
                        "subsystem": "dual_recover_B",
                    }
                    self.geometry_events.append(event)
                    return event

        return None

    # Introspection helper for debugging/tests
    def get_state(self) -> Dict[str, Any]:
        return {
            "pending_A": dict(self.pending_A),
            "last_counted": dict(self.last_counted),
            "last_side_A": dict(self.last_side_A),
            "last_side_B": dict(self.last_side_B),
        }

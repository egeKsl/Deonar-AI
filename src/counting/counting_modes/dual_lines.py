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
    side_hist: deque, need_frames: int, min_ratio: float = 0.6
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
    ):
        self.A = tuple(lineA_roi)
        self.B = tuple(lineB_roi)
        self.hyst = max(1, int(hyst_frames))
        self.window = max(1, int(window_frames))
        self.lock_frames = max(1, int(id_lock_frames))
        self.quiet = bool(quiet)
        self.debug_enabled = bool(debug_enabled)

        # per-tid histories and last confirmed side
        self.side_hist_A = defaultdict(lambda: deque(maxlen=max(3, self.hyst)))
        self.side_hist_B = defaultdict(lambda: deque(maxlen=max(3, self.hyst)))
        self.last_side_A: Dict[int, int] = {}
        self.last_side_B: Dict[int, int] = {}

        # pending A flips for verify mode: tid -> {"dir":"up"|"down", "frame": idx}
        self.pending_A: Dict[int, Dict[str, Any]] = {}

        # last frame when tid was counted (for lock)
        self.last_counted: Dict[int, int] = defaultdict(lambda: -(10**9))

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

        cA = _confirm_side(self.side_hist_A[tid], self.hyst)
        cB = _confirm_side(self.side_hist_B[tid], self.hyst)

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
                    self.last_counted[tid] = frame_idx
                    # consume pending and return event
                    try:
                        del self.pending_A[tid]
                    except KeyError:
                        pass
                    return {
                        "type": "count",
                        "subsystem": "dual_verify",
                        "direction": flipB,
                        "tid": tid,
                    }
                else:
                    # drop pending and log reason
                    reason = []
                    if not same_dir:
                        reason.append("dir_mismatch")
                    if not within_win:
                        reason.append("window_expired")
                    if not unlocked:
                        reason.append("locked")
                    try:
                        del self.pending_A[tid]
                    except KeyError:
                        pass
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
        cA = _confirm_side(self.side_hist_A[tid], self.hyst)
        if cA is not None:
            prevA = self.last_side_A.get(tid, None)
            self.last_side_A[tid] = cA
            flipA = self._direction_flip(prevA, cA)
            if flipA is not None:
                if (frame_idx - self.last_counted[tid]) >= self.lock_frames:
                    self.last_counted[tid] = frame_idx
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
                    return {
                        "type": "count",
                        "subsystem": "dual_recover_A",
                        "direction": flipA,
                        "tid": tid,
                    }

        # B side recovery (one-shot if lock has elapsed)
        sB = self._side_sign(cx, cy, self.B, margin_px)
        if sB != 0:
            self.side_hist_B[tid].append(sB)
        cB = _confirm_side(self.side_hist_B[tid], self.hyst)
        if cB is not None:
            prevB = self.last_side_B.get(tid, None)
            self.last_side_B[tid] = cB
            flipB = self._direction_flip(prevB, cB)
            if flipB is not None:
                if (frame_idx - self.last_counted[tid]) >= self.lock_frames:
                    self.last_counted[tid] = frame_idx
                    return {
                        "type": "count",
                        "subsystem": "dual_recover_B",
                        "direction": flipB,
                        "tid": tid,
                    }

        return None

    # Introspection helper for debugging/tests
    def get_state(self) -> Dict[str, Any]:
        return {
            "pending_A": dict(self.pending_A),
            "last_counted": dict(self.last_counted),
            "last_side_A": dict(self.last_side_A),
            "last_side_B": dict(self.last_side_B),
        }

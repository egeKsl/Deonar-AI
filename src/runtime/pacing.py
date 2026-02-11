# src/runtime/pacing.py
from __future__ import annotations
import time
import threading
import queue
from typing import Any, Dict, Optional
from src.utils.logger import log
from src.utils.pacing_contract_dict import _ensure_item_dict


class PacingController:
    """
    PacingController pulls frame items from capture_q and pushes paced frames
    to out_q. Non-blocking for capture (capture_q should be bounded and use
    drop-oldest on capture side).

    Config keys (from CONFIG["runtime"]):
      sync, playback_speed, autoskip, max_lag_s, sync_jitter_allowance_s,
      max_sleep_s, skip_policy, max_catchup_resync_s
    """

    def __init__(
        self,
        capture_q: "queue.Queue[Dict[str,Any]]",
        out_q: "queue.Queue[Dict[str,Any]]",
        cfg: Dict[str, Any],
        name: str = "PacingController",
        metrics: Optional[Any] = None,
    ):
        self.capture_q = capture_q
        self.out_q = out_q
        self.cfg = dict(cfg or {})
        self.sync = bool(self.cfg.get("sync", True))
        self.playback_speed = float(self.cfg.get("playback_speed", 1.0))
        self.autoskip = bool(self.cfg.get("autoskip", False))
        self.max_lag_s = float(self.cfg.get("max_lag_s", 0.75))
        self.sync_jitter_allowance_s = float(
            self.cfg.get("sync_jitter_allowance_s", 0.02)
        )
        self.max_sleep_s = float(self.cfg.get("max_sleep_s", 1.0))
        self.skip_policy = str(self.cfg.get("skip_policy", "drop_to_latest"))
        self.max_catchup_resync_s = float(self.cfg.get("max_catchup_resync_s", 5.0))

        self.name = name
        self._stop = threading.Event()
        self._join_timeout_default = float(self.cfg.get("pacer_join_timeout", 1.0))
        self._drain_limit = int(
            self.cfg.get("pacer_drain_limit", 10)
        )  # max items to drain when skipping
        self._thread: Optional[threading.Thread] = None

        # anchors for mapping source->real time
        self._start_source_ts: Optional[float] = None
        self._start_source_idx: Optional[int] = None
        self._start_real_ts: Optional[float] = None

        self.frames_received = 0
        self.frames_emitted = 0
        self.frames_skipped = 0
        self.frames_out_dropped = 0
        
        self.metrics = metrics

    def start(self):
        if self._thread and self._thread.is_alive():
            log.warn("PACE", "start() called but pacing thread already running")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=self.name)
        self._thread.start()
        log.info("PACE", f"started (sync={self.sync}, speed={self.playback_speed})")

    def stop(self, wait: bool = True, timeout: Optional[float] = None):
        """Signal stop. Optionally wait (join) for the thread to exit."""
        self._stop.set()
        if self._thread is None:
            return
        if wait:
            self.join(timeout if timeout is not None else self._join_timeout_default)
        log.info(
            "PACE",
            (
                "pacing thread joined cleanly successfully terminated"
                if not self._thread or not self._thread.is_alive()
                else "pacing thread still alive after join timeout"
            ),
        )

    def join(self, timeout: Optional[float] = None):
        if self._thread is None:
            return
        try:
            self._thread.join(
                timeout if timeout is not None else self._join_timeout_default
            )
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            log.warn("PACE", "join() timed out; pacing thread still alive")

    def _sleep_chunked(self, secs: float):
        rem = float(secs)
        while rem > 0 and not self._stop.is_set():
            chunk = min(0.02, rem)
            time.sleep(chunk)
            rem -= chunk

    def _get_source_ts(self, item: Dict[str, Any]) -> float:
        # Prefer explicit PTS/source_time, then capture_time (monotonic), then index/fps fallback (anchored)
        if item.get("source_time") is not None:
            try:
                return float(item["source_time"])
            except Exception:
                pass
        if item.get("capture_time") is not None:
            try:
                return float(item["capture_time"])
            except Exception:
                pass

        # fallback to frame_index/fps_hint; require fps_hint > 0 and an anchored start index
        idx = item.get("frame_index")
        fps = item.get("fps_hint")
        if idx is None or fps in (None, 0):
            # last resort: monotonic now (will cause fallback flow but still safe)
            return float(time.monotonic())

        # if we already anchored a source_ts and a start index, compute time relative to that anchor,
        # otherwise, treat idx/fps as an absolute timeline (best-effort)
        if self._start_source_idx is not None and self._start_source_ts is not None:
            try:
                delta_frames = float(idx - self._start_source_idx)
                return float(self._start_source_ts) + (delta_frames / float(fps))
            except Exception:
                return float(time.monotonic())
        else:
            # best-effort absolute estimate (less preferred)
            try:
                return float(idx) / float(fps)
            except Exception:
                return float(time.monotonic())

    def _drain_to_latest(self) -> Optional[Dict[str, Any]]:
        """
        Drain up to self._drain_limit items and return the latest item.
        IMPORTANT: do NOT call task_done() here; the caller will call task_done()
        after selecting/emitting the chosen item to keep get()/task_done() balanced
        in a single place.
        """
        latest = None
        drained = 0
        while drained < self._drain_limit:
            try:
                it = self.capture_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            latest = it
        # optionally log drain count if non-zero (debug)
        if drained:
            log.debug(
                "PACE",
                f"drained {drained} frames to latest (queue may have been backlogged)",
            )
        if latest is None:
            return None, 0
        return _ensure_item_dict(latest), drained

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    raw_item = self.capture_q.get(timeout=0.25)
                except queue.Empty:
                    continue

                item = _ensure_item_dict(raw_item)
                # ensure capture_time is monotonic; if None, set to now
                if item.get("capture_time") is None:
                    item["capture_time"] = time.monotonic()
                # Optionally add a wall_time for logs/CSV
                item.setdefault("wall_time", time.time())

                self.frames_received += 1
                source_ts = self._get_source_ts(item)
                frame_idx = item.get("frame_index")

                if self._start_source_ts is None:
                    # anchor both timestamp and index if available
                    self._start_source_ts = source_ts
                    self._start_real_ts = time.monotonic()
                    if frame_idx is not None:
                        self._start_source_idx = int(frame_idx)
                    log.info(
                        "PACE",
                        f"anchored timeline start_source_ts={self._start_source_ts} start_idx={self._start_source_idx}",
                    )

                # if not syncing, pass through (single place for task_done)
                if not self.sync:
                    self._emit_item(item)
                    try:
                        # mark the get() as processed now that we emitted
                        self.capture_q.task_done()
                    except Exception:
                        pass
                    continue

                # pacing decision loop (try to decide whether to sleep/skip/resync)
                while not self._stop.is_set():
                    elapsed_source = (source_ts - self._start_source_ts) / max(
                        1e-12, self.playback_speed
                    )
                    elapsed_real = time.monotonic() - self._start_real_ts
                    delta = elapsed_source - elapsed_real

                    # if source is ahead => sleep a bit (bounded)
                    if delta > self.sync_jitter_allowance_s:
                        to_sleep = min(delta, self.max_sleep_s)
                        self._sleep_chunked(to_sleep)
                        # then re-evaluate
                        continue

                    # if source is late beyond max_lag, consider skipping/catching-up
                    if delta < -self.max_lag_s:
                        # autoskip logic: drop_to_latest
                        if self.autoskip and self.skip_policy == "drop_to_latest":
                            latest, drained = self._drain_to_latest()
                            if latest:
                                self.frames_skipped += max(0, drained - 1)
                                # now treat the latest as the new candidate item
                                item = latest
                                source_ts = self._get_source_ts(item)
                                frame_idx = item.get("frame_index")

                                # compute how late the new candidate is relative to anchor
                                late_amount = (
                                    (source_ts - self._start_source_ts)
                                    / max(1e-12, self.playback_speed)
                                ) - (time.monotonic() - self._start_real_ts)
                                # if the new candidate is still way behind, resync carefully
                                if late_amount < -self.max_catchup_resync_s:
                                    # resync anchor to this new candidate and start real time now
                                    self._start_source_ts = source_ts
                                    self._start_real_ts = time.monotonic()
                                    if frame_idx is not None:
                                        self._start_source_idx = int(frame_idx)
                                    log.warn(
                                        "PACE-RESYNC",
                                        f"resynced to source_ts={source_ts} idx={frame_idx}",
                                    )
                                    # continue the inner loop to recompute delta vs new anchor
                                    continue
                                # otherwise, continue processing with the 'latest' as item
                                continue
                            else:
                                # queue was empty after drain attempt — nothing to do, break out to emit current item
                                break

                        # autoskip logic: drop_oldest - try to find first item within lag window
                        elif self.autoskip and self.skip_policy == "drop_oldest":
                            popped_ok = False
                            popped = 0
                            while popped < self._drain_limit:
                                try:
                                    nxt = self.capture_q.get_nowait()
                                except queue.Empty:
                                    break
                                popped += 1
                                nxt_ts = self._get_source_ts(nxt)
                                nxt_elapsed_source = (
                                    nxt_ts - self._start_source_ts
                                ) / max(1e-12, self.playback_speed)
                                nxt_delta = nxt_elapsed_source - (
                                    time.monotonic() - self._start_real_ts
                                )
                                if nxt_delta >= -self.max_lag_s:
                                    # found a candidate not too late
                                    item = nxt
                                    source_ts = nxt_ts
                                    frame_idx = nxt.get("frame_index")
                                    popped_ok = True
                                    break
                                else:
                                    self.frames_skipped += 1
                                    # since we used get_nowait() we must *not* call task_done() here; we will call
                                    # task_done for the final emitted item afterward (consistent single place).
                                    continue
                            if popped_ok:
                                # evaluate again with new item
                                continue
                            else:
                                # no candidate found — break to emit current or to fallback
                                break
                        else:
                            # not autoskipping -> just fall through and emit current item (maybe late)
                            break
                    # otherwise delta within allowed window -> emit
                    break

                # Emit chosen item and mark job as done
                self._emit_item(item)
                try:
                    self.capture_q.task_done()
                except Exception:
                    pass

            # final stats log
            log.info(
                "PACE",
                f"stopped; received={self.frames_received} emitted={self.frames_emitted} skipped={self.frames_skipped}",
            )
        except Exception as e:
            log.error("PACE", f"Fatal error in pacing thread: {e}")
            import traceback

            log.debug("PACE", traceback.format_exc())

    def _emit_item(self, item):
        try:
            self.out_q.put_nowait(item)
            self.frames_emitted += 1
            # successful put
            try:
                if hasattr(self, "metrics") and self.metrics is not None:
                    # Use monotonic runtime clock for stage timing consistency.
                    self.metrics.mark(
                        int(item.get("frame_index", -1)),
                        "pacer_emit",
                        ts=time.monotonic(),
                    )
            except Exception:
                pass

        except queue.Full:
            # Queue is full — drop the oldest frame
            try:
                _ = self.out_q.get_nowait()
                self.frames_out_dropped += 1
                # log.debug("PACER", "Output queue full — evicted oldest frame.")
            except Exception:
                pass
            try:
                self.out_q.put_nowait(item)
                self.frames_emitted += 1
                try:
                    if hasattr(self, "metrics") and self.metrics is not None:
                        # Emit mark for reinsertion path as well, so we do not undercount.
                        self.metrics.mark(
                            int(item.get("frame_index", -1)),
                            "pacer_emit",
                            ts=time.monotonic(),
                        )
                except Exception:
                    pass
            except Exception:
                log.warn("PACER", "Failed to reinsert frame after eviction.")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "frames_received": self.frames_received,
            "frames_emitted": self.frames_emitted,
            "frames_skipped": self.frames_skipped,
            "frames_out_dropped": self.frames_out_dropped,
        }

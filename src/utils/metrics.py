# src/utils/metrics.py
from __future__ import annotations
from sys import exc_info
import threading
import time
import collections
from src.utils.logger import log
import csv
from typing import Dict, Any, Optional
import os
import traceback


class MetricsCollector:
    """
    Thread-safe collector. Lightweight in-memory store + periodic CSV flush.
    Usage: metrics.mark(frame_id, "captured", ts, extra=...)
    """

    def __init__(self, csv_path: str = "metrics.csv"):

        self.lock = threading.Lock()
        self._csv_lock = threading.Lock()

        # Store events per frame_id: {frame_id: {event_name: (ts, extra)}}
        self._events: Dict[int, Dict[str, Any]] = {}
        self.counters = collections.Counter()

        # --- 1) Normalize path (handles whitespace, slashes, yaml weirdness) ---
        try:
            clean_path = str(csv_path).strip()
            clean_path = os.path.expanduser(clean_path)
            clean_path = os.path.abspath(clean_path)
            clean_path = os.path.normpath(clean_path)
            self.csv_path = clean_path
        except Exception as e:
            log.error("MetricsCollector",
                        f"❌ Failed to normalize CSV path '{csv_path}': {e}")
            raise RuntimeError(f"Invalid metrics path: {csv_path}")

        # --- 2) Ensure parent directory exists ---
        parent = os.path.dirname(self.csv_path) or "."
        try:
            if not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
                log.info("MetricsCollector",
                        f"📁 Created metrics directory: {parent}")
        except Exception as e:
            tb = traceback.format_exc()
            log.error("MetricsCollector",
                    f"❌ Failed to create metrics directory '{parent}': {e}")
            log.debug("MetricsCollector", tb)
            raise RuntimeError(
                f"Cannot create metrics directory '{parent}': {e}"
            )

        # --- 3) Open CSV safely and write header ---
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_id",
                    "captured",
                    "pacer_emit",
                    "infer_start",
                    "infer_end",
                    "result_queued",
                    "display_shown",
                    "e2e_ms",
                    "infer_ms",
                ])

            log.info("MetricsCollector",
                    f"🟢 Metrics CSV initialized at: {self.csv_path}")

        except Exception as e:
            tb = traceback.format_exc()
            log.error("MetricsCollector",
                    f"❌ Failed to initialize CSV at {self.csv_path}: {e}")
            log.debug("MetricsCollector", tb)
            raise RuntimeError(
                f"Failed to initialize metrics CSV '{self.csv_path}': {e}"
            )


    def mark(
        self,
        frame_id: int,
        event: str,
        ts: Optional[float] = None,
        extra: Optional[dict] = None,
    ):
        ts = float(ts or time.monotonic())
        with self.lock:
            d = self._events.setdefault(int(frame_id), {})
            d[event] = {"ts": ts, "extra": extra}
        # if this is a final event (display_shown), flush row to CSV
        if event == "display_shown":
            self._flush_row(frame_id)

    def incr(self, name: str, n: int = 1):
        with self.lock:
            self.counters[name] += n

    def get_snapshot(self):
        """Return aggregated snapshot (copy) for reporting."""
        with self.lock:
            events_copy = {k: dict(v) for k, v in self._events.items()}
            counters_copy = dict(self.counters)
        return events_copy, counters_copy

    def _flush_row(self, frame_id: int):
        with self.lock:
            row = self._events.pop(int(frame_id), None)
        if not row:
            return
        c_ts = row.get("captured", {}).get("ts")
        p_ts = row.get("pacer_emit", {}).get("ts")
        i0_ts = row.get("infer_start", {}).get("ts")
        i1_ts = row.get("infer_end", {}).get("ts")
        q_ts = row.get("result_queued", {}).get("ts")
        d_ts = row.get("display_shown", {}).get("ts")
        e2e_ms = (d_ts - c_ts) * 1000.0 if c_ts and d_ts else None
        infer_ms = (i1_ts - i0_ts) * 1000.0 if i0_ts and i1_ts else None

        try:
            with self._csv_lock:
                with open(self.csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            frame_id,
                            f"{c_ts:.6f}" if c_ts else "",
                            f"{p_ts:.6f}" if p_ts else "",
                            f"{i0_ts:.6f}" if i0_ts else "",
                            f"{i1_ts:.6f}" if i1_ts else "",
                            f"{q_ts:.6f}" if q_ts else "",
                            f"{d_ts:.6f}" if d_ts else "",
                            f"{e2e_ms:.3f}" if e2e_ms is not None else "",
                            f"{infer_ms:.3f}" if infer_ms is not None else "",
                        ]
                    )
        except Exception:
            # don't break pipeline on metrics failure
            pass

    def report(self):
        """Simple console report: compute p50/p90/p95 on E2E and infer."""
        events, counters = self.get_snapshot()
        e2es = []
        infers = []
        for _, ev in events.items():
            c = ev.get("captured", {}).get("ts")
            d = ev.get("display_shown", {}).get("ts")
            i0 = ev.get("infer_start", {}).get("ts")
            i1 = ev.get("infer_end", {}).get("ts")
            if c and d:
                e2es.append((d - c) * 1000.0)
            if i0 and i1:
                infers.append((i1 - i0) * 1000.0)

        def pct(data, p):
            if not data:
                return None
            data = sorted(data)
            k = (len(data) - 1) * (p / 100.0)
            f = int(k)
            c = min(f + 1, len(data) - 1)
            if f == c:
                return data[int(k)]
            d0 = data[f] * (c - k)
            d1 = data[c] * (k - f)
            return d0 + d1

        out = {
            "counts": counters,
            "e2e_p50": pct(e2es, 50),
            "e2e_p90": pct(e2es, 90),
            "e2e_p95": pct(e2es, 95),
            "infer_p50": pct(infers, 50),
            "infer_p90": pct(infers, 90),
            "infer_p95": pct(infers, 95),
            "samples_e2e": len(e2es),
            "samples_infer": len(infers),
        }
        return out

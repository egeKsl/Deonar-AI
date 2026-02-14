# src/slots/slot_csv_writer.py

from pathlib import Path
import csv
from typing import Optional

from src.slots.contracts import SlotCountEvent
from src.utils.logger import log


class SlotCsvWriter:
    """
    SlotCsvWriter
    -------------
    Responsible for persisting slot-specific counting events
    into a dedicated CSV file (one file per slot).

    Design principles:
    - Append-only
    - Flush-on-write (crash safe)
    - No business logic
    - No aggregation
    - One writer per slot lifecycle
    """

    CSV_COLUMNS = [
        "timestamp_s",
        "slot_id",
        "vendor_id",
        "vendor_name",
        "slot_count",
        "global_count",
        "direction",
        "track_id",
        "proc_frame_idx",
    ]

    def __init__(
        self,
        *,
        slots_dir: Path,
        slot_id: str,
        vendor_id: Optional[str],
        vendor_name: Optional[str],
    ):
        """
        Initialize slot CSV writer.

        Args:
            slots_dir: Path to outputs/runs/<run_id>/slots/
            slot_id: Unique slot/session identifier
            vendor_id: Optional vendor identifier
            vendor_name: Optional vendor name
        """
        self.slot_id = slot_id
        self.vendor_id = vendor_id
        self.vendor_name = vendor_name

        self._fh = None
        self._writer = None
        self._closed = False

        try:
            slots_dir = Path(slots_dir)
            slots_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = slots_dir / f"slot_{slot_id}_events.csv"

            self._fh = open(
                self.csv_path,
                mode="w",
                newline="",
                encoding="utf-8",
            )

            self._writer = csv.writer(self._fh)
            self._writer.writerow(self.CSV_COLUMNS)
            self._fh.flush()

            log.info(
                "SLOT-CSV",
                f"Slot CSV initialized: {self.csv_path}",
            )

        except Exception as e:
            log.error(
                "SLOT-CSV-ERROR",
                f"Failed to initialize slot CSV for slot_id={slot_id}: {e}",
            )
            self._safe_close()
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_event(self, event: SlotCountEvent) -> None:
        """
        Persist a single slot counting event.

        This method is expected to be called ONLY for accepted
        global count events when the slot is ACTIVE.

        Args:
            event: SlotCountEvent (Phase 0 contract)
        """
        if self._closed:
            log.warn(
                "SLOT-CSV",
                f"Ignored write_event on closed writer (slot_id={self.slot_id})",
            )
            return

        if not self._writer:
            log.error(
                "SLOT-CSV-ERROR",
                f"Writer not initialized for slot_id={self.slot_id}",
            )
            return

        try:
            self._writer.writerow(
                [
                    f"{event.timestamp.timestamp():.3f}",
                    event.slot_id,
                    self.vendor_id or "",
                    self.vendor_name or "",
                    event.slot_count,
                    event.global_count,
                    event.direction,
                    event.track_id if event.track_id is not None else "",
                    event.proc_frame_idx if event.proc_frame_idx is not None else "",
                ]
            )

            # CRITICAL: crash safety
            self._fh.flush()

        except Exception as e:
            log.error(
                "SLOT-CSV-ERROR",
                f"Failed to write slot event (slot_id={self.slot_id}): {e}",
            )
            raise

    def close(self) -> None:
        """
        Close the CSV writer safely.

        Safe to call multiple times.
        """
        if self._closed:
            return

        self._safe_close()
        self._closed = True

        log.info(
            "SLOT-CSV",
            f"Slot CSV closed cleanly (slot_id={self.slot_id})",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_close(self) -> None:
        """Internal close helper with exception safety."""
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        except Exception as e:
            log.warn(
                "SLOT-CSV-WARN",
                f"Failed during CSV close (slot_id={self.slot_id}): {e}",
            )
        finally:
            self._fh = None
            self._writer = None

from __future__ import annotations
from pathlib import Path
from typing import Callable

from typing import Optional, Dict
from datetime import datetime, timezone

from src.slots.contracts import (
    SlotStartPayload,
    SlotStopPayload,
    SlotCountEvent,
    SlotStatus,
)

from src.utils.logger import log
from src.slots.slot_csv_writer import SlotCsvWriter
from src.slots.slot_summary_generator import SlotSummaryGenerator


class SlotRuntime:
    """
    In-memory representation of a slot lifecycle.

    This object is MUTABLE only while status == ACTIVE.
    Once COMPLETED / ABORTED, it becomes immutable.
    """

    def __init__(
        self,
        slot_id: str,
        vendor_id: Optional[str],
        vendor_name: Optional[str],
        declared_count: Optional[int],
        start_time: datetime,
        start_global_count: int,
    ):
        self.slot_id = slot_id
        self.vendor_id = vendor_id
        self.vendor_name = vendor_name
        self.declared_count = declared_count

        self.start_time = start_time
        self.end_time: Optional[datetime] = None

        self.start_global_count = start_global_count
        self.end_global_count: Optional[int] = None

        self.slot_count: int = 0
        self.direction_breakdown: Dict[str, int] = {"up": 0, "down": 0}

        self.status: SlotStatus = "ACTIVE"

        # Event history (kept in memory only; persisted later by Phase 2)
        self.events: list[SlotCountEvent] = []

    # -----------------------------
    # Mutation helpers (ACTIVE only)
    # -----------------------------
    def add_event(self, event: SlotCountEvent) -> None:
        if self.status != "ACTIVE":
            raise RuntimeError(
                f"Cannot add event to slot {self.slot_id} (status={self.status})"
            )

        self.slot_count += 1
        self.direction_breakdown[event.direction] += 1
        self.events.append(event)

    def finalize(self, end_time: datetime, end_global_count: int, status: SlotStatus):
        self.end_time = end_time
        self.end_global_count = end_global_count
        self.status = status

    # -----------------------------
    # Read-only snapshot
    # -----------------------------
    def snapshot(self) -> dict:
        """Return a serializable snapshot of the slot runtime state."""
        return {
            "slot_id": self.slot_id,
            "vendor_id": self.vendor_id,
            "vendor_name": self.vendor_name,
            "declared_count": self.declared_count,
            "slot_count": self.slot_count,
            "direction_breakdown": dict(self.direction_breakdown),
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "start_global_count": self.start_global_count,
            "end_global_count": self.end_global_count,
        }


class SlotManager:
    """
    Phase 1 Slot Manager.

    Responsibilities:
    - Maintain exactly one ACTIVE slot at a time
    - Route accepted global count events into the active slot
    - Produce in-memory slot runtime data
    - NO persistence
    - NO API
    - NO UI
    """

    def __init__(
        self,
        *,
        slots_dir: Path,
        run_id: str,
        source: str,
        global_count_supplier: Callable[[], int],
    ):
        self._active_slot: Optional[SlotRuntime] = None
        self._csv_writer: Optional[SlotCsvWriter] = None
        self._slots_dir = slots_dir
        self._run_id = run_id
        self._source = source
        self._get_global_count = global_count_supplier

        # 🧠 lifecycle memory (NEW)
        self._last_started_slot_id: Optional[str] = None
        self._last_stopped_slot_id: Optional[str] = None

    # -----------------------------
    # Slot lifecycle
    # -----------------------------
    def start_slot(self, payload: SlotStartPayload) -> None:
        """
        Start a new slot.

        Idempotency rules:
        - Same slot_id while ACTIVE -> NO-OP
        - Different slot_id while ACTIVE -> ERROR
        - Restarting a previously stopped slot -> ERROR
        """

        # -------------------------------------------------
        # Idempotency: same slot already active
        # -------------------------------------------------
        if self._active_slot is not None:
            if payload.slot_id == self._active_slot.slot_id:
                log.warn(
                    "SLOT",
                    f"Duplicate start ignored for active slot {payload.slot_id}",
                )
                raise RuntimeError("Duplicate start ignored for active slot")
            raise RuntimeError(f"Slot already active: {self._active_slot.slot_id}")

        # -------------------------------------------------
        # Lifecycle rule: cannot restart same slot ID
        # -------------------------------------------------
        if payload.slot_id == self._last_stopped_slot_id:
            log.warn(
                "SLOT",
                f"Attempt to restart stopped slot {payload.slot_id} is not allowed",
            )
            raise RuntimeError(
                f"Slot {payload.slot_id} was already stopped and cannot be restarted"
            )

        log.info(
            "SLOT",
            f"Starting slot {payload.slot_id} "
            f"(vendor={payload.vendor_name}, declared={payload.declared_count})",
        )

        start_global_count = self._get_global_count()

        self._active_slot = SlotRuntime(
            slot_id=payload.slot_id,
            vendor_id=payload.vendor_id,
            vendor_name=payload.vendor_name,
            declared_count=payload.declared_count,
            start_time=payload.timestamp,
            start_global_count=start_global_count,
        )

        self._csv_writer = SlotCsvWriter(
            slots_dir=self._slots_dir,
            slot_id=payload.slot_id,
            vendor_id=payload.vendor_id,
            vendor_name=payload.vendor_name,
        )

        # remember lifecycle
        self._last_started_slot_id = payload.slot_id

    def stop_slot(self, payload: SlotStopPayload) -> SlotRuntime:
        """
        Stop the currently active slot and finalize it.

        Idempotency rules:
        - Stopping same slot twice -> NO-OP
        - Stopping when no slot active -> ERROR
        """

        # -------------------------------------------------
        # Idempotency: already stopped
        # -------------------------------------------------
        if self._active_slot is None:
            if payload.slot_id == self._last_stopped_slot_id:
                log.warn(
                    "SLOT",
                    f"Duplicate stop ignored for slot {payload.slot_id}",
                )
                raise RuntimeError(f"Slot {payload.slot_id} already stopped")
            raise RuntimeError("No active slot to stop")

        # -------------------------------------------------
        # Strict slot match
        # -------------------------------------------------
        if payload.slot_id != self._active_slot.slot_id:
            log.warn(
                "SLOT",
                f"Slot ID mismatch on stop: active={self._active_slot.slot_id}, "
                f"requested={payload.slot_id}",
            )
            raise RuntimeError(
                f"Slot ID mismatch: active={self._active_slot.slot_id}, "
                f"requested={payload.slot_id}"
            )

        slot = self._active_slot

        final_status: SlotStatus = payload.stop_type

        end_global_count = self._get_global_count()

        slot.finalize(
            end_time=payload.timestamp,
            end_global_count=end_global_count,
            status=final_status,
        )

        if self._csv_writer:
            try:
                self._csv_writer.close()
            except Exception as e:
                log.warn("SLOT", f"Failed to close slot CSV: {e}")
            finally:
                self._csv_writer = None

        # SlotRuntime status and SlotSummary status are intentionally different:
        # - runtime uses: ACTIVE | COMPLETED | ABORTED
        # - summary uses: OK | MISMATCH | ABORTED
        # Map explicitly to keep contracts consistent for downstream consumers.
        if slot.status == "ABORTED":
            summary_status = "ABORTED"
        else:
            # For completed slots, compare declared vs counted.
            # If declared_count is missing, treat as OK (no mismatch baseline).
            if slot.declared_count is None or slot.declared_count == slot.slot_count:
                summary_status = "OK"
            else:
                summary_status = "MISMATCH"

        SlotSummaryGenerator.instance().generate(
            slot_id=slot.slot_id,
            vendor_id=slot.vendor_id,
            vendor_name=slot.vendor_name,
            declared_count=slot.declared_count,
            counted_count=slot.slot_count,
            direction_breakdown=slot.direction_breakdown,
            slot_status=summary_status,
            slot_start_time=slot.start_time,
            slot_end_time=slot.end_time,
            start_global_count=slot.start_global_count,
            end_global_count=slot.end_global_count,
            run_id=self._run_id,
            source=self._source,
            output_root=self._slots_dir,
        )

        log.info(
            "SLOT",
            f"Stopped slot {slot.slot_id} "
            f"(counted={slot.slot_count}, status={slot.status})",
        )

        self._last_stopped_slot_id = slot.slot_id
        self._active_slot = None
        return slot

    # -----------------------------
    # Event routing
    # -----------------------------
    def on_global_count_event(
        self,
        *,
        global_count: int,
        direction: str,
        timestamp: datetime,
        track_id: Optional[int] = None,
        proc_frame_idx: Optional[int] = None,
    ) -> Optional[SlotCountEvent]:
        """
        Route an accepted global counting event into the active slot (if any).

        Returns:
            SlotCountEvent if routed, else None
        """
        if self._active_slot is None:
            return None

        event = SlotCountEvent(
            slot_id=self._active_slot.slot_id,
            slot_count=self._active_slot.slot_count + 1,
            global_count=global_count,
            direction=direction,
            timestamp=timestamp,
            track_id=track_id,
            proc_frame_idx=proc_frame_idx,
        )

        self._active_slot.add_event(event)

        log.debug(
            "SLOT",
            f"Slot {event.slot_id}: "
            f"+1 ({event.direction}) "
            f"[slot={event.slot_count}, global={event.global_count}]",
        )

        return event

    def abort_active_slot_if_any(self):
        if not self._active_slot:
            return

        log.warn(
            "SLOT",
            f"Aborting active slot {self._active_slot.slot_id} due to shutdown",
        )

        fake_payload = SlotStopPayload(
            slot_id=self._active_slot.slot_id,
            stop_type="ABORTED",
            stopped_by="system",
            # Keep shutdown/abort path timezone-aware and aligned with API paths.
            timestamp=datetime.now(timezone.utc),
            reason="process_shutdown",
        )

        try:
            self.stop_slot(fake_payload)
        except Exception as e:
            log.error("SLOT", f"Failed to abort slot cleanly: {e}")

    # -----------------------------
    # Introspection helpers
    # -----------------------------
    def is_slot_active(self) -> bool:
        return self._active_slot is not None

    def get_active_slot_snapshot(self) -> Optional[dict]:
        if self._active_slot is None:
            return None
        return self._active_slot.snapshot()

    def get_csv_writer(self) -> Optional[SlotCsvWriter]:
        return self._csv_writer

    def get_public_state(self) -> dict:
        """Public-facing slot API state used by lightweight status endpoints."""
        return {
            "active": self.is_slot_active(),
            "slot": self.get_active_slot_snapshot(),
        }

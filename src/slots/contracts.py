# src/slots/contracts.py

from dataclasses import dataclass
from typing import Optional, Literal, Dict
from datetime import datetime


# ============================================================
# Slot lifecycle state (runtime only)
# ============================================================

SlotStatus = Literal["ACTIVE", "COMPLETED", "ABORTED"]


# ============================================================
# API payload contracts (START / STOP)
# ============================================================


@dataclass(frozen=True)
class SlotStartPayload:
    """
    Payload received when admin/vendor presses START.
    """

    slot_id: str
    vendor_id: Optional[str]
    vendor_name: Optional[str]
    declared_count: Optional[int]
    started_by: str
    timestamp: datetime


@dataclass(frozen=True)
class SlotStopPayload:
    """
    Payload received when admin/vendor presses STOP.
    """

    slot_id: str
    stopped_by: str
    stop_type: Literal["COMPLETED", "ABORTED"]
    reason: Optional[str]
    timestamp: datetime


# ============================================================
# Slot runtime count event (one accepted crossing)
# ============================================================


@dataclass(frozen=True)
class SlotCountEvent:
    """
    Internal runtime representation of one accepted crossing
    that belongs to an active slot.
    """

    slot_id: str
    slot_count: int  # 1, 2, 3... (relative to slot)
    global_count: int  # absolute global count at this moment
    direction: Literal["up", "down"]
    timestamp: datetime # event timestamp (can be used for ordering, latency analysis, etc.)
    track_id: Optional[int]
    proc_frame_idx: Optional[int]


# ============================================================
# Slot Events CSV schema (LOCKED)
# ============================================================

# This CSV is APPEND-ONLY and contains ONLY per-event rows.
# No summary rows. No mixed semantics.

SLOT_EVENTS_CSV_COLUMNS = [
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


# ============================================================
# Slot Summary JSON contract (LOCKED)
# ============================================================


@dataclass(frozen=True)
class SlotSummary:
    """
    Authoritative summary written ONCE at slot completion.
    Serialized as JSON.
    """

    slot_id: str

    # vendor info
    vendor_id: Optional[str]
    vendor_name: Optional[str]

    # declared vs counted
    declared_count: Optional[int]
    counted_count: int
    difference: Optional[int]

    # final status
    slot_status: Literal["OK", "MISMATCH", "ABORTED"]

    # slot timing
    slot_start_time: datetime
    slot_end_time: datetime
    start_global_count: int
    end_global_count: int

    # direction stats
    direction_breakdown: Dict[str, int]  # {"up": X, "down": Y}

    # runtime context
    run_id: str
    source: str

    # audit
    generated_at: datetime

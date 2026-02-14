"""
Slot Summary Generator (Singleton)

Responsibility:
- Build an authoritative SlotSummary object from SlotManager state
- Serialize and write the summary as a single immutable JSON file
- Execute exactly once per slot completion

Design principles:
- No dependency on CSVs (CSV is audit trail only)
- No mutation of slot state
- Crash-safe (vision engine must not die if summary write fails)
- Human-readable + machine-readable JSON

This module intentionally combines:
- SlotSummaryBuilder (pure logic)
- SlotSummaryWriter  (I/O only)

Exposed as a SINGLETON for simplicity and consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from src.utils.logger import log
from src.slots.contracts import SlotSummary


# -------------------------------------------------------------------
# Helper: JSON encoder for datetime
# -------------------------------------------------------------------
class _DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that serializes datetime objects as ISO-8601 strings."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# -------------------------------------------------------------------
# SlotSummaryBuilder
# -------------------------------------------------------------------
class SlotSummaryBuilder:
    """
    Pure builder that constructs SlotSummary from provided slot state.

    This class:
    - contains NO file I/O
    - contains NO side effects
    - does NOT modify slot state
    """

    def build(
        self,
        *,
        slot_id: str,
        vendor_id: Optional[str],
        vendor_name: Optional[str],
        declared_count: Optional[int],
        counted_count: int,
        direction_breakdown: Dict[str, int],
        slot_status: str,
        slot_start_time: datetime,
        slot_end_time: datetime,
        start_global_count: int,
        end_global_count: int,
        run_id: str,
        source: str,
    ) -> SlotSummary:
        """
        Build SlotSummary object.

        Raises:
            ValueError: if required invariants are violated
        """

        if slot_end_time < slot_start_time:
            raise ValueError("slot_end_time cannot be earlier than slot_start_time")

        difference = (
            counted_count - declared_count if declared_count is not None else None
        )

        return SlotSummary(
            slot_id=slot_id,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            declared_count=declared_count,
            counted_count=counted_count,
            difference=difference,
            slot_status=slot_status,
            slot_start_time=slot_start_time,
            slot_end_time=slot_end_time,
            start_global_count=start_global_count,
            end_global_count=end_global_count,
            direction_breakdown=dict(direction_breakdown),
            run_id=run_id,
            source=source,
            generated_at=datetime.utcnow(),
        )


# -------------------------------------------------------------------
# SlotSummaryWriter
# -------------------------------------------------------------------
class SlotSummaryWriter:
    """
    Handles persistence of SlotSummary to disk as JSON.

    Safety guarantees:
    - Parent directories auto-created
    - fsync-style flush (best effort)
    - Failure does NOT crash vision engine
    """

    def write(self, summary: SlotSummary, out_dir: Path) -> Path:
        """
        Write summary JSON to disk.

        Returns:
            Path to written JSON file
        """

        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{summary.slot_id}_summary.json"

        try:
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    summary.__dict__,
                    fh,
                    cls=_DateTimeEncoder,
                    indent=2,
                    ensure_ascii=False,
                )
                fh.flush()

            log.success(
                "SLOT-SUMMARY",
                f"Slot summary written successfully → {out_path}",
            )

        except Exception as e:
            log.error(
                "SLOT-SUMMARY",
                f"Failed to write slot summary for slot_id={summary.slot_id}",
            )
            log.debug("SLOT-SUMMARY", f"Exception: {e}")
            # Do NOT raise — vision must survive
            return out_path

        return out_path


# -------------------------------------------------------------------
# Singleton Facade
# -------------------------------------------------------------------
class SlotSummaryGenerator:
    """
    Singleton facade combining builder + writer.

    Public API:
        SlotSummaryGenerator.instance().generate(...)
    """

    _instance: Optional["SlotSummaryGenerator"] = None

    def __init__(self):
        self._builder = SlotSummaryBuilder()
        self._writer = SlotSummaryWriter()

    @classmethod
    def instance(cls) -> "SlotSummaryGenerator":
        if cls._instance is None:
            cls._instance = cls()
            log.debug("SLOT-SUMMARY", "SlotSummaryGenerator singleton created")
        return cls._instance

    def generate(
        self,
        *,
        slot_id: str,
        vendor_id: Optional[str],
        vendor_name: Optional[str],
        declared_count: Optional[int],
        counted_count: int,
        direction_breakdown: Dict[str, int],
        slot_status: str,
        slot_start_time: datetime,
        slot_end_time: datetime,
        start_global_count: int,
        end_global_count: int,
        run_id: str,
        source: str,
        output_root: Path,
    ) -> Optional[Path]:
        """
        High-level API used by SlotManager on slot completion.

        This method:
        - builds SlotSummary
        - writes JSON to disk
        - returns written path (or None on failure)
        """

        try:
            summary = self._builder.build(
                slot_id=slot_id,
                vendor_id=vendor_id,
                vendor_name=vendor_name,
                declared_count=declared_count,
                counted_count=counted_count,
                direction_breakdown=direction_breakdown,
                slot_status=slot_status,
                slot_start_time=slot_start_time,
                slot_end_time=slot_end_time,
                start_global_count=start_global_count,
                end_global_count=end_global_count,
                run_id=run_id,
                source=source,
            )
        except Exception as e:
            log.error(
                "SLOT-SUMMARY",
                f"Failed to build SlotSummary for slot_id={slot_id}",
            )
            log.debug("SLOT-SUMMARY", f"Exception: {e}")
            return None

        out_dir = Path(output_root)
        # Guard against accidentally passing a file path instead of a directory.
        if out_dir.suffix.lower() == ".json":
            log.warn(
                "SLOT-SUMMARY",
                f"Expected directory for summary output, got file path '{out_dir}'. "
                "Using its parent directory instead.",
            )
            out_dir = out_dir.parent

        return self._writer.write(summary, out_dir)

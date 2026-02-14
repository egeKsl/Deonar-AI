# src/slots/api/routes.py

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from src.slots.api.schemas import (
    SlotStartRequest,
    SlotStopRequest,
    ApiResponse,
)
from src.slots.contracts import SlotStartPayload, SlotStopPayload
from src.utils.logger import log


def create_slot_router(slot_manager):
    """
    Factory that binds SlotManager into the router.
    """
    router = APIRouter(prefix="/api/slot", tags=["slots"])

    @router.post("/start", response_model=ApiResponse)
    def start_slot(req: SlotStartRequest):
        try:
            # Use server-generated UTC timestamp to keep slot lifecycle timing
            # consistent and avoid client timezone/clock drift issues.
            server_ts = datetime.now(timezone.utc)
            payload = SlotStartPayload(
                slot_id=req.slot_id,
                vendor_id=req.vendor_id,
                vendor_name=req.vendor_name,
                declared_count=req.declared_count,
                started_by=req.started_by,
                timestamp=server_ts,
            )
            slot_manager.start_slot(payload)

            log.info("SLOT-API", f"Slot started: {req.slot_id}")
            return ApiResponse(status="ok", message="Slot started")

        except Exception as e:
            log.error("SLOT-API", f"Start failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/stop", response_model=ApiResponse)
    def stop_slot(req: SlotStopRequest):
        try:
            # Use server-generated UTC timestamp to guarantee same timezone semantics
            # as start timestamps and internal event timestamps.
            server_ts = datetime.now(timezone.utc)
            payload = SlotStopPayload(
                slot_id=req.slot_id,
                stopped_by=req.stopped_by,
                stop_type="COMPLETED",
                reason=req.reason,
                timestamp=server_ts,
            )
            slot_manager.stop_slot(payload)

            log.info("SLOT-API", f"Slot stopped: {req.slot_id}")
            return ApiResponse(status="ok", message="Slot stopped")

        except Exception as e:
            log.error("SLOT-API", f"Stop failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/active", response_model=dict)
    def get_active_slot():
        """
        Lightweight status endpoint for UI.
        """
        return slot_manager.get_public_state()

    return router

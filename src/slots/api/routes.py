from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status

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

    Responsibilities of this layer:
    - Translate SlotManager lifecycle errors into proper HTTP responses
    - Enforce API-level semantics (idempotency visibility)
    - NEVER mutate slot state directly
    """

    router = APIRouter(prefix="/api/slot", tags=["slots"])

    # ---------------------------------------------------------
    # START SLOT
    # ---------------------------------------------------------
    @router.post(
        "/start",
        response_model=ApiResponse,
        status_code=status.HTTP_200_OK,
        summary="Start a slot session",
    )
    def start_slot(req: SlotStartRequest):
        """
        Start a new slot session.

        Behavior:
        - Idempotent if same slot_id is already ACTIVE
        - Rejects starting when another slot is ACTIVE
        - Rejects restarting an already stopped slot
        """

        server_ts = datetime.now(timezone.utc)

        payload = SlotStartPayload(
            slot_id=req.slot_id,
            vendor_id=req.vendor_id,
            vendor_name=req.vendor_name,
            declared_count=req.declared_count,
            started_by=req.started_by,
            timestamp=server_ts,
        )

        try:
            slot_manager.start_slot(payload)

            log.info("SLOT-API", f"Slot started: {req.slot_id}")

            return ApiResponse(
                status="ok",
                message="Slot started",
                slot_id=req.slot_id,
            )

        except RuntimeError as e:
            msg = str(e)

            # ---------------------------------------------
            # Idempotent duplicate start (already active)
            # ---------------------------------------------
            if "Duplicate start ignored" in msg or "already active" in msg:
                log.warn("SLOT-API", msg)
                return ApiResponse(
                    status="ok",
                    message="Slot already active",
                    slot_id=req.slot_id,
                )

            # ---------------------------------------------
            # Lifecycle conflict
            # ---------------------------------------------
            if "cannot be restarted" in msg or "Slot already active" in msg:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=msg,
                )

            # Fallback
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=msg,
            )

        except Exception as e:
            log.error("SLOT-API", f"Unhandled start error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error while starting slot",
            )

    # ---------------------------------------------------------
    # STOP SLOT
    # ---------------------------------------------------------
    @router.post(
        "/stop",
        response_model=ApiResponse,
        status_code=status.HTTP_200_OK,
        summary="Stop an active slot session",
    )
    def stop_slot(req: SlotStopRequest):
        """
        Stop the currently active slot session.

        Behavior:
        - Rejects stopping if no slot is active
        - Rejects mismatched slot IDs
        - Defensive against duplicate stop calls
        """

        server_ts = datetime.now(timezone.utc)

        payload = SlotStopPayload(
            slot_id=req.slot_id,
            stopped_by=req.stopped_by,
            stop_type="COMPLETED",
            reason=req.reason,
            timestamp=server_ts,
        )

        try:
            slot_manager.stop_slot(payload)

            log.info("SLOT-API", f"Slot stopped: {req.slot_id}")

            return ApiResponse(
                status="ok",
                message="Slot stopped",
                slot_id=req.slot_id,
            )

        except RuntimeError as e:
            msg = str(e)

            # ---------------------------------------------
            # Duplicate stop or lifecycle conflict
            # ---------------------------------------------
            if (
                "already stopped" in msg
                or "No active slot" in msg
                or "Slot ID mismatch" in msg
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=msg,
                )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=msg,
            )

        except Exception as e:
            log.error("SLOT-API", f"Unhandled stop error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error while stopping slot",
            )

    # ---------------------------------------------------------
    # ACTIVE SLOT STATUS
    # ---------------------------------------------------------
    @router.get(
        "/active",
        response_model=dict,
        status_code=status.HTTP_200_OK,
        summary="Get current slot state",
    )
    def get_active_slot():
        """
        Lightweight status endpoint for UI / WebRTC overlays.

        Returns:
        {
            "active": bool,
            "slot": { ... } | null
        }
        """
        return slot_manager.get_public_state()

    return router

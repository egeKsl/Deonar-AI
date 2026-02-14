# src/slots/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


# ----------------------------
# Slot START request
# ----------------------------
class SlotStartRequest(BaseModel):
    """
    Request payload for starting a slot session.

    Used by `POST /api/slot/start`.
    """

    slot_id: str = Field(..., description="Unique slot/session id")
    vendor_id: Optional[str] = None
    vendor_name: Optional[str] = None
    declared_count: Optional[int] = Field(None, ge=0)
    started_by: str


# ----------------------------
# Slot STOP request
# ----------------------------
class SlotStopRequest(BaseModel):
    """
    Request payload for stopping an active slot session.

    Used by `POST /api/slot/stop`.
    """

    slot_id: str
    stopped_by: str
    reason: Optional[str] = None


# ----------------------------
# Generic API response
# ----------------------------
class ApiResponse(BaseModel):
    """
    Standard API response envelope used by slot endpoints.

    `status` is typically `ok` or `error`; `message` carries user-facing context.
    """

    status: str
    message: str

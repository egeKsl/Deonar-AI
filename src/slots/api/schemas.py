"""
Slot API Schemas
================

This module defines **strict, defensive Pydantic models** for the Slot API.

Goals:
- Reject malformed / garbage payloads early (before SlotManager)
- Enforce clear invariants on identifiers and numeric fields
- Forbid unexpected keys (fail fast, no silent ignores)
- Keep contracts stable (NO behavior change)

NOTE:
- Slot IDs are still CLIENT-PROVIDED (server generation is a future phase)
- All timestamps are server-generated (not accepted from clients)
"""

from typing import Optional
from typing_extensions import TypeAlias
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Shared semantic aliases (NO validation here – typing only)
# ---------------------------------------------------------------------

SlotId: TypeAlias = str
ActorId: TypeAlias = str
VendorId: TypeAlias = str
VendorName: TypeAlias = str


# ---------------------------------------------------------------------
# Slot START request
# ---------------------------------------------------------------------
class SlotStartRequest(BaseModel):
    """
    Request payload for starting a slot session.

    Used by:
        POST /api/slot/start

    Validation guarantees:
    - slot_id is safe, bounded, and non-empty
    - declared_count (if present) is non-negative
    - no unexpected keys are accepted
    """

    slot_id: SlotId = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Unique slot/session identifier (client-provided)",
        example="SLOT_2025_08_14_01",
    )

    vendor_id: Optional[VendorId] = Field(
        None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Optional vendor identifier",
        example="VND-142",
    )

    vendor_name: Optional[VendorName] = Field(
        None,
        min_length=1,
        max_length=128,
        description="Optional vendor display name",
        example="Rahim Goat Supplier",
    )

    declared_count: Optional[int] = Field(
        None,
        ge=0,
        description="Declared expected count for the slot (>= 0)",
        example=94,
    )

    started_by: ActorId = Field(
        ...,
        min_length=2,
        max_length=64,
        pattern=r"^[A-Za-z0-9 _.-]+$",
        description="Identifier of the actor who started the slot",
        example="operator_1",
    )

    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True


# ---------------------------------------------------------------------
# Slot STOP request
# ---------------------------------------------------------------------
class SlotStopRequest(BaseModel):
    """
    Request payload for stopping an active slot session.

    Used by:
        POST /api/slot/stop

    Validation guarantees:
    - slot_id must match the active slot exactly
    - stopped_by is required and bounded
    - reason (if provided) is bounded and clean
    """

    slot_id: SlotId = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Slot/session identifier to stop",
        example="SLOT_2025_08_14_01",
    )

    stopped_by: ActorId = Field(
        ...,
        min_length=2,
        max_length=64,
        pattern=r"^[A-Za-z0-9 _.-]+$",
        description="Identifier of the actor who stopped the slot",
        example="operator_1",
    )

    reason: Optional[str] = Field(
        None,
        min_length=1,
        max_length=256,
        description="Optional reason for stopping / aborting the slot",
        example="Camera feed interrupted",
    )

    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True


# ---------------------------------------------------------------------
# Generic API response
# ---------------------------------------------------------------------
class ApiResponse(BaseModel):
    """
    Standard API response envelope.

    Notes:
    - `slot_id` is OPTIONAL and included only when relevant
    - `status` is intentionally a string for future extensibility
    """

    status: str = Field(
        ...,
        pattern=r"^(ok|error)$",
        description="Response status",
        example="ok",
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable message",
        example="Slot started successfully",
    )

    slot_id: Optional[SlotId] = Field(
        None,
        description="Slot identifier (included when applicable)",
        example="SLOT_2025_08_14_01",
    )

    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True

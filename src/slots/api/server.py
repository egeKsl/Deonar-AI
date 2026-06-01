# src/slots/api/server.py

from __future__ import annotations

import threading
import socket
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.slots.api.routes import create_slot_router
from src.utils.logger import log


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _is_port_free(host: str, port: int) -> bool:
    """Check whether a TCP port is free to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) != 0


# ------------------------------------------------------------------
# Slot API Runtime
# ------------------------------------------------------------------


class SlotApiRuntime:
    """
    Runtime handle for Slot Control API.

    This object is owned by the runner and is:
    - monitorable (is_alive)
    - stoppable (best-effort)
    - safe to ignore if disabled
    """

    def __init__(self, *, app: FastAPI, host: str, port: int):
        self.app = app
        self.host = host
        self.port = port

        self._thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._stopped = False

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------

    def start(self) -> None:
        """Start the FastAPI server in a daemon thread."""

        if self._thread is not None:
            log.warn("SLOT-API", "start() called but API already running")
            return

        def _run():
            log.info("SLOT-API", f"Starting API at http://{self.host}:{self.port}")
            try:
                uvicorn.run(
                    self.app,
                    host=self.host,
                    port=self.port,
                    log_level="warning",
                    access_log=False,
                )
            except Exception as e:
                log.error("SLOT-API", f"API server crashed: {e}")

        self._thread = threading.Thread(
            target=_run,
            name="SlotAPIThread",
            daemon=True,
        )
        self._thread.start()
        self._started_at = time.time()

    def is_alive(self) -> bool:
        """Return True if the API thread is alive."""
        if self._thread is None:
            return False
        return self._thread.is_alive()

    def stop(self) -> None:
        """
        Best-effort stop.

        NOTE:
        - uvicorn does not expose a clean programmatic shutdown
        - daemon thread will exit with process
        - this method exists for symmetry + future hardening
        """
        if self._stopped:
            return

        self._stopped = True

        if self._thread and self._thread.is_alive():
            log.info(
                "SLOT-API",
                "Stopping Slot API (best-effort; daemon thread will exit with process)",
            )
        else:
            log.debug("SLOT-API", "Slot API already stopped or never started")

    # --------------------------------------------------------------
    # Introspection
    # --------------------------------------------------------------

    def info(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "alive": self.is_alive(),
            "started_at": self._started_at,
        }


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def start_slot_api_if_enabled(config: dict, slot_manager) -> Optional[SlotApiRuntime]:
    """
    Create and start Slot API runtime IF slots config is present.

    Returns:
        SlotApiRuntime or None (if disabled or failed)
    """

    runtime_cfg = config.get("runtime", {}) if isinstance(config, dict) else {}
    runtime_slot_api = (
        runtime_cfg.get("slot_api", {}) if isinstance(runtime_cfg, dict) else {}
    )
    legacy_slots_cfg = config.get("slots", {}) if isinstance(config, dict) else {}

    slot_api_cfg = (
        runtime_slot_api
        if isinstance(runtime_slot_api, dict) and runtime_slot_api
        else legacy_slots_cfg
    )
    if not isinstance(slot_api_cfg, dict) or not slot_api_cfg:
        log.info("SLOT-API", "Slots API disabled (no slot_api config)")
        return None

    host = slot_api_cfg.get("host", "127.0.0.1")
    port = int(slot_api_cfg.get("port", 9091))

    if not _is_port_free(host, port):
        log.error("SLOT-API", f"Port {port} already in use")
        return None

    app = FastAPI(
        title="Goat Slot Control API",
        version="1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # Allow browser dashboard pages on the WebRTC port (8081) to call
    # the slot API on port 8090 without CORS errors.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "slot-api"}

    @app.get("/")
    def index():
        """Human-friendly landing endpoint for quick browser access."""
        return {
            "service": "slot-api",
            "status": "ok",
            "health": "/health",
            "docs": "/docs",
        }

    app.include_router(create_slot_router(slot_manager))

    runtime = SlotApiRuntime(app=app, host=host, port=port)

    try:
        runtime.start()
    except Exception as e:
        log.error("SLOT-API", f"Failed to start Slot API: {e}")
        return None

    return runtime

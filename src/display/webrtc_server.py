# src/display/webrtc_server.py
from __future__ import annotations

import asyncio
import threading
import traceback
import time
from typing import Dict, Optional, Set

import numpy as np
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

import queue as std_queue
from src.utils.logger import log


class _FrameQueueTrack(VideoStreamTrack):
    """
    WebRTC VideoTrack that pulls frames from an asyncio.Queue of numpy BGR frames.
    One queue per client; server pushes frames into all queues.
    """

    kind = "video"

    def __init__(self, frame_queue: asyncio.Queue, target_fps: float = 25.0):
        super().__init__()
        self._queue = frame_queue
        self._target_fps = float(target_fps)

    async def recv(self) -> VideoFrame:
        # Let base class manage timestamps
        pts, time_base = await self.next_timestamp()

        try:
            # Wait for a frame with a timeout; if no frame, we send a black one
            frame_bgr = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Fallback: black frame
            h, w = 360, 640
            frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert BGR (OpenCV) -> RGB
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            # If conversion fails, send black frame
            h, w = 360, 640
            frame_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        vframe = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        vframe.pts = pts
        vframe.time_base = time_base
        return vframe


class WebRTCServer:
    """
    WebRTC video server running in its own asyncio loop + thread.

    Features:
      - Serves a small HTML client at `/` (browser WebRTC viewer).
      - POST `/offer` for WebRTC SDP exchange (aiortc).
      - Optional POST `/control` endpoint to forward JSON commands to a std Queue.
      - Thread-safe `publish(frame)` API (numpy BGR) for DisplayWorker.
      - Clean `close()` for shutdown.

    Usage (from runner):
        from src.display.webrtc_server import WebRTCServer
        rtc = WebRTCServer(host='0.0.0.0', port=8082, target_fps=15, max_clients=2)
        ctrl_q = queue.Queue(maxsize=32)
        rtc.set_control_queue(ctrl_q)
        injected["webrtc_server"] = rtc
        injected["webrtc_control_q"] = ctrl_q

    From DisplayWorker:
        rtc = self.injected.get("webrtc_server")
        if rtc is not None:
            rtc.publish(full_disp)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8082,
        target_fps: float = 15.0,
        max_clients: Optional[int] = None,
        downscale_width: int = 960,
        downscale_height: int = 540,
    ):
        self.host = str(host)
        self.port = int(port)
        self.target_fps = float(target_fps)
        self.max_clients = None if max_clients is None else int(max_clients)
        self.downscale_width = int(downscale_width)
        self.downscale_height = int(downscale_height)

        # Async runtime objects
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="WebRTCServerLoop", daemon=True
        )
        self._closing = threading.Event()
        self._server_ready = threading.Event()

        # WebRTC state
        self._pcs: Set[RTCPeerConnection] = set()
        self._frame_queues: Dict[int, asyncio.Queue] = {}  # id(pc) -> queue

        # Control queue (to DisplayWorker / Runner)
        self._control_queue: Optional[std_queue.Queue] = None
        self._slot_manager = None
        # Cache slot-state snapshots to avoid hitting SlotManager on every browser poll.
        self._slot_state_cache = {"ts": 0.0, "data": {"active": False, "slot": None}}
        self._slot_state_cache_ttl_s = 2.0

        log.debug(
            "WEBRTC",
            f"Starting WebRTCServer thread on {self.host}:{self.port} (fps={self.target_fps})",
        )
        self._thread.start()

        if not self._server_ready.wait(timeout=5.0):
            log.warn(
                "WEBRTC",
                "WebRTCServer thread did not signal ready within 5s (server may still be starting).",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_control_queue(self, q: std_queue.Queue) -> None:
        """Attach a stdlib queue where /control JSON will be forwarded (non-blocking)."""
        self._control_queue = q
        log.debug("WEBRTC", "Control queue set")

    def publish(self, frame: np.ndarray) -> None:
        """
        Thread-safe: push latest frame (numpy BGR, HxWx3 uint8) to ALL connected clients.

        - Copies the frame once.
        - Inside asyncio loop, it will fan that frame out to per-client queues
          with drop-oldest policy to avoid unbounded growth.
        """
        if frame is None:
            return
        try:
            frame_copy = frame.copy()
        except Exception:
            log.debug("WEBRTC", "publish(): frame.copy() failed; skipping frame")
            return

        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._broadcast_frame_to_queues(frame_copy), self._loop
            )
        except Exception:
            log.debug(
                "WEBRTC", "publish() scheduling failed: " + traceback.format_exc()
            )

    def close(self, timeout: float = 3.0) -> None:
        """Synchronous shutdown: close peer connections and stop web server."""
        if self._closing.is_set():
            return
        self._closing.set()
        log.info("WEBRTC", "Closing WebRTCServer...")

        try:
            fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            try:
                fut.result(timeout=timeout)
            except Exception:
                log.debug(
                    "WEBRTC",
                    "Shutdown future result failed/timeout: " + traceback.format_exc(),
                )
        except Exception:
            log.error(
                "WEBRTC",
                "Failed to schedule shutdown: " + traceback.format_exc(),
            )

        # Join thread
        self._thread.join(timeout=timeout)
        log.info("WEBRTC", "WebRTCServer stopped")

    # ------------------------------------------------------------------
    # Internal: loop & app
    # ------------------------------------------------------------------
    def _run_loop(self):
        try:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._init_app())
            self._loop.run_forever()
        except Exception:
            log.error("WEBRTC", "WebRTCServer loop crashed: " + traceback.format_exc())
        finally:
            try:
                self._loop.run_until_complete(self._cleanup_loop())
            except Exception:
                log.debug("WEBRTC", "Cleanup exception: " + traceback.format_exc())
            finally:
                self._loop.close()
                log.debug("WEBRTC", "Async event loop closed")

    async def _init_app(self):
        app = web.Application()
        app["server"] = self  # allow handlers to access server
        app["slot_manager"] = self._slot_manager

        app.router.add_get("/", self._index_handler)
        app.router.add_post("/offer", self._offer_handler)
        app.router.add_post("/control", self._control_handler)
        app.router.add_get("/health", self._health_handler)
        app.router.add_get("/slot-state", self._slot_state_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=self.host, port=self.port)
        await site.start()

        log.info("WEBRTC", f"WebRTC HTTP server started on {self.host}:{self.port}")
        self._server_ready.set()

    async def _cleanup_loop(self):
        # Close all peer connections
        pcs = list(self._pcs)
        for pc in pcs:
            try:
                await pc.close()
            except Exception:
                pass
        self._pcs.clear()
        self._frame_queues.clear()
        log.debug("WEBRTC", "Cleanup done (pcs + frame queues cleared)")

    # ------------------------------------------------------------------
    # HTTP / WebRTC handlers
    # ------------------------------------------------------------------
    async def _index_handler(self, request: web.Request):
        """
        Minimal HTML client. Opens WebRTC connection and shows the stream.
        Uses POST /offer for signaling and POST /control for commands.
        """
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Goat Stream (WebRTC)</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #111;
      color: #eee;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }}
    .container {{
      background: #1b1b1b;
      border-radius: 10px;
      padding: 16px 20px 20px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.7);
      max-width: 960px;
      width: 100%;
    }}
    h3 {{
      margin: 0 0 10px;
      font-weight: 600;
      color: #f5f5f5;
    }}
    #video {{
      width: 100%;
      max-height: 540px;
      background: #000;
      border-radius: 6px;
      border: 1px solid #333;
    }}
    .controls {{
      margin-top: 10px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button {{
      padding: 6px 14px;
      border-radius: 4px;
      border: none;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      background: #2d6cdf;
      color: #fff;
      transition: background 0.2s, transform 0.1s;
    }}
    button:hover {{
      background: #3c7dff;
    }}
    button:active {{
      transform: scale(0.97);
    }}
    button.secondary {{
      background: #444;
    }}
    #status {{
      margin-left: auto;
      font-size: 13px;
      color: #0fdf7b;
      white-space: nowrap;
    }}
    #status.error {{
      color: #ff5c5c;
    }}
    /* Slot status colors */
    .slot-status-active {{
    color: #1ddf8b; /* green */
    font-weight: 600;
    }}

    .slot-status-inactive {{
    color: #ff5c5c; /* red */
    font-weight: 600;
    }}

    .slot-card {{
    margin-top: 12px;
    padding: 12px 16px;
    background: linear-gradient(135deg, #1f1f1f 0%, #232323 100%);
    border-left: 4px solid #1ddf8b;
    border-radius: 6px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.6);
    font-size: 14px;
    transition: opacity 0.2s ease, filter 0.2s ease, border-color 0.2s ease;
    }}

    .slot-card.hidden {{
    display: none;
    }}

    .slot-card.inactive {{
    border-left-color: #8a8a8a;
    opacity: 0.72;
    filter: saturate(0.8);
    }}

    .slot-row {{
    margin: 4px 0;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    }}

    .slot-title {{
    font-size: 15px;
    font-weight: 600;
    color: #1ddf8b;
    }}

    .slot-counts span,
    .slot-meta span {{
    font-family: monospace;
    font-weight: 600;
    }}

    .slot-divider {{
    height: 1px;
    background: #333;
    margin: 6px 0;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h3>Goat Stream (WebRTC)</h3>
    <video id="video" autoplay playsinline controls muted></video>
    <!-- Slot Info Card -->
    <div id="slot-card" class="slot-card hidden">
    <div class="slot-row slot-title">
        SLOT: <span id="slot-id">-</span>
    </div>

    <div class="slot-row">
        VENDOR ID: <span id="slot-vendor-id">-</span>
        VENDOR: <span id="slot-vendor-name">-</span>
    </div>

    <div class="slot-row">
        START: <span id="slot-start">-</span>
        END: <span id="slot-end">-</span>
    </div>

    <div class="slot-divider"></div>

    <div class="slot-row slot-counts">
        UP: <span id="slot-up">0</span>
        DOWN: <span id="slot-down">0</span>
        TOTAL: <span id="slot-total">0</span>
        DECLARED: <span id="slot-declared">-</span>
    </div>

    <div class="slot-divider"></div>

    <div class="slot-row slot-meta">
        STATUS: <span id="slot-status">-</span>
        START_GC: <span id="slot-start-gc">-</span>
        END_GC: <span id="slot-end-gc">-</span>
    </div>
    </div>
    <div class="controls">
      <button id="btn_start">Reconnect</button>
      <button id="btn_screenshot" class="secondary">Screenshot</button>
      <button id="btn_quit" class="secondary">Quit</button>
      <span id="status">idle</span>
    </div>
  </div>

  <script>
    const videoElem   = document.getElementById('video');
    const statusElem  = document.getElementById('status');
    const btnStart    = document.getElementById('btn_start');
    const btnShot     = document.getElementById('btn_screenshot');
    const btnQuit     = document.getElementById('btn_quit');

    let pc = null;

    function setStatus(text, isError = false) {{
      statusElem.textContent = text;
      if (isError) {{
        statusElem.classList.add('error');
      }} else {{
        statusElem.classList.remove('error');
      }}
    }}

    async function sendControl(cmd) {{
      try {{
        const resp = await fetch('/control', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ cmd }})
        }});
        if (!resp.ok) {{
          console.warn('control failed', resp.status);
        }}
      }} catch (e) {{
        console.error('control error', e);
      }}
    }}

    btnShot.onclick = () => {{
      sendControl('screenshot');
      setStatus('screenshot requested');
    }};

    btnQuit.onclick = () => {{
      sendControl('quit');
      setStatus('quit requested');
    }};

    btnStart.onclick = () => {{
      start().catch(err => {{
        console.error(err);
        setStatus('error: ' + err, true);
      }});
    }};

    async function start() {{
      // Clean up old connection if any
      if (pc) {{
        try {{ pc.close(); }} catch(e) {{}}
        pc = null;
      }}
      setStatus('connecting...');

      // Optional STUN server; can be omitted if everything is LAN / same host
      pc = new RTCPeerConnection({{
        iceServers: [
          {{ urls: 'stun:stun.l.google.com:19302' }}
        ]
      }});

      pc.ontrack = (event) => {{
        console.log('ontrack', event.track.kind);
        if (event.track.kind === 'video') {{
          const [stream] = event.streams;
          if (videoElem.srcObject !== stream) {{
            videoElem.srcObject = stream;
          }}
        }}
      }};

      pc.oniceconnectionstatechange = () => {{
        console.log('ice state:', pc.iceConnectionState);
        if (pc.iceConnectionState === 'connected') {{
          setStatus('connected');
        }} else if (pc.iceConnectionState === 'disconnected' ||
                   pc.iceConnectionState === 'failed') {{
          setStatus('connection lost', true);
        }}
      }};

      pc.onconnectionstatechange = () => {{
        console.log('conn state:', pc.connectionState);
        if (pc.connectionState === 'connected') {{
          setStatus('connected');
        }} else if (pc.connectionState === 'failed' ||
                   pc.connectionState === 'disconnected') {{
          setStatus('connection lost', true);
        }} else if (pc.connectionState === 'closed') {{
          setStatus('closed');
        }}
      }};

      // *** This is the important line ***
      // Request a RECV-ONLY video transceiver so the offer has a video m-line.
      pc.addTransceiver('video', {{ direction: 'recvonly' }});

      // Create and send offer to the server
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const resp = await fetch('/offer', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type
        }})
      }});

      if (!resp.ok) {{
        setStatus('offer failed (' + resp.status + ')', true);
        return;
      }}

      const answer = await resp.json();
      await pc.setRemoteDescription(answer);
      setStatus('streaming');
    }}

    // Auto-start once
    start().catch(err => {{
      console.error(err);
      setStatus('error: ' + err, true);
    }});

    // Cleanup when page is closed
    window.addEventListener('beforeunload', () => {{
      if (pc) {{
        try {{ pc.close(); }} catch(e) {{}}
        pc = null;
      }}
    }});

    const slotCard   = document.getElementById('slot-card');
    const slotId     = document.getElementById('slot-id');
    const slotVendorId   = document.getElementById('slot-vendor-id');
    const slotVendorName = document.getElementById('slot-vendor-name');
    const slotStart  = document.getElementById('slot-start');
    const slotEnd    = document.getElementById('slot-end');
    const slotUp     = document.getElementById('slot-up');
    const slotDown   = document.getElementById('slot-down');
    const slotTotal  = document.getElementById('slot-total');
    const slotDeclared = document.getElementById('slot-declared');
    const slotStatus = document.getElementById('slot-status');
    const slotStartGc = document.getElementById('slot-start-gc');
    const slotEndGc = document.getElementById('slot-end-gc');

    function formatTime(ts) {{
    if (!ts) return '—';
    const d = new Date(ts);
    return d.toLocaleString();
    }}

    let lastSlotSnapshot = null;
    let lastActiveSlotId = null;
    let inactiveEndTimeIso = null;

    function renderSlotCard(slot, isActive) {{
      if (!slot) {{
        slotCard.classList.add('hidden');
        return;
      }}

      const breakdown = slot.direction_breakdown || {{}};
      slotCard.classList.remove('hidden');
      slotCard.classList.toggle('inactive', !isActive);

      slotId.textContent         = slot.slot_id ?? '-';
      slotVendorId.textContent   = slot.vendor_id ?? '-';
      slotVendorName.textContent = slot.vendor_name ?? '-';
      slotStart.textContent      = formatTime(slot.start_time);
      // If inactive snapshot has no explicit end_time, show locally captured transition time.
      const resolvedEnd = slot.end_time ?? inactiveEndTimeIso;
      slotEnd.textContent        = formatTime(resolvedEnd);

      slotUp.textContent         = breakdown.up ?? 0;
      slotDown.textContent       = breakdown.down ?? 0;
      slotTotal.textContent      = slot.slot_count ?? 0;
      slotDeclared.textContent   = slot.declared_count ?? '-';

      slotStatus.textContent = isActive ? (slot.status ?? 'ACTIVE') : 'INACTIVE';

    slotStatus.classList.remove('slot-status-active', 'slot-status-inactive');
    slotStatus.classList.add(isActive ? 'slot-status-active' : 'slot-status-inactive');

      slotStartGc.textContent    = slot.start_global_count ?? '-';
      slotEndGc.textContent      = slot.end_global_count ?? '-';
    }}

    async function pollSlotState() {{
    // Avoid background-tab churn and unnecessary polling when page is hidden.
    if (document.hidden) return;
    try {{
        const resp = await fetch('/slot-state');
        if (!resp.ok) return;

        const data = await resp.json();

        if (data.active && data.slot) {{
          lastActiveSlotId = data.slot.slot_id ?? null;
          inactiveEndTimeIso = null;
        }}

        if (data.slot) {{
          lastSlotSnapshot = data.slot;
        }}
        if (data.active && data.slot) {{
          renderSlotCard(data.slot, true);
        }} else if (lastSlotSnapshot) {{
          // Capture a stable local end-time once when the slot transitions inactive.
          if (!inactiveEndTimeIso && lastActiveSlotId && lastSlotSnapshot.slot_id === lastActiveSlotId) {{
            inactiveEndTimeIso = new Date().toISOString();
          }}
          // Keep the latest slot visible in a dimmed state for operator context.
          renderSlotCard(lastSlotSnapshot, false);
        }} else {{
          slotCard.classList.add('hidden');
        }}
    }} catch (e) {{
        console.warn('slot poll failed', e);
    }}
    }}

    // Poll at a moderate interval to reduce API pressure on constrained devices.
    setInterval(pollSlotState, 3000);

  </script>
</body>
</html>
"""

        return web.Response(text=html, content_type="text/html")

    async def _offer_handler(self, request: web.Request):
        """
        Handle WebRTC offer from browser:
          - create RTCPeerConnection
          - attach a FrameQueueTrack with its own frame queue
          - return answer JSON
        """
        if self._closing.is_set():
            return web.json_response({"error": "server_closing"}, status=503)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        if "sdp" not in data or "type" not in data:
            return web.json_response({"error": "missing_sdp"}, status=400)

        if self.max_clients is not None and len(self._pcs) >= self.max_clients:
            return web.json_response({"error": "max_clients"}, status=503)

        # 1) Create PC and register it
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        pc = RTCPeerConnection()
        self._pcs.add(pc)
        conn_id = id(pc)

        log.info("WEBRTC", f"New peer connection (total={len(self._pcs)})")

        @pc.on("connectionstatechange")
        def on_state_change():
            st = pc.connectionState
            log.debug("WEBRTC", f"Peer {conn_id} state={st}")
            if st in ("failed", "closed"):
                asyncio.ensure_future(self._remove_peer(pc))

        try:
            # 2) First apply remote offer so transceivers are created
            await pc.setRemoteDescription(offer)

            # 3) Now create a per-client frame queue and attach our video track
            frame_q: asyncio.Queue = asyncio.Queue(maxsize=3)
            self._frame_queues[conn_id] = frame_q

            video_track = _FrameQueueTrack(frame_q, target_fps=self.target_fps)

            # Attach our track to the remote's video transceiver
            # (usually there is exactly one video transceiver)
            attached = False
            for t in pc.getTransceivers():
                if t.kind == "video" and not attached:
                    pc.addTrack(video_track)
                    attached = True

            if not attached:
                # No video transceiver in the offer -> nothing to stream
                log.error("WEBRTC", "Offer has no video transceiver; rejecting")
                await self._remove_peer(pc)
                return web.json_response({"error": "no_video"}, status=400)

            # 4) Create answer and set our local description
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

        except Exception:
            log.error(
                "WEBRTC",
                "Error during SDP exchange: " + traceback.format_exc(),
            )
            await self._remove_peer(pc)
            return web.json_response({"error": "sdp_failed"}, status=500)

        # 5) Send back answer SDP
        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

    async def _control_handler(self, request: web.Request):
        """
        Lightweight control endpoint.
        Browser POSTs JSON like: {"cmd": "screenshot"} or {"cmd": "quit"}
        Forward to std control_queue if configured.
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

        if self._control_queue is not None:
            try:
                self._control_queue.put_nowait(data)
            except Exception:
                log.debug("WEBRTC", "control_queue.put_nowait failed")
        else:
            log.debug("WEBRTC", f"Control received but no queue configured: {data}")

        return web.json_response({"ok": True})

    async def _health_handler(self, request: web.Request):
        try:
            info = {
                "ok": True,
                "closing": self._closing.is_set(),
                "peers": len(self._pcs),
                "frame_queues": len(self._frame_queues),
            }
            return web.json_response(info)
        except Exception:
            return web.json_response({"ok": False, "error": "health_failed"})

    def _get_slot_state_cached(self):
        """
        Return slot state with TTL caching to reduce repeated SlotManager access
        under frequent browser polling.
        """
        now = time.monotonic()
        last_ts = float(self._slot_state_cache.get("ts", 0.0))
        if (now - last_ts) < self._slot_state_cache_ttl_s:
            return self._slot_state_cache.get("data", {"active": False, "slot": None})

        slot_mgr = self._slot_manager
        data = {"active": False, "slot": None}
        try:
            if slot_mgr is not None:
                # Lightweight public snapshot only.
                data = slot_mgr.get_public_state()
        except Exception:
            # Keep endpoint resilient; never break WebRTC loop for slot-state failures.
            data = {"active": False, "slot": None}

        self._slot_state_cache["ts"] = now
        self._slot_state_cache["data"] = data
        return data

    async def _slot_state_handler(self, request: web.Request):
        try:
            return web.json_response(self._get_slot_state_cached())
        except Exception:
            return web.json_response({"active": False, "slot": None})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _broadcast_frame_to_queues(self, frame_bgr: np.ndarray):
        """
        Async: fan-out one frame to all per-client queues with drop-oldest policy.
        Runs inside asyncio loop.
        """
        if not self._frame_queues:
            return

        # Resize down for network if needed
        h, w = frame_bgr.shape[:2]
        if w > self.downscale_width or h > self.downscale_height:
            try:
                scale = min(
                    self.downscale_width / float(w),
                    self.downscale_height / float(h),
                )
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                frame_bgr = cv2.resize(
                    frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
            except Exception:
                # if resize fails, we still try to send original frame
                log.debug("WEBRTC", "resize failed, sending original frame")

        # Fan-out
        for conn_id, q in list(self._frame_queues.items()):
            try:
                if q.full():
                    try:
                        _ = q.get_nowait()
                    except Exception:
                        pass
                await q.put(frame_bgr)
            except Exception:
                # If queue fails, ignore; connection cleanup will handle stale queues
                pass

    async def _remove_peer(self, pc: RTCPeerConnection):
        conn_id = id(pc)
        if pc in self._pcs:
            self._pcs.discard(pc)
        if conn_id in self._frame_queues:
            try:
                del self._frame_queues[conn_id]
            except Exception:
                pass
        try:
            await pc.close()
        except Exception:
            pass
        log.info("WEBRTC", f"Peer {conn_id} removed (total={len(self._pcs)})")

    async def _shutdown(self):
        log.debug("WEBRTC", "Shutdown requested (async)")
        # Close all PCs
        pcs = list(self._pcs)
        for pc in pcs:
            try:
                await pc.close()
            except Exception:
                pass
        self._pcs.clear()
        self._frame_queues.clear()

        # Stop the loop
        self._loop.call_soon(self._loop.stop)
        log.debug("WEBRTC", "Event loop stop scheduled")

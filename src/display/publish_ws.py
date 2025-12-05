# src/display/publish_ws.py
"""
WSPublisher - lightweight websocket broadcaster for frames + JSON control/metadata.

Features implemented:
 - Single-threaded asyncio broadcaster running in a dedicated daemon thread.
 - Caches encoded JPEG bytes per-frame and re-uses until a new frame is published.
 - publish(frame) and publish_encoded(jpeg_bytes) APIs (thread-safe).
 - Per-client send timeout with immediate drop for slow clients.
 - Max clients limit with polite rejection.
 - Optional downscale before encoding (max_publish_width).
 - Configurable jpeg_quality and broadcast interval; adaptive behavior to avoid needless sends.
 - Control queue forwarding (thread-safe) for JSON commands from clients.
 - Light metrics counters for debugging/tuning.
 - Clean close()/shutdown behavior.

Usage (same as before).
"""

from __future__ import annotations

import asyncio
import threading
import time
import json
import traceback
from typing import Optional, Dict, Any, Set
import numpy as np
import cv2
from aiohttp import web, WSCloseCode, WSMsgType
import queue as std_queue

from src.utils.logger import log

# Default settings - tune as needed
DEFAULT_BROADCAST_INTERVAL = 0.04  # 25 Hz target default
DEFAULT_JPEG_QUALITY = 80
DEFAULT_SEND_TIMEOUT = 0.06  # seconds per-client send timeout; tune 0.03-0.2
DEFAULT_MAX_PUBLISH_WIDTH = 960  # downscale if frame wider than this (None to disable)
DEFAULT_MAX_CLIENTS = None  # None = unlimited
DEFAULT_MIN_BROADCAST_INTERVAL = 0.02  # never go faster than this


# Small helper
def _safe_close_ws(
    ws: web.WebSocketResponse, code=WSCloseCode.GOING_AWAY, msg: bytes = b""
):
    try:
        return ws.close(code=code, message=msg)
    except Exception:
        return None


class WSPublisher:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        broadcast_interval: float = DEFAULT_BROADCAST_INTERVAL,
        max_clients: Optional[int] = DEFAULT_MAX_CLIENTS,
        send_timeout: float = DEFAULT_SEND_TIMEOUT,
        max_publish_width: Optional[int] = DEFAULT_MAX_PUBLISH_WIDTH,
        min_broadcast_interval: float = DEFAULT_MIN_BROADCAST_INTERVAL,
    ):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.broadcast_interval = float(broadcast_interval)
        self.min_broadcast_interval = float(min_broadcast_interval)
        self.send_timeout = float(send_timeout)
        self.max_clients = None if max_clients is None else int(max_clients)
        self._max_publish_width = (
            None if max_publish_width is None else int(max_publish_width)
        )

        # latest frame management
        # _latest_frame: raw ndarray (copied) OR None
        # _latest_encoded: bytes OR None
        # _frame_seq: monotonic increasing integer that differentiates frames
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_encoded: Optional[bytes] = None
        self._frame_seq: int = 0
        self._frame_lock = threading.Lock()

        # cached metadata for broadcaster (protected by frame lock where needed)
        self._last_sent_seq_per_client: Dict[int, int] = (
            {}
        )  # key = id(ws) -> last_frame_seq
        self._last_broadcast_seq: int = -1  # seq that was last broadcast (global)

        # control queue from clients -> DisplayWorker
        self._control_queue: Optional[std_queue.Queue] = None

        # runtime state
        self._clients: Set[web.WebSocketResponse] = set()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._start_loop, name="WSPublisherLoop", daemon=True
        )
        self._closing = threading.Event()
        self._server_ready = threading.Event()

        # metrics (simple counters)
        self._metrics = {
            "encodes": 0,
            "broadcasts": 0,
            "bytes_sent": 0,
            "clients_dropped": 0,
            "clients_connected_total": 0,
        }

        log.debug(
            "WSPUBLISHER", f"Starting WSPublisher thread on {self.host}:{self.port}"
        )
        self._thread.start()
        if not self._server_ready.wait(timeout=5.0):
            log.warning(
                "WSPUBLISHER",
                "WSPublisher server thread did not signal ready within 5s",
            )

    # ---------------- public API ----------------
    def set_control_queue(self, q: std_queue.Queue):
        """Set a thread-safe queue where control dicts will be put (non-blocking)."""
        self._control_queue = q
        log.debug("WSPUBLISHER", "Control queue set")

    def publish(self, frame: np.ndarray) -> None:
        """Store latest raw ndarray frame (copied). Thread-safe and non-blocking.

        This invalidates the encoded cache and increments frame sequence.
        """
        if frame is None:
            return
        try:
            with self._frame_lock:
                # copy to avoid external mutation
                self._latest_frame = frame.copy()
                self._latest_encoded = None
                self._frame_seq += 1
        except Exception:
            log.error("WSPUBLISHER", "publish(frame) failed: " + traceback.format_exc())

    def publish_encoded(self, jpeg_bytes: bytes) -> None:
        """Store latest pre-encoded JPEG bytes. Thread-safe.

        Use this from DisplayWorker if you prefer to encode frames
        in the display thread (recommended for heavy workloads).
        """
        if jpeg_bytes is None:
            return
        try:
            with self._frame_lock:
                self._latest_encoded = bytes(jpeg_bytes)
                self._latest_frame = None
                self._frame_seq += 1
                self._metrics["encodes"] += 1
        except Exception:
            log.error(
                "WSPUBLISHER", "publish_encoded failed: " + traceback.format_exc()
            )

    def publish_json(self, obj: dict) -> None:
        """Broadcast a small JSON text message to all clients (async-safe)."""
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast_json(obj), self._loop)
        except Exception:
            log.error("WSPUBLISHER", "publish_json failed: " + traceback.format_exc())

    def close(self, timeout: float = 3.0) -> None:
        """Stop the webserver and broadcaster. This is synchronous and blocks until done or timeout."""
        if self._closing.is_set():
            return
        log.info("WSPUBLISHER", "Closing WSPublisher...")
        self._closing.set()
        try:
            fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            try:
                fut.result(timeout=timeout)
            except Exception:
                log.debug(
                    "WSPUBLISHER",
                    "Shutdown wait finished/failed: " + traceback.format_exc(),
                )
        except Exception:
            log.error(
                "WSPUBLISHER", "Failed to schedule shutdown: " + traceback.format_exc()
            )
        # join the thread
        self._thread.join(timeout=timeout)
        log.info("WSPUBLISHER", "WSPublisher stopped")

    # ---------------- internal event loop ----------------
    def _start_loop(self):
        try:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._init_app())
            self._loop.run_forever()
        except Exception:
            log.error(
                "WSPUBLISHER", "Publisher loop crashed: " + traceback.format_exc()
            )
        finally:
            try:
                self._loop.run_until_complete(self._cleanup_loop())
            except Exception:
                log.debug("WSPUBLISHER", "Cleanup exception: " + traceback.format_exc())
            finally:
                self._loop.close()
                log.debug("WSPUBLISHER", "Async loop closed")

    async def _init_app(self):
        app = web.Application()
        app.router.add_get("/", self._index_handler)
        app.router.add_get("/ws", self._ws_handler)
        app.router.add_get("/health", self._health_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=self.host, port=self.port)
        await site.start()
        log.info("WSPUBLISHER", f"HTTP server started on {self.host}:{self.port}")
        self._bcast_task = asyncio.create_task(self._broadcaster())
        self._server_ready.set()

    async def _cleanup_loop(self):
        try:
            if hasattr(self, "_bcast_task"):
                self._bcast_task.cancel()
                try:
                    await self._bcast_task
                except Exception:
                    pass
            to_close = list(self._clients)
            for ws in to_close:
                try:
                    await ws.close(
                        code=WSCloseCode.GOING_AWAY, message=b"Server shutdown"
                    )
                except Exception:
                    pass
            self._clients.clear()
        except Exception:
            log.debug("WSPUBLISHER", "cleanup exception: " + traceback.format_exc())

    # ---------------- simple dev HTML ----------------
    async def _index_handler(self, request):
        html = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>WSPublisher</title></head>
<body style="font-family:Arial,Helvetica,sans-serif">
<h3>WSPublisher Demo</h3>
<canvas id="c" style="max-width:100%;border:1px solid #333;"></canvas><br/>
<button onclick="sendCmd('screenshot')">Screenshot</button>
<button onclick="sendCmd('quit')">Quit</button>
<select id="quality" onchange="setQuality(this.value)">
  <option value="80">Quality 80</option>
  <option value="60">Quality 60</option>
  <option value="40">Quality 40</option>
</select>
<span id="status"></span>
<script>
  const status = document.getElementById('status');
  const ws = new WebSocket("ws://" + location.host + "/ws");
  ws.binaryType = 'arraybuffer';
  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('2d');
  ws.onopen = () => { status.innerText = 'connected'; };
  ws.onclose = () => { status.innerText = 'closed'; };
  ws.onerror = (e) => { status.innerText = 'error'; console.error(e); };
  ws.onmessage = (evt) => {
    if (typeof evt.data === 'string') {
      try { const obj = JSON.parse(evt.data); console.log('json-msg', obj); } catch(e){}
      return;
    }
    const blob = new Blob([evt.data], {type: 'image/jpeg'});
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(blob);
  };
  function sendCmd(cmd) { ws.send(JSON.stringify({cmd:cmd})); }
  function setQuality(q) { ws.send(JSON.stringify({cmd:'set_quality', quality: parseInt(q)})); }
</script>
</body>
</html>
"""
        return web.Response(text=html, content_type="text/html")

    # ---------------- health handler ----------------
    async def _health_handler(self, request):
        try:
            with self._frame_lock:
                have_frame = (
                    self._latest_frame is not None or self._latest_encoded is not None
                )
            info = {
                "ok": True,
                "clients": len(self._clients),
                "have_frame": bool(have_frame),
                "metrics": self._metrics,
            }
            return web.json_response(info)
        except Exception:
            return web.json_response({"ok": False, "error": "health-failed"})

    # ---------------- websocket handler ----------------
    async def _ws_handler(self, request):
        ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)

        # enforce max_clients if set
        if self.max_clients is not None and len(self._clients) >= self.max_clients:
            try:
                await ws.close(code=WSCloseCode.TRY_AGAIN_LATER, message=b"server_full")
            except Exception:
                pass
            log.debug("WSPUBLISHER", "Rejected client - max_clients reached")
            return ws

        self._clients.add(ws)
        self._metrics["clients_connected_total"] += 1
        log.debug("WSPUBLISHER", f"Client connected (total={len(self._clients)})")

        # ensure a mapping for last-sent seq
        try:
            self._last_sent_seq_per_client[id(ws)] = -1
        except Exception:
            pass

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    text = msg.data
                    try:
                        data = json.loads(text)
                        if self._control_queue is not None:
                            try:
                                self._control_queue.put_nowait(data)
                            except Exception:
                                log.debug(
                                    "WSPUBLISHER", "control_queue.put_nowait failed"
                                )
                        else:
                            # handle small built-in commands locally (e.g., quality change)
                            try:
                                cmd = data.get("cmd")
                                if cmd == "set_quality" and "quality" in data:
                                    q = int(data["quality"])
                                    with self._frame_lock:
                                        self.jpeg_quality = int(max(10, min(95, q)))
                                    log.info(
                                        "WSPUBLISHER",
                                        f"Client requested jpeg_quality={self.jpeg_quality}",
                                    )
                            except Exception:
                                pass
                    except Exception:
                        log.debug("WSPUBLISHER", "Invalid control JSON received")
                elif msg.type == WSMsgType.ERROR:
                    log.debug("WSPUBLISHER", f"ws connection error: {ws.exception()}")
        except Exception:
            log.debug(
                "WSPUBLISHER", "WS handler loop exception: " + traceback.format_exc()
            )
        finally:
            # cleanup client
            try:
                self._clients.discard(ws)
                if id(ws) in self._last_sent_seq_per_client:
                    try:
                        del self._last_sent_seq_per_client[id(ws)]
                    except Exception:
                        pass
                await ws.close()
            except Exception:
                pass
            log.debug(
                "WSPUBLISHER", f"Client disconnected (total={len(self._clients)})"
            )
        return ws

    # ---------------- broadcaster (core) ----------------
    async def _broadcaster(self):
        """Periodically send latest frame (binary JPEG) to all clients.

        This loop will:
        - encode frame if needed and cache the bytes
        - only broadcast when there's a new frame (new frame_seq) OR when heartbeat forces a resend
        - per-client send uses a short timeout; slow/blocked clients are dropped
        """
        log.debug("WSPUBLISHER", "Broadcaster started")
        try:
            while not self._closing.is_set():
                encoded = None
                seq = None

                # snapshot of latest encoded/cached bytes and sequence
                with self._frame_lock:
                    seq = int(self._frame_seq)
                    encoded = self._latest_encoded
                    # If encoded is None and we have a frame, we will encode below outside lock
                    frame = (
                        None
                        if (encoded is not None)
                        else (
                            self._latest_frame.copy()
                            if self._latest_frame is not None
                            else None
                        )
                    )

                # If no encoded bytes but a raw frame is present, encode once and cache
                if encoded is None and frame is not None:
                    try:
                        # optional downscale
                        if self._max_publish_width is not None:
                            h, w = frame.shape[:2]
                            if w > self._max_publish_width:
                                scale = float(self._max_publish_width) / float(w)
                                new_w = max(1, int(round(w * scale)))
                                new_h = max(1, int(round(h * scale)))
                                frame_to_enc = cv2.resize(
                                    frame,
                                    (new_w, new_h),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            else:
                                frame_to_enc = frame
                        else:
                            frame_to_enc = frame

                        ok, buf = cv2.imencode(
                            ".jpg",
                            frame_to_enc,
                            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                        )
                        if ok:
                            encoded = buf.tobytes()
                            # cache encoded bytes for reuse
                            with self._frame_lock:
                                # ensure no one else already encoded a newer frame
                                if self._frame_seq == seq:
                                    self._latest_encoded = encoded
                                # increment encode metric
                                self._metrics["encodes"] += 1
                    except Exception:
                        log.debug(
                            "WSPUBLISHER",
                            "JPEG encode failed: " + traceback.format_exc(),
                        )

                # decide whether to broadcast:
                # - send if encoded exists AND (new seq != last_broadcast_seq)
                # - OR if there are clients but no new seq and we want to heartbeat occasionally
                should_broadcast = False
                if encoded is not None:
                    if seq != self._last_broadcast_seq:
                        should_broadcast = True
                    else:
                        # heartbeat: we can choose to re-send occasionally to help late-joining clients
                        # but we keep it infrequent (send every N intervals)
                        # simplified: do not re-send unless there are zero broadcasts for a while
                        should_broadcast = False

                if encoded is not None and should_broadcast and self._clients:
                    to_drop = []
                    sent_count = 0
                    for ws in list(self._clients):
                        try:
                            # send with a small timeout so slow clients don't block everything
                            await asyncio.wait_for(
                                ws.send_bytes(encoded), timeout=self.send_timeout
                            )
                            sent_count += 1
                            self._metrics["bytes_sent"] += len(encoded)
                            # record per-client last sent seq
                            try:
                                self._last_sent_seq_per_client[id(ws)] = seq
                            except Exception:
                                pass
                        except asyncio.TimeoutError:
                            # client too slow -> drop
                            log.debug(
                                "WSPUBLISHER", "Client timed out (send) -> dropping"
                            )
                            to_drop.append(ws)
                        except Exception:
                            # other send error -> drop
                            to_drop.append(ws)

                    # cleanup dropped clients
                    for ws in to_drop:
                        try:
                            self._clients.discard(ws)
                            if id(ws) in self._last_sent_seq_per_client:
                                try:
                                    del self._last_sent_seq_per_client[id(ws)]
                                except Exception:
                                    pass
                            try:
                                await ws.close(
                                    code=WSCloseCode.GOING_AWAY, message=b"slow-client"
                                )
                            except Exception:
                                pass
                            self._metrics["clients_dropped"] += 1
                        except Exception:
                            pass

                    self._last_broadcast_seq = seq
                    self._metrics["broadcasts"] += 1
                    log.debug(
                        "WSPUBLISHER",
                        f"Broadcast seq={seq} to {sent_count}/{len(self._clients)+len(to_drop)} clients",
                    )
                # else: nothing to send or no new frame

                # sleep for broadcast interval (adaptive: cannot go below min interval)
                await asyncio.sleep(
                    max(self.min_broadcast_interval, self.broadcast_interval)
                )
        except asyncio.CancelledError:
            log.debug("WSPUBLISHER", "Broadcaster cancelled")
        except Exception:
            log.error("WSPUBLISHER", "Broadcaster crashed: " + traceback.format_exc())
        finally:
            log.debug("WSPUBLISHER", "Broadcaster exiting")

    # ---------------- broadcast JSON ----------------
    async def _broadcast_json(self, obj: dict):
        try:
            payload = json.dumps(obj)
            to_drop = []
            for ws in list(self._clients):
                try:
                    await asyncio.wait_for(
                        ws.send_str(payload), timeout=self.send_timeout
                    )
                except Exception:
                    to_drop.append(ws)
            for ws in to_drop:
                self._clients.discard(ws)
        except Exception:
            log.debug("WSPUBLISHER", "broadcast_json failed: " + traceback.format_exc())

    # ---------------- shutdown ----------------
    async def _shutdown(self):
        try:
            log.debug("WSPUBLISHER", "Shutdown requested (async)")
            self._closing.set()
            try:
                if hasattr(self, "_bcast_task"):
                    self._bcast_task.cancel()
                    try:
                        await self._bcast_task
                    except Exception:
                        pass
            except Exception:
                pass
            for ws in list(self._clients):
                try:
                    await ws.close(
                        code=WSCloseCode.GOING_AWAY, message=b"Server shutdown"
                    )
                except Exception:
                    pass
            self._clients.clear()
        except Exception:
            log.debug("WSPUBLISHER", "shutdown error: " + traceback.format_exc())
        finally:
            # stop the loop after small delay
            self._loop.call_soon(self._loop.stop)
            log.debug("WSPUBLISHER", "Event loop stop scheduled")

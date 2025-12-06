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
<button id="btn_screenshot">Screenshot</button>
<button id="btn_quit">Quit</button>
<select id="quality">
  <option value="80">Quality 80</option>
  <option value="60">Quality 60</option>
  <option value="40">Quality 40</option>
</select>
<span id="status" style="margin-left:12px;"></span>
<script>
  const status = document.getElementById('status');
  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('2d');
  const btnScreenshot = document.getElementById('btn_screenshot');
  const btnQuit = document.getElementById('btn_quit');
  const selQuality = document.getElementById('quality');

  let ws = new WebSocket("ws://" + location.host + "/ws");
  ws.binaryType = 'arraybuffer';

  // helper to disable UI once server closes
  function disableUI(reason) {
    try {
      btnScreenshot.disabled = true;
      btnQuit.disabled = true;
      selQuality.disabled = true;
    } catch (e) {}
    if (reason) {
      status.innerText = reason;
    }
  }

  ws.onopen = () => { status.innerText = 'connected'; };
  ws.onerror = (e) => { status.innerText = 'error'; console.error(e); };

  // onclose: display closed and disable UI
  ws.onclose = (evt) => {
    // If we already showed server_shutdown, keep that message.
    if (!status.innerText || status.innerText === 'connected' || status.innerText === 'error') {
      status.innerText = 'closed';
    }
    disableUI();
  };

  // onmessage: handle JSON control messages (text) and binary frames
  ws.onmessage = (evt) => {
    // Text messages: JSON commands from server
    if (typeof evt.data === 'string') {
      try {
        const obj = JSON.parse(evt.data);
        console.log('json-msg', obj);

        // server_shutdown: show friendly banner and close the socket
        if (obj && obj.cmd === 'server_shutdown') {
          const reason = obj.reason ? ` — ${obj.reason}` : '';
          status.innerText = 'Server closed' + reason;
          // Stop rendering further frames and close socket
          try { ws.close(); } catch(e){}
          disableUI('Server closed' + reason);
          return;
        }

        // other JSON messages can be handled here if needed
      } catch (e) {
        // not JSON or parse error — ignore
      }
      return;
    }

    // Binary frames: JPEG bytes
    try {
      const blob = new Blob([evt.data], {type: 'image/jpeg'});
      const img = new Image();
      img.onload = () => {
        // draw image to canvas
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(img.src);
      };
      img.src = URL.createObjectURL(blob);
    } catch (e) {
      console.error('frame decode error', e);
    }
  };

  // Safe send: only send if ws is open
  function safeSend(obj) {
    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(obj));
        return true;
      } else {
        console.warn('WebSocket not open — cannot send', obj);
        return false;
      }
    } catch (e) {
      console.warn('send failed', e);
      return false;
    }
  }

  // UI bindings
  btnScreenshot.addEventListener('click', () => {
    const ok = safeSend({cmd:'screenshot'});
    if (!ok) status.innerText = 'cannot send: disconnected';
  });

  btnQuit.addEventListener('click', () => {
    // ask server to quit; server will broadcast server_shutdown and close
    const ok = safeSend({cmd:'quit'});
    if (!ok) {
      status.innerText = 'cannot send quit: disconnected';
      disableUI('Disconnected');
    } else {
      status.innerText = 'quitting...';
    }
  });

  selQuality.addEventListener('change', (e) => {
    const q = parseInt(e.target.value || 80, 10);
    safeSend({cmd: 'set_quality', quality: q});
  });
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
                "closing": bool(self._closing.is_set()),
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
        """
        Periodically send latest frame (binary JPEG) to all clients.

        - Pre-encodes frame once per loop.
        - Resizes frames (optional) to remote_max_w x remote_max_h to reduce bandwidth.
        - Uses asyncio.wait_for on send_bytes to avoid blocking forever.
        - Drops clients that fail to receive quickly.
        - Adapts jpeg_quality / broadcast_interval when network is slow.
        """
        log.debug("WSPUBLISHER", "Broadcaster started")
        # tuning knobs (expose as ctor args if you like)
        remote_max_w = getattr(self, "remote_max_w", 640)  # scale down for remote viewers
        remote_max_h = getattr(self, "remote_max_h", 360)
        min_jpeg = 30
        max_jpeg = max(10, min(95, int(self.jpeg_quality)))
        quality = max_jpeg
        send_timeout = 0.35  # seconds to wait for a single client's send
        adapt_check_period = 5.0  # seconds between adapt checks
        samples = []
        last_adapt = time.monotonic()

        try:
            while not self._closing.is_set():
                start_loop = time.perf_counter()
                encoded = None

                # Grab latest frame/encoded safely
                with self._frame_lock:
                    if self._latest_encoded is not None:
                        encoded = self._latest_encoded
                    elif self._latest_frame is not None:
                        frame = self._latest_frame.copy()
                    else:
                        frame = None

                # If we have a raw frame, optionally resize to reduce bandwidth and encode once
                if encoded is None and frame is not None:
                    try:
                        h, w = frame.shape[:2]
                        if w > remote_max_w or h > remote_max_h:
                            scale = min(remote_max_w / float(w), remote_max_h / float(h))
                            new_w = max(1, int(round(w * scale)))
                            new_h = max(1, int(round(h * scale)))
                            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                        if ok:
                            encoded = buf.tobytes()
                    except Exception:
                        log.debug("WSPUBLISHER", "JPEG encode failed in broadcaster: " + traceback.format_exc())
                        encoded = None

                # If nothing to send, sleep a bit
                if encoded is None or not self._clients:
                    # small sleep - respects broadcast_interval but not too tight
                    await asyncio.sleep(self.broadcast_interval)
                    continue

                # Broadcast to clients, with per-client timeout
                send_start = time.perf_counter()
                to_drop = []
                # create list snapshot of clients to avoid mutation while iterating
                clients_snapshot = list(self._clients)
                for ws in clients_snapshot:
                    try:
                        # send with small timeout so slow clients don't block everyone
                        await asyncio.wait_for(ws.send_bytes(encoded), timeout=send_timeout)
                    except Exception as e:
                        # mark client to drop (timeout, connection reset, etc.)
                        to_drop.append(ws)
                # prune failed clients
                for ws in to_drop:
                    try:
                        self._clients.discard(ws)
                        try:
                            await ws.close(code=WSCloseCode.GOING_AWAY, message=b"Drop slow client")
                        except Exception:
                            pass
                    except Exception:
                        pass

                loop_send_time = time.perf_counter() - send_start
                samples.append(loop_send_time)
                # keep small sample window
                if len(samples) > 25:
                    samples.pop(0)

                # adaptive logic: if average send time high, lower quality or slow down broadcast
                if time.monotonic() - last_adapt >= adapt_check_period:
                    last_adapt = time.monotonic()
                    avg_send = sum(samples) / max(1, len(samples))
                    log.debug("WSPUBLISHER", f"avg_send_time={avg_send:.3f}s clients={len(self._clients)} quality={quality} interval={self.broadcast_interval}")
                    # If send is very slow, reduce quality or increase interval
                    if avg_send > 0.25 and quality > min_jpeg:
                        old = quality
                        quality = max(min_jpeg, int(quality * 0.8))
                        log.info("WSPUBLISHER", f"Reducing jpeg quality {old}->{quality} due to avg_send_time={avg_send:.3f}s")
                    elif avg_send < 0.08 and quality < max_jpeg:
                        # if network has slack, gently restore quality
                        old = quality
                        quality = min(max_jpeg, int(quality * 1.1) + 1)
                        if quality != old:
                            log.debug("WSPUBLISHER", f"Increasing jpeg quality {old}->{quality}")

                    # adjust interval if needed (more conservative)
                    if avg_send > 0.40:
                        # send slower (lower frame rate)
                        old_i = self.broadcast_interval
                        self.broadcast_interval = min(1.0, max(0.05, self.broadcast_interval * 1.5))
                        log.info("WSPUBLISHER", f"Increasing broadcast_interval {old_i:.3f}->{self.broadcast_interval:.3f}")
                    elif avg_send < 0.05:
                        # network is fast, you can lower interval if desired (don't go below 0.01)
                        old_i = self.broadcast_interval
                        self.broadcast_interval = max(0.01, self.broadcast_interval * 0.9)
                        if self.broadcast_interval != old_i:
                            log.debug("WSPUBLISHER", f"Decreasing broadcast_interval {old_i:.3f}->{self.broadcast_interval:.3f}")

                # sleep to respect broadcast_interval (account for time already spent)
                elapsed = time.perf_counter() - start_loop
                to_sleep = max(0.0, self.broadcast_interval - elapsed)
                if to_sleep > 0:
                    await asyncio.sleep(to_sleep)

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
            # Try to broadcast a final shutdown message to all clients (best-effort)
            try:
                await self._broadcast_json(
                    {"cmd": "server_shutdown", "reason": "server_shutdown"}
                )
            except Exception:
                log.debug(
                    "WSPUBLISHER",
                    "Final shutdown broadcast failed: " + traceback.format_exc(),
                )

            # cancel bcast task if present
            try:
                if hasattr(self, "_bcast_task"):
                    self._bcast_task.cancel()
                    try:
                        await self._bcast_task
                    except Exception:
                        pass
            except Exception:
                pass
            # close clients
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

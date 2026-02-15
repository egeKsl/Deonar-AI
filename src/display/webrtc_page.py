"""WebRTC index page HTML builder.

This module intentionally holds only UI markup/script so the server file can stay
focused on routing, signaling, and transport logic.
"""

from __future__ import annotations


def build_webrtc_index_html() -> str:
    """Return the HTML document served at the WebRTC root index route."""
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Goat Stream (WebRTC)</title>
  <style>
    body {
      margin: 0;
      padding: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #111;
      color: #eee;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .container {
      background: #1b1b1b;
      border-radius: 10px;
      padding: 16px 20px 20px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.7);
      max-width: 960px;
      width: 100%;
    }
    h3 {
      margin: 0 0 10px;
      font-weight: 600;
      color: #f5f5f5;
    }
    #video {
      width: 100%;
      max-height: 540px;
      background: #000;
      border-radius: 6px;
      border: 1px solid #333;
    }
    .controls {
      margin-top: 10px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    button {
      padding: 6px 14px;
      border-radius: 4px;
      border: none;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      background: #2d6cdf;
      color: #fff;
      transition: background 0.2s, transform 0.1s;
    }
    button:hover {
      background: #3c7dff;
    }
    button:active {
      transform: scale(0.97);
    }
    button.secondary {
      background: #444;
    }
    #status {
      margin-left: auto;
      font-size: 13px;
      color: #0fdf7b;
      white-space: nowrap;
    }
    #status.error {
      color: #ff5c5c;
    }
    /* Slot status colors */
    .slot-status-active {
    color: #1ddf8b; /* green */
    font-weight: 600;
    }

    .slot-status-inactive {
    color: #ff5c5c; /* red */
    font-weight: 600;
    }

    .slot-card {
    margin-top: 12px;
    padding: 12px 16px;
    background: linear-gradient(135deg, #1f1f1f 0%, #232323 100%);
    border-left: 4px solid #1ddf8b;
    border-radius: 6px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.6);
    font-size: 14px;
    transition: opacity 0.2s ease, filter 0.2s ease, border-color 0.2s ease;
    }

    .slot-card.hidden {
    display: none;
    }

    .slot-card.inactive {
    border-left-color: #8a8a8a;
    opacity: 0.72;
    filter: saturate(0.8);
    }

    .slot-row {
    margin: 4px 0;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    }

    .slot-title {
    font-size: 15px;
    font-weight: 600;
    color: #1ddf8b;
    }

    .slot-counts span,
    .slot-meta span {
    font-family: monospace;
    font-weight: 600;
    }

    .slot-divider {
    height: 1px;
    background: #333;
    margin: 6px 0;
    }
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

    function setStatus(text, isError = false) {
      statusElem.textContent = text;
      if (isError) {
        statusElem.classList.add('error');
      } else {
        statusElem.classList.remove('error');
      }
    }

    async function sendControl(cmd) {
      try {
        const resp = await fetch('/control', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cmd })
        });
        if (!resp.ok) {
          console.warn('control failed', resp.status);
        }
      } catch (e) {
        console.error('control error', e);
      }
    }

    btnShot.onclick = () => {
      sendControl('screenshot');
      setStatus('screenshot requested');
    };

    btnQuit.onclick = () => {
      sendControl('quit');
      setStatus('quit requested');
    };

    btnStart.onclick = () => {
      start().catch(err => {
        console.error(err);
        setStatus('error: ' + err, true);
      });
    };

    async function start() {
      // Clean up old connection if any
      if (pc) {
        try { pc.close(); } catch(e) {}
        pc = null;
      }
      setStatus('connecting...');

      // Optional STUN server; can be omitted if everything is LAN / same host
      pc = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' }
        ]
      });

      pc.ontrack = (event) => {
        console.log('ontrack', event.track.kind);
        if (event.track.kind === 'video') {
          const [stream] = event.streams;
          if (videoElem.srcObject !== stream) {
            videoElem.srcObject = stream;
          }
        }
      };

      pc.oniceconnectionstatechange = () => {
        console.log('ice state:', pc.iceConnectionState);
        if (pc.iceConnectionState === 'connected') {
          setStatus('connected');
        } else if (pc.iceConnectionState === 'disconnected' ||
                   pc.iceConnectionState === 'failed') {
          setStatus('connection lost', true);
        }
      };

      pc.onconnectionstatechange = () => {
        console.log('conn state:', pc.connectionState);
        if (pc.connectionState === 'connected') {
          setStatus('connected');
        } else if (pc.connectionState === 'failed' ||
                   pc.connectionState === 'disconnected') {
          setStatus('connection lost', true);
        } else if (pc.connectionState === 'closed') {
          setStatus('closed');
        }
      };

      // Request a RECV-ONLY video transceiver so the offer has a video m-line.
      pc.addTransceiver('video', { direction: 'recvonly' });

      // Create and send offer to the server
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const resp = await fetch('/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type
        })
      });

      if (!resp.ok) {
        setStatus('offer failed (' + resp.status + ')', true);
        return;
      }

      const answer = await resp.json();
      await pc.setRemoteDescription(answer);
      setStatus('streaming');
    }

    // Auto-start once
    start().catch(err => {
      console.error(err);
      setStatus('error: ' + err, true);
    });

    // Cleanup when page is closed
    window.addEventListener('beforeunload', () => {
      if (pc) {
        try { pc.close(); } catch(e) {}
        pc = null;
      }
    });

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

    function formatTime(ts) {
      if (!ts) return '-';
      const d = new Date(ts);
      return d.toLocaleString();
    }

    let lastSlotSnapshot = null;
    let lastActiveSlotId = null;
    let inactiveEndTimeIso = null;

    function renderSlotCard(slot, isActive) {
      if (!slot) {
        slotCard.classList.add('hidden');
        return;
      }

      const breakdown = slot.direction_breakdown || {};
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
    }

    async function pollSlotState() {
      // Avoid background-tab churn and unnecessary polling when page is hidden.
      if (document.hidden) return;
      try {
        const resp = await fetch('/slot-state');
        if (!resp.ok) return;

        const data = await resp.json();

        if (data.active && data.slot) {
          lastActiveSlotId = data.slot.slot_id ?? null;
          inactiveEndTimeIso = null;
        }

        if (data.slot) {
          lastSlotSnapshot = data.slot;
        }
        if (data.active && data.slot) {
          renderSlotCard(data.slot, true);
        } else if (lastSlotSnapshot) {
          // Capture a stable local end-time once when the slot transitions inactive.
          if (!inactiveEndTimeIso && lastActiveSlotId && lastSlotSnapshot.slot_id === lastActiveSlotId) {
            inactiveEndTimeIso = new Date().toISOString();
          }
          // Keep the latest slot visible in a dimmed state for operator context.
          renderSlotCard(lastSlotSnapshot, false);
        } else {
          slotCard.classList.add('hidden');
        }
      } catch (e) {
        console.warn('slot poll failed', e);
      }
    }

    // Poll at a moderate interval to reduce API pressure on constrained devices.
    setInterval(pollSlotState, 3000);

  </script>
</body>
</html>
"""

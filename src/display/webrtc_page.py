"""WebRTC index page HTML builder.

This module intentionally holds only UI markup/script so the server file can stay
focused on routing, signaling, and transport logic.

Pages:
  build_webrtc_index_html() — live stream viewer (port 8081/)
  build_dashboard_html()    — live monitoring dashboard (port 8081/dashboard)
  build_slots_html()        — slot management console   (port 8081/slots)
"""

from __future__ import annotations

# Pages: build_webrtc_index_html | build_dashboard_html | build_slots_html


def build_webrtc_index_html() -> str:
    """Return the HTML document served at the WebRTC root index route."""
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Goat Stream (WebRTC)</title>
  <style>
    /* --------------------------------------------------
      GLOBAL
    -------------------------------------------------- */

    *,*::before,*::after{
      box-sizing:border-box;
    }

    body{
      margin:0;
      background:#0f0f0f;
      color:#e0e0e0;
      min-height:100vh;
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
    }

    /* --------------------------------------------------
      NAVBAR
    -------------------------------------------------- */

    .navbar{
      background:#1b1b1b;
      border-bottom:1px solid #272727;

      display:flex;
      align-items:center;
      gap:10px;

      padding:10px 20px;

      position:sticky;
      top:0;
      z-index:100;
    }

    .navbar-brand{
      font-weight:700;
      color:#1ddf8b;
      margin-right:auto;
    }

    .nav-link{
      color:#888;
      text-decoration:none;
      font-size:.82rem;
      padding:5px 11px;
      border-radius:5px;
      transition:.15s;
    }

    .nav-link:hover{
      background:#252525;
      color:#ddd;
    }

    .nav-link.active{
      background:#1ddf8b22;
      color:#1ddf8b;
    }

    /* --------------------------------------------------
      PAGE
    -------------------------------------------------- */

    .main{
      max-width:1500px;
      margin:0 auto;
      padding:16px;
    }

    .stream-layout{
      display:flex;
      flex-direction:column;
      gap:14px;
    }

    /* --------------------------------------------------
      STREAM CARD
    -------------------------------------------------- */

    .stream-card{
      background:#1a1a1a;
      border:1px solid #262626;
      border-radius:12px;
      overflow:hidden;
    }

    .stream-header{
      padding:14px 16px;
      border-bottom:1px solid #262626;

      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:12px;
    }

    .stream-title{
      font-size:1rem;
      font-weight:600;
    }

    .status-pill{
      padding:6px 12px;
      border-radius:999px;
      font-size:.78rem;
      font-weight:600;

      background:#1ddf8b22;
      color:#1ddf8b;
    }

    .status-pill.error{
      background:#ff5c5c22;
      color:#ff5c5c;
    }

    /* --------------------------------------------------
      VIDEO
    -------------------------------------------------- */

    #video{
      width:100%;
      height:72vh;

      min-height:450px;
      max-height:900px;

      display:block;

      background:#000;

      object-fit:contain;
    }

    /* --------------------------------------------------
      SLOT CARD
    -------------------------------------------------- */

    .slot-card{
      background:#1a1a1a;
      border:1px solid #262626;
      border-left:4px solid #1ddf8b;

      border-radius:12px;

      padding:18px;
    }

    .slot-card.inactive{
      border-left-color:#666;
      opacity:.75;
    }

    .slot-title{
      font-size:1.1rem;
      font-weight:700;
      color:#1ddf8b;
      margin-bottom:6px;
    }

    .slot-sub{
      color:#999;
      margin-bottom:14px;
    }

    .slot-row{
      display:flex;
      gap:12px;
      flex-wrap:wrap;
      margin:6px 0;
    }

    .slot-divider{
      height:1px;
      background:#2a2a2a;
      margin:12px 0;
    }

    .slot-meta{
      font-size:.82rem;
      color:#888;
    }

    /* --------------------------------------------------
      METRICS
    -------------------------------------------------- */

    .metrics-grid{
      display:grid;
      grid-template-columns:repeat(4,1fr);
      gap:12px;
      margin-top:12px;
    }

    .metric{
      background:#121212;
      border:1px solid #262626;
      border-radius:8px;
      padding:14px;
      text-align:center;
    }

    .metric-value{
      font-size:1.6rem;
      font-weight:700;
    }

    .metric-label{
      margin-top:4px;
      font-size:.72rem;
      color:#777;
      text-transform:uppercase;
    }

    /* --------------------------------------------------
      META INFO
    -------------------------------------------------- */

    .meta-grid{
      display:grid;
      grid-template-columns:repeat(2,1fr);
      gap:10px;
      margin-top:14px;
    }

    .meta-item{
      color:#888;
      font-size:.82rem;
    }

    /* --------------------------------------------------
      CONTROLS
    -------------------------------------------------- */

    .controls{
      display:flex;
      gap:12px;
      flex-wrap:wrap;
    }

    button{
      min-width:140px;
      height:42px;

      border:none;
      border-radius:8px;

      cursor:pointer;

      font-weight:600;

      background:#2d6cdf;
      color:#fff;
    }

    button:hover{
      background:#3c7dff;
    }

    button.secondary{
      background:#444;
      color:white;
    }

    button.secondary:hover{
      background:#555;
    }

    /* --------------------------------------------------
      STATUS STATES
    -------------------------------------------------- */

    .slot-status-active{
      color:#1ddf8b;
      font-weight:600;
    }

    .slot-status-inactive{
      color:#ff5c5c;
      font-weight:600;
    }

    /* --------------------------------------------------
      MISMATCH
    -------------------------------------------------- */

    .slot-mismatch{
      color:#ffb347;
      animation:pulse 1.2s ease-in-out infinite;
    }

    @keyframes pulse{
      0%  { opacity:1;   }
      50% { opacity:.65; }
      100%{ opacity:1;   }
    }

    /* --------------------------------------------------
      UTILITY
    -------------------------------------------------- */

    .hidden{
      display:none !important;
    }

    /* --------------------------------------------------
      RESPONSIVE
    -------------------------------------------------- */

    @media(max-width:900px){

      .metrics-grid{
        grid-template-columns:repeat(2,1fr);
      }

      #video{
        height:55vh;
        min-height:300px;
      }

    }

    @media(max-width:600px){

      .metrics-grid{
        grid-template-columns:1fr;
      }

      .meta-grid{
        grid-template-columns:1fr;
      }

      .stream-header{
        flex-direction:column;
        align-items:flex-start;
      }

      .controls{
        flex-direction:column;
      }

      button{
        width:100%;
      }

    }
  </style>
</head>
<body>
  <div class="navbar">

    <span class="navbar-brand">
      🐐 DEONAR AI
    </span>

    <a href="/" class="nav-link active">
      Stream
    </a>

    <a href="/dashboard" class="nav-link">
      Dashboard
    </a>

    <a href="/slots" class="nav-link">
      Slots
    </a>

  </div>
  <div class="main">
    <div class="stream-layout">
      <div class="stream-card">
        <div class="stream-header">

          <div class="stream-title">
            Live Stream
          </div>

          <span id="status" class="status-pill">
            Connecting...
          </span>

        </div>

        <video id="video"
              autoplay
              playsinline
              controls
              muted></video>

      </div>

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

        <div class="metrics-grid">

          <div class="metric">
            <div class="metric-value" id="slot-up">0</div>
            <div class="metric-label">UP</div>
          </div>

          <div class="metric">
            <div class="metric-value" id="slot-down">0</div>
            <div class="metric-label">DOWN</div>
          </div>

          <div class="metric">
            <div class="metric-value" id="slot-total">0</div>
            <div class="metric-label">TOTAL</div>
          </div>

          <div class="metric">
            <div class="metric-value" id="slot-declared">0</div>
            <div class="metric-label">DECLARED</div>
          </div>

        </div>

        <div class="slot-divider"></div>

        <div class="slot-row slot-meta">
            STATUS: <span id="slot-status">-</span>
            START_GC: <span id="slot-start-gc">-</span>
            END_GC: <span id="slot-end-gc">-</span>
            DURATION: <span id="slot-duration">-</span>
        </div>
      </div>

      <div class="controls">
        <button id="btn_start">Reconnect</button>
        <button id="btn_screenshot" class="secondary">Screenshot</button>
        <button id="btn_quit" class="secondary">Quit</button>
      </div>

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
    const slotDuration = document.getElementById('slot-duration');

    function formatTime(ts) {
      if (!ts) return '-';
      const d = new Date(ts);
      return d.toLocaleString();
    }
    
    function formatDuration(ms) {
      if (!Number.isFinite(ms) || ms <= 0) return '-';
      const s = Math.floor(ms / 1000);
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      const sec = s % 60;
      return `${h}h ${m}m ${sec}s`;
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
      
      slotTotal.classList.remove('slot-mismatch');
      slotDeclared.classList.remove('slot-mismatch');

      const declaredNum = Number(slot.declared_count);
      const slotCountNum = Number(slot.slot_count);
      const hasComparableCounts =
        Number.isFinite(declaredNum) && Number.isFinite(slotCountNum);
      if (hasComparableCounts && declaredNum !== slotCountNum) {
        slotTotal.classList.add('slot-mismatch');
        slotDeclared.classList.add('slot-mismatch');
      }

      slotStatus.textContent = isActive ? (slot.status ?? 'ACTIVE') : 'INACTIVE';

      slotStatus.classList.remove('slot-status-active', 'slot-status-inactive');
      slotStatus.classList.add(isActive ? 'slot-status-active' : 'slot-status-inactive');

      slotStartGc.textContent    = slot.start_global_count ?? '-';
      slotEndGc.textContent      = slot.end_global_count ?? '-';
      
      if (slot.start_time && slotDuration) {
        const startMs = new Date(slot.start_time).getTime();
        const endMs = isActive
          ? Date.now()
          : new Date(resolvedEnd ?? Date.now()).getTime();
        slotDuration.textContent = formatDuration(endMs - startMs);
      } else if (slotDuration) {
        slotDuration.textContent = '-';
      }
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

    setInterval(() => {
      try {
        if (lastSlotSnapshot) {
          renderSlotCard(lastSlotSnapshot, slotStatus.classList.contains('slot-status-active'));
        }
      } catch (e) {
        console.warn('slot duration refresh failed', e);
      }
    }, 1000);
  </script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────────────────────
def build_dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard — Deonar AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js" defer></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:16px}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
     background:#0f0f0f;color:#e0e0e0;min-height:100vh}

/* ── NAV ── */
nav{background:#1b1b1b;border-bottom:1px solid #272727;padding:10px 20px;
    display:flex;align-items:center;gap:10px;flex-wrap:wrap;
    position:sticky;top:0;z-index:100;box-shadow:0 2px 8px #0008}
.brand{font-weight:700;font-size:1rem;color:#1ddf8b;margin-right:auto;
       display:flex;align-items:center;gap:6px}
nav a{color:#888;text-decoration:none;font-size:.82rem;padding:5px 11px;
      border-radius:5px;transition:.15s}
nav a:hover{background:#252525;color:#ddd}
nav a.active{background:#1ddf8b22;color:#1ddf8b;font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;background:#555;display:inline-block;transition:.3s}
.dot.on{background:#1ddf8b;box-shadow:0 0 6px #1ddf8b88;animation:pdot 1.8s ease-in-out infinite}
.dot.err{background:#ff5c5c}
@keyframes pdot{0%,100%{opacity:1}50%{opacity:.4}}
.live-time{font-size:.72rem;color:#555}

/* ── LAYOUT ── */
.main{padding:16px;max-width:1400px;margin:0 auto}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:14px}
.charts-grid{display:grid;grid-template-columns:1fr 320px;gap:12px;margin-bottom:14px}
.bottom-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:900px){.charts-grid{grid-template-columns:1fr}.bottom-grid{grid-template-columns:1fr}}
@media(max-width:480px){.stat-grid{grid-template-columns:1fr 1fr}}

/* ── CARDS ── */
.card{background:#1a1a1a;border:1px solid #262626;border-radius:10px;padding:14px 16px}
.card-title{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:#666;margin-bottom:10px}

/* ── STAT CARDS ── */
.stat-card{border-left:3px solid;transition:.2s}
.stat-card:hover{transform:translateY(-1px);box-shadow:0 4px 16px #0005}
.stat-val{font-size:2.4rem;font-weight:700;line-height:1;letter-spacing:-.02em}
.stat-label{font-size:.7rem;color:#888;margin-top:4px;text-transform:uppercase;letter-spacing:.06em}
.stat-delta{font-size:.7rem;margin-top:5px;opacity:.7}
.c-total{border-color:#e0e0e0;color:#e0e0e0}
.c-up   {border-color:#1ddf8b;color:#1ddf8b}
.c-down {border-color:#ff5c5c;color:#ff5c5c}
.c-fps  {border-color:#2d6cdf;color:#2d6cdf}

/* ── CHART CARD ── */
.chart-wrap{position:relative;width:100%;height:220px}

/* ── HEALTH ── */
.thread-list{display:flex;flex-direction:column;gap:7px}
.thread-row{display:flex;align-items:center;justify-content:space-between;
            font-size:.82rem}
.t-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.t-alive{background:#1ddf8b;box-shadow:0 0 5px #1ddf8b66;animation:pdot 2s ease-in-out infinite}
.t-dead{background:#ff5c5c}
.t-na{background:#555}
.t-name{flex:1;margin:0 8px;color:#aaa}
.t-status{font-size:.7rem}

/* ── QUEUE BARS ── */
.q-row{margin-bottom:9px}
.q-label{display:flex;justify-content:space-between;font-size:.75rem;color:#aaa;margin-bottom:4px}
.q-bar{height:8px;background:#222;border-radius:4px;overflow:hidden}
.q-fill{height:100%;border-radius:4px;transition:width .5s ease}

/* ── SLOT MINI ── */
.slot-mini{font-size:.82rem;line-height:1.7}
.slot-mini .sm-id{font-weight:700;color:#1ddf8b;font-size:.9rem}
.slot-mini .sm-row{display:flex;gap:16px;flex-wrap:wrap}
.slot-mini .sm-val{font-weight:700}
.slot-progress{height:6px;background:#222;border-radius:3px;margin:8px 0 4px;overflow:hidden}
.slot-progress-fill{height:100%;background:linear-gradient(90deg,#2d6cdf,#1ddf8b);
                    border-radius:3px;transition:width .6s ease}
.no-slot{color:#444;font-size:.82rem;text-align:center;padding:16px 0}

/* ── OFFLINE BANNER ── */
.offline-banner{display:none;background:#2a1010;border:1px solid #ff5c5c44;
                border-radius:6px;padding:8px 12px;font-size:.78rem;color:#ff9999;
                margin-bottom:12px;text-align:center}
.offline-banner.show{display:block}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <span class="brand">🐐 DEONAR AI</span>
  <a href="/">Stream</a>
  <a href="/dashboard" class="active">Dashboard</a>
  <a href="/slots">Slots</a>
  <span class="dot" id="liveDot"></span>
  <span class="live-time" id="liveTime"></span>
</nav>

<div class="main" x-data="dashboard()" x-init="init()">

  <!-- Offline banner -->
  <div class="offline-banner" :class="{show: offline}" id="offlineBanner">
    ⚠ Pipeline offline — showing last confirmed data
  </div>

  <!-- STAT CARDS -->
  <div class="stat-grid">
    <div class="card stat-card c-total">
      <div class="card-title">Total Counted</div>
      <div class="stat-val c-total" x-text="fmt(s.total)">—</div>
      <div class="stat-label">animals passed</div>
      <div class="stat-delta" x-text="deltaText(s.total, prev.total)"></div>
    </div>
    <div class="card stat-card c-up">
      <div class="card-title">Going In ↑</div>
      <div class="stat-val c-up" x-text="fmt(s.up)">—</div>
      <div class="stat-label" x-text="pct(s.up, s.total) + ' of total'">—</div>
      <div class="stat-delta" x-text="deltaText(s.up, prev.up)"></div>
    </div>
    <div class="card stat-card c-down">
      <div class="card-title">Going Out ↓</div>
      <div class="stat-val c-down" x-text="fmt(s.down)">—</div>
      <div class="stat-label" x-text="pct(s.down, s.total) + ' of total'">—</div>
      <div class="stat-delta" x-text="deltaText(s.down, prev.down)"></div>
    </div>
    <div class="card stat-card c-fps">
      <div class="card-title">Inference FPS</div>
      <div class="stat-val c-fps" x-text="s.infer_fps ?? '—'">—</div>
      <div class="stat-label" x-text="'e2e  ' + (s.e2e_fps ?? '—') + ' fps'">—</div>
      <div class="stat-delta" style="color:#555" x-text="offline ? 'offline' : 'live'"></div>
    </div>
  </div>

  <!-- CHARTS ROW -->
  <div class="charts-grid">
    <!-- Count over time -->
    <div class="card">
      <div class="card-title">Count Over Time (cumulative — last 2 min)</div>
      <div class="chart-wrap">
        <canvas id="countChart"></canvas>
      </div>
    </div>
    <!-- Pipeline health + queues -->
    <div style="display:flex;flex-direction:column;gap:12px">
      <div class="card" style="flex:1">
        <div class="card-title">Pipeline Health</div>
        <div class="thread-list">
          <template x-for="t in threadList()" :key="t.name">
            <div class="thread-row">
              <span class="t-dot" :class="t.cls"></span>
              <span class="t-name" x-text="t.name"></span>
              <span class="t-status" :style="'color:' + t.color" x-text="t.label"></span>
            </div>
          </template>
        </div>
      </div>
      <div class="card" style="flex:1">
        <div class="card-title">Queue Utilization</div>
        <template x-for="q in queueList()" :key="q.name">
          <div class="q-row">
            <div class="q-label">
              <span x-text="q.name"></span>
              <span x-text="q.fill + ' / ' + q.max"></span>
            </div>
            <div class="q-bar">
              <div class="q-fill" :style="'width:' + q.pct + '%;background:' + q.color"></div>
            </div>
          </div>
        </template>
      </div>
    </div>
  </div>

  <!-- BOTTOM ROW -->
  <div class="bottom-grid">
    <!-- Direction donut -->
    <div class="card">
      <div class="card-title">Direction Breakdown</div>
      <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
        <div style="width:140px;height:140px;flex-shrink:0">
          <canvas id="donutChart"></canvas>
        </div>
        <div style="font-size:.85rem;line-height:2">
          <div><span style="color:#1ddf8b;font-weight:700" x-text="fmt(s.up)"></span>
               <span style="color:#666"> ↑ Going In</span></div>
          <div><span style="color:#ff5c5c;font-weight:700" x-text="fmt(s.down)"></span>
               <span style="color:#666"> ↓ Going Out</span></div>
          <div style="margin-top:6px;color:#555;font-size:.75rem"
               x-text="s.total ? 'Ratio  ' + pct(s.up,s.total) + ' in / ' + pct(s.down,s.total) + ' out' : 'No data yet'"></div>
        </div>
      </div>
    </div>
    <!-- Active slot mini -->
    <div class="card">
      <div class="card-title">Active Slot</div>
      <template x-if="slotActive && slotData">
        <div class="slot-mini">
          <div class="sm-id" x-text="'SLOT: ' + slotData.slot_id"></div>
          <div x-text="slotData.vendor_name || slotData.vendor_id || '—'"></div>
          <div class="sm-row" style="margin-top:6px">
            <span>↑ <span class="sm-val" style="color:#1ddf8b"
              x-text="slotData.direction_breakdown?.up ?? 0"></span></span>
            <span>↓ <span class="sm-val" style="color:#ff5c5c"
              x-text="slotData.direction_breakdown?.down ?? 0"></span></span>
            <span>Total <span class="sm-val" x-text="slotData.slot_count ?? 0"></span></span>
            <template x-if="slotData.declared_count">
              <span>/ <span x-text="slotData.declared_count"></span> declared</span>
            </template>
          </div>
          <template x-if="slotData.declared_count">
            <div>
              <div class="slot-progress">
                <div class="slot-progress-fill"
                  :style="'width:' + Math.min(100,Math.round((slotData.slot_count||0)/slotData.declared_count*100)) + '%'">
                </div>
              </div>
              <div style="font-size:.7rem;color:#555"
                x-text="Math.min(100,Math.round((slotData.slot_count||0)/slotData.declared_count*100)) + '% complete'">
              </div>
            </div>
          </template>
          <div style="margin-top:6px;font-size:.75rem;color:#555">
            <a href="/slots" style="color:#2d6cdf;text-decoration:none">→ Manage slots</a>
          </div>
        </div>
      </template>
      <template x-if="!slotActive">
        <div class="no-slot">
          No active slot<br>
          <a href="/slots" style="color:#2d6cdf;font-size:.78rem;text-decoration:none">
            → Start a session
          </a>
        </div>
      </template>
    </div>
  </div>

</div><!-- /main -->

<script>
// ── Shared time ticker ────────────────────────────────────────────────────────
setInterval(()=>{
  var el=document.getElementById('liveTime');
  if(el) el.textContent=new Date().toLocaleTimeString();
},1000);
document.getElementById('liveTime').textContent=new Date().toLocaleTimeString();

// ── Alpine component ──────────────────────────────────────────────────────────
function dashboard(){
  return {
    s: {up:0, down:0, total:0, infer_fps:0, e2e_fps:0, threads:{}, queues:{}},
    prev: {up:0, down:0, total:0},
    slotActive: false,
    slotData: null,
    offline: false,
    _failCount: 0,
    _countChart: null,
    _donutChart: null,
    _history: [],   // [{ts, total, up, down}] last 60 points

    fmt(v){ return (v==null||v===undefined) ? '—' : Number(v).toLocaleString(); },
    pct(a,b){ return b ? Math.round(a/b*100)+'%' : '0%'; },
    deltaText(cur, prv){
      if(cur==null||prv==null) return '';
      var d=cur-prv; if(d===0) return ''; return (d>0?'+':'')+d+' since last poll';
    },

    threadList(){
      var t=this.s.threads||{};
      var names=['capture','pacer','infer','display','slot_api'];
      var labels={capture:'Capture',pacer:'Pacing',infer:'Inference',
                  display:'Display',slot_api:'Slot API'};
      return names.map(n=>{
        var v=t[n];
        if(v===null||v===undefined) return {name:labels[n],cls:'t-na',color:'#555',label:'N/A'};
        return v
          ? {name:labels[n],cls:'t-alive',color:'#1ddf8b',label:'ALIVE'}
          : {name:labels[n],cls:'t-dead', color:'#ff5c5c',label:'DEAD'};
      });
    },

    queueList(){
      var q=this.s.queues||{};
      var items=[
        {key:'cap',  name:'Capture Queue'},
        {key:'pacing',name:'Pacing Queue'},
        {key:'result',name:'Result Queue'},
      ];
      return items.map(i=>{
        var info=q[i.key]||{fill:0,max:0};
        var fill=info.fill||0, max=info.max||1;
        var pct=Math.round(fill/max*100);
        var color=pct<50?'#1ddf8b':pct<80?'#ffb347':'#ff5c5c';
        return {name:i.name,fill,max,pct,color};
      });
    },

    async pollStatus(){
      try{
        var r=await fetch('/status',{signal:AbortSignal.timeout(3000)});
        if(!r.ok) throw new Error('http '+r.status);
        var data=await r.json();
        this.prev={up:this.s.up,down:this.s.down,total:this.s.total};
        this.s=data;
        this._failCount=0;
        this.offline=false;
        document.getElementById('liveDot').className='dot on';
        // append to history
        this._history.push({ts:Date.now(),total:data.total||0,up:data.up||0,down:data.down||0});
        if(this._history.length>60) this._history.shift();
        this.updateCountChart();
        this.updateDonutChart();
      }catch(e){
        this._failCount++;
        if(this._failCount>=2){
          this.offline=true;
          document.getElementById('liveDot').className='dot err';
        }
      }
    },

    async pollSlot(){
      try{
        var r=await fetch('/slot-state',{signal:AbortSignal.timeout(3000)});
        if(!r.ok) return;
        var d=await r.json();
        this.slotActive=!!d.active;
        this.slotData=d.slot||null;
      }catch(e){}
    },

    initCountChart(){
      var ctx=document.getElementById('countChart').getContext('2d');
      this._countChart=new Chart(ctx,{
        type:'line',
        data:{
          labels:[],
          datasets:[
            {label:'Total',data:[],borderColor:'#e0e0e0',backgroundColor:'#e0e0e010',
             borderWidth:2,tension:.3,pointRadius:0,fill:true},
            {label:'In ↑',data:[],borderColor:'#1ddf8b',backgroundColor:'transparent',
             borderWidth:1.5,tension:.3,pointRadius:0},
            {label:'Out ↓',data:[],borderColor:'#ff5c5c',backgroundColor:'transparent',
             borderWidth:1.5,tension:.3,pointRadius:0},
          ]
        },
        options:{
          responsive:true,maintainAspectRatio:false,
          animation:{duration:400},
          scales:{
            x:{ticks:{color:'#444',maxTicksLimit:6,font:{size:10}},grid:{color:'#1f1f1f'}},
            y:{ticks:{color:'#666',font:{size:10}},grid:{color:'#1f1f1f'},beginAtZero:true}
          },
          plugins:{legend:{labels:{color:'#888',font:{size:11},boxWidth:12}}}
        }
      });
    },

    initDonutChart(){
      var ctx=document.getElementById('donutChart').getContext('2d');
      this._donutChart=new Chart(ctx,{
        type:'doughnut',
        data:{
          labels:['In ↑','Out ↓'],
          datasets:[{data:[0,0],backgroundColor:['#1ddf8b','#ff5c5c'],
                     borderColor:'#1a1a1a',borderWidth:2,hoverOffset:4}]
        },
        options:{
          responsive:true,maintainAspectRatio:false,
          cutout:'72%',
          animation:{duration:400},
          plugins:{legend:{display:false}}
        }
      });
    },

    updateCountChart(){
      if(!this._countChart) return;
      var labels=this._history.map(h=>new Date(h.ts).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}));
      this._countChart.data.labels=labels;
      this._countChart.data.datasets[0].data=this._history.map(h=>h.total);
      this._countChart.data.datasets[1].data=this._history.map(h=>h.up);
      this._countChart.data.datasets[2].data=this._history.map(h=>h.down);
      this._countChart.update('none');
    },

    updateDonutChart(){
      if(!this._donutChart) return;
      this._donutChart.data.datasets[0].data=[this.s.up||0, this.s.down||0];
      this._donutChart.update('none');
    },

    init(){
      this.initCountChart();
      this.initDonutChart();
      this.pollStatus();
      this.pollSlot();
      setInterval(()=>this.pollStatus(), 2000);
      setInterval(()=>this.pollSlot(), 3000);
    }
  };
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# SLOTS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def build_slots_html(slot_api_port: int = 8090) -> str:
    api = f"http://localhost:{slot_api_port}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Slots — Deonar AI</title>
<script src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js" defer></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
     background:#0f0f0f;color:#e0e0e0;min-height:100vh}}
nav{{background:#1b1b1b;border-bottom:1px solid #272727;padding:10px 20px;
    display:flex;align-items:center;gap:10px;flex-wrap:wrap;
    position:sticky;top:0;z-index:100;box-shadow:0 2px 8px #0008}}
.brand{{font-weight:700;font-size:1rem;color:#1ddf8b;margin-right:auto}}
nav a{{color:#888;text-decoration:none;font-size:.82rem;padding:5px 11px;
      border-radius:5px;transition:.15s}}
nav a:hover{{background:#252525;color:#ddd}}
nav a.active{{background:#2d6cdf22;color:#2d6cdf;font-weight:600}}
.dot{{width:8px;height:8px;border-radius:50%;background:#555;display:inline-block;transition:.3s}}
.dot.on{{background:#1ddf8b;box-shadow:0 0 6px #1ddf8b88;animation:pdot 1.8s ease-in-out infinite}}
@keyframes pdot{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.live-time{{font-size:.72rem;color:#555}}
.main{{padding:16px;max-width:1200px;margin:0 auto}}
.top-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
@media(max-width:700px){{.top-grid{{grid-template-columns:1fr}}}}
.card{{background:#1a1a1a;border:1px solid #262626;border-radius:10px;padding:16px}}
.card-title{{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
             color:#666;margin-bottom:12px}}
/* ACTIVE SLOT */
.slot-card{{border-left:3px solid #1ddf8b}}
.slot-card.inactive{{border-color:#444}}
.slot-id{{font-size:1.1rem;font-weight:700;color:#1ddf8b;margin-bottom:2px}}
.slot-vendor{{font-size:.85rem;color:#aaa;margin-bottom:10px}}
.counts-row{{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0}}
.count-item{{text-align:center}}
.count-val{{font-size:1.6rem;font-weight:700;line-height:1}}
.count-label{{font-size:.65rem;color:#666;text-transform:uppercase;letter-spacing:.05em}}
.c-up{{color:#1ddf8b}}
.c-down{{color:#ff5c5c}}
.c-tot{{color:#e0e0e0}}
.c-decl{{color:#ffb347}}
.prog-bar{{height:7px;background:#222;border-radius:4px;margin:10px 0 4px;overflow:hidden}}
.prog-fill{{height:100%;background:linear-gradient(90deg,#2d6cdf,#1ddf8b);
           border-radius:4px;transition:width .6s ease}}
.duration{{font-size:.75rem;color:#555;margin-top:6px}}
.no-slot{{color:#444;text-align:center;padding:20px 0;font-size:.85rem}}
/* STATUS BADGE */
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;
        font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em}}
.badge-active{{background:#1ddf8b22;color:#1ddf8b;border:1px solid #1ddf8b44}}
.badge-inactive{{background:#33333388;color:#666;border:1px solid #333}}
/* FORM */
.form-group{{margin-bottom:11px}}
label{{display:block;font-size:.72rem;color:#888;text-transform:uppercase;
       letter-spacing:.05em;margin-bottom:5px}}
input{{width:100%;background:#131313;border:1px solid #303030;border-radius:6px;
       padding:8px 10px;color:#e0e0e0;font-size:.875rem;transition:.2s;outline:none}}
input:focus{{border-color:#2d6cdf;box-shadow:0 0 0 2px #2d6cdf22}}
input::placeholder{{color:#3a3a3a}}
input:disabled{{opacity:.5;cursor:not-allowed}}
.btn{{width:100%;padding:10px;border:none;border-radius:7px;font-size:.875rem;
      font-weight:600;cursor:pointer;transition:.2s;margin-top:4px}}
.btn:disabled{{opacity:.45;cursor:not-allowed}}
.btn-start{{background:#1ddf8b;color:#0a0a0a}}
.btn-start:not(:disabled):hover{{background:#1af7a0}}
.btn-stop{{background:#ff5c5c22;color:#ff5c5c;border:1px solid #ff5c5c44;margin-top:8px}}
.btn-stop:not(:disabled):hover{{background:#ff5c5c33}}
.msg{{font-size:.78rem;padding:8px 10px;border-radius:5px;margin-top:8px;display:none}}
.msg.show{{display:block}}
.msg.ok{{background:#1ddf8b18;color:#1ddf8b;border:1px solid #1ddf8b33}}
.msg.err{{background:#ff5c5c18;color:#ff9999;border:1px solid #ff5c5c33}}
/* HISTORY TABLE */
.tbl-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.8rem}}
th{{background:#141414;color:#666;text-transform:uppercase;font-size:.68rem;
    letter-spacing:.06em;padding:8px 10px;text-align:left;border-bottom:1px solid #222}}
td{{padding:8px 10px;border-bottom:1px solid #1d1d1d;vertical-align:middle}}
tr:hover td{{background:#1e1e1e}}
.st-ok{{color:#1ddf8b}}
.st-mm{{color:#ffb347}}
.st-ab{{color:#ff5c5c}}
.st-badge{{display:inline-block;padding:2px 7px;border-radius:3px;font-size:.68rem;font-weight:600}}
.st-ok-b{{background:#1ddf8b18;color:#1ddf8b}}
.st-mm-b{{background:#ffb34718;color:#ffb347}}
.st-ab-b{{background:#ff5c5c18;color:#ff5c5c}}
.match-yes{{color:#1ddf8b;font-size:.85rem}}
.match-no{{color:#ffb347;font-size:.85rem}}
.empty-row{{text-align:center;color:#444;padding:20px}}
</style>
</head>
<body>
<nav>
  <span class="brand">🐐 DEONAR AI</span>
  <a href="/">Stream</a>
  <a href="/dashboard">Dashboard</a>
  <a href="/slots" class="active">Slots</a>
  <span class="dot" id="liveDot"></span>
  <span class="live-time" id="liveTime"></span>
</nav>

<div class="main" x-data="slotsApp()" x-init="init()">

  <div class="top-grid">

    <!-- LEFT: Active slot + controls -->
    <div style="display:flex;flex-direction:column;gap:12px">

      <!-- Active slot card -->
      <div class="card slot-card" :class="{{inactive: !slotActive}}">
        <div class="card-title">
          Active Session
          <span class="badge" :class="slotActive ? 'badge-active' : 'badge-inactive'"
                x-text="slotActive ? 'ACTIVE' : 'IDLE'" style="margin-left:8px"></span>
        </div>
        <template x-if="slotActive && slotData">
          <div>
            <div class="slot-id" x-text="'Slot: ' + slotData.slot_id"></div>
            <div class="slot-vendor" x-text="slotData.vendor_name || slotData.vendor_id || 'Unknown vendor'"></div>
            <div class="counts-row">
              <div class="count-item">
                <div class="count-val c-tot" x-text="slotData.slot_count ?? 0"></div>
                <div class="count-label">Counted</div>
              </div>
              <div class="count-item">
                <div class="count-val c-decl" x-text="slotData.declared_count ?? '—'"></div>
                <div class="count-label">Declared</div>
              </div>
              <div class="count-item">
                <div class="count-val c-up" x-text="slotData.direction_breakdown?.up ?? 0"></div>
                <div class="count-label">In ↑</div>
              </div>
              <div class="count-item">
                <div class="count-val c-down" x-text="slotData.direction_breakdown?.down ?? 0"></div>
                <div class="count-label">Out ↓</div>
              </div>
            </div>
            <template x-if="slotData.declared_count">
              <div>
                <div class="prog-bar">
                  <div class="prog-fill"
                    :style="'width:'+Math.min(100,Math.round((slotData.slot_count||0)/slotData.declared_count*100))+'%'">
                  </div>
                </div>
                <div class="duration"
                  x-text="Math.min(100,Math.round((slotData.slot_count||0)/slotData.declared_count*100))+'% complete'">
                </div>
              </div>
            </template>
            <div class="duration" x-text="'Started: ' + fmtTime(slotData.start_time)"></div>
            <div class="duration" x-text="'Duration: ' + liveDuration(slotData.start_time)"></div>
          </div>
        </template>
        <template x-if="!slotActive">
          <div class="no-slot">No active session — start one below</div>
        </template>
      </div>

      <!-- Stop button -->
      <button class="btn btn-stop" :disabled="!slotActive" @click="stopSlot()"
              x-text="stopping ? 'Stopping…' : '■  Stop Session'"></button>
      <div class="msg" :class="{{show: stopMsg, ok: stopOk, err: !stopOk}}"
           x-text="stopMsg"></div>

    </div><!-- /left -->

    <!-- RIGHT: Start form -->
    <div class="card">
      <div class="card-title">Start New Session</div>
      <div class="form-group">
        <label>Slot ID *</label>
        <input x-model="form.slot_id" placeholder="e.g. VND-001" :disabled="slotActive || starting">
      </div>
      <div class="form-group">
        <label>Vendor ID</label>
        <input x-model="form.vendor_id" placeholder="e.g. V001" :disabled="slotActive || starting">
      </div>
      <div class="form-group">
        <label>Vendor Name</label>
        <input x-model="form.vendor_name" placeholder="e.g. Ahmed Traders" :disabled="slotActive || starting">
      </div>
      <div class="form-group">
        <label>Declared Count</label>
        <input x-model="form.declared_count" type="number" min="1" placeholder="e.g. 50"
               :disabled="slotActive || starting">
      </div>
      <div class="form-group">
        <label>Started By</label>
        <input
            x-model="form.started_by"
            placeholder="e.g. Ubada"
            :disabled="slotActive || starting">
      </div>
      <button class="btn btn-start" :disabled="slotActive || starting || !form.slot_id.trim()"
              @click="startSlot()"
              x-text="starting ? 'Starting…' : '▶  Start Session'"></button>
      <div class="msg" :class="{{show: startMsg, ok: startOk, err: !startOk}}"
           x-text="startMsg"></div>
    </div>

  </div><!-- /top-grid -->

  <!-- HISTORY TABLE -->
  <div class="card">
    <div class="card-title">Session History
      <span style="color:#333;margin-left:6px" x-text="history.length ? '(' + history.length + ' sessions)' : ''"></span>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Slot ID</th>
            <th>Vendor</th>
            <th>Counted</th>
            <th>Declared</th>
            <th>Match</th>
            <th>Status</th>
            <th>Duration</th>
          </tr>
        </thead>
        <tbody>
          <template x-if="history.length === 0">
            <tr><td colspan="7" class="empty-row">No completed sessions yet this run</td></tr>
          </template>
          <template x-for="row in history" :key="row.slot_id + (row.start_time||'')">
            <tr>
              <td style="font-weight:600;color:#ccc" x-text="row.slot_id || '—'"></td>
              <td style="color:#aaa" x-text="row.vendor_name || row.vendor_id || '—'"></td>
              <td style="font-weight:600" :class="matchColor(row)" x-text="row.slot_count ?? '—'"></td>
              <td style="color:#aaa" x-text="row.declared_count ?? '—'"></td>
              <td x-html="matchIcon(row)"></td>
              <td x-html="statusBadge(row)"></td>
              <td style="color:#555;font-size:.75rem" x-text="calcDuration(row)"></td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>
  </div>

</div><!-- /main -->

<script>
setInterval(()=>{{
  var el=document.getElementById('liveTime');
  if(el) el.textContent=new Date().toLocaleTimeString();
}},1000);
document.getElementById('liveTime').textContent=new Date().toLocaleTimeString();

const SLOT_API='{api}';

function slotsApp(){{
  return {{
    slotActive:false, slotData:null,
    history:[],
    form:{{slot_id:'',vendor_id:'',vendor_name:'',declared_count:'',started_by:''}},
    starting:false, stopping:false,
    startMsg:'', startOk:true,
    stopMsg:'',  stopOk:true,
    
    currentOperator:'',

    fmtTime(ts){{
      if(!ts) return '—';
      try{{ return new Date(ts).toLocaleTimeString(); }}catch(e){{return ts;}}
    }},
    liveDuration(start){{
      if(!start) return '—';
      try{{
        var ms=Date.now()-new Date(start).getTime();
        if(ms<0) return '—';
        var s=Math.floor(ms/1000), h=Math.floor(s/3600),
            m=Math.floor((s%3600)/60), sec=s%60;
        return (h?h+'h ':'')+m+'m '+sec+'s';
      }}catch(e){{return '—';}}
    }},
    calcDuration(row){{
      if(!row.start_time) return '—';
      var end=row.end_time? new Date(row.end_time).getTime() : Date.now();
      var ms=end-new Date(row.start_time).getTime();
      if(isNaN(ms)||ms<0) return '—';
      var s=Math.floor(ms/1000), m=Math.floor(s/60);
      return m+'m '+(s%60)+'s';
    }},
    matchColor(row){{
      if(row.declared_count==null) return '';
      return row.slot_count===row.declared_count?'st-ok':'st-mm';
    }},
    matchIcon(row){{
      if(row.declared_count==null) return '<span style="color:#555">—</span>';
      return row.slot_count===row.declared_count
        ? '<span class="match-yes">✓</span>'
        : '<span class="match-no">⚠</span>';
    }},
    statusBadge(row){{
      var st=(row.summary_status||row.status||'').toUpperCase();
      if(st==='OK')       return '<span class="st-badge st-ok-b">OK</span>';
      if(st==='MISMATCH') return '<span class="st-badge st-mm-b">MISMATCH</span>';
      if(st==='ABORTED')  return '<span class="st-badge st-ab-b">ABORTED</span>';
      return '<span style="color:#555">'+st+'</span>';
    }},

    async pollSlot(){{
      try{{
        var r=await fetch('/slot-state',{{signal:AbortSignal.timeout(3000)}});
        if(!r.ok) return;
        var d=await r.json();
        this.slotActive=!!d.active;
        this.slotData=d.slot||null;
        document.getElementById('liveDot').className='dot on';
      }}catch(e){{
        document.getElementById('liveDot').className='dot';
      }}
    }},

    async loadHistory(){{
      try{{
        var r=await fetch('/slot-history',{{signal:AbortSignal.timeout(3000)}});
        if(!r.ok) return;
        var d=await r.json();
        this.history=d.history||[];
      }}catch(e){{}}
    }},

    async startSlot(){{
      var sid=this.form.slot_id.trim();
      if(!sid){{ this.startMsg='Slot ID is required.'; this.startOk=false; return; }}
      this.starting=true; this.startMsg='';
      try{{
        var body={{slot_id:sid, started_by: this.form.started_by.trim()}};
        if(this.form.vendor_id.trim())  body.vendor_id=this.form.vendor_id.trim();
        if(this.form.vendor_name.trim()) body.vendor_name=this.form.vendor_name.trim();
        if(this.form.declared_count)    body.declared_count=parseInt(this.form.declared_count)||null;
        var r=await fetch(SLOT_API+'/api/slot/start',{{
          method:'POST',
          headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify(body),
          signal:AbortSignal.timeout(5000)
        }});
        var d=await r.json();
        if(r.ok||r.status===200){{
          this.startMsg='Session started successfully.';
          this.startOk=true;
          this.form={{slot_id:'',vendor_id:'',vendor_name:'',declared_count:'',started_by:''}};
          await this.pollSlot();
          this.currentOperator = body.started_by;
        }}else{{
          this.startMsg='Error: '+(d.detail||d.message||r.status);
          this.startOk=false;
        }}
      }}catch(e){{
        this.startMsg='Could not reach slot API. Is the system running?';
        this.startOk=false;
      }}
      this.starting=false;
      setTimeout(()=>{{this.startMsg='';}}, 5000);
    }},

    async stopSlot(){{
      if(!this.slotData) return;
      this.stopping=true; this.stopMsg='';
      try{{
        var r=await fetch(SLOT_API+'/api/slot/stop',{{
          method:'POST',
          headers:{{'Content-Type':'application/json'}},
          body: JSON.stringify({{
            slot_id: this.slotData.slot_id,
            stopped_by: this.currentOperator,
            reason: "Manual stop from dashboard"
          }}),
          signal:AbortSignal.timeout(5000)
        }});
        var d=await r.json();
        if(r.ok||r.status===200){{
          this.stopMsg='Session stopped.';
          this.stopOk=true;
          await this.pollSlot();
          await this.loadHistory();
          this.currentOperator = '';
        }}else{{
          this.stopMsg='Error: '+(d.detail||d.message||r.status);
          this.stopOk=false;
        }}
      }}catch(e){{
        this.stopMsg='Could not reach slot API. Is the system running?';
        this.stopOk=false;
      }}
      this.stopping=false;
      setTimeout(()=>{{this.stopMsg='';}}, 5000);
    }},

    init(){{
      this.pollSlot();
      this.loadHistory();
      setInterval(()=>this.pollSlot(),   3000);
      setInterval(()=>this.loadHistory(), 8000);
    }}
  }};
}}
</script>
</body>
</html>
"""

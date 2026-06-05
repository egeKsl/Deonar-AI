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
.bottom-grid{display:grid;grid-template-columns:1fr;gap:12px}
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
      var names=['capture','pacer','infer','display'];
      var labels={capture:'Capture',pacer:'Pacing',infer:'Inference',
                  display:'Display'};
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
      setInterval(()=>this.pollStatus(), 2000);
    }
  };
}
</script>
</body>
</html>
"""



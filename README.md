# 🐐 Goat Detection & Counting System

A production‑grade computer vision pipeline for **real‑time goat detection, tracking, and counting** on farm CCTV feeds and recorded video. Built for rugged, noisy environments, the system combines **YOLO detection**, **ByteTrack tracking**, and **multi‑mode counting** (line, dual‑line, zone) with strong logging and visual validation outputs.

---

## ✨ What This Project Delivers

- ✅ **Accurate counting** in real farm conditions
- ✅ **Live CCTV support** (RTSP / HTTP / webcam)
- ✅ **Offline batch processing** for recordings
- ✅ **Dual‑line counting** for robust direction validation
- ✅ **CSV audit trails** for verification and analytics
- ✅ **Annotated video** for evidence and debugging
- ✅ **WebRTC live preview** for remote monitoring

---

## 🎯 Real‑World Use Case (Chute Monitoring)

Goats pass through a chute with two common camera views:

- **Standing view (vertical)**
  - **Up**: bottom → top (entry → exit)
  - **Down**: top → bottom (exit → entry)

- **Slipping view (horizontal)**
  - **Up**: right → left (entry → exit)
  - **Down**: left → right (exit → entry)

This system is hardened to handle **jitter near the line**, **occlusion**, and **tracking instability** while still preserving direction correctness.

---

## 🧠 Counting Modes (Core Logic)

### 1) **Single‑Line Counting**
Counts when a track crosses a single line.
- Uses **side‑history confirmation** to avoid flicker
- Uses **cooldown frames** to prevent double counts

### 2) **Dual‑Line Counting (Recommended)**
Counts only when a track crosses **Line A → Line B** in a consistent direction.
- ✅ **Verify mode**: requires A then B within a time window
- ✅ **Recover mode**: counts on A, but can recover missed A with B
- ✅ **Directional hardening**: motion consistency validation

### 3) **Zone Counting**
Defines a rectangular region and counts entries/exits.
- Great for pens, gates, or enclosure monitoring
- Supports **entry/exit inference**

---

## 🏗 Pipeline Architecture

**Offline (Recorded Video)**
1. Read video
2. Crop ROI
3. Detect + track goats
4. Apply counting mode
5. Write CSV events & timeseries
6. Save annotated output video

**Online (Live CCTV / RTSP)**
1. Capture frames
2. Apply pacing (sync + autoskip)
3. Detect + track goats
4. Apply counting logic
5. Stream live annotated feed (WebRTC)
6. Write CSV audit logs in real time

---

## 🧩 Tech Stack

- **Language**: Python 3.10+
- **Detection**: YOLO (Ultralytics)
- **Tracking**: ByteTrack
- **Core CV**: OpenCV
- **Live Streaming**: WebRTC (aiortc)
- **Logging**: Rich + CSV output

---

## 📂 Project Structure

- `main.py` → entrypoint
- `src/app/` → runners (single‑threaded & multi‑threaded)
- `src/infer/` → model loading + tracking
- `src/counting/` → counting logic + state
- `src/display/` → drawing + WebRTC
- `src/viz` -> output video UI like HUD and countings animation
- `src/io/` → CSV + output writing
- `configs/config.yaml` → configuration

---

## ⚙️ Installation (Beginner Friendly)

### 1) Create virtual environment
```bash
python -m venv .venv
```

### 2) Activate it
**Windows PowerShell:**
```bash
.venv/Script/Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3) Install project (editable)
```bash
python -m pip install -e . -v
```

### 4) Run enhanced installer (recommended)
```bash
setup-installer --install-cuda-python --install-nvidia-ml --auto-detect-torch --always-progress --verbose --run-after python main.py
```

---

## 🔧 Configuration (configs/config.yaml)

### ✅ Required
- `paths.weights`: YOLO model file
- `paths.source`: video file or RTSP link

### ✅ ROI & Geometry
Adjust crop to focus only on chute:
```yaml
geometry:
  roi:
    xr: 0.0000
    yr: 0.1514
    wr: 1.0000
    hr: 0.8472
```

### ✅ Counting Mode
```yaml
counting:
  mode: line     # line | zone
```

### ✅ Dual‑Line (Robust)
```yaml
counting:
  dual:
    enabled: true
    mode: recover
    profile: slipping_view   # or standing_view
```

---

## 🧪 Dual‑Line Profiles (Automatic)

Profiles are auto‑applied when `counting.dual.profile` is set:

- `standing_view`
- `slipping_view`

Each profile adjusts:
- hysteresis frames
- non‑zero evidence threshold
- motion consistency checks

---

## 📊 Outputs (Audit‑Friendly)

### ✅ CSV Events
- One row per counted goat
- Includes timestamps + both frame indices

### ✅ CSV Timeseries
- Per‑second cumulative counts

### ✅ CSV Decisions
- Motion + geometry decisions for debug

### ✅ Annotated Video
- Frame overlay with IDs, lines, and counts

---

## 🛰 Live Preview (WebRTC)

Turn on in config:
```yaml
webrtc:
  enable: true
  host: "0.0.0.0"
  port: 8081
```

Visit in browser:
```
http://<host>:8081
```

---

## ✅ Common Usage

Run with YAML config:
```bash
python main.py
```

Switch dual profile:
```yaml
counting:
  dual:
    profile: standing_view
```

---

## 🚑 Troubleshooting

- **Wrong direction?** Check line orientation and `profile`.
- **Counts missing?** Lower motion thresholds.
- **Too many false counts?** Increase motion consistency.
- **RTSP stutters?** Enable autoskip in runtime.

---

## 📜 License
MIT License.

## 🙏 Credits
- Ultralytics YOLO
- ByteTrack
- Engineering & Integration by Ubada Ghavte

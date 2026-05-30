# 🐐 Goat Detection & Counting System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HuggingFace Models](https://img.shields.io/badge/🤗%20Models-ubada11%2Fgoat--detection--yolov11-yellow)](https://huggingface.co/ubada11/goat-detection-yolov11)
[![YOLOv11](https://img.shields.io/badge/Detection-YOLOv11%20%7C%20YOLOv12-orange)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-red)](https://github.com/ifzhang/ByteTrack)

A production-grade computer vision pipeline for **real-time goat detection, tracking, and counting** on farm CCTV feeds and recorded video. Built and deployed on-site at **Deonar abattoir, Mumbai** — the world's largest animal market — replacing manual head-counting with an automated AI system.

> **Built by:** [Ubada Ghawte](https://github.com/ubada11) · Final Year B.E. (Electronics & CS), Rizvi College of Engineering, Mumbai
> **R&D Partner:** MI Tradings & General Suppliers (BMC contractor)
> **Duration:** 12 months of active development (2024–2025)

---

## 🎬 Demo

> _Demo video coming soon — live RTSP pipeline with dual-line counting and WebRTC preview_

---

## ✨ What This System Does

- ✅ **Real-time goat counting** on live CCTV (RTSP) with < 5% miss rate on 180-goat sessions
- ✅ **Dual-line counting** with motion gating — eliminates false counts from jitter and occlusion
- ✅ **Slot management API** — per-vendor counting sessions with CSV audit trail
- ✅ **WebRTC live preview** — monitor annotated feed from any browser
- ✅ **Multi-threaded pipeline** — separate capture / pacing / inference / display threads
- ✅ **Offline batch processing** for recorded videos
- ✅ **Custom CUDA-aware installer** — one command setup on any GPU machine
- ✅ **Full audit outputs** — events CSV, timeseries CSV, decisions CSV, annotated video

---

## 🤗 Pre-trained Models

Models trained on a custom dataset of **~20,000 annotated goat images** across real farm conditions.
All four variants achieve **~99% mAP@50** on the validation set.

| Model | mAP@50 | mAP@50-95 | Size | Best For |
|---|---|---|---|---|
| YOLOv11-nano | 98.99% | 78.05% | ~5.4 MB | ✅ Production / live CCTV |
| YOLOv11-small | 99.05% | 79.00% | ~19 MB | High accuracy offline |
| YOLOv12-nano | 99.04% | 78.12% | ~5.4 MB | Research / comparison |
| YOLOv12-small | 99.09% | 79.43% | ~19 MB | Best accuracy overall |

**➡️ Download from HuggingFace:** [ubada11/goat-detection-yolov11](https://huggingface.co/ubada11/goat-detection-yolov11)

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="ubada11/goat-detection-yolov11",
    filename="goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt",
    local_dir="models/"
)
```

---

## 📦 Dataset

The training dataset (~20,000 images) was collected and annotated entirely by the project team over 2+ months of on-site work at Deonar abattoir.

- **Annotation tool:** CVAT (self-hosted)
- **Scene coverage:** top view, side view, inclined/vertical/horizontal chutes, night, dust, morning light, mixed animals, empty frames
- **Classes:** 1 (`goat`)

The dataset is **not publicly distributed** to protect the integrity of the work. It is available for legitimate research purposes on request.

📧 **Request access:** [ubadaghawte2005@gmail.com](mailto:ubadaghawte2005@gmail.com) · [LinkedIn](https://linkedin.com/in/ubada-ghawte)

---

## 🎯 Real-World Use Case

Goats pass through a chute single-file. The camera is fixed above or beside the chute. The system detects each goat, assigns it a unique track ID, and counts it exactly once as it crosses the counting line — in the correct direction.

Two supported camera orientations:

| View | "Up" direction | "Down" direction |
|---|---|---|
| Standing (vertical) | bottom → top | top → bottom |
| Slipping (horizontal) | right → left | left → right |

The system is hardened for **jitter near the line**, **occlusion**, and **tracking ID switches** through dual-line confirmation and motion intent analysis.

---

## 🧠 Counting Modes

### 1. Single-Line Counting
Counts when a track crosses a single line. Uses side-history confirmation (anti-flicker) and cooldown frames to prevent double counts.

### 2. Dual-Line Counting _(Recommended)_
Counts only when a track crosses **Line A → Line B** in a consistent direction. Two sub-modes:
- **Verify** — requires A then B crossing within a time window
- **Recover** — counts on A immediately; B recovers a missed A as a fallback

Both modes run a **motion intent analyzer** that checks displacement, direction consistency, and axis alignment before confirming any count.

### 3. Zone Counting
Define a rectangular region; count entries and exits. Useful for pen monitoring or wide gate views.

---

## 🏗 Pipeline Architecture

```
LIVE (RTSP / CCTV)
─────────────────────────────────────────────────────
  ThreadedVideoCapture  →  PacingController  →  InferenceWorker  →  DisplayWorker
  (frames from RTSP)       (sync + autoskip)   (YOLO + ByteTrack)  (count + draw + stream)
       ↓                                                                    ↓
  capture_queue                                                     WebRTC / OpenCV
                                                                    CSV audit logs
                                                                    SlotManager API

OFFLINE (recorded video)
─────────────────────────────────────────────────────
  cv2.VideoCapture  →  ROIStream  →  track_once()  →  counting logic  →  annotated .mp4
```

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Detection | YOLOv11 / YOLOv12 (Ultralytics) |
| Tracking | ByteTrack / BoT-SORT |
| Core CV | OpenCV |
| Live streaming | WebRTC (aiortc) |
| Slot API | FastAPI |
| Logging | Rich (console) + JSONL (file) |
| Installer | Custom CUDA-aware CLI tool |

---

## 📂 Project Structure

```
├── main.py                          # Entry point
├── configs/config.yaml              # All configuration (single source of truth)
├── models/                          # Model weights (download from HuggingFace)
├── src/
│   ├── app/                         # Pipeline runners (single & multi-threaded)
│   ├── capture/                     # Threaded RTSP capture
│   ├── runtime/                     # Frame pacing controller
│   ├── infer/                       # Model loading + YOLO inference
│   ├── counting/                    # Counting logic (line / dual-line / zone)
│   ├── display/                     # Drawing, WebRTC server, display worker
│   ├── slots/                       # Slot management + REST API
│   ├── viz/                         # HUD, animations, color management
│   ├── io/                          # CSV writers
│   ├── geometry/                    # ROI math, line building
│   ├── utils/                       # Logger, metrics, video recorder
│   ├── runtime_configs/             # Config loader + tracker YAML builder
│   └── setup_installer_enhanced/    # Custom CUDA/PyTorch installer CLI
└── utils/
    ├── fake_producer.bat            # Fake RTSP stream for testing
    └── upload_to_huggingface.py     # Model upload script
```

---

## ⚙️ Installation

### Option A — Enhanced Installer (recommended for GPU machines)

```bash
# 1. Clone
git clone https://github.com/ubada11/goat-detection-counting-pipeline
cd goat-detection-counting-pipeline

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# 3. Install project
python -m pip install -e . -v

# 4. Run the enhanced installer (auto-detects CUDA, installs correct PyTorch)
setup-installer --install-cuda-python --install-nvidia-ml --auto-detect-torch --always-progress --verbose

# 5. Download model weights
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ubada11/goat-detection-yolov11',
    filename='goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt',
    local_dir='models/'
)
"
```

### Option B — Manual (CPU or known CUDA version)

```bash
pip install ultralytics opencv-python aiortc fastapi uvicorn rich pyyaml huggingface_hub
```

---

## 🔧 Quick Start

### 1. Set your source and model in `configs/config.yaml`

```yaml
paths:
  weights: goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt
  source: rtsp://your-camera-ip/stream    # or path/to/video.mp4
```

### 2. Run

```bash
python main.py
```

### 3. View live feed (if WebRTC enabled)

```
http://localhost:8081
```

---

## 🔧 Key Configuration Options

```yaml
# ROI — crop to chute area only
geometry:
  roi:
    xr: 0.0    # left edge (0–1)
    yr: 0.15   # top edge (0–1)
    wr: 1.0    # width fraction
    hr: 0.85   # height fraction

# Counting mode
counting:
  mode: line          # line | zone
  dual:
    enabled: true
    mode: recover     # verify | recover
    profile: slipping_view  # slipping_view | standing_view

# Slot API (per-vendor counting sessions)
runtime:
  slots_enabled: true
  slot_api:
    host: "127.0.0.1"
    port: 8090
```

---

## 📊 Outputs

All outputs are organized under `outputs/runs/<run_id>/`:

| File | Contents |
|---|---|
| `events/goat_cross_events.csv` | One row per counted goat — timestamp, frame index, track ID, direction, centroid |
| `timeseries/goat_counts_timeseries.csv` | Per-second cumulative up/down/total counts |
| `decisions/goat_decisions.csv` | Per-frame motion + geometry decisions for debugging |
| `metrics/goat_metrics.csv` | End-to-end latency per frame (capture → inference → display) |
| `video/output.mp4` | Annotated output video |
| `slots/<slot_id>/` | Per-vendor slot CSV + summary JSON + video |

---

## 🛰 Slot Management API

Start/stop per-vendor counting sessions via REST:

```bash
# Start a slot
curl -X POST http://localhost:8090/api/slot/start \
  -H "Content-Type: application/json" \
  -d '{"slot_id": "vendor_001", "vendor_name": "Ali Traders", "declared_count": 50}'

# Stop a slot
curl -X POST http://localhost:8090/api/slot/stop \
  -H "Content-Type: application/json" \
  -d '{"slot_id": "vendor_001"}'

# Check active slot
curl http://localhost:8090/api/slot/active
```

---

## 🚑 Troubleshooting

| Symptom | Fix |
|---|---|
| Wrong direction counted | Check `profile` setting (`slipping_view` vs `standing_view`) |
| Counts missing | Lower `motion_min_displacement_px` or `min_nonzero_abs` |
| Too many false counts | Increase `motion_dir_consistency_ratio` |
| RTSP feed stuttering | Enable `autoskip: true` in runtime config |
| High latency | Reduce `imgsz` or switch to nano model |
| GPU not detected | Run `setup-installer --auto-detect-torch` to reinstall correct PyTorch |

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Credits

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — multi-object tracker
- [aiortc](https://github.com/aiortc/aiortc) — WebRTC implementation
- **Engineering, dataset, training & integration:** Ubada Ghawte
- **Team:** Adil, Raafe
- **Academic guide:** Prof. Farhan, Rizvi College of Engineering

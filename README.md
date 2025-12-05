# 🐐 Goat Detection & Counting System

An advanced computer vision pipeline for **real-time goat detection, tracking, and counting**.
Built with **YOLO detection + ByteTrack tracking + multi-mode counting (line, dual-line, zone)**, the system is designed for **high accuracy, persistence, and usability** in real farm conditions.

We didn’t just stop at raw detection — this project includes a full visualization & logging stack, with configurable drawing modes, robust CSV outputs, and professional-grade overlays.

---

## ✨ Features

### 🎯 Detection & Tracking

* **YOLO-based detection** with class filtering.
* **ByteTrack tracker** for persistent IDs, tuned for minimal ID switches.
* Adjustable **stride** for frame skipping control (default: `1` = every frame).

### 📊 Counting Modes

* **Single-Line Counting**

  * Count objects crossing a line (up/down).
  * Anti-flicker hysteresis (`min_side_frames`).
  * Cooldown frames to prevent double-counting.

* **Dual-Line Counting**

  * **Verify mode**: counts only when the same ID crosses A → B consistently.
  * **Recover mode**: recovers missed detections with Line B.
  * Configurable offset (`DUAL_OFFSET_PX`) and width (`DUAL_WIDTH_PX`).

* **Zone Counting**

  * Define a rectangular ROI zone.
  * Counts entries/exits across edges.
  * Supports backfill projection visualization for debugging.

### 🎨 Visualization

* **Ultra-Pretty Drawing Engine** (`viz/draw.py`)

  * Configurable modes: `bbox` | `centroid` | `auto`.
  * **Rounded translucent boxes** with pill-shaped ID chips.
  * **Minimal centroids** mode for crowded scenes.
  * Dynamic color hashing per track ID.
  * Automatic counted-ID recoloring (blue/persist/window modes).
  * **Readable text labels** with contrast-aware chips.

* **HUD overlays**

  * Live counts (Up / Down / Total).
  * Frame indices (source & ROI).
  * Mode indicator (Line / Dual / Zone).

* **Animations**

  * CrossAnimator for `+1` bursts when counts fire.
  * Pulsing rings and floating text.

### 📂 Outputs

* **CSV Events** (`csv_events.csv`)

  * Deduplicated per `track_id` (no double counting).
  * Logs: `timestamp_s, src_frame_idx, proc_frame_idx, track_id, direction, cx, cy`.

* **CSV Timeseries** (`csv_timeseries.csv`)

  * Per-second log of cumulative counts: `timestamp_s, up, down, total`.

* **Video Output**

  * Full annotated video saved with progress tracking.
  * HUD shows effective FPS, ETA, etc.

* **Screenshots**

  * Press `s` to save a snapshot into `/outputs/images/`.

### ⚡ Pipeline Architecture

* Modularized pipeline under `src/pipeline/`:

  * `runner.py` → orchestrator (`run()`).
  * `init.py` → ROI, dual-line, and zone initialization.
  * `counts.py` → line, dual-line, and zone counting logic.
  * `viz/draw.py` → drawing utilities (boxes, centroids, HUDs, animations).
  * `output/io.py` → CSV writing, video writer, progress bar.

### 🛠 Tech Stack

* **Language:** Python 3.10+
* **Core CV:** OpenCV
* **Detection:** YOLOv11 (custom-trained)
* **Tracking:** ByteTrack (configurable thresholds/buffer)
* **Data:** CSV event + timeseries logging
* **Visualization:** Custom OpenCV rendering with pretty overlays

---

## ⚙️ Configuration

All major settings are controlled via `.env` or CLI args.

**Example `.env`:**

```ini
# Counting
COUNT_MODE=line                 # line | dual | zone
COUNT_LINE_ROI=0.309,0.411,0.546,0.411

# Dual-line mode
DUAL_LINES_ENABLED=true
DUAL_MODE=verify                # verify | recover
DUAL_OFFSET_PX=24
DUAL_WIDTH_PX=120               # custom width for line B
DUAL_HYST_FRAMES=2
DUAL_WINDOW_FRAMES=30
DUAL_ID_LOCK_FRAMES=60

# Zone mode
ZONE_RECT=0.2,0.3,0.5,0.4       # relative [x,y,w,h]
ZONE_BORN_INSIDE=count_entry

# Detection/Tracking
WEIGHTS=weights/yolov8.pt
STRIDE=1
CONF=0.5
IOU=0.5

# Drawing
DRAW_MODE=auto                  # bbox | centroid | auto
CROWD_SWITCH=6
BOX_COLOR=0,255,0
COUNT_BOX_COLOR=255,0,0
COUNT_BOX_MODE=persist          # persist | window | off
```

---

## 🚀 Usage

1. **Clone repo and install requirements:**

   ```bash
   git clone https://github.com/yourname/goat-detection-ai
   cd goat-detection-ai
   pip install -r requirements.txt
   ```

2. **Place your video under `inputs/`** or use a live camera.

3. **Run:**

   ```bash
   python -m src.main \
     --source inputs/goats.mp4 \
     --csv-events outputs/events.csv \
     --csv-timeseries outputs/timeseries.csv \
     --weights weights/yolov8.pt
   ```

4. **Optional flags:**

   * `--dual-lines-enabled true` → enable dual-line mode.
   * `--count-mode zone` → switch to zone counting.
   * `--draw-mode centroid` → force centroid drawing.
   * `--stride 2` → skip every other frame.

---

## 🖼 Example Visuals

![IMAGE 01](outputs/images/test_20250830_005135_741760_f1397.jpg)
![IMAGE 02](outputs/images/test2_20250829_163214_006946_f3714.jpg)
![IMAGE 03](outputs/images/test2_20250829_163225_029370_f3759.jpg)
![IMAGE 04](outputs/images/test3_20250830_003507_182071_f1350.jpg)
![IMAGE 05](outputs/images/test3_20250830_003527_033105_f1421.jpg)
![IMAGE 06](outputs/images/test3_20250830_004436_819709_f1423.jpg)

---

## 🧑‍💻 Development Notes

* **Stride tradeoff:** higher stride = faster but less persistent IDs.
* **Occlusion robustness:** ByteTrack config tuned; DeepSORT/BoT-SORT can be swapped in later.
* **Persistence:** Dual-line verify ensures true counts even if IDs flicker.
* **Neatness:** Drawing layer separated into `viz/draw.py` with `PrettyDrawConfig`.

---

## 📌 Roadmap

* [ ] Add DeepSORT/BoT-SORT option for re-ID-based tracking.
* [ ] Export to dashboard (real-time counts + video).
* [ ] Web-based config editor.
* [ ] ONNX / TensorRT acceleration.

---

## 📜 License

MIT License © 2025 Your Name

---

## 💡 Credits

* **YOLOv8:** [Ultralytics](https://github.com/ultralytics/ultralytics)
* **ByteTrack:** [Yifu Zhang et al.](https://github.com/ifzhang/ByteTrack)
* **Design & Engineering:** Ubada Ghavte

---
# Goat Detection & Counting System — Complete Documentation

**Project:** Real-time livestock detection, tracking, and counting pipeline
**Author:** Ubada Ghawte · Rizvi College of Engineering, Mumbai
**R&D Partner:** MI Tradings & General Suppliers (BMC contractor)
**Duration:** 12 months (2024–2025)
**GitHub:** [goat-detection-counting-pipeline](https://github.com/ubada11/goat-detection-counting-pipeline)
**Models:** [ubada11/goat-detection-yolov11](https://huggingface.co/ubada11/goat-detection-yolov11)

---

## Table of Contents

### Part 1 — The Story
1. [The Beginning — An Unexpected Meeting](#1-the-beginning--an-unexpected-meeting)
2. [Visiting Deonar — Understanding the Real Problem](#2-visiting-deonar--understanding-the-real-problem)
3. [Six Months of Data — The Hardest Part](#3-six-months-of-data--the-hardest-part)
4. [Training — Learning What the Numbers Actually Mean](#4-training--learning-what-the-numbers-actually-mean)
5. [The Pipeline Battle — Where Real Engineering Began](#5-the-pipeline-battle--where-real-engineering-began)
6. [On-Site Testing — The Real Validation](#6-on-site-testing--the-real-validation)
7. [The Slot System — Thinking Past the Demo](#7-the-slot-system--thinking-past-the-demo)
8. [The End — And What Was Gained](#8-the-end--and-what-was-gained)

### Part 2 — System Architecture
9. [Big Picture](#9-big-picture)
10. [Why This Is More Than a YOLO Pipeline](#10-why-this-is-more-than-a-yolo-pipeline)
11. [Live Pipeline — Thread Architecture](#11-live-pipeline--thread-architecture)
12. [Component Deep Dives](#12-component-deep-dives)
13. [Offline Pipeline](#13-offline-pipeline)
14. [Coordinate System & ROI](#14-coordinate-system--roi)
15. [Configuration-Driven Design (Scalability)](#15-configuration-driven-design-scalability)
16. [Counting System Architecture](#16-counting-system-architecture)
17. [Slot Management System](#17-slot-management-system)
18. [Output System](#18-output-system)
19. [Metrics & Observability](#19-metrics--observability)
20. [Custom Installer](#20-custom-installer)

### Part 3 — Counting Logic Deep Dive
21. [The Core Problem](#21-the-core-problem)
22. [Counting Architecture Overview](#22-counting-architecture-overview)
23. [Centroid Computation](#23-centroid-computation)
24. [Side Sign Computation](#24-side-sign-computation)
25. [Side History Confirmation (Anti-Flicker)](#25-side-history-confirmation-anti-flicker)
26. [Single-Line Counting](#26-single-line-counting)
27. [Dual-Line Counting](#27-dual-line-counting)
28. [Phase 2 — Motion Intent Analyzer](#28-phase-2--motion-intent-analyzer)
29. [Zone Counting](#29-zone-counting)
30. [Anti-Flicker Parameters Reference](#30-anti-flicker-parameters-reference)
31. [Tuning Guide](#31-tuning-guide)
32. [Data Flow — From Pixel to CSV Row](#32-data-flow--from-pixel-to-csv-row)

### Part 4 — FAQ
33. [Frequently Asked Questions](#33-frequently-asked-questions)

---

# Part 1 — The Story

*How a final-year engineering student ended up building a real-time AI counting system for the world's largest animal market.*

---

## 1. The Beginning — An Unexpected Meeting

In May 2024, three final-year B.E. students from Rizvi College of Engineering in Mumbai — Ubada Ghawte, Adil, and Raafe — visited the office of MI Tradings & General Suppliers in Mahim. The visit wasn't about this project. They had come to pitch their college project: a flood management system with XAI-based flow prediction, built for BMC contractors.

The owner, Abdul Hamid, listened politely and said something that changed the direction of the next 12 months: *"These kinds of projects don't sustain. But if your team can build AI systems, I have something that actually needs to exist."*

He described the problem at **Deonar abattoir** — a 260-acre facility in Mumbai that handles over 80,000 animals per year. At the busiest times, hundreds of goats pass through a narrow chute in a matter of hours. Every single one needs to be counted — by direction, by vendor, with an audit trail. At the time, this was done manually. A person stood at the chute gate and clicked a counter.

The business case was clear: automate this with AI, get BMC approval, win the tender. The deal: no upfront payment, no salary, no guarantee. If the system works and gets approved, the tender money pays everyone. If it fails, nothing. Pure R&D risk.

They took the deal.

---

## 2. Visiting Deonar — Understanding the Real Problem

Before writing a single line of code, the team made multiple visits to Deonar to understand exactly what they were dealing with.

Deonar is not a clean, controlled environment. It is loud, dusty, chaotic. Animals move unpredictably. Lighting changes from harsh morning sun to deep shadow within the same hour. The chutes are narrow — sometimes multiple goats try to pass simultaneously. Cameras are fixed to walls or overhead rigs at angles that weren't designed for machine vision.

What they observed:
- Goats don't walk steadily through the chute. They hesitate, back up, push against each other.
- The same chute has two common camera views — a top-down "slipping" view where goats move horizontally, and a side-mounted "standing" view where they move vertically.
- Occlusion is frequent. One goat can completely hide another for several frames.
- The lighting at Deonar is difficult — direct sunlight, shadows, dust particles that the camera picks up as noise.
- Counting needs to be directional — you need to know how many went in vs. how many came back out.

This on-site understanding shaped every subsequent technical decision. The dual-line counting mode exists because of goats that hesitate and back up. The motion intent analyzer exists because of goats that move sideways without crossing. The two camera profiles exist because the team measured both views in person.

---

## 3. Six Months of Data — The Hardest Part

The team had the problem. Now they needed data.

Abdul Hamid provided CCTV recordings from the facility. The initial dataset was small — a few hours of footage, chopped into clips. They trained an early YOLOv8-nano model on about 1,000 annotated images. Results were poor: missed detections, hallucinations on background objects, poor performance in low-light frames.

A professional ML developer the team consulted gave direct advice: *"Your model performance is a dataset problem, not a model problem. You need 10x the data, and it needs to cover every condition your production environment will throw at it."*

This started two months of intensive data collection. The team reviewed hours of footage and extracted frames covering:
- Top view (slipping) and side view (standing)
- Morning, afternoon, and evening lighting
- Dust conditions
- Multiple animals in frame simultaneously
- Partial occlusion
- Animals moving in both directions
- Empty chutes and near-empty chutes
- Non-goat objects (workers, equipment, other animals)

They ended up with **~20,000 images** that needed to be annotated — bounding boxes drawn around every goat in every frame.

Annotation was done using **CVAT**, a professional annotation tool. The team self-hosted CVAT on a $5/month VPS, set up a Tailscale private network so all three could access it securely from anywhere, and divided the annotation work. It took weeks of evenings and weekends. 20,000 boxes. All manual.

---

## 4. Training — Learning What the Numbers Actually Mean

Training the first proper model on the full dataset was when the real learning began.

Ultralytics YOLO produces a rich set of training outputs: loss curves, precision-recall curves, confusion matrices, mAP scores at every epoch. At first, these were just numbers. The team had to learn what each one meant and why it mattered.

**mAP@50** — the primary metric. Measures how well the model detects objects at 50% IoU (intersection over union) threshold. A score of 95%+ means the model is finding almost every goat and placing the bounding box reasonably accurately.

**Precision vs Recall** — the fundamental tradeoff. High precision means when the model says "goat," it's almost certainly a goat. High recall means the model finds almost every goat in the frame. Tuning the confidence threshold moves the operating point along this curve.

**Loss curves** — the story of how training progressed. Box loss (how accurate the bounding box position is), class loss (how confident the class prediction is), DFL loss (distribution focal loss — how precise the box edges are). Watching these converge correctly vs. diverge or plateau early revealed what the model was struggling with.

Training was done on Kaggle (free T4/P100 GPU access). Multiple runs across YOLOv11-nano, YOLOv11-small, YOLOv12-nano, and YOLOv12-small. Final results: all four models achieved **~99% mAP@50** on the validation set. The nano models, despite being 3.5x smaller than the small models, reached within 0.1% of the same accuracy — a testament to the dataset quality.

---

## 5. The Pipeline Battle — Where Real Engineering Began

With a working model, the team moved to deployment. This is where the project went from "student project" to "real engineering problem."

The initial approach was the obvious one: use Ultralytics' built-in streaming mode. `model.track(source="rtsp://...", stream=True)`. Simple, clean, and completely inadequate for production.

On a GTX 1650 with 4GB VRAM, `model.track()` takes 93–178ms per frame. The camera produces frames at 25fps — one every 40ms. The pipeline falls behind immediately. By the time you process frame 10, the camera is at frame 40. You are counting goats that passed 1.5 seconds ago.

The drawing code made it worse. Rendering bounding boxes, HUD text, and count overlays in the same thread as inference added another 15–30ms per frame. On a consumer GPU, this gap between source rate and processing rate compounds rapidly.

The team rebuilt the pipeline from scratch with a multi-threaded architecture:
- A **capture thread** that reads from RTSP continuously and feeds frames into a queue
- A **pacing controller** that synchronizes frame timing to the source clock
- An **inference thread** that runs YOLO + ByteTrack on queued frames
- A **display thread** that handles drawing, CSV writing, and streaming

Each thread communicates through bounded queues with drop-oldest policies. When the system falls behind, old frames are dropped automatically — the pipeline always stays current with the live feed.

Then came TCP vs. UDP. RTSP can use either transport. TCP provides reliable delivery but adds latency — every dropped packet is retransmitted. For live monitoring, this means frames can arrive 300–500ms late. UDP drops packets without retransmission, keeping latency low but occasionally breaking frames. The team switched to UDP transport (`rtsp://...` with CAP_FFMPEG backend) and handled broken frames at the application level.

Then came the pacing logic. Without it, a burst of 5 frames arriving simultaneously would flood the inference queue. The pacing controller measures the source timestamp of each frame and sleeps when the pipeline is ahead of the source, drops frames when it's behind. This sounds simple. Getting it right — handling autoskip, resync, two skip policies — took weeks of iteration.

The final system achieved real-time processing on a GTX 1650. Inference latency p50: 93ms. End-to-end p90: 156ms. At 25fps source, this means the displayed frame is at most 1.5 source-frames behind the live feed.

---

## 6. On-Site Testing — The Real Validation

With the pipeline working in the lab, the team tested it on-site at Deonar using the actual CCTV feeds.

The first sessions revealed issues that didn't appear in lab testing:
- Goats that hesitate at the line were being counted 2–3 times as they crossed, backed up, and re-crossed
- Tracking ID switches (where ByteTrack loses a track and assigns a new ID to the same animal) caused missed counts near the line
- Direction was sometimes wrong — goats moving diagonally were occasionally classified as crossing in the wrong direction

Each of these led to a specific fix:
- **Double counts** → added `cooldown_frames` (a per-track lockout after each count) and the motion intent analyzer (rejecting counts where the goat didn't move enough in the right direction)
- **ID switch misses** → added `min_age` requirement (new track IDs must exist for at least 60 frames before they can count — filters out ghost tracks)
- **Wrong direction** → added the dual-line system with direction consistency checking, and the two camera profiles (`standing_view`, `slipping_view`) with tuned motion parameters

After several rounds of iteration, on-site results showed less than 5% miss rate on 180-goat counting sessions.

---

## 7. The Slot System — Thinking Past the Demo

Midway through development, Abdul Hamid raised a new requirement: each vendor at Deonar brings their own animals. The counting system needs to track not just the total count for the day, but how many animals each individual vendor sent through the chute during their assigned time slot.

This added a completely new layer to the system: the slot management API.

A slot represents one vendor's counting session. An operator calls `POST /api/slot/start` when a vendor's animals begin entering the chute. The system starts routing all new counting events into that vendor's slot — separate CSV file, separate count, separate video recording. When the vendor's animals are done, `POST /api/slot/stop` finalizes the slot and generates a summary: declared count vs. actual count, direction breakdown, timestamps.

This is the feature that transformed the project from "a counting demo" into "an auditable livestock management system."

---

## 8. The End — And What Was Gained

When Bakra Eid approached in 2025, MI Tradings pitched the system to BMC. BMC's response: the project was technically sound, but the infrastructure cost for widespread deployment (dedicated GPU servers at every facility, network integration with existing CCTV systems) was too high for the current budget cycle. The tender was not approved. No payment was ever made.

The project was shelved.

What the team came away with:

**Technical:** Real-world multi-threaded pipeline design. Production RTSP handling. Custom dataset collection and annotation at scale (20,000 images). Model training, evaluation, and hyperparameter selection. Motion-based counting algorithm design. REST API design and thread-safe state management. WebRTC streaming. CUDA-aware deployment automation.

**Non-technical:** How to navigate a real client relationship with ambiguous requirements. How to do on-site research before building. How to manage a team of three across months of unglamorous work (annotating 20,000 images is not exciting — doing it anyway is what separates people who ship from people who prototype).

**The experience letter:** MI Tradings provided an official R&D experience letter confirming 12 months of work on a BMC AI project. The first real work experience, earned on a zero-budget project that never shipped to production.

---

*The system still works. The code is clean. The models are accurate. The pipeline runs on consumer hardware. Somewhere, there are 20,000 annotated images of goats from Deonar that took months to collect and label. The project never went to production, but everything that was built here is real.*

---

# Part 2 — System Architecture

*The complete technical reference for the pipeline — every component, how they communicate, why they were designed this way, and how the system scales beyond goats to any detection + counting use case.*

---

## 9. Big Picture

At its core this system solves a deceptively hard problem: **count objects crossing a line in a live video stream, reliably, in real time, on consumer hardware.**

Detection alone (YOLO) is not enough. Tracking alone is not enough. A line crossing check alone is not enough. Each of these pieces fails in isolation when confronted with real-world conditions: occlusion, lighting changes, tracking ID switches, goats partially crossing and backing up, camera jitter, RTSP buffer lag, and GPU contention.

This system layers solutions on top of each other:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM LAYERS                                │
│                                                                     │
│  Layer 5: Slot Management    ← per-vendor session isolation         │
│  Layer 4: Motion Gating      ← physics-based count confirmation     │
│  Layer 3: Dual-Line Logic    ← geometry-based count confirmation    │
│  Layer 2: ByteTrack          ← persistent identity across frames    │
│  Layer 1: YOLOv11            ← per-frame object detection           │
│  Layer 0: Pipeline           ← real-time frame delivery             │
└─────────────────────────────────────────────────────────────────────┘
```

Each layer handles a class of failure that the layer below cannot handle. This is what separates a research prototype from a production system.

---

## 10. Why This Is More Than a YOLO Pipeline

A naive implementation using Ultralytics out of the box looks like this:

```python
model = YOLO("best.pt")
for result in model.track(source="rtsp://...", stream=True):
    # count crossings
```

This works in demos. It fails in production for the following reasons:

**Problem 1 — `model.track()` is synchronous and monolithic.**
Ultralytics bundles capture + pre-processing + forward pass + post-processing + ByteTrack into one blocking call. On a GTX 1650, this takes 93–178ms per frame. At 25fps the source produces a frame every 40ms. The pipeline falls behind within seconds, causing a growing buffer of stale frames. By the time you process frame N, the camera is already at frame N+30. You are counting goats that passed 1.2 seconds ago.

**Problem 2 — The main thread doing display work starves inference.**
Drawing bounding boxes, rendering the HUD, writing to OpenCV windows — all CPU-heavy. If these happen in the same thread as inference, every frame takes even longer, compounding the lag.

**Problem 3 — RTSP streams deliver frames in bursts, not at a steady rate.**
Jitter in network delivery means frames sometimes arrive in clusters. Without a pacing layer, the inference thread gets flooded with 5 frames at once then starves for 200ms.

**Problem 4 — A raw line crossing check produces false counts.**
A goat that hesitates near the line, partially crosses and backs up, or is occluded mid-crossing can trigger 2–4 false counts with a naive side-flip check. On a 180-goat session, this adds up fast.

**This system solves all four problems** with a multi-threaded pipeline, a pacing controller, and a two-phase counting architecture (geometry + motion).

---

## 11. Live Pipeline — Thread Architecture

```
                         LIVE PIPELINE
═══════════════════════════════════════════════════════════════════════

  ┌──────────────────┐     capture_queue      ┌──────────────────┐
  │  ThreadedVideo   │  ──── [bounded=3] ───▶ │  PacingControl   │
  │  Capture         │                         │  ler             │
  │                  │                         │                  │
  │  • Opens RTSP    │                         │  • Syncs source  │
  │  • Reads frames  │                         │    timestamps to │
  │  • Reconnects on │                         │    real time     │
  │    failure       │                         │  • Autoskip when │
  │  • drop-oldest   │                         │    lagging       │
  │    policy        │                         │  • drop-oldest   │
  └──────────────────┘                         │    on output q   │
         Thread 1                              └──────────────────┘
                                                      Thread 2
                                                          │
                                               pacing_out_q [bounded=12]
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │  InferenceWorker │
                                               │                  │
                                               │  • YOLO forward  │
                                               │    pass          │
                                               │  • ByteTrack     │
                                               │  • ROI crop      │
                                               │  • Area filter   │
                                               └──────────────────┘
                                                      Thread 3
                                                          │
                                                result_queue [bounded=12]
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │  DisplayWorker   │
                                               │                  │
                                               │  • Line crossing │
                                               │    check         │
                                               │  • Motion gating │
                                               │  • Draw HUD      │
                                               │  • WebRTC push   │
                                               │  • CSV write     │
                                               │  • Slot routing  │
                                               └──────────────────┘
                                                      Thread 4
                                                    │         │
                                              WebRTC      SlotAPI
                                              Server      (FastAPI)
                                            (browser)   Thread 5

═══════════════════════════════════════════════════════════════════════
           Monitor Thread (Thread 6) — watches all threads,
           logs state changes, restarts pacer/infer if they crash
═══════════════════════════════════════════════════════════════════════
```

### Queue Design

All queues are **bounded** — this is critical. A bounded queue applies backpressure. If the inference thread is slow, the pacing queue fills up and the pacing controller drops old frames automatically. The system stays real-time instead of processing stale data.

| Queue | Max Size | Policy on Full |
|---|---|---|
| `capture_queue` | 3 | Drop oldest (capture thread) |
| `pacing_out_q` | 12 | Drop oldest (pacer) |
| `result_queue` | 12 | Drop oldest (inference worker) |

The capture queue is intentionally tiny (3 frames). This ensures the pacing controller always works with fresh frames.

### Thread Lifecycle

Threads are daemon threads — they die automatically when the main process exits. The monitor thread watches liveness every 1 second and:
- Logs the exact moment a thread dies (not repeatedly — only on state transition)
- Logs recovery when a thread comes back
- Attempts a single restart for the pacer and inference worker
- Never restarts the capture thread (reconnect logic is built into it) or the display thread (stateful — restarting would corrupt counts)

---

## 12. Component Deep Dives

### 12.1 ThreadedVideoCapture

Opens `cv2.VideoCapture` for any source — RTSP URL, HTTP stream, local file, or webcam index (e.g. `"0"` → `int(0)`). Runs in its own thread so the main pipeline is never blocked waiting for a frame.

Key behaviors:
- **Auto-reconnect:** if `cap.read()` fails, releases the capture and sleeps `reconnect_delay` seconds before retrying. This handles RTSP stream drops without crashing.
- **Metadata publication:** after the first successful open, writes `cap_info` dict (`fps`, `width`, `height`, `total`) which the rest of the pipeline reads to compute geometry.
- **Drop-oldest policy:** queue is bounded. If inference is slow and the queue fills up, the oldest frame is evicted. The pipeline always processes the newest available frame.
- **CAP_PROP_BUFFERSIZE=2:** reduces OpenCV's internal frame buffer to minimize capture-to-display latency.

### 12.2 PacingController

The most underrated component. Its job: ensure the pipeline processes frames at the correct rate relative to the source, not at whatever rate inference happens to run.

Without pacing, a 25fps source feeding a 10fps inference pipeline would result in processing frames from 1.5 seconds ago by the time you reach the 15th frame. Counts would be timestamped incorrectly and the live view would lag badly.

**How it works:**

```
For each frame:
  source_ts = timestamp when this frame was captured (monotonic clock)
  expected_real_time = (source_ts - start_source_ts) / playback_speed
  elapsed_real_time = now - start_real_ts

  if expected > elapsed + jitter_allowance:
      sleep(expected - elapsed)   ← source is ahead, wait

  if elapsed > expected + max_lag_s:
      if autoskip:
          drain queue, grab latest frame  ← we're behind, skip ahead
      else:
          emit frame as-is (late but correct)
```

**Autoskip policies:**
- `drop_to_latest` — drain the entire queue, process only the most recent frame. Best for live monitoring where you always want the latest view.
- `drop_oldest` — scan forward through the queue to find the first frame that's not too late. Best when you want continuity but need to catch up.

### 12.3 InferenceWorker

Consumes paced frames, runs detection and tracking, produces detection results.

**Initialization is deferred to inside the thread.** Model loading (which downloads weights, initializes CUDA context, compiles YOLO layers) takes 3–10 seconds. Doing this in the main thread would block everything. Instead, the first frame to arrive triggers `_init()`.

**ROI crop happens here.** The full frame is kept for display, but only the cropped ROI is sent to the model. This means:
- The model only sees the relevant portion of the frame (the chute)
- Background objects outside the chute are never detected
- Model inference is faster because the input region is smaller

**Model loading has two fallback levels:**
1. Try the project's enhanced loader (`load_model_threaded`) — handles CUDA half-precision, layer fusion, device normalization
2. Fall back to Ultralytics' default `YOLO.load()` — always works but no optimizations

**Output contract** — the result dict pushed to `result_queue` has a guaranteed stable shape:
```python
{
    "frame_id": int,
    "timestamp": float,       # monotonic capture time
    "frame": np.ndarray,      # full frame (HWC uint8)
    "roi": np.ndarray,        # cropped ROI
    "dets": list,             # [(box, track_id, conf, class, mask), ...]
    "feeder": FeederLike,     # carries frame counters + geometry
    "infer_fps": float,       # smoothed inference FPS (8-frame rolling average)
    ...                       # queue fill levels for monitoring
}
```

### 12.4 DisplayWorker

The busiest thread. Receives detection results and does everything visual and counting-related:

1. **Runs counting logic** (line cross check, motion gating, dual-line state machine)
2. **Writes CSV events** when a count is confirmed
3. **Draws the annotated frame** (bounding boxes, track IDs, lines, HUD, animations)
4. **Pushes to WebRTC** for browser viewing
5. **Routes counts to SlotManager** if a slot is active
6. **Records video** to disk (global or per-slot)
7. **Falls back to OpenCV windows** if WebRTC is not enabled

Counting state (`up_count`, `down_count`, track histories) lives here. This is intentional — counting is a display-side concern because it requires per-frame geometry (ROI dimensions, line coordinates) that are only available once the frame arrives.

### 12.5 WebRTC Server

Runs a minimal HTTP + WebSocket server (aiortc) that:
- Serves an HTML page at `http://<host>:<port>` with a live video element
- Streams annotated frames via WebRTC peer connection (browser-compatible)
- Accepts control commands from the browser (`screenshot`, `quit`)
- Downscales frames to configurable resolution before streaming (default 960×540) to reduce bandwidth

This means the live annotated feed is viewable from any browser on the same network — no VNC, no display server required. Critical for headless GPU machines.

---

## 13. Offline Pipeline

For recorded video files, a simpler single-threaded pipeline is used:

```
cv2.VideoCapture(file)
       │
       ▼
   ROIStream (iterator)
   - Applies stride (process every Nth frame)
   - Applies extra_skip for autoskip
   - Yields ROI crops
       │
       ▼
   track_once(model, roi, args, tracker_yaml)
   - Single YOLO forward pass + ByteTrack update
       │
       ▼
   Counting logic (same as live)
       │
       ▼
   _compose_frames() → draw + write to VideoWriter
```

Single-threaded is correct for offline because:
- The source is a file — no network jitter, no reconnection needed
- Processing speed doesn't matter for real-time delivery
- Simplicity means fewer failure modes

The offline pipeline produces identical CSV outputs and annotated video as the live pipeline. This was used extensively during development to validate counting logic against known recordings before testing on live RTSP.

---

## 14. Coordinate System & ROI

The system uses **two coordinate spaces** and converting between them is a core part of the architecture.

```
Full Frame (W × H pixels)
┌────────────────────────────────────┐
│                                    │
│   rx,ry ┌──────────────┐           │
│         │              │           │
│         │   ROI        │ rh        │
│         │   (rw × rh)  │           │
│         └──────────────┘           │
│              rw                    │
└────────────────────────────────────┘
```

All geometry in `configs/config.yaml` is expressed as **normalized ratios (0.0–1.0)**:

```yaml
geometry:
  roi:
    xr: 0.0     # left edge = 0% of frame width
    yr: 0.1514  # top edge = 15.14% of frame height
    wr: 1.0     # width = 100% of frame width
    hr: 0.8472  # height = 84.72% of frame height
```

At runtime these are converted to pixels using the actual frame dimensions from `cap_info`. This makes the configuration **resolution-independent** — the same config works whether the camera outputs 720p or 1080p.

Counting lines are defined in **ROI-relative ratios** (not full-frame):

```yaml
counting:
  line_roi: "0.1875, 0.1754, 0.1875, 0.8984"
  #          ax      ay      bx      by
  #          (all as fractions of ROI width/height)
```

This means the line position is expressed relative to the cropped chute region, not the full camera frame. When the ROI changes, the line stays in the same relative position inside the chute.

---

## 15. Configuration-Driven Design (Scalability)

**This system is not limited to goats.** It is a general-purpose object detection + directional counting pipeline. The only thing that makes it "for goats" is the model weights and the ROI coordinates.

To adapt it to a different use case:

| What to change | Where | Effect |
|---|---|---|
| Swap model weights | `paths.weights` | Detect different objects (sheep, cattle, people, vehicles) |
| Adjust ROI | `geometry.roi.*` | Focus on any region of the camera frame |
| Reposition counting line | `counting.line_roi` | Count crossings at any position within the ROI |
| Switch counting mode | `counting.mode` | `line` / `zone` |
| Enable/disable dual-line | `counting.dual.enabled` | Add or remove the second verification line |
| Change camera orientation | `counting.dual.profile` | `standing_view` or `slipping_view` — tunes all motion parameters |
| Tighten/loosen counting | Motion parameters | Control sensitivity vs. false-positive tradeoff |
| Add per-vendor sessions | `runtime.slots_enabled` | Slot API automatically partitions counts by vendor |

**Real examples of reuse without code changes:**

- **People counting at a doorway:** swap to a person detection model, set the line at the door threshold, use `slipping_view` profile (horizontal movement)
- **Vehicle counting at a gate:** swap to a vehicle model, wider ROI, `dual` mode with a larger `offset_px` between lines
- **Cattle pen monitoring:** use `zone` mode instead of line mode, draw a rectangle around the pen entry point

The architecture deliberately puts all tuneable behavior in `configs/config.yaml` so that operators can adjust the system without touching code.

---

## 16. Counting System Architecture

### Why Counting Is Hard

Detection tells you: "there is a goat at position (x, y) in this frame."
Tracking tells you: "this goat has ID #42 and was also at position (x', y') last frame."

Neither of these tells you: "goat #42 has definitively crossed the line in the upward direction and should be counted exactly once."

The gap between tracking and counting is where most systems fail. This system fills that gap with a two-phase architecture.

### Phase 1 — Geometry (Eyes)

**Side sign computation:**
For each tracked object, compute which side of the counting line its centroid is on, using the cross product of the line vector and the point vector:

```
side = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
```

Positive → one side. Negative → the other side. Near-zero (within `line_margin_px`) → the object is too close to the line to determine side (ignored).

**Side history confirmation (anti-flicker):**
Rather than reacting to a single side reading, the system maintains a rolling history of side readings per track ID. A side is only "confirmed" when a minimum number of non-zero readings in the history agree. This prevents counting when a goat oscillates near the line without truly crossing.

**Flip detection:**
A count is a geometry event when the confirmed side changes: `prev_side != current_side`. The direction (`up` or `down`) is determined by the sign of the flip.

### Phase 2 — Motion Gating (Brain)

A geometry flip is a necessary condition for a count, but not sufficient. Before committing the count, the motion intent analyzer checks the track's recent movement history:

```
1. Require minimum frames of history (default: 8 frames)
2. Compute displacement vector (dx, dy) over recent history
3. Check total displacement >= min_displacement_px (default: 16px)
4. Identify dominant axis (x or y) based on line orientation
5. Check axis displacement >= axis_min_displacement_px (default: 12px)
6. Check direction consistency — at least 70% of frame-to-frame
   steps must point in the same direction as the overall displacement
7. Project displacement onto line normal → compute motion direction
8. Compare motion direction to geometry direction → must agree
```

Outcomes:
- **Accept** — motion supports geometry, count is committed
- **Reject** — motion contradicts geometry, count is discarded, logged with reason
- **Defer** — insufficient motion history yet, decision postponed

### Dual-Line Modes

**Verify mode:**
```
Line A crossing → create "pending A" record (direction, frame index)
Line B crossing → if same direction AND within window_frames → COMMIT
                  else → drop pending, start over
```

**Recover mode:**
```
Line A crossing → immediately signal as geometry intent (no commit yet)
                  run motion gating → if accepted → COMMIT on A
Line B crossing → if A was missed for this track → COMMIT on B (recovery)
```

In practice, **Recover mode is used in production** because missed crossings hurt accuracy more than the marginal increase in false positives that Verify mode prevents.

---

## 17. Slot Management System

The slot system allows an operator to partition the global count into per-vendor sessions.

```
Slot Lifecycle:

  POST /api/slot/start  →  SlotManager.start_slot()
  │                              │
  │                         Creates SlotRuntime
  │                         Opens SlotCsvWriter
  │                         Fires on_slot_start callbacks
  │                              │
  │                    ┌─────────▼──────────┐
  │                    │   ACTIVE SLOT      │
  │                    │                    │
  counting events ────▶│  on_global_count_  │
  (from DisplayWorker) │  event()           │
  │                    │  → adds to         │
  │                    │    slot_count      │
  │                    │  → writes to       │
  │                    │    slot CSV        │
  │                    └─────────┬──────────┘
  │                              │
  POST /api/slot/stop  →  SlotManager.stop_slot()
                                 │
                            Finalizes slot
                            Generates summary JSON
                            Closes slot CSV
                            Fires on_slot_stop callbacks
                            Writes status.txt
```

**Thread safety:** The slot manager is accessed from two threads simultaneously — the display thread (counting events) and the FastAPI thread (API requests). All public methods are protected by a `threading.Lock`.

**Idempotency:** Duplicate start requests for the same active slot are silently ignored. Duplicate stop requests raise a 409. This makes the API safe to call from unreliable clients.

**Per-slot video recording:** When slot video recording is enabled, `DisplayWorker.on_slot_start()` opens a new `VideoRecorder` for the slot directory. Frames are written to `SLOT_<id>_<timestamp>/slot_video.mp4` during the active slot. On stop or abort, the recorder is closed and a `status.txt` is written.

---

## 18. Output System

Every run creates a directory under `outputs/runs/<run_id>/` where `run_id = <timestamp>_<source_name>`.

```
outputs/runs/2026-02-19_15-32-10_mystream/
├── events/
│   └── goat_cross_events.csv       ← one row per counted crossing
├── timeseries/
│   └── goat_counts_timeseries.csv  ← cumulative counts per second
├── decisions/
│   └── goat_decisions.csv          ← every motion gating decision
├── metrics/
│   └── goat_metrics.csv            ← per-frame latency breakdown
│   └── goat_metrics_summary.json   ← p50/p90/p95 latency summary
├── video/
│   └── output.mp4                  ← annotated output video
└── slots/
    └── SLOT_vendor_001__20260219_153500/
        ├── slot_events.csv         ← this vendor's crossings only
        ├── slot_summary.json       ← declared vs counted, status
        ├── slot_video.mp4          ← video during this slot
        └── status.txt              ← COMPLETED / ABORTED
```

**CSV Events schema:**

| Column | Description |
|---|---|
| `timestamp_s` | Seconds from video start when crossing occurred |
| `src_frame_idx` | Original source frame index (before any skipping) |
| `proc_frame_idx` | Processed frame index (after pacing/skipping) |
| `track_id` | ByteTrack-assigned persistent track ID |
| `direction` | `up` or `down` |
| `cx`, `cy` | Centroid position in ROI pixels at crossing moment |

The two frame indices allow reconstructing exactly when in the source video a crossing occurred, even if the pipeline skipped frames.

---

## 19. Metrics & Observability

The `MetricsCollector` records timing marks at four pipeline stages for every frame:

```
Frame lifecycle:
  captured        ← ThreadedVideoCapture puts frame in queue
  pacer_emit      ← PacingController forwards to inference queue
  infer_start     ← InferenceWorker begins forward pass
  infer_end       ← InferenceWorker completes, pushes result
  display_shown   ← DisplayWorker renders and pushes to WebRTC
```

From these marks, end-to-end latency per frame is computable. Summary statistics (p50, p90, p95) are written to `goat_metrics_summary.json` at the end of each run.

The monitor thread logs queue utilization every 5 seconds:
```
QUEUES: cap_q=2/3  pacing_q=4/12  res_q=1/12
```

This gives a real-time view of which stage is the bottleneck during a live run.

---

## 20. Custom Installer

Installing the correct version of PyTorch for a given CUDA driver version is one of the most common failure modes when deploying deep learning systems.

The `setup-installer` CLI tool automates this:

1. Reads the installed NVIDIA driver version (`nvidia-smi`)
2. Maps driver version to maximum supported CUDA version
3. Selects the correct PyTorch wheel from the PyTorch index
4. Installs PyTorch + torchvision + torchaudio + ultralytics with zero version conflicts
5. Optionally installs `cuda-python` and `pynvml` for advanced GPU monitoring
6. Runs a post-install verification step

```bash
setup-installer \
  --auto-detect-torch \
  --install-cuda-python \
  --install-nvidia-ml \
  --always-progress \
  --verbose \
  --run-after "python main.py"
```

This works on Windows, Linux, and any CUDA version from 11.x to 12.x.

---

# Part 3 — Counting Logic Deep Dive

*Exactly how the counting system works — from raw detections to a confirmed, direction-labelled count event written to CSV.*

---

## 21. The Core Problem

YOLO tells you there is a goat at pixel coordinates (x1, y1, x2, y2) in this frame.
ByteTrack tells you this goat has track ID #42 and was at (x1', y1', x2', y2') in the previous frame.

Neither of these answers: *Has this goat definitively crossed the counting line, in a known direction, and should be counted exactly once?*

Answering that question reliably is harder than it looks:

| Failure mode | Symptom | Naive fix | Why it fails |
|---|---|---|---|
| Goat hesitates at line | 2–5 counts per crossing | Wider margin | Delays count but doesn't prevent flicker |
| Goat backs up after crossing | Double count | Cooldown timer | Timer too short → still double counts |
| Tracking ID switch at line | Missed count or wrong direction | Lower track threshold | More false tracks created |
| Goat moves diagonally | Wrong direction | Ignore | Diagonal motion can look like crossing |
| Camera shake | Random side flips | Smooth position | Smoothing adds latency |
| Occlusion mid-crossing | ID loss, new ID assigned | Nothing | Miss is unavoidable without backup line |

The system handles all of these. Here's how.

---

## 22. Counting Architecture Overview

```
Per frame:
  for each tracked detection (box, track_id, conf, class, mask):

    1. Compute centroid (cx, cy)
       └── from mask if available, else box center

    2. Update motion history
       └── append (frame_idx, cx, cy) to per-track deque

    3. Phase 1 — Geometry
       └── Compute side sign vs counting line
       └── Append to side history
       └── Confirm stable side (anti-flicker)
       └── Detect side flip → geometry event

    4. Phase 2 — Motion gating
       └── Analyze recent motion history
       └── Accept / Reject / Defer

    5. If ACCEPT:
       └── Increment up_count or down_count
       └── Write to events CSV
       └── Trigger animation
       └── Mark track color as "counted"
       └── Route to active slot (if any)
```

---

## 23. Centroid Computation

The system prefers **mask centroids** over bounding box centers when segmentation masks are available:

```python
if mask is not None:
    ys, xs = np.where(mask)
    if xs.size > 0 and ys.size > 0:
        cx, cy = xs.mean(), ys.mean()
else:
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
```

Why mask centroids? A bounding box center can be significantly off from the actual animal center when the box is large (two animals close together) or the animal is partially occluded. The mask centroid is always inside the detected object.

All counting line checks use the centroid. This means the "moment of crossing" is determined by when the animal's center of mass crosses the line, not when its bounding box edge touches it.

---

## 24. Side Sign Computation

Given a counting line defined by two points A=(ax, ay) and B=(bx, by), the side of point P=(cx, cy) is:

```
raw = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
```

This is the cross product of vector AB and vector AP. Positive means one side, negative means the other.

**Margin zone:** Points within `line_margin_px` pixels of the line (perpendicular distance) return sign = 0, meaning "too close to determine side."

```python
dist = |((bx-ax)*(ay-cy) - (by-ay)*(ax-cx))| / length(AB)

if dist < line_margin_px:
    return 0    # in the dead zone — ignore
return +1 if raw > 0 else -1
```

The margin zone is critical. Without it, a goat standing exactly on the line would oscillate between +1 and -1 every frame, generating dozens of false count events.

---

## 25. Side History Confirmation (Anti-Flicker)

Rather than reacting to a single side reading, the system maintains a deque of recent side readings per track ID. The deque has a maximum length of `max(3, hyst_frames)`.

A side is only "confirmed" when the history passes this check:

```python
def _confirm_side(side_hist, need_frames, min_ratio=0.6,
                  min_nonzero_abs=1, min_nonzero_ratio=0.0):

    window = last need_frames elements of side_hist
    nonzeros = [s for s in window if s != 0]

    # Must have enough non-zero samples
    min_n = max(min_nonzero_abs, ceil(need_frames * min_nonzero_ratio))
    if len(nonzeros) < min_n:
        return None

    # STRICT: all non-zero samples identical AND count == need_frames
    if len(nonzeros) == need_frames and all same:
        return that sign

    # LOOSE: majority >= min_ratio
    pos = count of +1 in nonzeros
    if pos/total >= min_ratio: return +1
    if neg/total >= min_ratio: return -1

    return None
```

**Strict confirmation** requires every non-zero sample in the window to agree. This is used when the history is clean — no noise.

**Loose confirmation** requires a supermajority (default 60%) of non-zero samples to agree. This handles cases where a few frames were in the margin zone (returned 0) but the majority clearly show one side.

The two-tier approach means the system can confirm a side even when some frames in the window were ambiguous, while still requiring strong evidence.

---

## 26. Single-Line Counting

This is the baseline counting mode. One line, count every crossing.

```
State per track:
  side_hist[tid]    ← deque of recent side signs
  last_side[tid]    ← last confirmed side
  last_counted[tid] ← frame index of last count for this track
  age_frames[tid]   ← frames this track has existed

Per frame, per track:
  1. sgn = side_sign(cx, cy, line, margin)
  2. if sgn != 0: side_hist[tid].append(sgn)
  3. confirmed = confirm_side(side_hist[tid])
  4. if confirmed is None: continue

  5. prev = last_side[tid]
  6. last_side[tid] = confirmed

  7. flipped = (prev is not None and prev != confirmed)
  8. ready_age = age_frames[tid] >= min_age
  9. ready_cooldown = (current_frame - last_counted[tid]) >= cooldown_frames

  10. if flipped AND ready_age AND ready_cooldown:
        direction = "up" if prev < confirmed else "down"
        → COMMIT COUNT
```

**`min_age`** (default: 60 frames at 25fps = 2.4 seconds): A new track must exist for at least this many frames before it can count. This eliminates ghost tracks — false detections that appear briefly and happen to be near the line. A real goat walking through the chute will be tracked for far longer than 60 frames.

**`cooldown_frames`** (default: 15 frames = 0.6 seconds): After a count is committed for a track, that track cannot count again for this many frames. This prevents a goat that stops exactly on the line from being counted every time it oscillates slightly.

---

## 27. Dual-Line Counting

Two parallel lines, A and B, placed at a fixed offset (`offset_px`) in the direction of movement. B is downstream of A.

```
Camera view (slipping / horizontal):

         Line A          Line B
           │               │
           │    offset_px  │
     ──────┼───────────────┼──────  (goat moves right → left)
           │               │
           │               │
```

The dual-line system requires a goat to cross both lines in the same direction within a time window. This makes it geometrically impossible for a goat that hesitates at line A, backs up, and re-crosses to generate a count — it would need to then cross line B in the same direction within the window, which backing-up prevents.

### Verify Mode

```
State per track:
  pending_A[tid] = {"dir": direction, "frame": frame_idx}

On A flip:
  Store in pending_A[tid]

On B flip (if pending exists):
  Check: same_dir? AND within window_frames? AND unlocked?

  If all three:
    → COMMIT COUNT (trigger: B)
    Clear pending_A[tid]

  Else:
    Drop pending_A[tid]
    Log reason (dir_mismatch / window_expired / locked)
```

**Verify mode is conservative.** A missed B crossing (tracking loss, occlusion at line B) means the count is missed entirely. Best for low-traffic, clear-view scenarios where false positives are worse than missed counts.

### Recover Mode

```
On A flip:
  If unlocked:
    Call on_A_count_cb() → record geometry intent
    Run Phase 2 motion gating
    If accepted → COMMIT COUNT (trigger: A)

On B flip:
  If this track has not been counted recently:
    Run Phase 2 motion gating
    If accepted → COMMIT COUNT (trigger: B, "recovered at B")
```

**Recover mode is robust.** Line A counts immediately when crossed. Line B acts as a safety net — if a goat's A crossing was missed (tracking loss), B catches it. This means every goat that physically passes through the chute is counted, even if it moved quickly or tracking was briefly lost.

Recover mode is used in production at Deonar because missed counts are more damaging than rare false positives in high-throughput scenarios.

---

## 28. Phase 2 — Motion Intent Analyzer

This is the "brain" behind every count. Before any count is committed — regardless of which mode — the motion history is analyzed.

```python
def analyze(motion_history, tid, event_frame,
            geometry_direction, line_vector):

    hist = motion_history.get(tid)
    past = [frames before event_frame in hist]

    # Check 1: enough history
    if len(past) < min_frames:
        return defer("insufficient motion frames")

    # Keep only recent window
    past = past[-max_lookback:]
    (f0, x0, y0) = past[0]
    (f1, x1, y1) = past[-1]

    dx = x1 - x0
    dy = y1 - y0
    disp = sqrt(dx² + dy²)

    # Check 2: enough total displacement
    if disp < min_displacement_px:
        return reject(dx, dy, "motion too small")

    # Check 3: project motion onto line normal
    # line_vector = (lx, ly) — direction along the line
    # normal = (-ly, lx) / |line| — direction perpendicular to line
    nx, ny = -ly/|line|, lx/|line|
    projection = dx*nx + dy*ny
    motion_dir = "up" if projection > 0 else "down"

    # Check 4: dominant axis displacement
    dominant_axis = "x" if |nx| >= |ny| else "y"
    axis_disp = |dx| if x-axis else |dy|
    if axis_disp < axis_min_displacement_px:
        return reject(dx, dy, "axis displacement too small")

    # Check 5: direction consistency across frames
    steps = [frame-to-frame deltas on dominant axis]
    consistent = count of steps in same direction as overall delta
    if consistent/total < dir_consistency_ratio:
        return reject(dx, dy, "direction inconsistent")

    # Check 6: motion direction must match geometry direction
    if motion_dir != geometry_direction:
        return reject(dx, dy, "motion contradicts geometry")

    return accept(confidence=min(1.0, disp/(min_disp*3)))
```

**Why project onto the line normal?**
The counting line has an orientation. For a vertical line (standing view), movement perpendicular to the line is horizontal (left-right). For a horizontal line (slipping view), it is vertical (up-down). The motion analyzer needs to check movement in the direction that matters for crossing, not absolute 2D movement. The line normal gives exactly this.

**Why check direction consistency?**
A goat that moves forward 20px then sideways 5px then forward 10px shows consistent forward motion. A goat that moves forward 10px, sideways 8px, backward 4px, forward 6px is jittery — likely camera noise or tracking instability. The consistency check requires at least `dir_consistency_ratio` (default 70%) of frame-to-frame steps to point in the same direction as the overall displacement.

**Outcomes:**

| Decision | Meaning | Effect |
|---|---|---|
| `accept` | Motion fully supports the geometry crossing | Count is committed |
| `reject` | Motion contradicts or is too weak | Count discarded, logged to decisions CSV |
| `defer` | Not enough motion history yet | Decision postponed to next frame |

All three outcomes are logged to `goat_decisions.csv` with full detail: delta, dominant axis, confidence, reason. This makes the system fully debuggable — you can replay any run and see exactly why every count was accepted or rejected.

---

## 29. Zone Counting

Alternative to line counting for monitoring enclosed areas (pens, gates, doorways).

A zone is a rectangle in ROI-relative coordinates. The system determines whether each tracked object is inside or outside the zone each frame.

```
Entry = object transitions from outside → inside
Exit  = object transitions from inside → outside

born_inside policy:
  "ignore"       → objects that start inside the zone are not counted on exit
  "count_entry"  → objects that start inside are back-projected as having entered
```

The `backfill_wait` parameter delays this decision — if a new track appears inside the zone, the system waits `backfill_wait` frames before deciding whether to count it as an entry. This handles the case where detection starts mid-crossing.

Zone counting does not use the motion intent analyzer — direction is determined by the inside/outside transition, which is unambiguous.

---

## 30. Anti-Flicker Parameters Reference

| Parameter | Default | Effect |
|---|---|---|
| `min_age` | 60 frames | Minimum track age before counting eligibility |
| `min_side_frames` | 2 frames | Frames needed for strict side confirmation |
| `cooldown_frames` | 15 frames | Per-track lockout frames after each count |
| `line_margin_px` | 8 px | Dead zone width around the line |
| `min_nonzero_abs` | 4 samples | Minimum non-zero side samples for loose confirmation |
| `min_nonzero_ratio` | 0.67 | Fraction of non-zero samples that must agree |
| `hyst_frames` | 5 frames | Side history window length |
| `id_lock_frames` | 80 frames | Per-track lockout in dual-line mode |
| `window_frames` | 40 frames | A→B pairing window in verify mode |

---

## 31. Tuning Guide

**Too many false counts (over-counting):**
- Increase `motion_min_displacement_px` — require more physical movement
- Increase `motion_dir_consistency_ratio` — require more uniform direction
- Increase `id_lock_frames` — longer lockout after each count
- Increase `min_nonzero_abs` — require more non-zero side samples

**Missed counts (under-counting):**
- Decrease `motion_min_displacement_px` — accept counts with smaller displacement
- Decrease `min_age` — allow newer tracks to count sooner
- Switch from `verify` to `recover` dual-line mode
- Decrease `hyst_frames` — confirm sides faster

**Wrong direction being counted:**
- Check camera orientation matches `profile` setting (`standing_view` vs `slipping_view`)
- Verify line coordinates in config — the line direction affects which side is "up"
- Increase `motion_axis_min_displacement_px` to filter diagonal crossings

**Counts happening too far from the line:**
- Decrease `line_margin_px` — narrow the dead zone
- Verify ROI coordinates are correct — the line is defined in ROI space

---

## 32. Data Flow — From Pixel to CSV Row

```
cap.read() → frame (full resolution, HWC uint8)
           │
           ▼
InferenceWorker crops ROI: frame[ry:ry+rh, rx:rx+rw]
           │
           ▼
track_once(model, roi, args, tracker_yaml)
  → YOLO forward pass on ROI
  → ByteTrack update
  → filter by min_area_ratio, class filter
  → returns: [(box_roi, track_id, conf, class, mask), ...]
           │
           ▼
DisplayWorker receives result dict
  → derives geometry (rx, ry, rw, rh, W, H)
  → builds lines_roi from config string
  → initializes DualLineCounter (once, cached)
           │
           ▼
For each detection (box_roi, tid, conf, class, mask):
  cx, cy = centroid(box_roi, mask)   ← in ROI pixel space
  motion.update(tid, frame_idx, cx, cy)
  geometry_event = dual.update_recover(tid, cx, cy, ...)
           │
           ▼
If geometry_event:
  decision = motion_analyzer.analyze(...)
  If decision == "accept":
    state.up_count += 1  (or down_count)
    csvs.write_event(ts_s, src_frame, proc_frame, tid, dir, cx, cy)
    animator.trigger(cx, cy, lineB_roi, lineB_full, dir, frame_idx)
    colorer.mark_counted(tid, frame_idx)
    slot_mgr.on_global_count_event(...)   ← if slot active
           │
           ▼
_compose_frames() renders annotated ROI + full frame
  → pushed to WebRTC or OpenCV window
  → written to VideoRecorder
```

Every step from pixel coordinates to CSV row is traceable. The decisions CSV records every motion gating decision. The events CSV records every committed count. Combined, these allow post-hoc analysis of any session.

---

# Part 4 — FAQ

## 33. Frequently Asked Questions

**Q: Is this only for goats?**
No. The system detects and counts whatever the model is trained to detect. Swap `paths.weights` to a different model and adjust the ROI and counting line. The pipeline, counting logic, slot system, and outputs all work identically.

**Q: Does it work on CPU?**
Yes. Set `inference.device: "cpu"` and `inference.half: false`. Inference will be slow (~500–1000ms per frame) but the pipeline handles it — the pacing controller will drop frames to stay real-time rather than building a backlog.

**Q: How does it handle a goat that backs up and crosses the line multiple times?**
The `cooldown_frames` parameter (default: 15) prevents the same track ID from being counted again until at least 15 frames have passed. The `id_lock_frames` in dual-line mode (default: 80) provides a longer lock. Combined with motion gating, a backing-up goat would fail the direction consistency check even if it triggers a geometry flip.

**Q: What happens if ByteTrack loses a track ID and reassigns a new ID to the same goat?**
The new track ID starts with no history. It needs to accumulate `min_age` frames (default: 60) before it is eligible to count. This means a goat that loses its track ID very close to the line will likely not be counted on that crossing — it will be counted on the next full crossing. This is the correct conservative behavior.

**Q: What if the RTSP stream drops during a live session?**
The `ThreadedVideoCapture` detects the read failure, releases the capture, waits `reconnect_delay` seconds (default: 3s), and reconnects automatically. The pipeline continues without intervention. The slot system remains active; counts resume when the stream recovers.

**Q: Can multiple cameras be monitored simultaneously?**
Not natively in the current configuration. Each `python main.py` process handles one source. To monitor N cameras, run N processes with N separate config files and N separate slot API ports. Outputs are organized by run ID so they don't conflict.

**Q: How is direction determined?**
Direction is determined by which side of the line the track was on before vs after the crossing. "Up" and "down" are abstract labels — their real-world meaning depends on camera orientation. Use `counting.dual.profile: standing_view` for vertical chutes (up = bottom-to-top) and `slipping_view` for horizontal chutes (up = right-to-left). Each profile tunes the motion analysis parameters for the expected movement axis.

**Q: Why dual-line instead of just a wider margin on a single line?**
A wider margin just means you count later. It doesn't confirm the crossing — a goat that partially enters the margin and backs out will still trigger a count. Dual-line requires the goat to pass through a confirmed spatial sequence (A then B), which physically cannot happen without a complete crossing.

**Q: What does the WebRTC server serve exactly?**
A self-contained HTML page with a live `<video>` element. The annotated frames (with bounding boxes, track IDs, counting lines, HUD, and count overlay) are streamed via WebRTC peer connection at up to 25fps. Any browser on the same network can view it. No plugins required.

**Q: Why is `min_age = 60` frames before a track is eligible to count?**
At 25fps, 60 frames = 2.4 seconds. This ensures that "ghost" tracks created by momentary detections of background objects are never counted. A real goat moving through the chute will be tracked for many more frames than this. A false detection that appears for 3–5 frames will age out before reaching the line.

**Q: Can I use this for people counting or vehicle counting?**
Yes — swap the model weights and adjust the ROI and line coordinates in config. No code changes required. The pipeline architecture, counting algorithm, slot system, and all outputs are fully generic. The only goat-specific element is the trained model.

**Q: What is the minimum hardware requirement?**
Any NVIDIA GPU with 4GB+ VRAM runs the nano models comfortably. CPU-only mode works but inference drops to ~1–2fps. For production use on live CCTV, a GTX 1650 or better is recommended.

**Q: Why does the system use bounded queues with drop-oldest instead of blocking?**
Blocking queues would cause the pipeline to process stale frames. If inference falls behind, blocking would mean displaying footage from 10 seconds ago while the camera is live. Drop-oldest ensures the system always shows and counts from the most recent available frames — this is critical for a live monitoring application.

**Q: How are slot summary files generated?**
When `POST /api/slot/stop` is called, `SlotSummaryGenerator` computes declared count vs. actual counted, direction breakdown (up/down), slot start/end timestamps, and final status (OK / MISMATCH / ABORTED). This is written as a JSON file in the slot's output directory alongside the CSV events and video.

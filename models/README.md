---
license: mit
language:
- en
tags:
- object-detection
- object-tracking
- computer-vision
- yolo
- yolov11
- yolov12
- livestock
- goat
- real-time
- bytetrack
- cctv
datasets:
- custom
pipeline_tag: object-detection
---

# Goat Detection Models — YOLOv11 / YOLOv12 (Custom Trained)

Custom-trained YOLO models for **real-time goat detection** on farm CCTV footage.
Built for the [goat-detection-counting-pipeline](https://github.com/ubada11/goat-detection-counting-pipeline) project — a production-grade livestock counting system developed for **Deonar abattoir, Mumbai** (world's largest animal market).

---

## Model Variants

| File | Architecture | Image Size | Batch | LR | Optimizer | Best Epoch | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| `goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt` | YOLOv11-nano | 1024 | 16 | 0.0033 | SGD | 189 / 197 | **98.99%** | 78.05% |
| `goat_yolo11s_img1024_bs12_lr0.0025_sgd_best.pt` | YOLOv11-small | 1024 | 12 | 0.0025 | SGD | 75 / 115 | **99.05%** | 79.00% |
| `goat_yolo12n_img1024_bs12_lr0.0075_sgd_best.pt` | YOLOv12-nano | 1024 | 12 | 0.0075 | SGD | 162 / 202 | **99.04%** | 78.12% |
| `goat_yolo12s_img1024_bs12_lr0.0025_sgd_best.pt` | YOLOv12-small | 1024 | 12 | 0.0025 | SGD | 73 / 110 | **99.09%** | 79.43% |

> **Recommended for production:** `goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt`
> Best real-world performance in the live pipeline (tested on-site at Deonar). Fastest inference, lowest VRAM usage, nearly identical mAP to larger models.

---

## Dataset

- **~20,000 annotated images** — collected and annotated manually by the project team over 2+ months
- **Annotation tool:** CVAT (self-hosted on a $5/month VPS with Tailscale private network)
- **Platform:** Trained on Kaggle (P100 GPU)
- **Scene diversity:** top view, side view, inclined chute, vertical chute, horizontal chute, night conditions, dust, morning light, mixed animals, empty frames, garbage-heavy backgrounds
- **Augmentation:** mosaic (0.2), mixup (0.1), HSV shift, horizontal flip (0.4), scale (0.10), erasing (0.2), rotation (±5°)
- **Classes:** 1 (`goat`)
- **Split:** standard train/val split

---

## Training Configuration

All models trained using [Ultralytics](https://github.com/ultralytics/ultralytics) with the following shared config:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # swap for yolo11s / yolo12n / yolo12s
model.train(
    data="data.yaml",
    imgsz=1024,
    epochs=300,
    batch=16,            # 12 for s/n variants except yolo11n
    optimizer="SGD",
    lr0=0.0033,          # varies per model (see table above)
    momentum=0.937,
    weight_decay=0.0005,
    patience=20,         # early stopping
    single_cls=True,
    pretrained=True,
    amp=True,            # mixed precision
    cache="disk",
    workers=8,
    # Augmentation
    hsv_h=0.0, hsv_s=0.15, hsv_v=0.25,
    degrees=5.0, translate=0.05, scale=0.10,
    fliplr=0.4, flipud=0.0,
    mosaic=0.2, mixup=0.1, erasing=0.2,
)
```

---

## Per-Model Results

All metrics are on the validation set at the best checkpoint epoch.

---

### YOLOv11-nano (`goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt`)

**Training:** 197 epochs total, early stopped at best epoch 189 — total training time ~4.4 hours

| Metric | Value |
|---|---|
| Precision | 96.32% |
| Recall | 96.61% |
| mAP@50 | **98.99%** |
| mAP@50-95 | 78.05% |
| Val Box Loss | 0.686 |
| Val Cls Loss | 0.726 |
| Val DFL Loss | 0.859 |

**Convergence progression:**

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| 1 | 90.64% | 86.68% | 94.29% | 60.76% |
| 10 | 93.91% | 92.45% | 97.65% | 70.30% |
| 50 | 95.98% | 96.00% | 98.89% | 76.87% |
| **189 (best)** | **96.32%** | **96.61%** | **98.99%** | **78.05%** |

---

### YOLOv11-small (`goat_yolo11s_img1024_bs12_lr0.0025_sgd_best.pt`)

**Training:** 115 epochs total, early stopped at best epoch 75 — total training time ~5.1 hours

| Metric | Value |
|---|---|
| Precision | 96.69% |
| Recall | 96.75% |
| mAP@50 | **99.05%** |
| mAP@50-95 | 79.00% |
| Val Box Loss | 0.660 |
| Val Cls Loss | 0.673 |
| Val DFL Loss | 0.836 |

**Convergence progression:**

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| 1 | 92.44% | 89.33% | 96.17% | 63.77% |
| 10 | 95.01% | 94.31% | 98.44% | 73.31% |
| 50 | 96.33% | 96.60% | 99.02% | 78.73% |
| **75 (best)** | **96.69%** | **96.75%** | **99.05%** | **79.00%** |

---

### YOLOv12-nano (`goat_yolo12n_img1024_bs12_lr0.0075_sgd_best.pt`)

**Training:** 202 epochs total, early stopped at best epoch 162 — total training time ~9.2 hours

| Metric | Value |
|---|---|
| Precision | 96.61% |
| Recall | 96.60% |
| mAP@50 | **99.04%** |
| mAP@50-95 | 78.12% |
| Val Box Loss | 0.681 |
| Val Cls Loss | 0.690 |
| Val DFL Loss | 0.851 |

**Convergence progression:**

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| 1 | 90.64% | 86.49% | 94.56% | 57.15% |
| 10 | 93.26% | 92.27% | 97.23% | 68.70% |
| 50 | 95.87% | 96.10% | 98.93% | 76.98% |
| **162 (best)** | **96.61%** | **96.60%** | **99.04%** | **78.12%** |

---

### YOLOv12-small (`goat_yolo12s_img1024_bs12_lr0.0025_sgd_best.pt`)

**Training:** 110 epochs total, early stopped at best epoch 73 — total training time ~7.8 hours

| Metric | Value |
|---|---|
| Precision | 96.50% |
| Recall | 96.97% |
| mAP@50 | **99.09%** |
| mAP@50-95 | 79.43% |
| Val Box Loss | 0.658 |
| Val Cls Loss | 0.854 |
| Val DFL Loss | 0.882 |

**Convergence progression:**

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| 1 | 93.13% | 91.39% | 97.06% | 66.74% |
| 10 | 95.09% | 94.06% | 98.31% | 72.44% |
| 50 | 96.35% | 96.77% | 99.07% | 79.03% |
| **73 (best)** | **96.50%** | **96.97%** | **99.09%** | **79.43%** |

---

## Model Comparison Summary

| Model | Params | mAP@50 | mAP@50-95 | Best Epoch | Train Time | Recommended For |
|---|---|---|---|---|---|---|
| YOLOv11-nano | ~2.6M | 98.99% | 78.05% | 189/197 | ~4.4h | ✅ Production (live CCTV) |
| YOLOv11-small | ~9.4M | 99.05% | 79.00% | 75/115 | ~5.1h | High accuracy offline |
| YOLOv12-nano | ~2.6M | 99.04% | 78.12% | 162/202 | ~9.2h | Comparison/research |
| YOLOv12-small | ~9.4M | **99.09%** | **79.43%** | 73/110 | ~7.8h | Best accuracy overall |

**Key insight:** All four models converge to nearly identical mAP@50 (~99%), demonstrating the quality of the 20k dataset. The nano models achieve 98.99–99.04% — within 0.1% of the small models — making them the clear choice for real-time edge deployment.

---

## Live Pipeline Performance

Tested on-site at Deonar abattoir using GTX 1650 (4GB VRAM), 8GB RAM:

| Metric | Value |
|---|---|
| Inference latency p50 | ~93 ms (~10 FPS) |
| Inference latency p90 | ~156 ms |
| Inference latency p95 | ~178 ms |
| On-site miss rate | < 5% on 180-goat sessions |
| Stream type | Live RTSP from farm CCTV |

---

## How to Use

### With the counting pipeline (recommended)

```bash
git clone https://github.com/ubada11/goat-detection-counting-pipeline
cd goat-detection-counting-pipeline

# Download the recommended model
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ubada11/goat-detection-yolov11',
    filename='goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt',
    local_dir='models/'
)
"

# Update configs/config.yaml:
# paths:
#   weights: goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt
python main.py
```

### Standalone inference

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="ubada11/goat-detection-yolov11",
    filename="goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt",
    local_dir="models/"
)

model = YOLO(path)
results = model.predict("your_video.mp4", conf=0.25, iou=0.45)
```

---

## Project Context

This model was developed as part of an R&D project for **MI Tradings & General Suppliers** (a BMC contractor) to automate livestock counting at **Deonar abattoir, Mumbai** — replacing manual head-counting with a real-time AI pipeline.

The project involved 12 months of work: on-site visits to study camera angles and chute geometry, 2+ months of data collection across all lighting and weather conditions, manual annotation of 20,000 images using CVAT, iterative training on Kaggle, and live on-site testing with real CCTV feeds.

The full pipeline includes dual-line crossing detection, ByteTrack multi-object tracking, a slot management REST API for per-vendor counting sessions, and a WebRTC live preview server.

**Team:** Ubada Ghawte (lead engineer & sole coder), Adil, Raafe
**Guide:** Prof. Farhan (Rizvi College of Engineering, Mumbai)
**Duration:** 2024–2025

---

## License

MIT — free to use, modify, and distribute.

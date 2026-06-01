<h1 align="center">DEONAR AI</h1>

<p align="center">
Real-Time Livestock Detection, Tracking & Counting Platform
</p>

<p align="center">
Built for Deonar Abattoir, Mumbai
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch">
  <img src="https://img.shields.io/badge/YOLOv11-Detection-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/YOLOv12-Detection-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/ByteTrack-Tracking-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/OpenCV-ComputerVision-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge">
  <img src="https://img.shields.io/badge/WebRTC-LiveStreaming-purple?style=for-the-badge">
  <img src="https://img.shields.io/badge/HuggingFace-ModelHosting-yellow?style=for-the-badge">
</p>

---

# Overview

DEONAR AI is a production-grade livestock counting platform developed to automate animal counting operations at Deonar Abattoir, Mumbai.

The system combines YOLO-based detection, multi-object tracking, motion-aware counting intelligence, vendor session management, and real-time monitoring to replace manual counting processes with an accurate, auditable, and scalable AI solution.

### Highlights

* Real-time CCTV counting
* YOLOv11 & YOLOv12 detection models
* ByteTrack & BoT-SORT tracking
* Dual-line counting intelligence
* Vendor slot management
* WebRTC live monitoring
* Multi-threaded processing pipeline
* CSV audit trail & reporting
* Production deployment ready

---

# System Overview

<p align="center">
  <img src="assets/architecture/hero-overview.png" width="100%">
</p>

<p align="center">
  <i>
  End-to-end overview of the Deonar AI livestock counting platform.
  </i>
</p>

---

## Complete System Architecture

<p align="center">
  <img src="assets/architecture/system-architecture.png" width="100%">
</p>

<p align="center">
  <i>
  Complete system architecture showing video ingestion, AI processing, counting intelligence, vendor management and reporting.
  </i>
</p>

### Core Workflow

```text
CCTV Streams
      ↓
YOLO Detection
      ↓
Multi-Object Tracking
      ↓
Counting Intelligence
      ↓
Vendor Slot Mapping
      ↓
Reports & Audit Trail
```

---

# Counting Intelligence Engine

<p align="center">
  <img src="assets/architecture/counting-engine.png" width="100%">
</p>

<p align="center">
  <i>
  Motion-aware counting engine designed to ensure each animal is counted exactly once.
  </i>
</p>

The counting engine combines:

* Object Detection
* Multi-Object Tracking
* Dual-Line Verification
* Motion Validation
* Direction Analysis
* Count Confirmation Logic

This architecture significantly reduces false counts caused by occlusion, jitter, direction changes, and tracking ID switches.

---

# Deployment Architecture

<p align="center">
  <img src="assets/architecture/deployment-architecture.png" width="100%">
</p>

<p align="center">
  <i>
  Production deployment architecture used for real-time livestock counting operations.
  </i>
</p>

---

# Technology Stack

### AI & Deep Learning

* PyTorch
* Ultralytics
* YOLOv11
* YOLOv12

### Tracking

* ByteTrack
* BoT-SORT

### Computer Vision

* OpenCV
* NumPy

### Live Streaming

* WebRTC
* aiortc
* aiohttp

### Backend & APIs

* FastAPI
* Uvicorn
* Pydantic

### Monitoring & Logging

* Rich
* JSONL Logging
* CSV Reporting

### Infrastructure

* CVAT
* Vast.ai (RTX 5090)
* HuggingFace Hub
* Tailscale

---

# Dataset & Training

| Metric                  | Value             |
| ----------------------- | ----------------- |
| Dataset Size            | 20,000+ Images    |
| Annotation Platform     | CVAT              |
| Training Infrastructure | Vast.ai RTX 5090  |
| Detection Models        | YOLOv11 & YOLOv12 |
| Validation Accuracy     | ~99% mAP@50       |

The dataset was collected and annotated from real operational environments at Deonar Abattoir under varying lighting, density, camera angles and environmental conditions.

---

# Project Structure

```text
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

# Results

| Metric             | Performance              |
| ------------------ | ------------------------ |
| Detection Accuracy | ~99% mAP@50              |
| Miss Rate          | < 5%                     |
| Processing Mode    | Real-Time                |
| Tracking           | ByteTrack / BoT-SORT     |
| Streaming          | WebRTC                   |
| Reporting          | CSV + Video + Audit Logs |

---

# Real-World Deployment

The platform was developed and tested for livestock counting operations at Deonar Abattoir, Mumbai.

Key capabilities include:

* Automated livestock counting
* Vendor-wise session management
* Real-time monitoring
* Audit-ready reporting
* Operational analytics
* Production deployment support

---

# Contributors

**Ubada Ghawte**
AI Engineer & Project Lead

### Team Members

* Adil
* Raafe

### Academic Guide

* Prof. Farhan
* Rizvi College of Engineering

### Industry Partner

MI Tradings & General Suppliers

---

# License

This project is released under the Apache License 2.0.
For details, see the LICENSE file included in the repository.

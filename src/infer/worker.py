# src/infer/worker.py
"""
InferenceWorker (threaded).

Responsibilities:
 - Load model (prefer helper loader if available)
 - Convert capture frames into safe numpy HWC uint8
 - Crop ROI (using cap_info if provided or falling back to sample-frame geometry)
 - Call track_once(...) and produce a stable minimal result dict pushed to result_queue

Design notes:
 - Keep initialization light in __init__; do heavy init in _init() inside the thread.
 - Provide consistent result dict keys so DisplayWorker can rely on a stable contract.
 - Defensive logging and graceful shutdown on exceptions.
"""

import threading, queue, time, traceback, torch
from typing import Optional, Any

import numpy as np

from src.utils.logger import log
from src.infer.yolo_infer import track_once
from src.infer.loader import load_model_threaded, load_model as yolo_load_model
from src.runtime_configs.bytetrack_cfg import make_bytetrack_yaml
from src.geometry.geom import _prepare_geometry, clamp_roi
from src.utils.pacing_contract_dict import _ensure_item_dict


class _FeederLike:
    """
    Minimal feeder-like object used to carry frame and geometry info through the pipeline.

    Attributes (set and used by InferenceWorker and DisplayWorker):
        frame_in (int): legacy frame input index.
        out_index (int): legacy processed frame index.
        roi (np.ndarray): cropped ROI image.
        full (np.ndarray): full-frame image.
        rx, ry, rw, rh (int): ROI rectangle (x, y, width, height).
        W, H (int): full-frame resolution.
        Additional attributes may be attached dynamically by calling code (e.g., src_frame_idx).
    """

    def __init__(self):
        self.frame_in = 0
        self.out_index = 0
        self.roi = None
        self.full = None
        self.rx = None
        self.ry = None
        self.rw = None
        self.rh = None
        self.W = None
        self.H = None


class InferenceWorker(threading.Thread):
    """
    Threaded inference worker.

    The thread:
      - consumes items from pacing_out_q (pacing -> inference),
      - ensures model and geometry initialized inside the thread (_init),
      - converts incoming frames to HWC uint8 numpy arrays,
      - crops ROI, runs track_once(...),
      - composes a stable result dict and attempts to place it in result_queue.

    Constructor arguments:
        pacing_out_q (queue.Queue): queue with capture frames (from PacingController).
        result_queue (queue.Queue): queue where inference results are pushed.
        stop_event (threading.Event): event used to signal graceful shutdown.
        args: parsed args or SimpleNamespace with runtime and inference knobs.
        cap_info (Optional[dict]): optional capture metadata (width/height/total).

    Important behavior notes:
      - Heavy initialization (model loading, tracker YAML, ROI computation) is deferred
        to _init(sample_frame) and executed inside the thread to avoid import-time work.
      - Results use a stable contract (keys include: frame_id, timestamp, frame, roi, dets, feeder, avg_fps, queue stats).
      - Uses a small FPS rolling buffer for infer FPS smoothing.
    """

    def __init__(
        self,
        pacing_out_q: "queue.Queue",
        result_queue: "queue.Queue",
        stop_event: threading.Event,
        args,
        cap_info: Optional[dict] = None,
        metrics: Optional[Any] = None,
    ):
        super().__init__(daemon=True)
        self.pacing_out_q = pacing_out_q
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.args = args
        self.cap_info = cap_info or {}

        # model fields
        self.model = None
        self.model_loader_used = None
        self.device = args.device
        self.half = args.half
        self.fuse = args.fuse
        self.tracker_yaml = None

        # ROI geometry (set during _init)
        self.rx = self.ry = self.rw = self.rh = None
        self.roi_area = None

        # fps buffer
        self._fps_buf = []
        self._fps_buf_len = 8

        # counters (persist across frames)
        self.frame_in_counter = 0
        self.out_index_counter = 0

        # init flag
        self._worker_initialized = False
        
        self.metrics = metrics

    # ---------------- conversion helpers ----------------
    def _normalize_dtype_and_channels(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize numpy array dtype and channels to HWC uint8.

        - Handles channels-first -> HWC transpose guess.
        - Converts grayscale to 3-channel.
        - Clips and converts floating arrays scaled to 0..1 or 0..255.
        - Ensures returned dtype is uint8 with 3 channels.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("_normalize_dtype_and_channels expects numpy array")

        # channels-first -> HWC guess
        if (
            img.ndim == 3
            and img.shape[0] in (1, 3, 4)
            and img.shape[2] not in (1, 3, 4)
        ):
            img = np.transpose(img, (1, 2, 0))

        # grayscale -> 3-channel
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # ensure 3 channels
        if img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        # dtype handling
        if np.issubdtype(img.dtype, np.floating):
            mx = float(np.nanmax(img)) if img.size > 0 else 0.0
            if mx <= 1.5:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        else:
            if img.dtype != np.uint8:
                img = img.astype(np.int32).clip(0, 255).astype(np.uint8)

        return img

    def _to_numpy_image(self, x) -> np.ndarray:
        """
        Convert supported input types to HWC uint8 numpy image.

        Accepts:
          - torch.Tensor (CPU/GPU tensors are detached and moved to CPU)
          - numpy.ndarray
          - PIL.Image.Image
          - Any object convertible via numpy.array()

        Raises:
            TypeError for unsupported types.
        """
        # torch tensor
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            if t.ndim == 4 and t.shape[0] == 1:
                t = t[0]
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                t = t.permute(1, 2, 0).contiguous()
            try:
                arr = t.numpy()
            except Exception:
                arr = t.cpu().float().numpy()
            return self._normalize_dtype_and_channels(arr)

        # numpy
        if isinstance(x, np.ndarray):
            return self._normalize_dtype_and_channels(x)

        # PIL fallback and generic
        try:
            import PIL.Image as PILImage  # local import

            if isinstance(x, PILImage.Image):
                arr = np.array(x)
                return self._normalize_dtype_and_channels(arr)
        except Exception:
            pass

        try:
            arr = np.array(x)
            if isinstance(arr, np.ndarray):
                return self._normalize_dtype_and_channels(arr)
        except Exception as e:
            raise TypeError(
                f"Unsupported frame type for conversion to numpy: {type(x).__name__} - {e}"
            )

    # ---------------- initialization inside thread ----------------
    def _init(self, sample_frame: np.ndarray):
        """
        Heavy initialization performed inside the worker thread.

        Actions:
          - Resolve weights path from args (supports legacy names).
          - Attempt helper loader (load_model_threaded) first; on failure fall back to yolo_load_model.
          - Prepare ByteTrack YAML (if configured).
          - Compute ROI geometry using cap_info (if present) or via _prepare_geometry(sample_frame).
          - Sets self._worker_initialized = True on success.

        Raises and logs any model load failures.
        """
        weights = getattr(self.args, "weights", getattr(self.args, "model_path", None))
        if weights is None:
            weights = getattr(self.args, "WEIGHTS", None)

        # 1) Try helper loader first if present
        if load_model_threaded is not None:
            try:
                loader_type, model_obj, effective_device = load_model_threaded(
                    weights, device_arg=self.device, half=self.half, fuse=self.fuse
                )
                self.model = model_obj
                self.model_loader_used = f"helper:{loader_type}"
                log.info(
                    "INFER-MODEL",
                    f"Loaded model via helper loader (type={loader_type}) -> device={effective_device}",
                )
            except Exception as e:
                log.warn(
                    "INFER-MODEL",
                    f"Helper model_loader failed ({e}); falling back to yolo_infer loader",
                )
                log.debug("INFER-MODEL", traceback.format_exc())
                self.model = None
                self.model_loader_used = None

        # 2) fallback to yolo_infer.load_model (original behavior)
        if self.model is None:
            try:
                model_obj, effective_device = yolo_load_model(
                    weights, self.device, self.half, fuse=False, quiet=True
                )
                self.model = model_obj
                self.model_loader_used = "yolo_infer"
                log.info(
                    "INFER-MODEL",
                    f"Loaded model via yolo_infer -> device={effective_device}",
                )
            except Exception as e:
                log.error("INFER-MODEL", f"Failed to load model: {e}")
                log.debug("INFER-MODEL", traceback.format_exc())
                raise

        # tracker yaml
        try:
            self.tracker_yaml = make_bytetrack_yaml(self.args)
        except Exception:
            self.tracker_yaml = None

        # compute ROI geometry: use cap_info if available, else infer from sample_frame
        try:
            if getattr(self, "cap_info", None):
                W = int(self.cap_info.get("width", sample_frame.shape[1]))
                H = int(self.cap_info.get("height", sample_frame.shape[0]))
                rx = int(W * float(self.args.roi_xr))
                ry = int(H * float(self.args.roi_yr))
                rw = int(W * float(self.args.roi_wr))
                rh = int(H * float(self.args.roi_hr))
                rx, ry, rw, rh = clamp_roi(rx, ry, rw, rh, W, H)
                self.rx, self.ry, self.rw, self.rh = rx, ry, rw, rh
                self.roi_area = max(1, rw * rh)
            else:
                rx, ry, rw, rh = _prepare_geometry(
                    sample_frame,
                    self.args.roi_xr,
                    self.args.roi_yr,
                    self.args.roi_wr,
                    self.args.roi_hr,
                )
                self.rx, self.ry, self.rw, self.rh = rx, ry, rw, rh
                self.roi_area = max(1, rw * rh)
        except Exception:
            h, w = sample_frame.shape[:2]
            self.rx, self.ry, self.rw, self.rh = 0, 0, w, h
            self.roi_area = max(1, w * h)

        self._worker_initialized = True
        log.debug(
            "INFER-MODEL",
            f"InferenceWorker initialization complete (loader={self.model_loader_used})",
        )

    # ---------------- queue helper ----------------
    def _safe_put_result(self, res: dict):
        """
        Put result dict into result_queue non-blocking.

        Behavior:
          - Tries put_nowait().
          - If queue.Full, evicts the oldest item (get_nowait()) and retries once.
          - If still failing, logs a warning and drops the result.
        """
        try:
            self.result_queue.put_nowait(res)
            return True
        except queue.Full:
            try:
                _ = self.result_queue.get_nowait()
            except Exception:
                pass
            try:
                self.result_queue.put_nowait(res)
                return True
            except Exception:
                log.warn(
                    "INFER-MODEL",
                    f"Dropping result {res.get('frame_id')} due to full result_queue",
                )
                return False

    # ---------------- main run loop ----------------
    def run(self):
        """
        Thread entrypoint.

        Main loop:
          - Consume item from pacing_out_q (with timeout).
          - Ensure worker initialization with a safe sample frame.
          - Convert incoming frame to numpy HWC uint8, crop ROI.
          - Call track_once(...) to get detections.
          - Build stable result dict and call _safe_put_result().
          - Maintain counters and fps buffer.
          - Mark pacing_out_q.task_done() when appropriate.
        """
        log.info("INFER-WORKER", "InferenceWorker started")
        while not self.stop_event.is_set():
            try:
                raw_item = self.pacing_out_q.get(timeout=0.5)
            except queue.Empty:
                continue

            item = _ensure_item_dict(raw_item)

            try:
                # item contract:
                # { "frame": np.ndarray, "frame_index": int, "capture_time": float, "source_time": Optional[float], ... }
                frame = item.get("frame")
                frame_id = item.get("frame_index", getattr(item, "frame_index", None))
                # prefer capture_time (monotonic) else source_time then fallback to now()
                ts = (
                    item.get("capture_time")
                    or item.get("source_time")
                    or time.monotonic()
                )

                # persistent counters (one increment per frame read)
                self.frame_in_counter += 1
                
                if hasattr(self, "metrics") and self.metrics is not None:
                    self.metrics.mark(int(frame_id or -1), "infer_start")

                # ensure initialized (model + geometry) inside thread using a safe sample
                if not self._worker_initialized:
                    try:
                        sample = (
                            frame
                            if isinstance(frame, np.ndarray)
                            else (
                                np.array(frame)
                                if frame is not None
                                else np.zeros((480, 640, 3), dtype=np.uint8)
                            )
                        )
                        sample = self._to_numpy_image(sample)
                        self._init(sample)
                    except Exception:
                        log.error(
                            "INFER-MODEL",
                            "Initial setup failed in InferenceWorker; stopping.",
                        )
                        log.debug("INFER-MODEL", traceback.format_exc())
                        self.stop_event.set()
                        break

                # normalize to numpy HWC uint8
                try:
                    full_frame = self._to_numpy_image(frame)
                except Exception as e:
                    log.error(
                        "INFER-MODEL", f"Failed converting capture frame to numpy: {e}"
                    )
                    log.debug("INFER-MODEL", traceback.format_exc())
                    # skip frame
                    continue

                # crop ROI cleanly
                rx, ry, rw, rh = (
                    int(self.rx or 0),
                    int(self.ry or 0),
                    int(self.rw or full_frame.shape[1]),
                    int(self.rh or full_frame.shape[0]),
                )
                Hf, Wf = full_frame.shape[:2]
                rx = max(0, min(rx, Wf - 1))
                ry = max(0, min(ry, Hf - 1))
                rw = max(1, min(rw, Wf - rx))
                rh = max(1, min(rh, Hf - ry))
                roi = full_frame[ry : ry + rh, rx : rx + rw].copy()

                # prepare feeder-like object (used by processing functions in DisplayWorker)
                feeder = _FeederLike()
                feeder.src_frame_idx = int(
                    item.get("frame_index")
                    if item.get("frame_index") is not None
                    else 0
                )
                feeder.proc_frame_idx = int(self.out_index_counter)
                # keep compatibility — set legacy attributes but prefer explicit names
                feeder.frame_in = feeder.src_frame_idx
                feeder.out_index = feeder.proc_frame_idx
                feeder.roi = roi
                feeder.full = full_frame
                feeder.rx = rx
                feeder.ry = ry
                feeder.rw = rw
                feeder.rh = rh
                feeder.W = Wf
                feeder.H = Hf

                # call model tracking
                try:
                    t0 = time.perf_counter()
                    dets = track_once(
                        self.model,
                        roi,
                        self.args,
                        self.tracker_yaml,
                        self.roi_area,
                        None,
                    )
                    t1 = time.perf_counter()
                    infer_time = t1 - t0
                    if infer_time and infer_time > 0:
                        self._fps_buf.append(1.0 / infer_time)
                        if len(self._fps_buf) > self._fps_buf_len:
                            self._fps_buf.pop(0)
                except Exception as e:
                    log.error(
                        "INFER-MODEL", f"track_once failed on frame {frame_id}: {e}"
                    )
                    log.debug("INFER-MODEL", traceback.format_exc())
                    dets = []
                    infer_time = None

                # build stable result payload
                result = {
                    "frame_id": frame_id,
                    "timestamp": ts,
                    "frame": full_frame,
                    "roi": roi,
                    "dets": dets,
                    "feeder": feeder,
                    "frame_in_counter": self.frame_in_counter,
                    "out_index_counter": self.out_index_counter,
                    "avg_fps": (
                        (sum(self._fps_buf) / len(self._fps_buf))
                        if len(self._fps_buf) > 0
                        else 0.0
                    ),
                    "pacing_out_q_fill": (
                        self.pacing_out_q.qsize()
                        if hasattr(self.pacing_out_q, "qsize")
                        else None
                    ),
                    "pacing_out_q_max": (
                        self.pacing_out_q.maxsize
                        if hasattr(self.pacing_out_q, "maxsize")
                        else None
                    ),
                    "result_q_fill": (
                        self.result_queue.qsize()
                        if hasattr(self.result_queue, "qsize")
                        else None
                    ),
                    "result_q_max": (
                        self.result_queue.maxsize
                        if hasattr(self.result_queue, "maxsize")
                        else None
                    ),
                }

                # push result (non-blocking)
                frame_proc = self._safe_put_result(result)

                if frame_proc:
                    self.out_index_counter += (
                        1  # Frame successfully processed and sent to result_queue
                    )
                    if hasattr(self, "metrics") and self.metrics is not None:
                        self.metrics.mark(int(frame_id or -1), "infer_end", ts=time.perf_counter(), extra={"infer_wall_s": t1-t0})
                else:
                    if hasattr(self, "metrics") and self.metrics is not None:
                        self.metrics.incr("infer_result_dropped")
                    # dropped result; do not increment out_index_counter
                    pass
            except Exception as e:
                log.error("INFER-MODEL", f"Exception processing item: {e}")
                log.debug("INFER-MODEL", traceback.format_exc())
                # continue (we still want to call task_done() for the consumed item)
            finally:
                # mark the pacing_out_q item as processed (safe-guarded)
                try:
                    if hasattr(self.pacing_out_q, "task_done"):
                        self.pacing_out_q.task_done()
                except Exception:
                    pass

        log.info("INFER-WORKER", "InferenceWorker exiting")

    def get_stats(self) -> dict:
        """
        Return a small status dictionary summarizing inference activity.

        Keys:
          - infer_frames_received
          - infer_frames_processed
          - infer_frames_dropped
          - avg_infer_fps
        """
        status = {
            "infer_frames_received": self.frame_in_counter,
            "infer_frames_processed": self.out_index_counter,
            "infer_frames_dropped": self.frame_in_counter - self.out_index_counter,
            "avg_infer_fps": (
                (sum(self._fps_buf) / len(self._fps_buf))
                if len(self._fps_buf) > 0
                else 0.0
            ),
        }
        return status

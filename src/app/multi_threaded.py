# src/app/multi_threaded.py

import time, traceback, threading, queue

from src.display.drawing import _prepare_drawing
from src.io.io import CsvWriters
from src.capture.worker import ThreadedVideoCapture
from src.runtime.pacing import PacingController
from src.infer.worker import InferenceWorker
from src.display.worker import DisplayWorker
from src.utils.metrics import MetricsCollector
from src.utils.logger import log

from src.display.webrtc_server import WebRTCServer
import queue as std_queue

from pathlib import Path


# ===============================================================
# 🧩 Helper Functions — Modularized from run_threaded()
# Each helper keeps *identical logic* from the original monolithic function.
# Only docstrings and clarity improvements added.
# ===============================================================


def _normalize_source(args):
    """
    Normalize the video source input.

    - Converts backslashes to forward slashes for consistency.
    - Fixes malformed RTSP URLs missing '//' (e.g., "rtsp:..." -> "rtsp://...").
    - Converts numeric strings like "0" into integer webcam indices.
    - Returns the normalized or converted source value.
    """
    src_raw = args.source
    if isinstance(src_raw, str):
        src_norm = src_raw.replace("\\", "/")
        if src_norm.lower().startswith("rtsp:") and not src_norm.lower().startswith(
            "rtsp://"
        ):
            rest = src_norm[len("rtsp:") :].lstrip("/\\")
            src_norm = "rtsp://" + rest
    else:
        src_norm = src_raw

    if isinstance(src_norm, str) and src_norm.isdigit():
        src_final = int(src_norm)
    else:
        src_final = src_norm

    return src_final


def _create_capture_thread(
    src_final, capture_queue, stop_event, args, cap_backend, metrics=None
):
    """
    Create and start the video capture thread safely.

    - Initializes ThreadedVideoCapture with the given source and queues.
    - Attempts to start capture immediately.
    - Logs all progress and errors.
    - Returns the capture thread object.
    """
    try:
        capture = ThreadedVideoCapture(
            src_final,
            capture_queue,
            stop_event,
            cap_backend=cap_backend,
            reconnect_delay=float(getattr(args, "reconnect_delay", 3.0)),
            metrics=metrics,
        )
        log.debug("RUNNER", "Starting capture thread...")
        capture.start()
        return capture
    except Exception as e:
        log.error("RUNNER", f"Failed to create/start capture thread: {e}")
        raise


def _wait_for_cap_info(capture, wait_timeout, poll_interval):
    """
    Wait for the capture thread to provide video metadata (cap_info).

    - Polls capture.cap_info periodically until available or timeout.
    - Detects if capture thread exits unexpectedly.
    - Returns cap_info dict or None.
    """
    cap_info = None
    wait_t = 0.0
    while wait_t < wait_timeout:
        cap_info = getattr(capture, "cap_info", None)
        if cap_info:
            break
        if not getattr(capture, "is_alive", lambda: True)():
            log.error("RUNNER", "Capture thread exited before providing cap_info.")
            break
        time.sleep(poll_interval)
        wait_t += poll_interval

    if cap_info:
        log.debug("RUNNER", f"Got cap_info from capture thread: {cap_info}")
    else:
        log.warn(
            "RUNNER",
            f"cap_info not available after {wait_timeout}s — inference will infer geometry from first frame.",
        )
    return cap_info


def _wait_for_vision_ready(display, timeout=5.0, poll=0.05) -> bool:
    """
    Block until DisplayWorker is alive and state is initialized.

    This defines 'vision system READY'.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            if (
                display
                and display.is_alive()
                and hasattr(display, "state")
                and display.state is not None
            ):
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False


def _prepare_injected_context(args, cap_info):
    """
    Prepare injected reusable components for DisplayWorker.

    - Prepares drawing resources (animator, colorer, pretty_cfg).
    - Initializes CSV writers for events and time-series outputs.
    - Injects total frame count (if available) into context.
    - Returns (injected_dict, injected_csvs).
    """
    injected = {}
    injected_csvs = None
    try:
        try:
            animator, colorer, pretty_cfg = _prepare_drawing(args)
            injected["animator"] = animator
            injected["colorer"] = colorer
            injected["pretty_cfg"] = pretty_cfg
            log.debug(
                "RUNNER", "Prepared drawing context for DisplayWorker (injected)."
            )
        except Exception as e:
            log.warn("RUNNER", f"_prepare_drawing failed for injection: {e}")
            log.debug("RUNNER", traceback.format_exc())

        try:
            decisions_path = (
                args.csv_decisions if hasattr(args, "dual_lines_enabled") else None
            )
            injected_csvs = CsvWriters(
                events_path=args.csv_events,
                ts_path=args.csv_timeseries,
                decisions_path=decisions_path,
            )
            injected["csvs"] = injected_csvs
            log.debug("RUNNER", "Prepared CsvWriters for DisplayWorker (injected).")
        except Exception as e:
            log.warn("RUNNER", f"CsvWriters creation failed for injection: {e}")
            log.debug("RUNNER", traceback.format_exc())

        if cap_info:
            injected["total"] = int(cap_info.get("total", 0) or 0)
            injected["fps"] = float(cap_info.get("fps", 25.0) or 25.0)
        else:
            injected["total"] = getattr(args, "total_frames", 0)
            injected["fps"] = float(getattr(args, "source_fps", 25.0) or 25.0)
    except Exception:
        injected = {}

    return injected, injected_csvs


def _build_runtime_cfg(args):
    """
    Build or retrieve runtime pacing configuration.

    - Uses args.runtime_cfg if provided.
    - Otherwise constructs defaults from args attributes.
    - Returns runtime_cfg dictionary.
    """
    runtime_cfg = getattr(args, "runtime_cfg", None)
    if runtime_cfg is None:
        log.debug("RUNNER", "No runtime_cfg provided; building from args defaults")
        runtime_cfg = {
            "sync": getattr(args, "sync", True),
            "playback_speed": getattr(args, "playback_speed", 1.0),
            "autoskip": getattr(args, "autoskip", False),
            "max_lag_s": getattr(args, "max_lag_s", 0.75),
            "skip_policy": getattr(args, "skip_policy", "drop_to_latest"),
            "sync_jitter_allowance_s": getattr(args, "sync_jitter_allowance_s", 0.02),
            "max_sleep_s": getattr(args, "max_sleep_s", 1.0),
            "max_catchup_resync_s": getattr(args, "max_catchup_resync_s", 5.0),
            "cap_qsize": getattr(args, "cap_qsize", 3),
            "res_qsize": getattr(args, "res_qsize", 12),
        }
    return runtime_cfg


def _create_and_start_pacer(capture_queue, pacing_out_q, runtime_cfg, metrics=None):
    """
    Initialize and start the PacingController thread.

    - Controls frame pacing between capture and inference queues.
    - Logs configuration and startup status.
    - Returns the PacingController instance.
    """
    pacer = PacingController(
        capture_q=capture_queue, out_q=pacing_out_q, cfg=runtime_cfg, metrics=metrics
    )
    try:
        log.debug(
            "RUNNER", f"Starting pacing controller, with config: {str(runtime_cfg)}"
        )
        pacer.start()
    except Exception as e:
        log.error("RUNNER", f"Failed to start pacing controller: {e}")
    return pacer


def _create_workers(
    pacing_out_q, result_queue, stop_event, args, cap_info, injected, metrics=None
):
    """
    Create inference and display worker threads.

    - Initializes InferenceWorker (for detection/tracking).
    - Initializes DisplayWorker (for visualization and CSV saving).
    - Returns (infer, display) worker objects.
    """
    infer, display = None, None
    try:
        infer = InferenceWorker(
            pacing_out_q,
            result_queue,
            stop_event,
            args,
            cap_info=cap_info,
            metrics=metrics,
        )
        display = DisplayWorker(result_queue, stop_event, args, injected=injected)
    except Exception as e:
        log.error("RUNNER", f"ERROR creating worker objects: {e}")
        log.debug("RUNNER", traceback.format_exc())
    return infer, display


def _start_workers(infer, display):
    """
    Start inference and display threads safely.

    - Starts each worker if available.
    - Logs startup success or failure for each thread.
    """
    if infer is not None:
        try:
            log.debug("RUNNER", "Starting inference thread...")
            infer.start()
        except Exception as e:
            log.error("RUNNER", f"Failed to start inference: {e}")
    else:
        log.error("RUNNER", "Inference worker is None; skipping start()")

    if display is not None:
        try:
            log.debug("RUNNER", "Starting display thread...")
            display.start()
        except Exception as e:
            log.error("RUNNER", f"Failed to start display: {e}")
    else:
        log.error("RUNNER", "Display worker is None; skipping start()")


def _monitor_threads(
    capture,
    pacer,
    infer,
    display,
    capture_queue,
    pacing_out_q,
    result_queue,
    stop_event,
    cap_info,
    args,
    metrics=None,
):
    """
    Monitor and restart threads if they crash unexpectedly.

    - Monitors thread liveness and queue utilization every 1s (internal check loop uses 0.1s sleep).
    - Emits an error log the moment a thread is detected dead (no repeated spam).
    - Emits an info log when a thread recovers.
    - Restarts pacer or inference worker once if they die.
    - Returns possibly updated (pacer, infer) references.
    """
    infer_restarted = False
    pacer_restarted = False

    # track last-known liveness to avoid repeated logs
    last_alive = {
        "capture": None,
        "pacer": None,
        "infer": None,
        "display": None,
    }
    # track whether we've already emitted an error for a dead thread (to avoid spam)
    error_reported = {
        "capture": False,
        "pacer": False,
        "infer": False,
        "display": False,
    }

    last_monitor = 0.0
    last_queue_log = 0.0
    try:
        while not stop_event.is_set():
            now = time.time()
            if now - last_monitor >= 1.0:
                last_monitor = now
                try:
                    # helper to probe liveness safely
                    def is_capture_alive():
                        return getattr(capture, "is_alive", lambda: False)()

                    def is_pacer_alive():
                        # pacer may be None or may expose either _thread or is_alive
                        try:
                            if pacer is None:
                                return False
                            # if pacer has an is_alive method
                            if hasattr(pacer, "is_alive"):
                                return getattr(pacer, "is_alive")()
                            # fallback to _thread if present
                            if hasattr(pacer, "_thread") and pacer._thread is not None:
                                return getattr(
                                    pacer._thread, "is_alive", lambda: False
                                )()
                            return False
                        except Exception:
                            return False

                    def is_infer_alive():
                        return (
                            getattr(infer, "is_alive", lambda: False)()
                            if infer is not None
                            else False
                        )

                    def is_disp_alive():
                        return (
                            getattr(display, "is_alive", lambda: False)()
                            if display is not None
                            else False
                        )

                    cap_alive = is_capture_alive()
                    pacer_alive = is_pacer_alive()
                    inf_alive = is_infer_alive()
                    disp_alive = is_disp_alive()

                    # Only log queue utilization every 5 seconds to avoid spamming
                    if now - last_queue_log >= 5.0:
                        last_queue_log = now
                        try:
                            cap_q_fill = (
                                capture_queue.qsize()
                                if hasattr(capture_queue, "qsize")
                                else None
                            )
                            pacing_q_fill = (
                                pacing_out_q.qsize()
                                if hasattr(pacing_out_q, "qsize")
                                else None
                            )
                            res_q_fill = (
                                result_queue.qsize()
                                if hasattr(result_queue, "qsize")
                                else None
                            )

                            cap_q_max = (
                                capture_queue.maxsize
                                if hasattr(capture_queue, "maxsize")
                                else None
                            )
                            pacing_q_max = (
                                pacing_out_q.maxsize
                                if hasattr(pacing_out_q, "maxsize")
                                else None
                            )
                            res_q_max = (
                                result_queue.maxsize
                                if hasattr(result_queue, "maxsize")
                                else None
                            )

                            pacer_stats = None
                            try:
                                if hasattr(pacer, "get_stats"):
                                    pacer_stats = pacer.get_stats()
                            except Exception:
                                pacer_stats = None

                            infer_stats = None
                            try:
                                if hasattr(infer, "get_stats"):
                                    infer_stats = infer.get_stats()
                            except Exception:
                                infer_stats = None

                            log.debug(
                                "RUNNER-MONITOR",
                                (
                                    f"QUEUES utilization: cap_q={cap_q_fill}/{cap_q_max} "
                                    f"pacing_q={pacing_q_fill}/{pacing_q_max} res_q={res_q_fill}/{res_q_max}"
                                    + (
                                        f" PACER_STATS={pacer_stats}"
                                        if pacer_stats
                                        else ""
                                    )
                                    + (
                                        f" INFER_STATS={infer_stats}"
                                        if infer_stats
                                        else ""
                                    )
                                ),
                            )
                        except Exception as e:
                            log.error(
                                "RUNNER-MONITOR",
                                f"Monitor error (collecting queues): {e}",
                            )

                    # now check transitions and only log on changes or errors
                    def check_and_report(name, alive):
                        prev = last_alive.get(name)
                        if prev is None:
                            # first observation: only log if dead to avoid startup noise
                            if not alive:
                                log.error(
                                    "RUNNER-MONITOR",
                                    f"{name.upper()} not alive on first check",
                                )
                                error_reported[name] = True
                        else:
                            # state transition: alive -> dead => immediate error
                            if prev and not alive:
                                log.error(
                                    "RUNNER-MONITOR", f"{name.upper()} became not alive"
                                )
                                error_reported[name] = True
                            # state transition: dead -> alive => info (recovery)
                            elif not prev and alive:
                                log.info(
                                    "RUNNER-MONITOR",
                                    f"{name.upper()} recovered and is now alive",
                                )
                                error_reported[name] = False
                            # otherwise: no repeated logs for same state
                        last_alive[name] = alive

                        # If currently dead and not yet reported, report error now
                        if not alive and not error_reported.get(name, False):
                            log.error("RUNNER-MONITOR", f"{name.upper()} not alive")
                            error_reported[name] = True

                    check_and_report("capture", cap_alive)
                    check_and_report("pacer", pacer_alive)
                    check_and_report("infer", inf_alive)
                    check_and_report("display", disp_alive)
                except Exception as e:
                    log.error(
                        "RUNNER-MONITOR", f"Monitor error (collecting status): {e}"
                    )

                # Restart Pacer if died (existing logic) — also clear reported error if restart succeeds
                try:
                    if (
                        (pacer is not None)
                        and hasattr(pacer, "_thread")
                        and not getattr(pacer._thread, "is_alive", lambda: False)()
                        and not pacer_restarted
                        and not stop_event.is_set()
                    ):
                        log.warn(
                            "RUNNER-MONITOR", "Pacer not alive — attempting one restart"
                        )
                        try:
                            pacer = PacingController(
                                capture_q=capture_queue,
                                out_q=pacing_out_q,
                                cfg=getattr(args, "runtime_cfg", {}),
                                metrics=metrics,
                            )
                            pacer.start()
                            pacer_restarted = True
                            # clear error flag so future restarts/errors will be reported anew
                            error_reported["pacer"] = False
                            log.info(
                                "RUNNER-MONITOR", "Pacing controller restarted once"
                            )
                        except Exception as e:
                            log.error(
                                "RUNNER-MONITOR",
                                f"Failed to restart pacing controller: {e}",
                            )
                            log.debug("RUNNER-MONITOR", traceback.format_exc())
                except Exception as e:
                    # defensive: any unexpected error probing pacer internals should not crash monitor loop
                    log.error(
                        "RUNNER-MONITOR",
                        f"Error while attempting pacer-restart checks: {e}",
                    )
                    log.debug("RUNNER-MONITOR", traceback.format_exc())

                # Restart Inference worker if died (existing logic) — clear reported error on success
                try:
                    infer_alive_check = (
                        getattr(infer, "is_alive", lambda: False)()
                        if infer is not None
                        else False
                    )
                    if (
                        (not infer_alive_check)
                        and infer is not None
                        and not infer_restarted
                        and not stop_event.is_set()
                    ):
                        log.warn(
                            "RUNNER-MONITOR",
                            "Inference thread not alive — attempting one restart",
                        )
                        try:
                            infer_src_q = (
                                pacing_out_q
                                if getattr(pacer, "is_alive", lambda: False)()
                                else capture_queue
                            )
                            infer = InferenceWorker(
                                infer_src_q,
                                result_queue,
                                stop_event,
                                args,
                                cap_info=cap_info,
                                metrics=metrics,
                            )
                            infer.start()
                            infer_restarted = True
                            error_reported["infer"] = False
                            log.info(
                                "RUNNER-MONITOR",
                                f"Inference worker restarted once (source_q={'pacing_out_q' if infer_src_q is pacing_out_q else 'capture_queue'})",
                            )
                        except Exception as e:
                            log.error(
                                "RUNNER-MONITOR",
                                f"Failed to restart inference worker: {e}",
                            )
                            log.debug("RUNNER-MONITOR", traceback.format_exc())
                except Exception as e:
                    log.error(
                        "RUNNER-MONITOR",
                        f"Error while attempting infer-restart checks: {e}",
                    )
                    log.debug("RUNNER-MONITOR", traceback.format_exc())

            time.sleep(0.1)
    except KeyboardInterrupt:
        log.info(
            "RUNNER-MONITOR", "Received KeyboardInterrupt, stopping threaded runner..."
        )
    return pacer, infer


def _cleanup(
    pacer,
    capture,
    infer,
    display,
    injected,
    injected_csvs,
    stop_event,
    metrics=None,
):
    """
    Gracefully stop all threads and release resources.

    - Signals shutdown to all threads via stop_event.
    - Attempts clean stop for pacer, capture, inference, and display threads.
    - Closes injected CSV writers.
    - Logs final termination message.
    """
    stop_event.set()
    try:
        pacer.stop(wait=True, timeout=1.0)
        if hasattr(pacer, "is_alive") and pacer.is_alive():
            log.warn("RUNNER", "Pacing controller still alive after stop timeout")
    except Exception as e:
        log.warn("RUNNER", f"Error stopping pacing controller cleanly: {e}")

    for obj in (capture, infer, display):
        try:
            if obj is not None:
                obj.join(timeout=1.0)
        except Exception:
            pass

    # NEW: flush & close metrics (best-effort)
    try:
        if metrics is not None:
            metrics.close(flush=True)
    except Exception:
        pass

    try:
        if injected_csvs is not None:
            injected_csvs.close()
    except Exception:
        pass

    try:
        webrtc_server = injected.get("webrtc_server")
    except Exception:
        webrtc_server = None

    if webrtc_server is not None:
        try:
            webrtc_server.close()
        except Exception:
            log.debug("RUNNER", "Failed to close WebRTCServer", exc_info=True)

    log.info("RUNNER", "Threaded runner stopped")


# ===============================================================
# 🎬 Main Entry Point — Threaded Pipeline Runner
# ===============================================================


def run_threaded(args):
    """
    Main orchestrator for the multi-threaded goat counting pipeline.

    Execution flow:
    1️⃣ Create capture, pacing, inference, and display threads.
    2️⃣ Wait for capture metadata (cap_info).
    3️⃣ Prepare drawing and CSV contexts.
    4️⃣ Start all threads in correct sequence.
    5️⃣ Monitor health; auto-restart inference/pacer once if they fail.
    6️⃣ Handle graceful shutdown on interrupt or error.

    All logs are routed through src.utils.logger for colorized real-time feedback.
    """
    # --------------------------------------------------
    # 1) Core queues + stop signal
    # --------------------------------------------------
    # capture_queue: raw frames from capture thread
    # pacing_out_q : paced frames for inference
    # result_queue : inference outputs for display/counting
    cap_qsize = int(args.cap_qsize)
    res_qsize = int(args.res_qsize)
    capture_queue = queue.Queue(maxsize=cap_qsize)
    pacing_out_q = queue.Queue(maxsize=res_qsize)
    result_queue = queue.Queue(maxsize=res_qsize)
    stop_event = threading.Event()

    # Prefer FFmpeg capture backend when OpenCV provides it.
    cap_backend = None
    try:
        import cv2 as _cv2

        if hasattr(_cv2, "CAP_FFMPEG"):
            cap_backend = _cv2.CAP_FFMPEG
    except Exception:
        cap_backend = None

    # --------------------------------------------------
    # 2) Start capture + gather source metadata
    # --------------------------------------------------
    src_final = _normalize_source(args)
    log.info("RUNNER", f"Starting threaded pipeline with source={src_final}")

    # Shared metrics sink for threaded pipeline stages.
    metrics = MetricsCollector(
        csv_path=getattr(args, "csv_metrics", "outputs/metrics/metrics.csv")
    )
    capture = _create_capture_thread(
        src_final, capture_queue, stop_event, args, cap_backend, metrics=metrics
    )
    cap_info = _wait_for_cap_info(
        capture,
        float(getattr(args, "cap_info_wait_timeout", 6.0)),
        float(getattr(args, "cap_info_poll_interval", 0.05)),
    )

    # --------------------------------------------------
    # 3) Build injected context consumed by DisplayWorker
    # --------------------------------------------------
    injected, injected_csvs = _prepare_injected_context(args, cap_info)
    injected["metrics"] = metrics

    # --------------------------------------------------
    # 4) Optional WebRTC surface for remote viewing/control
    # --------------------------------------------------
    webrtc_server = None
    webrtc_control_q = None
    try:
        if getattr(args, "webrtc_enable", False):
            rtc_host = getattr(args, "webrtc_host", "0.0.0.0")
            rtc_port = int(getattr(args, "webrtc_port", 8080))
            rtc_fps = float(getattr(args, "webrtc_fps", 25.0))
            rtc_max_clients = getattr(args, "webrtc_max_clients", 2)
            rtc_downscale_height = int(getattr(args, "webrtc_downscale_height", 540))
            rtc_downscale_width = int(getattr(args, "webrtc_downscale_width", 960))

            webrtc_server = WebRTCServer(
                host=rtc_host,
                port=rtc_port,
                target_fps=rtc_fps,
                max_clients=rtc_max_clients,
                downscale_height=rtc_downscale_height,
                downscale_width=rtc_downscale_width,
            )

            webrtc_control_q = std_queue.Queue(maxsize=32)
            webrtc_server.set_control_queue(webrtc_control_q)

            injected["webrtc_server"] = webrtc_server
            injected["webrtc_control_q"] = webrtc_control_q

            log.info("RUNNER", f"WebRTCServer started on {rtc_host}:{rtc_port}")
    except Exception:
        log.error("RUNNER", "Failed to start WebRTCServer: " + traceback.format_exc())

    # --------------------------------------------------
    # Slot system initialization (LIVE only)
    # --------------------------------------------------
    slot_manager = None
    slot_api = None
    slots_enabled = bool(getattr(args, "slots_enabled", False))

    # --------------------------------------------------
    # 5) Start pacing + workers
    # --------------------------------------------------
    runtime_cfg = _build_runtime_cfg(args)
    pacer = _create_and_start_pacer(
        capture_queue, pacing_out_q, runtime_cfg, metrics=metrics
    )
    infer, display = _create_workers(
        pacing_out_q,
        result_queue,
        stop_event,
        args,
        cap_info,
        injected,
        metrics=metrics,
    )
    _start_workers(infer, display)

    # Live global count supplier used by SlotManager.
    # Falls back to 0 if display state is not yet readable.
    def get_global_count():
        try:
            return display.state.up_count + display.state.down_count
        except Exception:
            return 0

    # --------------------------------------------------
    # 6) Gate slot startup on vision readiness
    # --------------------------------------------------
    # SlotManager depends on live counts from DisplayWorker state;
    # avoid starting slot control before display state is available.
    vision_ready = _wait_for_vision_ready(display)
    if not vision_ready:
        log.error("RUNNER", "Vision system did not become ready — slots disabled")

    # Slot system disabled (removed)
    slot_manager = None
    slot_api = None

    # ── Wire live status provider into WebRTC server ──────────────────────────
    # Reads only in-memory values; zero pipeline overhead.
    if webrtc_server is not None:
        def _get_live_status():
            status = {}
            try:
                st = display.state if display is not None else None
                status["up"]    = int(st.up_count)   if st else 0
                status["down"]  = int(st.down_count) if st else 0
                status["total"] = int(st.up_count + st.down_count) if st else 0
            except Exception:
                status["up"] = status["down"] = status["total"] = 0

            try:
                infer_stats = infer.get_stats() if (infer is not None and hasattr(infer, "get_stats")) else {}
                status["infer_fps"] = round(float(infer_stats.get("avg_infer_fps", 0.0)), 1)
            except Exception:
                status["infer_fps"] = 0.0

            try:
                fps_buf = getattr(display, "_e2e_fps_buf", None)
                if fps_buf and len(fps_buf) > 0:
                    status["e2e_fps"] = round(sum(fps_buf) / len(fps_buf), 1)
                else:
                    status["e2e_fps"] = 0.0
            except Exception:
                status["e2e_fps"] = 0.0

            try:
                def _alive(obj):
                    try:
                        return bool(getattr(obj, "is_alive", lambda: False)())
                    except Exception:
                        return False
                status["threads"] = {
                    "capture":  _alive(capture),
                    "pacer":    _alive(pacer._thread) if (pacer and hasattr(pacer, "_thread")) else False,
                    "infer":    _alive(infer),
                    "display":  _alive(display),
                    "slot_api": False,
                }
            except Exception:
                status["threads"] = {}

            try:
                def _qinfo(q):
                    if q is None:
                        return {"fill": 0, "max": 0}
                    return {
                        "fill": q.qsize() if hasattr(q, "qsize") else 0,
                        "max":  q.maxsize if hasattr(q, "maxsize") else 0,
                    }
                status["queues"] = {
                    "cap":    _qinfo(capture_queue),
                    "pacing": _qinfo(pacing_out_q),
                    "result": _qinfo(result_queue),
                }
            except Exception:
                status["queues"] = {}

            return status

        try:
            webrtc_server.set_status_provider(_get_live_status)
            log.debug("RUNNER", "Status provider wired into WebRTC server")
        except Exception as e:
            log.warn("RUNNER", f"Failed to set status provider: {e}")

        pass

    # --------------------------------------------------
    # 7) Monitor + graceful teardown
    # --------------------------------------------------
    pacer, infer = _monitor_threads(
        capture,
        pacer,
        infer,
        display,
        capture_queue,
        pacing_out_q,
        result_queue,
        stop_event,
        cap_info,
        args,
        metrics=metrics,
    )
    _cleanup(
        pacer,
        capture,
        infer,
        display,
        injected,
        injected_csvs,
        stop_event,
        metrics=metrics,
    )

    log.info("RUNNER", "Threaded pipeline execution complete.")


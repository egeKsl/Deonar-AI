# src/infer/loader.py
"""
Clean, modular wrapper around YOLO model loader for both threads and single thread.
Preserves original public API and behavior but improves structure, logging,
and defensive checks.

Public functions preserved:
 - load_model(weights, device_arg, half, fuse, quiet=False) -> (model, effective_device_str)
 - load_model_threaded(path, device_arg=None, prefer_ultralytics=True) -> (loader_type, model, effective_device)
"""

import torch
from typing import Any, List, Optional, Tuple
from src.utils.logger import log


# ---------------------------
# Helpers: device / cuda info
# ---------------------------
def _cuda_inventory_text() -> List[str]:
    """Return diagnostic lines describing CUDA availability and devices."""
    lines: List[str] = []
    try:
        if not torch.cuda.is_available():
            lines.append("CUDA available: False (PyTorch cannot use any GPUs)")
            return lines
        n = torch.cuda.device_count()
        lines.append(f"CUDA available: True  |  CUDA device count: {n}")
        for i in range(n):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = f"CUDA:{i}"
            try:
                cap = torch.cuda.get_device_capability(i)
            except Exception:
                cap = ("?", "?")
            try:
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            except Exception:
                mem = 0.0
            lines.append(
                f"  CUDA:{i} -> {name} | compute cap {cap[0]}.{cap[1]} | VRAM {mem:.1f} GB"
            )
        lines.append(
            "Note: Windows Task Manager GPU indices (GPU 0/1/…) may not match CUDA indices."
        )
        lines.append(
            "      Only NVIDIA GPUs show up here. AMD GPUs are NOT used by PyTorch."
        )
    except Exception as e:
        lines.append(f"(Could not query CUDA inventory: {e})")
    return lines


def _pretty_device_request(req: Optional[str]) -> str:
    if req is None or str(req).strip() == "":
        return "auto"
    return str(req)


def _normalize_device_arg(device_arg: Optional[str]) -> Optional[str]:
    """
    Normalize common device shorthands to forms acceptable by torch/ultralytics.
    - None or "auto" -> None (loader will pick best)
    - "0" / "1" -> "cuda:0" / "cuda:1"
    - "cuda" -> "cuda:0"
    - "cpu" -> "cpu"
    - "cuda:0" -> unchanged
    """
    if device_arg is None:
        return None
    s = str(device_arg).strip().lower()
    if s == "" or s in ("auto", "none"):
        return None
    if s == "cpu":
        return "cpu"
    if s.isdigit():
        return f"cuda:{s}"
    if s == "cuda":
        return "cuda:0"
    return s


def _patch_posixpath_for_windows():
    """
    Patch pathlib.PosixPath so Linux-trained torch checkpoints
    can be safely loaded on Windows.
    """
    import os

    if os.name != "nt":
        return  # only needed on Windows

    import pathlib

    if hasattr(pathlib, "_patched_posixpath"):
        return  # already patched

    class _WindowsPosixPath(pathlib.WindowsPath):
        pass

    pathlib.PosixPath = _WindowsPosixPath  # type: ignore
    pathlib._patched_posixpath = True


# ---------------------------
# Model loading - threaded-friendly loader (ultralytics preferred)
# ---------------------------
def load_model_threaded(
    path: str,
    device_arg: Optional[str] = None,
    prefer_ultralytics: bool = True,
    half: bool = False,
    fuse: bool = False,
) -> Tuple[str, Any, str]:
    """
    Thread-safe model loader returning (loader_type, model_obj, effective_device_str).

    loader_type: "ultralytics" or "yolov5_hub"
    model_obj: the loaded model wrapper/object
    effective_device_str: a string like "cuda:0" or "cpu"
    """
    device_hint = _normalize_device_arg(device_arg)

    # decide default: CUDA if available else CPU
    if device_hint is None:
        device_hint = "cuda:0" if torch.cuda.is_available() else "cpu"

    # If CUDA was requested but not available, fall back
    if device_hint.startswith("cuda") and not torch.cuda.is_available():
        log.warn(
            "MODEL-LOADER",
            f"CUDA requested ('{device_hint}') but not available. Falling back to CPU.",
        )
        device_hint = "cpu"

    effective_device = device_hint

    # Try Ultralytics first (preferred)
    if prefer_ultralytics:
        try:
            from ultralytics import YOLO  # lazy import

            # ---- OS-safe PosixPath patch (CRITICAL) ----
            _patch_posixpath_for_windows()

            # ---- Normalize weights path (string-level hygiene) ----
            from pathlib import Path

            weights = str(Path(path).expanduser().resolve())

            log.info(
                "MODEL-LOADER",
                f"Loading Ultralytics YOLO weights: {weights} -> device={effective_device}",
            )
            model = YOLO(weights)

            # attempt to move to device (ultralytics wrapper supports .to in many versions)
            # 1) Move model to device if supported
            try:
                model.to(effective_device)
            except Exception:
                # non-fatal: some ultralytics manage device internally
                log.debug(
                    "MODEL-LOADER",
                    "model.to(...) call failed or not needed for Ultralytics wrapper (continuing).",
                )

            # 2) Fuse conv+bn BEFORE any dtype conversion (important!)
            try:
                # model.model may be None or have different APIs across versions — guard
                if (
                    hasattr(model, "model")
                    and model.model is not None
                    and hasattr(model.model, "fuse")
                    and fuse
                ):
                    log.debug(
                        "MODEL-LOADER",
                        "Attempting model.model.fuse() (float dtype expected)",
                    )
                    # Fuse while params are still float (default) to avoid dtype mismatch
                    model.model.fuse()
                    log.info("MODEL-LOADER", "model.model.fuse() succeeded")
                else:
                    log.debug(
                        "MODEL-LOADER",
                        "model.model.fuse() not available; skipping fuse",
                    )
            except Exception as e_fuse:
                log.warn(
                    "MODEL-LOADER",
                    f"model.model.fuse() raised an exception (continuing): {e_fuse}",
                )
                log.debug("MODEL-LOADER", "fuse traceback:", exc_info=True)

            # 3) Convert to FP16 only if CUDA is in use (optional and guarded)
            try:
                if (
                    effective_device is not None
                    and str(effective_device).startswith("cuda")
                    and half
                    and fuse
                ):
                    # Only attempt half() on CUDA devices
                    if hasattr(model, "model") and model.model is not None:
                        # Some Ultralytics internals may already be half; guard with try/except
                        model.model.half()
                        log.info(
                            "MODEL-LOADER",
                            "model.model.half() call succeeded (using FP16).",
                        )
                    else:
                        log.debug(
                            "MODEL-LOADER",
                            "model.model not present; skipping model.model.half()",
                        )
                else:
                    log.debug(
                        "MODEL-LOADER",
                        "Not using CUDA device; skipping FP16 conversion.",
                    )
            except Exception as e_half:
                log.warn(
                    "MODEL-LOADER",
                    f"model.model.half() call failed (continuing without FP16): {e_half}",
                )
                log.debug("MODEL-LOADER", "half traceback:", exc_info=True)

            return "ultralytics", model, effective_device
        except Exception as e:
            log.warn(
                "MODEL-LOADER",
                f"Ultralytics loader failed ({e}); trying torch.hub fallback.",
            )
            log.debug("MODEL-LOADER", f"Ultralytics loader traceback:", exc_info=True)

    # Torch hub fallback (ultralytics/yolov5 hub)
    try:
        log.info(
            "MODEL-LOADER",
            f"Loading torch.hub yolov5 custom weights: {path} -> device={effective_device}",
        )
        model = torch.hub.load("ultralytics/yolov5", "custom", path, force_reload=False)
        # convert effective_device to torch.device if possible
        try:
            torch_device = torch.device(effective_device)
        except Exception:
            log.warn(
                "MODEL-LOADER",
                f"Invalid device '{effective_device}' for torch.device(); falling back to 'cpu'.",
            )
            torch_device = torch.device("cpu")
            effective_device = "cpu"
        try:
            model.to(torch_device)
        except Exception as e:
            log.warn(
                "MODEL-LOADER",
                f"model.to({torch_device}) failed: {e} — continuing with model.device={getattr(model,'device',None)}",
            )
        return "yolov5_hub", model, effective_device
    except Exception as e:
        # Critical failure: both loaders failed
        raise RuntimeError(f"Failed to load model with available loaders: {e}")


# ---------------------------
# Original (single-thread) loader kept for compatibility
# ---------------------------
def load_model(
    weights: str,
    device_arg: Optional[str],
    half: bool,
    fuse: bool,
    quiet: bool = False,
) -> Tuple[Any, str]:
    """
    Classic loader used by the single-threaded path.
    Returns (model, effective_device_str).

    Preserves original behaviour: uses Ultralytics YOLO API directly and attempts
    to move/fuse/half the underlying model where supported.
    """
    # ---- OS-safe PosixPath patch (CRITICAL) ----
    _patch_posixpath_for_windows()

    # ---- Normalize weights path (string-level hygiene) ----
    from pathlib import Path

    weights = str(Path(weights).expanduser().resolve())

    if not quiet:
        log.info(
            "INFER-RUNTIME", f" Device request: {_pretty_device_request(device_arg)}"
        )
        for ln in _cuda_inventory_text():
            log.info("INFER-RUNTIME", ln)

    # Construct Ultralytics model (will raise if ultralytics not installed)
    from ultralytics import YOLO  # lazy import

    model = YOLO(weights)

    # Decide runtime device: if device_arg is None -> '0' if CUDA available else 'cpu'
    if device_arg is None:
        dev = "0" if torch.cuda.is_available() else "cpu"
    else:
        dev = device_arg

    # Try to move model
    target = None
    try:
        target = f"cuda:{dev}" if isinstance(dev, str) and dev.isdigit() else dev
        model.to(target)
    except Exception as e:
        if not quiet:
            log.info(
                "INFER-MODEL",
                f" Could not move to '{dev}' ({e}); staying on {model.device}.",
            )
        target = str(model.device)

    # set eval, fuse, dtype similar to original
    try:
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
    except Exception:
        pass

    if fuse and hasattr(model.model, "fuse"):
        try:
            model.model.fuse()
            if not quiet:
                log.info("INFER-MODEL", f" Fused Conv+BN for faster inference.")
        except Exception as e:
            if not quiet:
                log.info("INFER-MODEL", f" Fuse skipped ({e}).")

    effective_device_str = str(model.device)
    if half and "cuda" in effective_device_str.lower():
        try:
            model.model.half()
            if not quiet:
                log.info("INFER-MODEL", f" FP16 enabled.")
        except Exception as e:
            if not quiet:
                log.info("INFER-MODEL", f" FP16 skipped ({e}).")
    else:
        try:
            model.model.float()
        except Exception:
            pass

    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    if not quiet:
        log.info(
            "INFER-RUNTIME",
            f" Using device: {effective_device_str} | half={('cuda' in effective_device_str.lower()) and half} | fuse={fuse}",
        )

    # Return the model and the device hint (preserve old return semantics)
    # Note: older code expects (model, device)
    return model, (dev if dev else effective_device_str)

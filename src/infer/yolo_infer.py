# src/infer/yolo_infer.py
"""
Clean, modular wrapper around YOLO model single-frame tracking.
Preserves original public API and behavior but improves structure, logging,
and defensive checks.

Public functions preserved:
 - load_model(weights, device_arg, half, fuse, quiet=False) -> (model, effective_device_str)
 - load_model_threaded(path, device_arg=None, prefer_ultralytics=True) -> (loader_type, model, effective_device)
 - resolve_class_filter(model, arg_str) -> set[int] | None
 - track_once(model, roi_frame, args, tracker_yaml, roi_area, keep_class_ids) -> list of det tuples
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Any

from src.utils.logger import log


# ---------------------------
# Class filter resolution
# ---------------------------
def resolve_class_filter(model: Any, arg_str: Optional[str]):
    """
    Accepts a model and a comma-separated argument string and returns a set
    of integer class IDs to keep, or None to keep all classes.

    Behaviour preserved from original:
     - '*' or 'all' or empty -> None
     - numeric tokens -> parsed as int class ids
     - name tokens -> map to model.model.names if available; log warnings if not found
    """
    if not arg_str or arg_str.strip() == "*" or arg_str.strip().lower() == "all":
        return None

    names_map: Dict[int, str] = {}
    try:
        if hasattr(model, "model") and hasattr(model.model, "names"):
            names_map = model.model.names
    except Exception:
        names_map = {}

    toks = [t.strip() for t in str(arg_str).split(",") if t.strip()]
    keep_ids = set()
    for t in toks:
        if t.isdigit() or (t.startswith("-") and t[1:].isdigit()):
            keep_ids.add(int(t))
            continue
        # not numeric -> treat as class name
        found = None
        for k, v in names_map.items():
            if str(v).lower() == t.lower():
                found = k
                break
        if found is not None:
            keep_ids.add(int(found))
        else:
            log.warn(
                "INFER-RUNTIME",
                f" class name '{t}' not found in model names: {names_map}",
            )
            log.warn(
                "INFER-RUNTIME", f" Available names: {list(names_map.values())}"
            )
    return keep_ids


# ---------------------------
# Single-frame tracking wrapper (unchanged logic)
# ---------------------------
def track_once(
    model: Any,
    roi_frame: np.ndarray,
    args: Any,
    tracker_yaml: Optional[Any],
    roi_area: float,
    keep_class_ids,
) -> List[tuple]:
    """
    Run model.track on a single ROI numpy image and convert results to the
    expected list of detection tuples:
       (xyxy_array, track_id, conf, class_id, mask_or_None)

    Preserves original filtering by area ratio and class filter.
    """
    # Run the model.track call (relies on ultralytics model API)
    results = model.track(
        source=roi_frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=(args.device if args.device else None),
        half=args.half,
        tracker=(tracker_yaml if tracker_yaml else "bytetrack.yaml"),
        persist=True,
        stream=False,
        verbose=False,
    )

    dets: List[tuple] = []
    if not results:
        return dets

    r0 = results[0]
    if r0 is None or getattr(r0, "boxes", None) is None:
        return dets

    boxes = r0.boxes
    if boxes is None or len(boxes) == 0:
        return dets

    # Extract arrays defensively
    xyxy = getattr(boxes, "xyxy", None)
    confs = getattr(boxes, "conf", None)
    ids = getattr(boxes, "id", None)
    cls = getattr(boxes, "cls", None)

    try:
        xyxy_arr = xyxy.cpu().numpy() if xyxy is not None else np.zeros((0, 4))
    except Exception:
        xyxy_arr = np.array([]).reshape(0, 4)

    try:
        conf_arr = (
            confs.cpu().numpy() if confs is not None else np.zeros((xyxy_arr.shape[0],))
        )
    except Exception:
        conf_arr = np.zeros((xyxy_arr.shape[0],))

    try:
        id_arr = (
            ids.cpu().numpy().astype(int)
            if ids is not None
            else np.array([-1] * xyxy_arr.shape[0])
        )
    except Exception:
        id_arr = np.array([-1] * xyxy_arr.shape[0])

    try:
        cls_arr = (
            cls.cpu().numpy().astype(int)
            if cls is not None
            else np.zeros((xyxy_arr.shape[0],), dtype=int)
        )
    except Exception:
        cls_arr = np.zeros((xyxy_arr.shape[0],), dtype=int)

    # masks (optional)
    masks = None
    try:
        masks = getattr(r0, "masks", None)
        if masks is not None:
            # masks.data is sometimes the numpy/torch array
            mdata = getattr(masks, "data", None)
            if mdata is not None:
                masks_list = mdata.cpu().numpy()
            else:
                # fallback: masks itself may be a list-like
                masks_list = masks
            # ensure length matches boxes
            if hasattr(masks_list, "__len__") and len(masks_list) == xyxy_arr.shape[0]:
                masks = list(masks_list)
            else:
                masks = [None] * xyxy_arr.shape[0]
        else:
            masks = [None] * xyxy_arr.shape[0]
    except Exception:
        masks = [None] * xyxy_arr.shape[0]

    # Compose detection tuples and apply filters (area ratio & class)
    for idx in range(xyxy_arr.shape[0]):
        b = xyxy_arr[idx]
        c = float(conf_arr[idx]) if idx < len(conf_arr) else 0.0
        i_d = int(id_arr[idx]) if idx < len(id_arr) else -1
        cl = int(cls_arr[idx]) if idx < len(cls_arr) else 0
        mask = masks[idx] if idx < len(masks) else None

        # area filter (preserve original logic)
        area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        try:
            if roi_area and (area / float(roi_area) < args.min_area_ratio):
                continue
        except Exception:
            # if something odd about roi_area/args, skip area filter
            pass

        if keep_class_ids is not None and (int(cl) not in keep_class_ids):
            continue

        dets.append((b, i_d, c, cl, mask))

    return dets

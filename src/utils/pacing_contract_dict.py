# src/utils/pacing_contract_dict.py

from typing import Any, Dict


# small helper for validating/coercing capture items
def _ensure_item_dict(item: Any) -> Dict[str, Any]:
    """
    Ensure item is a dict with the expected capture contract.
    Accepts either:
      - dict already: returns it (with defaults filled), or
      - tuple/list of (frame_index, capture_time, frame) -> coerces to dict (back-compat)
    Expected dict keys:
      - "frame" : np.ndarray
      - "frame_index": int
      - "capture_time": float (monotonic)
      - "source_time": Optional[float] (decoder PTS or None)
      - "fps_hint": Optional[float]
      - "meta": dict (width,height,source)
    """
    if isinstance(item, dict):
        # minimal defaults
        out = dict(item)
        out.setdefault("frame", None)
        out.setdefault("frame_index", None)
        out.setdefault("capture_time", None)
        out.setdefault("source_time", None)
        out.setdefault("fps_hint", None)
        out.setdefault("meta", {})
        return out
    # back-compat: tuple-like (frame_index, ts, frame) or (frame, ts, frame_id)
    try:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            # detect order: many older codes used (frame_id, ts, frame)
            a, b, c = item[0], item[1], item[2]
            # prefer (frame_id, ts, frame)
            if isinstance(a, int) or (isinstance(a, (str,)) and str(a).isdigit()):
                frame_index = int(a)
                capture_time = float(b)
                frame = c
            else:
                # fallback: (frame, ts, idx) unlikely but handle
                frame = a
                capture_time = float(b)
                frame_index = int(c) if isinstance(c, int) else None
            return {
                "frame": frame,
                "frame_index": frame_index,
                "capture_time": capture_time,
                "source_time": None,
                "fps_hint": None,
                "meta": {},
            }
    except Exception:
        pass
    # last resort: wrap as dict with minimal fields
    return {
        "frame": getattr(item, "frame", None),
        "frame_index": getattr(item, "frame_index", None),
        "capture_time": getattr(item, "capture_time", None),
        "source_time": getattr(item, "source_time", None),
        "fps_hint": getattr(item, "fps_hint", None),
        "meta": getattr(item, "meta", {}) or {},
    }

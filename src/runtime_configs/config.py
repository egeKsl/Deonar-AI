# src/config.py
"""
Configuration loader for Goat Detection & Counting.

Behavior:
 - Prefer YAML at configs/config.yaml (recommended).
 - If YAML missing, fall back to legacy configs/.env (line-based KEY=VALUE).
 - Validate types/values and resolve common path shorthands (weights -> models/,
   source -> data/).
 - Expose:
     CONFIG (dict)  - validated hierarchical config
     ARGS   (SimpleNamespace) - flat namespace kept for backwards compatibility
 - Provide helpers to convert ROI-relative coords to pixel coords:
     roi_pixels_from_config(cfg_geometry, cap_info)
     roi_line_to_full_pixels(line_str, rx, ry, rw, rh)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Tuple, Union

# Try to use project logger if available; otherwise fallback to print
try:
    from src.utils.logger import log
except Exception:

    class _FallbackLog:
        def info(self, tag, msg):
            print(f"[INFO] [{tag}] {msg}")

        def success(self, tag, msg):
            print(f"[SUCCESS] [{tag}] {msg}")

        def warn(self, tag, msg):
            print(f"[WARN] [{tag}] {msg}")

        def warning(self, tag, msg):
            print(f"[WARN] [{tag}] {msg}")

        def error(self, tag, msg):
            print(f"[ERROR] [{tag}] {msg}")

        def debug(self, tag, msg):
            print(f"[DEBUG] [{tag}] {msg}")

    log = _FallbackLog()  # type: ignore

# Try to import yaml; if not present, YAML loading will fail early with helpful message.
try:
    import yaml  # pyyaml
except Exception:
    yaml = None  # type: ignore


# -------------------------
# Utility converters & parsers
# -------------------------
def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _as_int(
    v: Any,
    default: Optional[int] = None,
    minv: Optional[int] = None,
    maxv: Optional[int] = None,
    name: str = "int",
) -> Optional[int]:
    if v is None or v == "":
        return default
    try:
        x = int(v)
    except Exception:
        raise ValueError(f"{name} must be integer, got: {v}")
    if minv is not None and x < minv:
        raise ValueError(f"{name} must be >= {minv}, got {x}")
    if maxv is not None and x > maxv:
        raise ValueError(f"{name} must be <= {maxv}, got {x}")
    return x


def _as_float(
    v: Any,
    default: Optional[float] = None,
    minv: Optional[float] = None,
    maxv: Optional[float] = None,
    name: str = "float",
) -> Optional[float]:
    if v is None or v == "":
        return default
    try:
        x = float(v)
    except Exception:
        raise ValueError(f"{name} must be float, got: {v}")
    if minv is not None and x < minv:
        raise ValueError(f"{name} must be >= {minv}, got {x}")
    if maxv is not None and x > maxv:
        raise ValueError(f"{name} must be <= {maxv}, got {x}")
    return x


def _as_str(
    v: Any, default: Optional[str] = None, allow_blank: bool = False, name: str = "str"
) -> Optional[str]:
    if v is None:
        return default
    s = str(v)
    if not allow_blank and s.strip() == "":
        if default is not None:
            return default
        raise ValueError(f"{name} cannot be blank")
    return s


def _as_choice(
    v: Any, choices: Iterable[str], default: Optional[str] = None, name: str = "choice"
) -> Optional[str]:
    if v is None:
        return default
    val = str(v).strip().lower()
    normalized = {c.lower(): c for c in choices}
    if val not in normalized:
        raise ValueError(f"{name} must be one of {list(choices)}, got: {v}")
    return normalized[val]


def _parse_bgr(v: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if v is None:
        return default
    if isinstance(v, (list, tuple)) and len(v) == 3:
        b, g, r = v
    else:
        s = str(v)
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"BGR must be three ints 'B,G,R', got: {v}")
        b, g, r = [int(p) for p in parts]
    for name, val in zip(("B", "G", "R"), (b, g, r)):
        if not (0 <= val <= 255):
            raise ValueError(f"{name} must be 0..255, got {val}")
    return int(b), int(g), int(r)


def _resolve_path(path_str: Optional[str], default_dir: str) -> Optional[str]:
    if path_str is None:
        return None
    p = Path(str(path_str))
    # treat bare filenames (no parent) as to be resolved under default_dir
    if p.is_absolute() or p.parent != Path("."):
        return str(p)
    # if a single name like 'best.pt' -> place under default_dir
    return str(Path(default_dir) / p.name)


def _parse_lines(line_val: Any) -> Tuple[str, list]:
    """
    Accepts string like "ax,ay,bx,by;ax,ay,bx,by"
    Returns (clean_string, list_of_tuples)
    """
    if line_val is None:
        raise ValueError("COUNT_LINE_ROI is required when using line mode.")
    s = str(line_val).split("#", 1)[0].strip()
    if not s:
        raise ValueError("COUNT_LINE_ROI is required when using line mode.")
    out = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Each line must have 4 numbers ax,ay,bx,by. Got: {chunk}")
        nums = [float(p) for p in parts]
        for n in nums:
            if n < 0.0 or n > 1.0:
                raise ValueError(f"Line coordinates must be 0..1 ratios. Got: {nums}")
        out.append(tuple(nums))
    return s, out


# helper: create runtime_cfg for PacingController from CONFIG
def build_runtime_cfg_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a normalized runtime config dict for runtime modules (PacingController, runners).
    Expects 'cfg' to be the full CONFIG dict returned by load_config.
    Returns keys:
      - sync (bool)
      - playback_speed (float)
      - autoskip (bool)
      - max_lag_s (float)
      - skip_policy (str)             # e.g. "drop_to_latest" | "drop_oldest" | "none"
      - sync_jitter_allowance_s (float)
      - max_sleep_s (float)
      - max_catchup_resync_s (float)
      - cap_qsize (int)
      - res_qsize (int)
      - cap_info_wait_timeout (float)
      - cap_info_poll_interval (float)
      - stride (int)
      - no_roi (bool)
      - no_full (bool)
      - playback_speed (float)
    """
    runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}

    # use your _as_* helpers available in this module (assumes same file)
    # NOTE: replace with the proper imported helpers if structure differs.
    sync = _as_bool(runtime.get("sync"), True)
    playback_speed = _as_float(
        runtime.get("playback_speed"), 1.0, minv=0.01, name="playback_speed"
    )
    autoskip = _as_bool(runtime.get("autoskip"), False)
    max_lag_s = _as_float(runtime.get("max_lag_s"), 0.75, minv=0.0, name="max_lag_s")
    # skip policy: defensive default
    skip_policy_raw = runtime.get("skip_policy", runtime.get("skip", "drop_to_latest"))
    skip_policy = (
        str(skip_policy_raw).strip().lower()
        if skip_policy_raw is not None
        else "drop_to_latest"
    )
    if skip_policy not in ("drop_to_latest", "drop_oldest", "none"):
        # accept a small set; fallback to safe default
        skip_policy = "drop_to_latest"

    sync_jitter_allowance_s = _as_float(
        runtime.get("sync_jitter_allowance_s"),
        0.02,
        minv=0.0,
        name="sync_jitter_allowance_s",
    )
    max_sleep_s = _as_float(
        runtime.get("max_sleep_s"), 1.0, minv=0.0, name="max_sleep_s"
    )
    max_catchup_resync_s = _as_float(
        runtime.get("max_catchup_resync_s"), 5.0, minv=0.0, name="max_catchup_resync_s"
    )

    cap_qsize = _as_int(runtime.get("cap_qsize"), 3, minv=1, name="cap_qsize")
    res_qsize = _as_int(runtime.get("res_qsize"), 12, minv=1, name="res_qsize")

    cap_info_wait_timeout = _as_float(
        runtime.get("cap_info_wait_timeout"),
        6.0,
        minv=0.0,
        name="cap_info_wait_timeout",
    )
    cap_info_poll_interval = _as_float(
        runtime.get("cap_info_poll_interval"),
        0.05,
        minv=0.0,
        name="cap_info_poll_interval",
    )

    stride = _as_int(runtime.get("stride"), 1, minv=1, name="stride")
    no_roi = _as_bool(runtime.get("no_roi"), False)
    no_full = _as_bool(runtime.get("no_full"), False)

    return {
        "sync": sync,
        "playback_speed": playback_speed,
        "autoskip": autoskip,
        "max_lag_s": max_lag_s,
        "skip_policy": skip_policy,
        "sync_jitter_allowance_s": sync_jitter_allowance_s,
        "max_sleep_s": max_sleep_s,
        "max_catchup_resync_s": max_catchup_resync_s,
        "cap_qsize": cap_qsize,
        "res_qsize": res_qsize,
        "cap_info_wait_timeout": cap_info_wait_timeout,
        "cap_info_poll_interval": cap_info_poll_interval,
        "stride": stride,
        "no_roi": no_roi,
        "no_full": no_full,
    }


# -------------------------
# Legacy .env loader (fallback)
# -------------------------
def _load_dotenv(path: Union[str, Path]) -> Dict[str, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f".env file not found: {path}")
    d: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


# -------------------------
# YAML loader
# -------------------------
def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML not installed. Install 'pyyaml' to use YAML configs."
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML configuration not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


# -------------------------
# ROI helpers (pixel mapping)
# -------------------------
def roi_pixels_from_config(
    cfg_geometry: Dict[str, Any], cap_info: Dict[str, Any]
) -> Tuple[int, int, int, int]:
    """
    Convert normalized geometry.roi (xr,yr,wr,hr) -> pixel coords (rx,ry,rw,rh)
    cap_info must contain 'width' and 'height' (ints)
    """
    xr = float(cfg_geometry.get("roi", {}).get("xr", 0.0))
    yr = float(cfg_geometry.get("roi", {}).get("yr", 0.0))
    wr = float(cfg_geometry.get("roi", {}).get("wr", 1.0))
    hr = float(cfg_geometry.get("roi", {}).get("hr", 1.0))

    W = int(cap_info.get("width", 0) or 0)
    H = int(cap_info.get("height", 0) or 0)
    if W <= 0 or H <= 0:
        raise ValueError(
            "cap_info must contain positive 'width' and 'height' to compute ROI pixels."
        )

    rx = int(round(xr * W))
    ry = int(round(yr * H))
    rw = int(round(wr * W))
    rh = int(round(hr * H))

    # clamp
    rx = max(0, min(rx, W - 1))
    ry = max(0, min(ry, H - 1))
    rw = max(0, min(rw, W - rx))
    rh = max(0, min(rh, H - ry))
    return rx, ry, rw, rh


def roi_line_to_full_pixels(
    line_str: str, rx: int, ry: int, rw: int, rh: int
) -> Tuple[int, int, int, int]:
    """
    Convert ROI-relative line string "ax,ay,bx,by" (ratios in 0..1) into full-frame pixel endpoints.
    """
    parts = [p.strip() for p in str(line_str).split(",")]
    if len(parts) != 4:
        raise ValueError("line string must be 'ax,ay,bx,by'")
    ax, ay, bx, by = [float(p) for p in parts]
    ax_px = int(round(rx + ax * rw))
    ay_px = int(round(ry + ay * rh))
    bx_px = int(round(rx + bx * rw))
    by_px = int(round(ry + by * rh))
    return ax_px, ay_px, bx_px, by_px


# -------------------------
# Main loader: returns (CONFIG dict, ARGS namespace)
# -------------------------
def load_config(
    yaml_path: str = "configs/config.yaml", env_path: str = "configs/.env"
) -> Tuple[Dict[str, Any], SimpleNamespace]:
    """
    Loads and validates configuration from YAML (preferred) or ENV (fallback).
    Returns (CONFIG, ARGS)
        - CONFIG: hierarchical dict (geometry, runtime, inference, counting, viz, debug, csv, logging, paths, tracking)
        - ARGS: SimpleNamespace with flattened keys used by the pipeline (keeps backwards compatibility)
    """

    # ---------------------------
    # Project-root-aware base_dir
    # ---------------------------
    # We want a single source-of-truth `base_dir` that points to the project root
    # (one level *above* src/). That way all later calls that used `base_dir`
    # will resolve consistently (models/, data/, outputs/, configs/ at project root).

    this_file = Path(__file__).resolve()
    # project_root -> <repo>/ (two parents above this file: runtime_configs/config.py)
    base_dir = this_file.parents[2]
    # src_root -> <repo>/src
    src_root = this_file.parents[1]

    # Compose final absolute paths for YAML/.env lookup
    # Accept incoming yaml_path/env_path as either:
    #  - "configs/config.yaml" (relative to project root)
    #  - "/abs/path/to/..." (absolute)
    #  - "config.yaml" -> prefer base_dir/configs/config.yaml
    def _locate_config_file(path_arg: str, default_subdir: str = "configs") -> Path:
        p = Path(path_arg)
        if p.is_absolute():
            return p
        # If path_arg already includes a subdir, check it relative to project root and src_root
        if "/" in path_arg or "\\" in path_arg:
            cand = base_dir / path_arg
            if cand.exists():
                return cand.resolve()
            cand2 = src_root / path_arg
            if cand2.exists():
                return cand2.resolve()
        # If just a filename like "config.yaml", prefer project_root/configs/<name>
        cand = base_dir / default_subdir / path_arg
        if cand.exists():
            return cand.resolve()
        # fallback: project_root/<path_arg>
        fallback = base_dir / path_arg
        if fallback.exists():
            return fallback.resolve()
        # still not found: return the canonical project_root/configs/<name> (caller will detect .exists())
        return (base_dir / default_subdir / path_arg).resolve()

    yaml_full = _locate_config_file(yaml_path, default_subdir="configs")
    env_full = _locate_config_file(env_path, default_subdir="configs")

    # `base_dir` is now available for use throughout the loader. Example:
    # _resolve_path(csv_events, str(base_dir / "outputs" / "events"))

    raw: Dict[str, Any] = {}
    used_yaml = False

    # Prefer YAML
    try:
        if yaml_full.exists():
            raw = _load_yaml(yaml_full)
            used_yaml = True
            log.info("CONFIG", f"Loaded YAML config: {yaml_full}")
        else:
            # fallback to .env
            if env_full.exists():
                raw_lines = _load_dotenv(env_full)
                # convert flat env into structured raw dict similar to YAML layout
                # We'll preserve many legacy keys to keep backward compat.
                raw = {"legacy_env": raw_lines}
                log.info("CONFIG", f"Loaded legacy .env (fallback): {env_full}")
            else:
                raise FileNotFoundError(
                    f"Neither YAML ({yaml_full}) nor .env ({env_full}) found."
                )
    except Exception as e:
        log.error("CONFIG", f"Failed to load configuration: {e}")
        raise

    # Provide defaults for top-level sections if missing
    cfg: Dict[str, Any] = {}
    # Default geometry
    cfg["geometry"] = raw.get(
        "geometry", {"roi": {"xr": 0.0, "yr": 0.0, "wr": 1.0, "hr": 1.0}}
    )
    # runtime
    cfg["runtime"] = raw.get("runtime", {})
    # paths
    cfg["paths"] = raw.get("paths", {})
    # inference
    cfg["inference"] = raw.get("inference", {})
    # tracking
    cfg["tracking"] = raw.get("tracking", {})
    # counting
    cfg["counting"] = raw.get("counting", {})
    # viz
    cfg["viz"] = raw.get("viz", {})
    # debug
    cfg["debug"] = raw.get("debug", {})
    # csv
    cfg["csv"] = raw.get("csv", {})
    # logging
    cfg["logging"] = raw.get("logging", {})
    # webrtc
    cfg["webrtc"] = raw.get("webrtc", {})
    # output_video
    cfg["output_video"] = raw.get("output_video", {})

    # legacy_env if fallback used
    if not used_yaml:
        cfg["legacy_env"] = raw.get("legacy_env", {})

    # --- Resolve / normalize many common knobs, with validation and helpful logs ---
    # Paths: weights, source
    try:
        # prefer YAML-style: cfg["paths"].weights
        raw_weights = (
            cfg["paths"].get("weights") if isinstance(cfg["paths"], dict) else None
        )
        raw_source = (
            cfg["paths"].get("source") if isinstance(cfg["paths"], dict) else None
        )

        # if fallback to .env, look into legacy_env keys
        if raw_weights is None and "legacy_env" in cfg:
            raw_weights = cfg["legacy_env"].get("WEIGHTS")
        if raw_source is None and "legacy_env" in cfg:
            raw_source = cfg["legacy_env"].get("SOURCE")

        weights = _as_str(raw_weights, name="WEIGHTS")
        source = _as_str(raw_source, name="SOURCE")
        if not weights:
            raise ValueError("WEIGHTS (model path) is required in config.")
        if not source:
            raise ValueError("SOURCE (input path/rtsp) is required in config.")
        # Resolve bare names
        weights_res = _resolve_path(weights, str(base_dir / "models"))
        source_res = _resolve_path(source, str(base_dir / "data"))
    except Exception as e:
        log.error("CONFIG", f"Missing/invalid paths: {e}")
        raise

    # Inference knobs
    inf = cfg.get("inference", {})
    device = _as_str(
        (
            inf.get("device")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DEVICE", None)
        ),
        default=None,
        allow_blank=True,
        name="DEVICE",
    )
    # normalize GPU shorthand: "0" -> "cuda:0"
    if device is not None:
        sd = str(device).strip().lower()
        if sd == "cpu":
            device_norm = "cpu"
        elif sd == "cuda":
            device_norm = "cuda:0"
        elif sd.isdigit():
            device_norm = f"cuda:{sd}"
        else:
            device_norm = device
    else:
        device_norm = None

    half = _as_bool(
        inf.get("half") if used_yaml else cfg.get("legacy_env", {}).get("HALF"), False
    )
    fuse = _as_bool(
        inf.get("fuse") if used_yaml else cfg.get("legacy_env", {}).get("FUSE"), True
    )
    imgsz = _as_int(
        inf.get("imgsz") if used_yaml else cfg.get("legacy_env", {}).get("IMGSZ"),
        1024,
        minv=64,
        name="IMGSZ",
    )
    conf = _as_float(
        inf.get("conf") if used_yaml else cfg.get("legacy_env", {}).get("CONF"),
        0.25,
        minv=0.0,
        maxv=1.0,
        name="CONF",
    )
    iou = _as_float(
        inf.get("iou") if used_yaml else cfg.get("legacy_env", {}).get("IOU"),
        0.45,
        minv=0.0,
        maxv=1.0,
        name="IOU",
    )

    # Runtime / pacing / UI
    run = cfg.get("runtime", {})
    runtime_live = _as_bool(
        run.get("live") if used_yaml else cfg.get("legacy_env", {}).get("LIVE"), True
    )
    runtime_quiet = _as_bool(
        run.get("quiet") if used_yaml else cfg.get("legacy_env", {}).get("QUIET"), False
    )
    runtime_verbose = _as_bool(
        run.get("verbose") if used_yaml else cfg.get("legacy_env", {}).get("VERBOSE"),
        False,
    )
    runtime_sync = _as_bool(
        run.get("sync") if used_yaml else cfg.get("legacy_env", {}).get("SYNC"), True
    )
    runtime_autoskip = _as_bool(
        run.get("autoskip") if used_yaml else cfg.get("legacy_env", {}).get("AUTOSKIP"),
        False,
    )
    runtime_max_lag_s = _as_float(
        (
            run.get("max_lag_s")
            if used_yaml
            else cfg.get("legacy_env", {}).get("MAX_LAG_S")
        ),
        0.75,
        minv=0.0,
    )
    runtime_stride = _as_int(
        run.get("stride") if used_yaml else cfg.get("legacy_env", {}).get("STRIDE"),
        1,
        minv=1,
    )
    runtime_playback_speed = _as_float(
        (
            run.get("playback_speed")
            if used_yaml
            else cfg.get("legacy_env", {}).get("PLAYBACK_SPEED")
        ),
        1.0,
        minv=0.01,
    )
    runtime_cap_qsize = _as_int(
        (
            run.get("cap_qsize")
            if used_yaml
            else cfg.get("legacy_env", {}).get("cap_qsize")
        ),
        3,
        minv=1,
    )
    runtime_res_qsize = _as_int(
        (
            run.get("res_qsize")
            if used_yaml
            else cfg.get("legacy_env", {}).get("res_qsize")
        ),
        12,
        minv=1,
    )
    runtime_no_roi = _as_bool(
        run.get("no_roi") if used_yaml else cfg.get("legacy_env", {}).get("NO_ROI"),
        False,
    )
    runtime_no_full = _as_bool(
        run.get("no_full") if used_yaml else cfg.get("legacy_env", {}).get("NO_FULL"),
        False,
    )
    runtime_save_out = _as_str(
        run.get("save_out") if used_yaml else cfg.get("legacy_env", {}).get("SAVE_OUT"),
        default="",
        allow_blank=True,
    )
    runtime_progress_every = _as_int(
        (
            run.get("progress_every")
            if used_yaml
            else cfg.get("legacy_env", {}).get("PROGRESS_EVERY")
        ),
        30,
        minv=1,
    )

    # Counting
    counting = cfg.get("counting", {})
    count_mode = _as_choice(
        (
            counting.get("mode")
            if used_yaml
            else cfg.get("legacy_env", {}).get("COUNT_MODE")
        ),
        ["line", "zone"],
        default="line",
        name="COUNT_MODE",
    )
    # Lines
    line_roi_raw = (
        counting.get("line_roi")
        if used_yaml
        else cfg.get("legacy_env", {}).get("COUNT_LINE_ROI")
    )
    if line_roi_raw is None and count_mode == "line":
        # legacy expects COUNT_LINE_ROI required for line mode
        raise ValueError(
            "COUNT_LINE_ROI (or counting.line_roi) is required for line counting mode."
        )
    count_line_roi_str = None
    count_line_roi_parsed = []
    if line_roi_raw is not None:
        try:
            count_line_roi_str, count_line_roi_parsed = _parse_lines(line_roi_raw)
        except Exception as e:
            log.error("CONFIG", f"Invalid counting.line_roi: {e}")
            raise

    # zone rect
    zone = counting.get("zone", {})
    zone_rect_raw = (
        zone.get("rect") if used_yaml else cfg.get("legacy_env", {}).get("ZONE_RECT")
    )
    zone_rect_ratios = None
    if zone_rect_raw:
        try:
            parts = [float(x.strip()) for x in str(zone_rect_raw).split(",")]
            if len(parts) != 4:
                raise ValueError("zone.rect must be 'rx,ry,rw,rh'")
            for val in parts:
                if not (0.0 <= val <= 1.0):
                    raise ValueError("zone.rect values must be 0..1")
            zone_rect_ratios = tuple(parts)
        except Exception as e:
            log.warn(
                "CONFIG",
                f"Invalid zone.rect: {e}; ignoring zone_rect and falling back to line mode.",
            )
            zone_rect_ratios = None
            count_mode = "line"

    # dual line
    dual_cfg = (
        counting.get("dual", {}) if isinstance(counting.get("dual", {}), dict) else {}
    )
    dual_enabled = _as_bool(
        (
            dual_cfg.get("enabled")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DUAL_LINES_ENABLED")
        ),
        False,
    )
    dual_mode = _as_choice(
        (
            dual_cfg.get("mode")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DUAL_MODE")
        ),
        ["verify", "recover"],
        default="verify",
    )
    dual_offset_px = _as_int(
        (
            dual_cfg.get("offset_px")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DUAL_OFFSET_PX")
        ),
        150,
        minv=0,
    )
    dual_thick_px = _as_int(
        (
            dual_cfg.get("thick_px")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DUAL_THICK_PX")
        ),
        2,
        minv=1,
    )
    dual_hyst_frames = _as_int(
        (
            dual_cfg.get("hyst_frames")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DUAL_HYST_FRAMES")
        ),
        2,
        minv=1,
    )
    dual_window_frames = _as_int(
        (
            dual_cfg.get("window_frames")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DUAL_WINDOW_FRAMES")
        ),
        80,
        minv=1,
    )
    dual_id_lock_frames = _as_int(
        (
            dual_cfg.get("id_lock_frames")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DUAL_ID_LOCK_FRAMES")
        ),
        60,
        minv=1,
    )
    dual_width_px = _as_int(
        (
            dual_cfg.get("width_px")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DUAL_WIDTH_PX")
        ),
        550,
        minv=0,
    )

    # viz
    viz = cfg.get("viz", {})
    viz_mode = _as_choice(
        viz.get("mode") if used_yaml else cfg.get("legacy_env", {}).get("VIZ_MODE"),
        ["bbox", "centroid", "auto"],
        default="auto",
    )
    viz_crowd_switch = _as_int(
        (
            viz.get("crowd_switch")
            if used_yaml
            else cfg.get("legacy_env", {}).get("VIZ_CROWD_SWITCH")
        ),
        6,
        minv=1,
    )
    viz_box_radius = _as_int(
        (
            viz.get("box_radius")
            if used_yaml
            else cfg.get("legacy_env", {}).get("VIZ_BOX_RADIUS")
        ),
        6,
        minv=0,
    )
    viz_box_thick = _as_int(
        (
            viz.get("box_thick")
            if used_yaml
            else cfg.get("legacy_env", {}).get("VIZ_BOX_THICK")
        ),
        2,
        minv=1,
    )
    viz_fill_alpha = _as_float(
        (
            viz.get("fill_alpha")
            if used_yaml
            else cfg.get("legacy_env", {}).get("VIZ_FILL_ALPHA")
        ),
        0.22,
        minv=0.0,
        maxv=1.0,
    )
    viz_font_scale = _as_float(
        (
            viz.get("font_scale")
            if used_yaml
            else cfg.get("legacy_env", {}).get("VIZ_FONT_SCALE")
        ),
        0.45,
        minv=0.1,
    )
    viz_centroid_radius = _as_int(
        (
            viz.get("centroid_radius")
            if used_yaml
            else cfg.get("legacy_env", {}).get("VIZ_CENTROID_RADIUS")
        ),
        3,
        minv=1,
    )
    count_box_mode = _as_choice(
        (
            viz.get("count_box_mode")
            if used_yaml
            else cfg.get("legacy_env", {}).get("COUNT_BOX_MODE")
        ),
        ["persist", "window", "off"],
        default="persist",
    )
    count_box_frames = _as_int(
        (
            viz.get("count_box_frames")
            if used_yaml
            else cfg.get("legacy_env", {}).get("COUNT_BOX_FRAMES")
        ),
        120,
        minv=1,
    )
    counted_bgr = _parse_bgr(
        (
            viz.get("counted_bgr")
            if used_yaml
            else cfg.get("legacy_env", {}).get("COUNT_BOX_COLOR")
        ),
        (255, 0, 0),
    )
    base_bgr = _parse_bgr(
        (
            viz.get("base_bgr")
            if used_yaml
            else cfg.get("legacy_env", {}).get("BOX_COLOR")
        ),
        (0, 255, 0),
    )

    # debug
    debug = cfg.get("debug", {})
    debug_enabled = _as_bool(
        (
            debug.get("enabled")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DEBUG_ENABLED")
        ),
        False,
    )
    debug_backproj_vis = _as_bool(
        (
            debug.get("backproj_vis")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DEBUG_BACKPROJ_VIS")
        ),
        True,
    )
    debug_trace = _as_bool(
        (
            debug.get("trace")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DEBUG_TRACE")
        ),
        False,
    )
    debug_trail_len = _as_int(
        (
            debug.get("trail_len")
            if used_yaml
            else cfg.get("legacy_env", {}).get("DEBUG_TRAIL_LEN")
        ),
        20,
        minv=1,
    )

    # csv outputs
    csvs = cfg.get("csv", {})
    csv_events = _as_str(
        (
            csvs.get("events")
            if used_yaml
            else cfg.get("legacy_env", {}).get("CSV_EVENTS")
        ),
        default="goat_cross_events.csv",
        allow_blank=True,
    )
    csv_timeseries = _as_str(
        (
            csvs.get("timeseries")
            if used_yaml
            else cfg.get("legacy_env", {}).get("CSV_TIMESERIES")
        ),
        default="goat_counts_timeseries.csv",
        allow_blank=True,
    )
    csv_metrics = _as_str(
        (
            csvs.get("metrics")
            if used_yaml
            else cfg.get("legacy_env", {}).get("CSV_METRICS")
        ),
        default="goat_metrics.csv",
        allow_blank=True,
    )
    csv_events = (
        _resolve_path(csv_events, str(base_dir / "outputs" / "events"))
        if csv_events
        else None
    )
    csv_timeseries = (
        _resolve_path(csv_timeseries, str(base_dir / "outputs" / "timeseries"))
        if csv_timeseries
        else None
    )
    csv_metrics = (
        _resolve_path(csv_metrics, str(base_dir / "outputs" / "metrics"))
        if csv_metrics
        else None
    )

    # logging config
    logging_cfg = cfg.get("logging", {})
    log_dir = _as_str(
        (
            logging_cfg.get("dir")
            if used_yaml
            else cfg.get("legacy_env", {}).get("LOG_DIR")
        ),
        default=str(base_dir / "logs"),
    )
    log_file = _as_str(
        (
            logging_cfg.get("file")
            if used_yaml
            else cfg.get("legacy_env", {}).get("LOG_FILE")
        ),
        default=str(Path(log_dir) / "log.jsonl"),
    )
    log_max_bytes = _as_int(
        (
            logging_cfg.get("max_bytes")
            if used_yaml
            else cfg.get("legacy_env", {}).get("LOG_MAX_BYTES")
        ),
        5 * 1024 * 1024,
    )
    log_backup_count = _as_int(
        (
            logging_cfg.get("backup_count")
            if used_yaml
            else cfg.get("legacy_env", {}).get("LOG_BACKUP_COUNT")
        ),
        3,
    )
    log_queue_max = _as_int(
        (
            logging_cfg.get("queue_max")
            if used_yaml
            else cfg.get("legacy_env", {}).get("QUEUE_MAX")
        ),
        8192,
    )
    show_banner = _as_bool(
        (
            logging_cfg.get("show_banner")
            if used_yaml
            else cfg.get("legacy_env", {}).get("SHOW_BANNER")
        ),
        True,
    )

    # anti-flicker & filters
    antif = counting.get("antiflicker", {})
    min_age = _as_int(
        antif.get("min_age") if used_yaml else cfg.get("legacy_env", {}).get("MIN_AGE"),
        60,
        minv=0,
    )
    min_side_frames = _as_int(
        (
            antif.get("min_side_frames")
            if used_yaml
            else cfg.get("legacy_env", {}).get("MIN_SIDE_FRAMES")
        ),
        2,
        minv=1,
    )
    cooldown_frames = _as_int(
        (
            antif.get("cooldown_frames")
            if used_yaml
            else cfg.get("legacy_env", {}).get("COOLDOWN_FRAMES")
        ),
        15,
        minv=0,
    )
    line_margin_px = _as_int(
        (
            antif.get("line_margin_px")
            if used_yaml
            else cfg.get("legacy_env", {}).get("LINE_MARGIN_PX")
        ),
        8,
        minv=0,
    )
    min_area_ratio = _as_float(
        (
            counting.get("filters", {}).get("min_area_ratio")
            if used_yaml
            else cfg.get("legacy_env", {}).get("MIN_AREA_RATIO")
        ),
        0.003,
        minv=0.0,
        maxv=1.0,
    )

    # classes
    classes_raw = (
        counting.get("classes")
        if used_yaml
        else cfg.get("legacy_env", {}).get("COUNT_CLASSES", "*")
    )
    classes = _as_str(classes_raw, default="*", allow_blank=True)

    # webrtc config
    webrtc = cfg.get("webrtc", {})
    webrtc_enable = _as_bool(
        (webrtc.get("enable")),
        False,
    )
    webrtc_host = _as_str(
        (webrtc.get("host")),
        default="0.0.0.0",
    )
    webrtc_port = _as_int(
        (webrtc.get("port")),
        default=8080,
    )
    webrtc_fps = _as_int(
        (webrtc.get("fps")),
        default=25,
    )
    webrtc_max_clients = _as_int(
        (webrtc.get("max_clients")),
        default=2,
    )
    webrtc_downscale_width = _as_int(
        (webrtc.get("downscale_width")),
        default=960,
    )
    webrtc_downscale_height = _as_int(
        (webrtc.get("downscale_height")),
        default=540,
    )

    # output video config
    output_video = cfg.get("output_video", {})
    record_video = _as_bool(
        (output_video.get("enable")),
        False,
    )
    video_path = _as_str(
        (output_video.get("filename")),
        default="output_video.mp4",
    )
    video_fps = _as_int(
        (output_video.get("fps")),
        default=30,
    )
    video_fourcc = _as_str(
        (output_video.get("fourcc")),
        default="mp4v",
    )
    video_path = (
        _resolve_path(video_path, str(base_dir / "outputs" / "video"))
        if video_path
        else None
    )
    overwrite_video = _as_bool(
        (output_video.get("overwrite")),
        default=False,
    )

    # Compose final CONFIG dict (structured)
    CONFIG: Dict[str, Any] = {
        "geometry": cfg.get("geometry"),
        "runtime": {
            "live": runtime_live,
            "quiet": runtime_quiet,
            "verbose": runtime_verbose,
            "sync": runtime_sync,
            "autoskip": runtime_autoskip,
            "max_lag_s": runtime_max_lag_s,
            "stride": runtime_stride,
            "playback_speed": runtime_playback_speed,
            "cap_qsize": runtime_cap_qsize,
            "res_qsize": runtime_res_qsize,
            "cap_info_wait_timeout": run.get("cap_info_wait_timeout", 6.0),
            "cap_info_poll_interval": run.get("cap_info_poll_interval", 0.05),
            "no_roi": runtime_no_roi,
            "no_full": runtime_no_full,
            "save_out": runtime_save_out,
            "progress_every": runtime_progress_every,
        },
        "paths": {
            "weights": weights_res,
            "source": source_res,
        },
        "inference": {
            "device": device_norm,
            "half": half,
            "fuse": fuse,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
        },
        "tracking": cfg.get("tracking", {}),
        "counting": {
            "mode": count_mode,
            "line_roi_str": count_line_roi_str,
            "line_roi": count_line_roi_parsed,
            "zone_rect_ratios": zone_rect_ratios,
            "dual": {
                "enabled": dual_enabled,
                "mode": dual_mode,
                "offset_px": dual_offset_px,
                "thick_px": dual_thick_px,
                "hyst_frames": dual_hyst_frames,
                "window_frames": dual_window_frames,
                "id_lock_frames": dual_id_lock_frames,
                "width_px": dual_width_px,
            },
            "antiflicker": {
                "min_age": min_age,
                "min_side_frames": min_side_frames,
                "cooldown_frames": cooldown_frames,
                "line_margin_px": line_margin_px,
            },
            "filters": {"min_area_ratio": min_area_ratio},
            "classes": classes,
        },
        "viz": {
            "mode": viz_mode,
            "crowd_switch": viz_crowd_switch,
            "box_radius": viz_box_radius,
            "box_thick": viz_box_thick,
            "fill_alpha": viz_fill_alpha,
            "font_scale": viz_font_scale,
            "centroid_radius": viz_centroid_radius,
            "count_box_mode": count_box_mode,
            "count_box_frames": count_box_frames,
            "counted_bgr": counted_bgr,
            "base_bgr": base_bgr,
        },
        "debug": {
            "enabled": debug_enabled,
            "backproj_vis": debug_backproj_vis,
            "trace": debug_trace,
            "trail_len": debug_trail_len,
        },
        "csv": {
            "events": csv_events,
            "timeseries": csv_timeseries,
            "metrics": csv_metrics,
        },
        "logging": {
            "live": runtime_live,
            "dir": log_dir,
            "file": log_file,
            "max_bytes": log_max_bytes,
            "backup_count": log_backup_count,
            "queue_max": log_queue_max,
            "show_banner": show_banner,
        },
        "webrtc": {
            "enable": webrtc_enable,
            "host": webrtc_host,
            "port": webrtc_port,
            "fps": webrtc_fps,
            "max_clients": webrtc_max_clients,
            "downscale_width": webrtc_downscale_width,
            "downscale_height": webrtc_downscale_height,
        },
        "output_video": {
            "enable": record_video,
            "filename": video_path,
            "fps": video_fps,
            "fourcc": video_fourcc,
            "overwrite": overwrite_video,
        },
    }

    # Flatten tracking bytetrack overrides (may be None)
    bt = (
        CONFIG.get("tracking", {}).get("bytetrack", {})
        if isinstance(CONFIG.get("tracking", {}), dict)
        else {}
    )

    ARGS = SimpleNamespace(
        # Required
        weights=CONFIG["paths"]["weights"],
        source=CONFIG["paths"]["source"],
        # Core knobs (flat)
        device=CONFIG["inference"]["device"],
        half=CONFIG["inference"]["half"],
        fuse=CONFIG["inference"]["fuse"],
        imgsz=CONFIG["inference"]["imgsz"],
        conf=CONFIG["inference"]["conf"],
        iou=CONFIG["inference"]["iou"],
        stride=CONFIG["runtime"]["stride"],
        playback_speed=CONFIG["runtime"]["playback_speed"],
        # runtime
        sync=CONFIG["runtime"]["sync"],
        autoskip=CONFIG["runtime"]["autoskip"],
        max_lag_s=CONFIG["runtime"]["max_lag_s"],
        no_roi=CONFIG["runtime"]["no_roi"],
        no_full=CONFIG["runtime"]["no_full"],
        quiet=CONFIG["runtime"]["quiet"],
        verbose=CONFIG["runtime"]["verbose"],
        live=CONFIG["runtime"]["live"],
        save_out=CONFIG["runtime"]["save_out"],
        progress_every=CONFIG["runtime"]["progress_every"],
        cap_qsize=CONFIG["runtime"]["cap_qsize"],
        res_qsize=CONFIG["runtime"]["res_qsize"],
        cap_info_wait_timeout=CONFIG["runtime"].get("cap_info_wait_timeout"),
        cap_info_poll_interval=CONFIG["runtime"].get("cap_info_poll_interval"),
        # Counting
        count_mode=CONFIG["counting"]["mode"],
        count_line_roi=CONFIG["counting"]["line_roi_str"],  # raw string
        count_line_roi_parsed=CONFIG["counting"]["line_roi"],  # parsed list of tuples
        zone_rect_ratios=CONFIG["counting"]["zone_rect_ratios"],
        # Dual-line (flat)
        dual_lines_enabled=CONFIG["counting"]["dual"]["enabled"],
        dual_mode=CONFIG["counting"]["dual"]["mode"],
        dual_offset_px=CONFIG["counting"]["dual"]["offset_px"],
        dual_thick_px=CONFIG["counting"]["dual"]["thick_px"],
        dual_hyst_frames=CONFIG["counting"]["dual"]["hyst_frames"],
        dual_window_frames=CONFIG["counting"]["dual"]["window_frames"],
        dual_id_lock_frames=CONFIG["counting"]["dual"]["id_lock_frames"],
        dual_width_px=CONFIG["counting"]["dual"]["width_px"],
        # Viz
        viz_mode=CONFIG["viz"]["mode"],
        viz_crowd_switch=CONFIG["viz"]["crowd_switch"],
        viz_box_radius=CONFIG["viz"]["box_radius"],
        viz_box_thick=CONFIG["viz"]["box_thick"],
        viz_fill_alpha=CONFIG["viz"]["fill_alpha"],
        viz_font_scale=CONFIG["viz"]["font_scale"],
        viz_centroid_radius=CONFIG["viz"]["centroid_radius"],
        count_box_mode=CONFIG["viz"]["count_box_mode"],
        count_box_frames=CONFIG["viz"]["count_box_frames"],
        count_box_color=tuple(CONFIG["viz"].get("counted_bgr", counted_bgr)),
        box_color=tuple(CONFIG["viz"].get("base_bgr", base_bgr)),
        # count_box_color=count_box_color_str,
        # box_color=box_color_str,
        # Debug
        debug_enabled=CONFIG["debug"]["enabled"],
        debug_backproj_vis=CONFIG["debug"]["backproj_vis"],
        debug_trace=CONFIG["debug"]["trace"],
        debug_trail_len=CONFIG["debug"]["trail_len"],
        # Anti-flicker & filters
        min_age=CONFIG["counting"]["antiflicker"]["min_age"],
        min_side_frames=CONFIG["counting"]["antiflicker"]["min_side_frames"],
        cooldown_frames=CONFIG["counting"]["antiflicker"]["cooldown_frames"],
        line_margin_px=CONFIG["counting"]["antiflicker"]["line_margin_px"],
        min_area_ratio=CONFIG["counting"]["filters"]["min_area_ratio"],
        # CSV outputs (flat)
        csv_events=CONFIG["csv"]["events"],
        csv_timeseries=CONFIG["csv"]["timeseries"],
        csv_metrics=CONFIG["csv"]["metrics"],
        # ByteTrack flat overrides (legacy API)
        bt_profile=bt.get("profile", None),
        bt_high=bt.get("high", None),
        bt_low=bt.get("low", None),
        bt_new=bt.get("new", None),
        bt_match=bt.get("match", None),
        bt_buffer=bt.get("buffer", None),
        bt_min_area=bt.get("min_area", None),
        bt_mot20=bt.get("mot20", False),
        # Keep full tracking dict accessible too (modern API)
        tracking=CONFIG.get("tracking", {}),
        # Geometry / ROI (both tuple and flat values)
        geometry=CONFIG.get("geometry", {}),
        roi_xr=CONFIG.get("geometry", {}).get("roi", {}).get("xr"),
        roi_yr=CONFIG.get("geometry", {}).get("roi", {}).get("yr"),
        roi_wr=CONFIG.get("geometry", {}).get("roi", {}).get("wr"),
        roi_hr=CONFIG.get("geometry", {}).get("roi", {}).get("hr"),
        # Colors/classes
        count_classes=CONFIG["counting"]["classes"],
        # WebRTC
        webrtc_enable=CONFIG["webrtc"]["enable"],
        webrtc_host=CONFIG["webrtc"]["host"],
        webrtc_port=CONFIG["webrtc"]["port"],
        webrtc_fps=CONFIG["webrtc"]["fps"],
        webrtc_max_clients=CONFIG["webrtc"]["max_clients"],
        webrtc_downscale_width=CONFIG["webrtc"]["downscale_width"],
        webrtc_downscale_height=CONFIG["webrtc"]["downscale_height"],
        # Output video
        record_video=CONFIG["output_video"]["enable"],
        video_path=CONFIG["output_video"]["filename"],
        video_fps=CONFIG["output_video"]["fps"],
        video_fourcc=CONFIG["output_video"]["fourcc"],
        overwrite_video=CONFIG["output_video"]["overwrite"],
    )
    # Ensure logs directory exists
    try:
        Path(CONFIG["logging"]["dir"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    log.info("CONFIG", f"Configuration loaded successfully (yaml={used_yaml})")
    return CONFIG, ARGS


# Provide a convenient module-level load on import if desired:
# CONFIG, ARGS = load_config()   # <-- DON'T auto-load to avoid surprising IO on import
# Callers should call load_config() explicitly from their entrypoint (main).

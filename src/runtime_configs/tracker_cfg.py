"""
Tracker configuration helpers.

This module builds a temporary YAML config for Ultralytics `model.track(...)`.
It supports multiple trackers while preserving ByteTrack behavior.

Public:
 - make_bytetrack_yaml(args, fps) -> path | None
 - make_botsort_yaml(args, fps) -> path
 - make_tracker_yaml(args, fps) -> path | None
"""

from __future__ import annotations

import json
import tempfile
from src.utils.logger import log


def make_bytetrack_yaml(args, fps):
    """
    Returns path to a temp ByteTrack YAML if any tuning is requested,
    otherwise returns None to use the built-in 'bytetrack.yaml'.
    """
    presets = {
        "default": dict(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            match_thresh=0.88,
            track_buffer=90,
            min_box_area=1,
            mot20=False,
        ),
        "crowd": dict(
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            new_track_thresh=0.7,
            match_thresh=0.76,
            track_buffer=120,
            min_box_area=80,
            mot20=False,
        ),
        "very_crowd": dict(
            track_high_thresh=0.65,
            track_low_thresh=0.12,
            new_track_thresh=0.72,
            match_thresh=0.74,
            track_buffer=150,
            min_box_area=120,
            mot20=False,
        ),
    }
    cfg = presets[args.bt_profile].copy()

    # User overrides
    if args.bt_high is not None:
        cfg["track_high_thresh"] = float(args.bt_high)
    if args.bt_low is not None:
        cfg["track_low_thresh"] = float(args.bt_low)
    if args.bt_new is not None:
        cfg["new_track_thresh"] = float(args.bt_new)
    if args.bt_match is not None:
        cfg["match_thresh"] = float(args.bt_match)
    if args.bt_buffer is not None:
        cfg["track_buffer"] = int(args.bt_buffer)
    if args.bt_min_area is not None:
        cfg["min_box_area"] = float(args.bt_min_area)
    if args.bt_mot20:
        cfg["mot20"] = True

    cfg["frame_rate"] = int(max(1, round(fps)))

    changed = (args.bt_profile != "default") or any(
        v is not None
        for v in [
            args.bt_high,
            args.bt_low,
            args.bt_new,
            args.bt_match,
            args.bt_buffer,
            args.bt_min_area,
            args.bt_mot20,
        ]
    )
    if not changed:
        return None  # use built-in YAML

    # IMPORTANT: include fuse_score for your Ultralytics version
    fuse_score = True  # or False if you want pure IoU matching

    yaml_text = (
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {cfg['track_high_thresh']}\n"
        f"track_low_thresh: {cfg['track_low_thresh']}\n"
        f"new_track_thresh: {cfg['new_track_thresh']}\n"
        f"track_buffer: {cfg['track_buffer']}\n"
        f"match_thresh: {cfg['match_thresh']}\n"
        f"frame_rate: {cfg['frame_rate']}\n"
        f"min_box_area: {cfg['min_box_area']}\n"
        f"mot20: {str(cfg['mot20']).lower()}\n"
        f"fuse_score: {str(bool(fuse_score)).lower()}\n"
    )

    tmp = tempfile.NamedTemporaryFile(
        prefix="bytetrack_", suffix=".yaml", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(yaml_text)
    tmp.flush()
    tmp.close()

    if not args.quiet:
        log.info(
            "TRACKER",
            f" Using ByteTrack cfg:\n{json.dumps({**cfg, 'fuse_score': fuse_score}, indent=2)}",
        )
        log.info("TRACKER", f" YAML path: {tmp.name}")
    return tmp.name


def make_botsort_yaml(args, fps):
    """
    Returns path to a temp BoT-SORT YAML. BoT-SORT is always driven
    by explicit config values, so we always emit a YAML file.
    """
    cfg = {
        "tracker_type": getattr(args, "bs_tracker_type", "botsort"),
        "frame_rate": int(max(1, round(fps))),
        "mot20": bool(getattr(args, "bs_mot20", False)),
        "model": str(getattr(args, "bs_model", "auto")),
        # Detection gating
        "track_high_thresh": float(getattr(args, "bs_track_high_thresh", 0.6)),
        "track_low_thresh": float(getattr(args, "bs_track_low_thresh", 0.1)),
        "new_track_thresh": float(getattr(args, "bs_new_track_thresh", 0.7)),
        # Association
        "match_thresh": float(getattr(args, "bs_match_thresh", 0.8)),
        "fuse_score": bool(getattr(args, "bs_fuse_score", True)),
        "proximity_thresh": float(getattr(args, "bs_proximity_thresh", 0.5)),
        # Track lifecycle
        "track_buffer": int(getattr(args, "bs_track_buffer", 75)),
        "min_box_area": float(getattr(args, "bs_min_box_area", 80)),
        # Re-ID
        "with_reid": bool(getattr(args, "bs_with_reid", True)),
        "reid_model": str(getattr(args, "bs_reid_model", "osnet_x0_25_msmt17.pt")),
        "appearance_thresh": float(getattr(args, "bs_appearance_thresh", 0.25)),
        # Global Motion Compensation
        "gmc_method": str(getattr(args, "bs_gmc_method", "sparseOptFlow")),
    }

    yaml_text = (
        "tracker_type: botsort\n"
        f"frame_rate: {cfg['frame_rate']}\n"
        f"mot20: {str(cfg['mot20']).lower()}\n"
        f"model: {cfg['model']}\n"
        f"track_high_thresh: {cfg['track_high_thresh']}\n"
        f"track_low_thresh: {cfg['track_low_thresh']}\n"
        f"new_track_thresh: {cfg['new_track_thresh']}\n"
        f"match_thresh: {cfg['match_thresh']}\n"
        f"fuse_score: {str(cfg['fuse_score']).lower()}\n"
        f"proximity_thresh: {cfg['proximity_thresh']}\n"
        f"track_buffer: {cfg['track_buffer']}\n"
        f"min_box_area: {cfg['min_box_area']}\n"
        f"with_reid: {str(cfg['with_reid']).lower()}\n"
        f"reid_model: {cfg['reid_model']}\n"
        f"appearance_thresh: {cfg['appearance_thresh']}\n"
        f"gmc_method: {cfg['gmc_method']}\n"
    )

    tmp = tempfile.NamedTemporaryFile(
        prefix="botsort_", suffix=".yaml", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(yaml_text)
    tmp.flush()
    tmp.close()

    if not args.quiet:
        log.info("TRACKER", f" Using BoT-SORT cfg:\n{json.dumps(cfg, indent=2)}")
        log.info("TRACKER", f" YAML path: {tmp.name}")
    return tmp.name


def make_tracker_yaml(args, fps):
    """
    Unified entrypoint: select tracker based on args.tracker_type.

    - bytetrack -> uses make_bytetrack_yaml (returns None if unchanged)
    - botsort  -> uses make_botsort_yaml (always returns a YAML path)
    """
    tracker = getattr(args, "tracker_type", "bytetrack")
    if str(tracker).lower() == "botsort":
        return make_botsort_yaml(args, fps)
    return make_bytetrack_yaml(args, fps)

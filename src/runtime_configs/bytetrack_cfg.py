# src/bytetrack_cli.py
import tempfile
from src.utils.logger import log
import json


def make_bytetrack_yaml(args, fps):
    """
    Returns path to a temp ByteTrack YAML if any tuning is requested;
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

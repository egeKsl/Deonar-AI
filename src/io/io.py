# src/io/io.py
import os, cv2, csv
from pathlib import Path
import traceback
from src.utils.logger import log

VALID_EXTS = {".mp4"}


# ---------- Helpers ----------
def safe_print_error(msg: str, exc: Exception | None = None):
    """
    Log a clean error message with optional exception details.
    Uses the rich-based Logger for pretty formatting.
    """
    # Base error message
    log.error("SYSTEM", msg)

    if exc:
        # Exception class + message
        log.error("SYSTEM", f" → {exc.__class__.__name__}: {exc}")

        # Short traceback details
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        log.debug("SYSTEM", f" (details: {tb})")


def infer_out_path(inp_path):
    base, ext = os.path.splitext(os.path.basename(inp_path))
    # Create a subfolder under project root
    out_dir = Path("outputs") / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build the final output path
    out_file = f"{base}.annotated{ext or '.mp4'}"
    return str(out_dir / out_file)


def _ask_user_confirmation(default_path: str) -> str:
    """Ask user to confirm or change the output path with validation."""
    while True:
        log.info("IO-SAVE", f" Default save path: {default_path}")
        resp = input("Do you want to save here? (y/n): ").strip().lower()

        if resp in ("y", "yes"):
            # Confirm overwrite if file already exists
            if os.path.exists(default_path):
                if not _confirm_overwrite(default_path):
                    continue  # ask again
            return default_path

        elif resp in ("n", "no"):
            while True:  # loop until valid path is given
                new_path = input("Enter new output path or filename: ").strip()
                if not new_path:
                    log.warn("IO-SAVE", "No input provided. Keeping default path.")
                    return default_path

                # If user gives only a filename → join with default folder
                if not os.path.isabs(new_path):
                    folder = str(Path(default_path).parent)
                    new_path = os.path.join(folder, new_path)

                # Validate extension
                ext = Path(new_path).suffix.lower()
                if ext not in VALID_EXTS:
                    log.error(
                        "IO-SAVE",
                        f"Invalid extension '{ext}'. "
                        f"Accepted: {', '.join(VALID_EXTS)}",
                    )
                    continue  # ask again

                # Confirm overwrite if file exists
                if os.path.exists(new_path):
                    if not _confirm_overwrite(new_path):
                        continue

                return new_path

        else:
            log.warn("IO-SAVE", "Invalid input. Please type 'y' or 'n'.")


def _confirm_overwrite(path: str) -> bool:
    """Ask user before overwriting existing file."""
    while True:
        resp = (
            input(f"⚠️ File already exists: {path}. Overwrite? (y/n): ").strip().lower()
        )
        if resp in ("y", "yes"):
            return True
        elif resp in ("n", "no"):
            return False
        else:
            log.warn("IO-SAVE", " Invalid input. Please type 'y' or 'n'.")


def setup_output(args, W, H, fps):
    """Initialize video writer safely, with confirmation + validation."""
    if args.live:
        return None, True, False, None

    # default output path
    out_path = args.save_out if args.save_out else infer_out_path(args.source)
    out_path = _ask_user_confirmation(out_path)

    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        safe_print_error(f"Failed to create output directory for {out_path}", e)
        raise SystemExit(1)

    # effective fps (adjusted by playback speed)
    speed = args.playback_speed or 1.0
    eff_fps = max(1.0, (fps if fps and fps > 0 else 25.0) * float(speed))

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, eff_fps, (W, H))

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open writer for {out_path}")

        if not args.quiet:
            log.success(
                "IO-SAVE",
                f"Confirmed save path: {out_path} "
                f"(fps={eff_fps:.2f}, speed×{speed:.2f})",
            )
        return writer, False, True, out_path

    except Exception as e:
        safe_print_error(f"Failed to initialize video writer: {out_path}", e)
        raise SystemExit(1)


class CsvWriters:
    def __init__(self, events_path=None, ts_path=None):
        self.ev_writer = self.ts_writer = None
        self.ev_fh = self.ts_fh = None
        self.ev_seen_ids = set()

        # Event CSV
        if events_path:
            try:
                Path(events_path).parent.mkdir(parents=True, exist_ok=True)
                self.ev_fh = open(events_path, "w", newline="", encoding="utf-8")
                self.ev_writer = csv.writer(self.ev_fh)
                self.ev_writer.writerow(
                    [
                        "timestamp_s",
                        "src_frame_idx",
                        "proc_frame_idx",
                        "track_id",
                        "direction",
                        "cx",
                        "cy",
                    ]
                )
            except Exception as e:
                safe_print_error(f"Failed to open event CSV: {events_path}", e)
                raise SystemExit(1)

        # Timeseries CSV
        if ts_path:
            try:
                Path(ts_path).parent.mkdir(parents=True, exist_ok=True)
                self.ts_fh = open(ts_path, "w", newline="", encoding="utf-8")
                self.ts_writer = csv.writer(self.ts_fh)
                self.ts_writer.writerow(["timestamp_s", "up", "down", "total"])
            except Exception as e:
                safe_print_error(f"Failed to open timeseries CSV: {ts_path}", e)
                raise SystemExit(1)

    def write_event(
        self, ts_s, src_frame_idx, proc_frame_idx, tid, direction, cx, cy
    ) -> bool:
        try:
            if self.ev_writer:
                if tid in self.ev_seen_ids:
                    return False
                self.ev_seen_ids.add(tid)
                self.ev_writer.writerow(
                    [
                        f"{ts_s:.3f}",
                        src_frame_idx,
                        proc_frame_idx,
                        tid,
                        direction,
                        f"{cx:.1f}",
                        f"{cy:.1f}",
                    ]
                )
                return True
        except Exception as e:
            safe_print_error("Failed to write event row", e)
            raise SystemExit(1)
        return False

    def write_timeseries(self, sec, up, down):
        try:
            if self.ts_writer:
                self.ts_writer.writerow([sec, up, down, up + down])
        except Exception as e:
            safe_print_error("Failed to write timeseries row", e)
            raise SystemExit(1)

    def close(self):
        for fh, label in [(self.ev_fh, "events CSV"), (self.ts_fh, "timeseries CSV")]:
            if fh:
                try:
                    fh.close()
                except Exception as e:
                    safe_print_error(f"Failed to close {label}", e)

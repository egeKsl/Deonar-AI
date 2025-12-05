# src/app/runner.py

import os

from src.app.single_threaded import run_original
from src.app.multi_threaded import run_threaded

from src.utils.logger import log


def _is_live_source(src):
    if src is None:
        return False

    if isinstance(src, str):
        # If it's a URL or webcam index
        if (
            src.strip()
            .lower()
            .startswith(("rtsp://", "rtmp://", "http://", "https://", "rtsp"))
        ):
            return True
        if src.strip().isdigit():
            return True

        # If it's a file path
        if not os.path.exists(src):
            log.error("RUNNER", f"File not found: {src}")
            return False  # report as invalid, not live source

        return False  # existing file → not live

    return False


def run(args):
    """Dispatch to original file-based run or threaded live run based on args.source presence."""
    src_arg = args.source
    if _is_live_source(src_arg):
        log.info(
            "RUNNER",
            f"Detected live source '{src_arg}'. Starting threaded live pipeline.",
        )
        run_threaded(args)
    else:
        # call previous file-based runner (renamed to _run_original)
        try:
            run_original(args)
        except NameError:
            raise RuntimeError("Original run implementation not found.")

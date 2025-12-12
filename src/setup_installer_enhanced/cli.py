"""Thin CLI wrapper: parse args here and call core.Installer.

Keeping CLI + parse_args in this module prevents build-time import of the
heavy installer logic. This module is safe to reference from pyproject scripts.
"""

from __future__ import annotations
import argparse
import os
import signal
import sys

from .utils import ensure_rich, log
from .constants import Config
from . import core  # import here so CLI imports core only when CLI runs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ultra-enhanced setup installer")
    ap.add_argument(
        "--torch-version",
        default=None,
        help="(optional) pin base torch version e.g. 2.7.1",
    )
    ap.add_argument(
        "--torchvision-version", default=None, help="(optional) torchvision pin"
    )
    ap.add_argument(
        "--torchaudio-version", default=None, help="(optional) torchaudio pin"
    )
    ap.add_argument(
        "--cuda-tag", default="cu118", help="default CUDA tag used in wheel filenames"
    )
    ap.add_argument(
        "--index-url",
        default="https://download.pytorch.org/whl/cu118",
        help="Primary index for PyTorch wheels",
    )
    ap.add_argument(
        "--extra-index-url",
        default="https://pypi.org/simple",
        help="Extra index (PyPI)",
    )
    ap.add_argument("--install-cuda-python", action="store_true")
    ap.add_argument("--install-nvidia-ml", action="store_true")
    ap.add_argument("--opencv-min", default="4.8.0")
    ap.add_argument("--no-deps", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--auto-detect-torch",
        action="store_true",
        help="auto-detect CUDA runtime and attempt torch trio install",
    )
    ap.add_argument(
        "--force-reinstall",
        action="store_true",
        help="force uninstall+reinstall of torch trio if mismatch",
    )
    ap.add_argument(
        "--metrics-file",
        default=Config().metrics_file,
        help="Path to write install metrics",
    )
    ap.add_argument(
        "--metrics-dir",
        default=Config().metrics_dir,
        help="Directory where install logs and metrics JSON will be stored.",
    )
    ap.add_argument(
        "--retries", type=int, default=3, help="Retry attempts for failed installs"
    )
    ap.add_argument(
        "--heartbeat",
        type=int,
        default=4,
        help="Seconds of silence before heartbeat message",
    )
    ap.add_argument(
        "--prefetch-sizes",
        action="store_true",
        help="(opt-in) try to prefetch wheel sizes via HEAD (experimental)",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Show raw pip output as it arrives"
    )
    ap.add_argument(
        "--always-progress",
        action="store_true",
        dest="always_progress",
        help="Force pip --progress-bar=on and -v when possible",
    )
    return ap.parse_args()


def _signal_handler(sig, frame):
    log("error", "Interrupted by user. Exiting...")
    sys.exit(1)


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, _signal_handler)

    # metrics dir handling (identical behaviour to original main)
    metrics_dir = os.path.abspath(getattr(args, "metrics_dir", None) or "./logs")
    try:
        os.makedirs(metrics_dir, exist_ok=True)
    except Exception as e:
        log(
            "warn",
            f"Could not create metrics dir '{metrics_dir}': {e} — falling back to CWD",
        )
        metrics_dir = os.getcwd()

    metrics_fname = os.path.basename(
        getattr(args, "metrics_file", None) or "install_metrics.json"
    )
    args.metrics_file = os.path.join(metrics_dir, metrics_fname)

    # make sure rich is available (may install it)
    ensure_rich(getattr(args, "no_deps", False))

    # bind rich UI if available is performed inside core.Installer.execute (same as original)
    installer = core.Installer(args)
    try:
        installer.execute()
    finally:
        installer.cleanup()


if __name__ == "__main__":
    main()
    """Entry point for CLI."""
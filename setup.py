#!/usr/bin/env python3
"""
setup_installer.py

Enhanced, robust two-phase installer with:
- Phase-1: safe install of non-machine deps (but now skips incompatible packages to avoid bulk failure)
- Phase-2: machine-dependent installs with validation
- Python-version compatibility checks and clear user-friendly logging
- Timestamped, colored logs with emojis (rich if available, graceful ANSI fallback)

Behavior changes vs earlier version:
- Before bulk-install, the script checks Python compatibility for each non-machine package.
  If a package is likely incompatible with the current interpreter, it will be skipped
  (moved to "skipped_non_machine"), and a clear warning will be shown instead of
  letting pip fail the whole Phase-1. This prevents abrupt aborts like aiortc failing
  on older Python versions.
- The script prints a concise summary of which packages were installed, skipped, or failed.

Usage:
  python setup_installer.py
  python setup_installer.py --dry-run
  python setup_installer.py --torch-version 2.7.0 --install-cuda-python

"""

from __future__ import annotations
import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --------------------- Phase-1: non-machine deps ---------------------
NON_MACHINE_DEPS = [
    "ultralytics>=8.0.0",
    "numpy>=1.26.0",
    "av>=10.0.0",
    "Pillow>=10.0.0",
    "aiohttp>=3.8.0",
    "aiortc>=1.14.0",
    "boto3>=1.28.0",
    "flask>=3.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
    "rich>=13.6.0",
    "filterpy>=1.4.5",
    "scipy>=1.11.0",
    "lap>=0.4.0",
    "tqdm>=4.66.1",
    # Dev/testing (optional but harmless to install in Phase-1)
    "pytest>=7.4.0",
    "black>=23.9.1",
    "isort>=5.12.0",
    "flake8>=6.1.0",
]

# --------------------- Python version support map ---------------------
PYTHON_SUPPORT_MAP = {
    "numpy": (3, 8, 3, 12),
    "scipy": (3, 8, 3, 12),
    "torch": (3, 8, 3, 12),
    "torchvision": (3, 8, 3, 12),
    "torchaudio": (3, 8, 3, 12),
    "opencv": (3, 8, 3, 12),
    "av": (3, 8, 3, 12),
    "aiortc": (3, 8, 3, 12),  # aiortc 3.x requires newer Python in many builds
    "aiohttp": (3, 8, 3, 12),
    "Pillow": (3, 8, 3, 12),
    "boto3": (3, 8, 3, 12),
    "flask": (3, 8, 3, 12),
    "pyyaml": (3, 8, 3, 12),
    "rich": (3, 8, 3, 12),
    "filterpy": (3, 8, 3, 12),
    "lap": (3, 8, 3, 12),
    "tqdm": (3, 7, 3, 12),
    "cuda-python": (3, 8, 3, 12),
    "nvidia-ml-py": (3, 7, 3, 12),
}

# --------------------- Logging helpers ---------------------


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# Try to import rich for visually pleasing logs; fallback to ANSI
try:
    from rich.console import Console
    from rich.text import Text
    from rich.style import Style

    RCON = Console()

    def log(level: str, msg: str):
        emoji = {"info": "💡", "ok": "✅", "warn": "⚠️", "error": "❌"}.get(level, "")
        style = {"info": "cyan", "ok": "green", "warn": "yellow", "error": "red"}.get(
            level, ""
        )
        RCON.print(f"[{_now_ts()}] {emoji} ", end="")
        RCON.print(msg, style=style)

    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False
    ANSI = {
        "cyan": "[96m",
        "green": "[92m",
        "yellow": "[93m",
        "red": "[91m",
        "bold": "[1m",
        "end": "[0m",
    }

    def log(level: str, msg: str):
        emoji = {"info": "[i]", "ok": "[+]", "warn": "[!]", "error": "[-]"}.get(
            level, ""
        )
        col = {"info": "cyan", "ok": "green", "warn": "yellow", "error": "red"}.get(
            level, ""
        )
        print(f"[{_now_ts()}] {ANSI.get(col, '')}{emoji} {msg}{ANSI['end']}")


# --------------------- Utilities ---------------------


def run(
    cmd: List[str], capture: bool = True, env=None, check: bool = False
) -> Tuple[int, str, str]:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            env=env,
        )
        out, err = proc.communicate()
        stdout = out.decode(errors="ignore") if out else ""
        stderr = err.decode(errors="ignore") if err else ""
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=stdout, stderr=stderr
            )
        return proc.returncode, stdout, stderr
    except FileNotFoundError:
        return 127, "", f"Executable not found: {cmd[0]}"


def version_tuple() -> Tuple[int, int]:
    return sys.version_info.major, sys.version_info.minor


def version_in_range(cur: Tuple[int, int], rng: Tuple[int, int, int, int]) -> bool:
    cur_major, cur_minor = cur
    min_major, min_minor, max_major, max_minor = rng
    if (cur_major, cur_minor) < (min_major, min_minor):
        return False
    if (cur_major, cur_minor) > (max_major, max_minor):
        return False
    return True


def check_package_python_support(
    pkg_name: str,
) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
    cur = version_tuple()
    for key, rng in PYTHON_SUPPORT_MAP.items():
        if key.lower() in pkg_name.lower():
            ok = version_in_range(cur, rng)
            return ok, rng
    return True, None


# --------------------- Phase-1 installer (safe) ---------------------


def install_non_machine_deps(
    dry_run: bool = False, no_deps: bool = False
) -> Tuple[bool, List[str]]:
    """Install NON_MACHINE_DEPS but skip packages that are incompatible with current Python.
    Returns (success, skipped_list).
    """
    cur = version_tuple()
    skipped = []
    to_install = []

    for spec in NON_MACHINE_DEPS:
        # extract package token (before comparison operators)
        token = spec.split()[0].split("=")[0].split(">")[0]
        ok, rng = check_package_python_support(token)
        if not ok:
            log(
                "warn",
                f"Skipping {token} for Python {cur[0]}.{cur[1]} (supported: {rng[0]}.{rng[1]} - {rng[2]}.{rng[3]})",
            )
            skipped.append(token)
            continue
        to_install.append(spec)

    if dry_run:
        log(
            "info",
            "Dry-run mode: the following non-machine packages would be installed:",
        )
        for s in to_install:
            log("info", f"  - {s}")
        if skipped:
            log(
                "warn",
                "The following packages would be skipped due to Python-version incompatibility: "
                + ", ".join(skipped),
            )
        return True, skipped

    if not to_install:
        log("warn", "No non-machine packages to install after compatibility filtering.")
        return True, skipped

    pip_cmd = [sys.executable, "-m", "pip", "install"] + to_install
    if no_deps:
        pip_cmd.append("--no-deps")

    log(
        "info",
        "Installing non-machine-dependent packages (filtered for compatibility)...",
    )
    code, out, err = run(pip_cmd, capture=True)
    if code == 0:
        log("ok", "Non-machine dependencies installed successfully.")
        return True, skipped
    else:
        log(
            "error", "Failed to install non-machine dependencies. See pip output below:"
        )
        log("info", out + " " + err)
        return False, skipped


# --------------------- After Phase-1 we re-try rich import ---------------------
try:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    CONSOLE = Console()
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False


# --------------------- System detection and installer class ---------------------


def is_root() -> bool:
    if os.name == "nt":
        return False
    try:
        return os.geteuid() == 0
    except Exception:
        return False


def detect_nvidia_smi() -> Optional[dict]:
    code, out, err = run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,count",
            "--format=csv,noheader",
        ],
        capture=True,
    )
    if code != 0:
        return None
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    names = []
    driver = None
    total_gpus = 0
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 3:
            name, driver_version, count = parts[0], parts[1], parts[2]
            names.append(name)
            driver = driver_version
            try:
                total_gpus += int(count)
            except Exception:
                total_gpus += 1
    return {"gpus": total_gpus, "names": names, "driver": driver}


def detect_ffmpeg() -> bool:
    code, out, err = run(["ffmpeg", "-version"], capture=True)
    return code == 0


def is_headless() -> bool:
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return False
        return True
    return False


@dataclass
class PackageTask:
    name: str
    wheel_spec: Optional[str] = None
    install_args: Optional[List[str]] = None
    optional: bool = False
    reason: Optional[str] = None


class Installer:
    def __init__(self, args):
        self.args = args
        self.queue: List[PackageTask] = []
        self.tempdir = tempfile.mkdtemp(prefix="setup_installer_")
        self.start_time = time.time()

    def detect_system(self):
        cur = version_tuple()
        styled_range = f"{self.args.py_min_major}.{self.args.py_min_minor} - {self.args.py_max_major}.{self.args.py_max_minor}"
        log(
            "info",
            f"Current Python: {cur[0]}.{cur[1]} — recommended range: {styled_range}",
        )
        n = detect_nvidia_smi()
        if n:
            log(
                "ok",
                f"nvidia-smi found: GPUs={n['gpus']} driver={n['driver']} names={n['names']}",
            )
        else:
            log(
                "warn",
                "nvidia-smi not available or returned non-zero. GPU info unknown.",
            )
        ff = detect_ffmpeg()
        if ff:
            log("ok", "ffmpeg detected on PATH.")
        else:
            log(
                "warn",
                "ffmpeg not detected. Some packages (av, aiortc) may require ffmpeg installed on system.",
            )
        headless = is_headless()
        if headless:
            log(
                "warn",
                "Headless environment detected. Will recommend opencv-python-headless.",
            )
        else:
            log(
                "info", "Display server detected. GUI-capable OpenCV wheel recommended."
            )
        return {"nvidia": n, "ffmpeg": ff, "headless": headless}

    def plan_queue(self, env):
        torch_ver = self.args.torch_version
        cuda_tag = self.args.cuda_tag
        index_url = self.args.index_url
        extra_index = self.args.extra_index_url

        if torch_ver:
            torch_spec = (
                f"torch=={torch_ver}+{cuda_tag}" if cuda_tag else f"torch=={torch_ver}"
            )
            tv_spec = (
                f"torchvision=={self.args.torchvision_version}+{cuda_tag}"
                if self.args.torchvision_version
                else None
            )
            ta_spec = (
                f"torchaudio=={self.args.torchaudio_version}+{cuda_tag}"
                if self.args.torchaudio_version
                else None
            )
            self.queue.append(
                PackageTask(
                    name="torch",
                    wheel_spec=torch_spec,
                    install_args=["-i", index_url, "--extra-index-url", extra_index],
                )
            )
            if tv_spec:
                self.queue.append(
                    PackageTask(
                        name="torchvision",
                        wheel_spec=tv_spec,
                        install_args=[
                            "-i",
                            index_url,
                            "--extra-index-url",
                            extra_index,
                        ],
                    )
                )
            if ta_spec:
                self.queue.append(
                    PackageTask(
                        name="torchaudio",
                        wheel_spec=ta_spec,
                        install_args=[
                            "-i",
                            index_url,
                            "--extra-index-url",
                            extra_index,
                        ],
                    )
                )
        else:
            log(
                "warn",
                "No torch_version provided; skipping PyTorch automatic install. User must install manually.",
            )

        if self.args.install_cuda_python:
            self.queue.append(
                PackageTask(
                    name="cuda-python",
                    optional=True,
                    reason="Provides Python bindings for CUDA runtime — may not be necessary if user uses system CUDA.",
                )
            )
        if self.args.install_nvidia_ml:
            self.queue.append(
                PackageTask(
                    name="nvidia-ml-py",
                    optional=True,
                    reason="Used to query GPU telemetry via Python (pynvml).",
                )
            )

        opencv_pkg = "opencv-python-headless" if env["headless"] else "opencv-python"
        self.queue.append(
            PackageTask(
                name=opencv_pkg,
                wheel_spec=f"{opencv_pkg}>={self.args.opencv_min}",
                optional=False,
            )
        )

        self.queue.append(
            PackageTask(
                name="ffmpeg",
                optional=True,
                reason="System package — not installed via pip. Script will only validate presence and provide install hints.",
            )
        )

        return self.queue

    def validate_pytorch_wheels(self, task: PackageTask) -> bool:
        if not task.wheel_spec:
            return False
        log("info", f"Validating availability of {task.wheel_spec} from index...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            task.wheel_spec,
            "-d",
            self.tempdir,
        ]
        if task.install_args:
            cmd.extend(task.install_args)
        code, out, err = run(cmd, capture=True)
        files = os.listdir(self.tempdir)
        if code == 0 and any(f.endswith(".whl") or ".tar.gz" in f for f in files):
            log("ok", "Wheel found and downloaded to temp dir.")
            for f in files:
                try:
                    os.remove(os.path.join(self.tempdir, f))
                except Exception:
                    pass
            return True
        else:
            log(
                "error",
                f"Could not find wheel for {task.wheel_spec}. pip output:{out} {err}",
            )
            for f in os.listdir(self.tempdir):
                try:
                    os.remove(os.path.join(self.tempdir, f))
                except Exception:
                    pass
            return False

    def install_task(self, task: PackageTask) -> bool:
        ok, declared = check_package_python_support(task.name)
        cur = version_tuple()
        if declared is not None and not ok:
            log(
                "error",
                f"Incompatible: {task.name} does not support Python {cur[0]}.{cur[1]}. Supported: {declared[0]}.{declared[1]} - {declared[2]}.{declared[3]}. Skipping install.",
            )
            return task.optional
        elif declared is None:
            log(
                "warn",
                f"No explicit Python-range metadata for {task.name}; proceeding with caution.",
            )

        if task.name == "ffmpeg":
            ok = detect_ffmpeg()
            if ok:
                log("ok", "ffmpeg is present on PATH.")
                return True
            log(
                "warn",
                "ffmpeg not found. Please install system ffmpeg. Common commands:",
            )
            system = platform.system().lower()
            if system == "linux":
                log(
                    "info",
                    "apt (Debian/Ubuntu): sudo apt update && sudo apt install ffmpeg -y",
                )
                log(
                    "info",
                    "yum (CentOS/RHEL): sudo yum install epel-release && sudo yum install ffmpeg -y",
                )
            elif system == "darwin":
                log("info", "brew install ffmpeg")
            elif system == "windows":
                log(
                    "info",
                    "choco install ffmpeg -y  OR download static builds from ffmpeg.org and add to PATH",
                )
            return False

        pip_cmd = [sys.executable, "-m", "pip", "install"]
        if task.wheel_spec:
            pip_cmd.append(task.wheel_spec)
        else:
            pip_cmd.append(task.name)
        if task.install_args:
            pip_cmd.extend(task.install_args)
        if self.args.no_deps:
            pip_cmd.append("--no-deps")

        log("info", f"Installing {task.name}...")
        code, out, err = run(pip_cmd, capture=True)
        if code == 0:
            log("ok", f"Installed {task.name} successfully.")
            return True
        else:
            log("error", f"Failed to install {task.name}.")
            log("info", f"Stdout: {out} Stderr: {err}")
            return False

    def execute(self):
        env = self.detect_system()
        queue = self.plan_queue(env)

        if RICH_AVAILABLE:
            table = Table(title="Planned install queue", box=box.SIMPLE_HEAVY)
            table.add_column("#", style="bold")
            table.add_column("Package")
            table.add_column("Spec")
            table.add_column("Optional")
            table.add_column("Notes")
            for i, t in enumerate(queue, 1):
                table.add_row(
                    str(i),
                    t.name,
                    t.wheel_spec or "-",
                    str(t.optional),
                    t.reason or "-",
                )
            CONSOLE.print(table)
        else:
            log("info", "Planned install queue:")
            for i, t in enumerate(queue, 1):
                log(
                    "info",
                    f"{i}. {t.name}  spec={t.wheel_spec or '-'} optional={t.optional} notes={t.reason or '-'}",
                )

        if self.args.dry_run:
            log("warn", "Dry-run: no packages will actually be installed. Exiting.")
            return

        for t in [q for q in queue if q.name == "torch"]:
            ok = self.validate_pytorch_wheels(t)
            if not ok:
                log(
                    "error",
                    "PyTorch wheel validation failed. Will not attempt to install torch automatically.",
                )
                queue = [
                    q
                    for q in queue
                    if q.name not in ("torch", "torchvision", "torchaudio")
                ]
                break

        successes = []
        failures = []
        for t in queue:
            success = self.install_task(t)
            if success:
                successes.append(t.name)
            else:
                failures.append(t.name)
                if not t.optional:
                    log(
                        "error",
                        f"Fatal: required package {t.name} failed to install. Stopping further installs.",
                    )
                    break
                else:
                    log("warn", f"Optional package {t.name} failed — continuing.")

        total_time = time.time() - self.start_time
        log(
            "info",
            f"Summary: Installed: {successes}  Failed: {failures}. Time: {total_time:.1f}s",
        )

    def cleanup(self):
        try:
            shutil.rmtree(self.tempdir)
        except Exception:
            pass


# --------------------- CLI and main ---------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description="Two-phase installer: stage1 installs non-machine deps, stage2 installs machine-dependent libs."
    )
    ap.add_argument(
        "--torch-version",
        default=None,
        help="PyTorch base version (e.g., 2.7.0) — omit to skip auto-install",
    )
    ap.add_argument(
        "--torchvision-version",
        default=None,
        help="torchvision version to match (optional)",
    )
    ap.add_argument(
        "--torchaudio-version",
        default=None,
        help="torchaudio version to match (optional)",
    )
    ap.add_argument(
        "--cuda-tag",
        default="cu118",
        help="CUDA tag used in wheel filenames (e.g., cu118, cu121)",
    )
    ap.add_argument(
        "--index-url",
        default="https://download.pytorch.org/whl/cu118",
        help="Primary index for PyTorch wheels",
    )
    ap.add_argument(
        "--extra-index-url",
        default="https://pypi.org/simple",
        help="Extra index for fallback pip packages",
    )
    ap.add_argument(
        "--install-cuda-python",
        action="store_true",
        help="Attempt to pip install cuda-python (optional)",
    )
    ap.add_argument(
        "--install-nvidia-ml",
        action="store_true",
        help="Attempt to pip install nvidia-ml-py (optional)",
    )
    ap.add_argument("--opencv-min", default="4.8.0", help="Minimum OpenCV version")
    ap.add_argument(
        "--no-deps",
        action="store_true",
        help="Pass --no-deps to pip installs (useful for minimal installs)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not actually perform installs; only show plan",
    )
    # Python sweet spot overrides
    ap.add_argument(
        "--py-min-major",
        type=int,
        default=3,
        help="Minimum Python major version (default 3)",
    )
    ap.add_argument(
        "--py-min-minor",
        type=int,
        default=8,
        help="Minimum Python minor version (default 8)",
    )
    ap.add_argument(
        "--py-max-major",
        type=int,
        default=3,
        help="Maximum Python major version (default 3)",
    )
    ap.add_argument(
        "--py-max-minor",
        type=int,
        default=12,
        help="Maximum Python minor version (default 12)",
    )
    return ap.parse_args()


def _signal_handler(sig, frame):
    log("error", "Interrupted by user. Exiting...")
    sys.exit(1)


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, _signal_handler)

    ok, skipped = install_non_machine_deps(dry_run=args.dry_run, no_deps=args.no_deps)
    if not ok:
        log("error", "Phase-1 failed. Aborting.")
        sys.exit(1)
    if skipped:
        log(
            "warn",
            "Phase-1 skipped packages due to Python incompatibility: "
            + ", ".join(skipped),
        )

    # Re-import rich (if available) for prettier tables in phase 2
    global RICH_AVAILABLE, CONSOLE, box, Table

    try:
        from rich import box as rich_box
        from rich.console import Console
        from rich.table import Table as rich_Table

        box = rich_box
        Table = rich_Table
        CONSOLE = Console()

        RICH_AVAILABLE = True
    except Exception:
        RICH_AVAILABLE = False

    installer = Installer(args)
    try:
        installer.execute()
    finally:
        installer.cleanup()


if __name__ == "__main__":
    main()

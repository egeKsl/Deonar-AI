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


def ensure_rich_installed(no_deps: bool = False) -> bool:
    """
    Guarantee `rich` is importable. If not, attempt `pip install rich` alone.
    Return True if rich is importable afterwards; False otherwise.
    """
    try:
        # quick check
        import rich  # type: ignore

        return True
    except Exception:
        pass

    # Try installing rich alone (safer than re-running whole bulk install)
    pip_cmd = [sys.executable, "-m", "pip", "install", "rich>=13.6.0"]
    if no_deps:
        pip_cmd.append("--no-deps")
    log("info", "Attempting to install `rich` separately so phase-2 UI works...")
    code, out, err = run(pip_cmd, capture=True)
    if code != 0:
        log("error", f"Failed to install `rich`. pip output: {out} {err}")
        return False

    # Try import again
    try:
        import importlib

        importlib.invalidate_caches()
        importlib.import_module("rich")
        log("ok", "`rich` installed and importable.")
        return True
    except Exception as e:
        log("error", f"`rich` still not importable after install: {e}")
        return False


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
        elif not self.args.auto_detect_torch:
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

        # --------------------- CUDA / PyTorch auto-detection helpers (instance methods) ---------------------

    def detect_cuda_runtime(self) -> Optional[str]:
        """Return CUDA runtime version string like '12.8' from nvidia-smi or nvcc, else None.
        This is robust to variations across platforms (Windows / Linux) and different nvidia-smi fields.
        """
        # 1) Try nvidia-smi CSV query (works on newer nvidia-smi)
        for query in [
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            [
                "nvidia-smi",
                "--query-gpu=driver_version,cuda_version",
                "--format=csv,noheader",
            ],
        ]:
            code, out, err = run(query, capture=True)
            if code == 0 and out.strip():
                # if single field returned "12.8" or "570.158.01, 12.8"
                first = out.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",") if p.strip()]
                # prefer the cuda part if present
                for p in reversed(parts):
                    if p and any(ch.isdigit() for ch in p):
                        # quick sanity: "12.8" or "12"
                        if "." in p or p.isdigit():
                            return p
        # 2) Fallback: parse `nvidia-smi -q` output for "CUDA Version"
        code, out, err = run(["nvidia-smi", "-q"], capture=True)
        if code == 0 and out:
            for line in out.splitlines():
                if "CUDA Version" in line:
                    try:
                        return line.split("CUDA Version:")[-1].strip()
                    except Exception:
                        pass
        # 3) Fallback: nvcc --version (if nvcc available)
        code, out, err = run(["nvcc", "--version"], capture=True)
        if code == 0 and out:
            # nvcc shows a line like: "Cuda compilation tools, release 12.1, V12.1.105"
            for line in out.splitlines():
                if "release" in line:
                    try:
                        seg = line.split("release")[-1].split(",")[0].strip()
                        # e.g. "12.1"
                        if seg:
                            return seg
                    except Exception:
                        pass
        return None

    def cuda_runtime_to_tags(self, cuda_runtime: Optional[str]) -> List[str]:
        """Return prioritized list of candidate PyTorch wheel tags based on detected cuda runtime.
        Examples:
          '12.8' -> ['cu128','cu121','cu118']
          None  -> ['cu130','cu128','cu121','cu118']
        Always end with 'cpu' fallback.
        """
        candidates: List[str] = []
        if not cuda_runtime:
            # try a broader set so we don't miss common distro tags
            candidates = ["cu130", "cu128", "cu121", "cu118"]
        else:
            ver = cuda_runtime.strip()
            seg = ver.split(".")
            try:
                major = int(seg[0])
                minor = int(seg[1]) if len(seg) > 1 else 0
                # produce cu{major}{minor} like cu128
                tag = f"cu{major}{minor}"
                candidates.append(tag)
                # add a smaller minor fallback and common LTS tags
                if minor >= 1:
                    candidates.append(f"cu{major}{minor-1}")
                if major == 13:
                    candidates += ["cu130", "cu128", "cu121"]
                elif major == 12:
                    candidates += ["cu128", "cu121", "cu118"]
                elif major == 11:
                    candidates += ["cu118", "cu117", "cu116"]
                else:
                    candidates += ["cu118", "cu121"]
            except Exception:
                candidates = ["cu130", "cu128", "cu121", "cu118"]
        # dedupe while preserving order, then add CPU fallback
        seen = []
        for c in candidates:
            if c not in seen:
                seen.append(c)
        seen.append("cpu")
        return seen

    def try_install_torch_trio(
        self,
        torch_version: Optional[str],
        candidates: List[str],
        index_base: str,
        extra_index: str,
        no_deps: bool = False,
    ) -> bool:
        """
        Try to automatically pick and install torch + torchvision + torchaudio.
        Behavior:
          - If torch_version provided: try pinned `torch==<ver>+<tag>` first; if not found, try unpinned `torch` on that index.
          - If no torch_version provided: install unpinned packages from the candidate index; pip will choose the matching wheel for Python.
        Returns True if installation succeeded.
        """
        log(
            "info",
            f"Attempting automatic PyTorch install (torch_version={torch_version or 'auto'})",
        )

        tv_ver = self.args.torchvision_version
        ta_ver = self.args.torchaudio_version

        for cand in candidates:
            # determine index args for candidate; for 'cpu' we use PyPI (extra_index)
            if cand == "cpu":
                index_args = ["-i", extra_index]
                index_label = extra_index
            else:
                index_args = ["-i", f"https://download.pytorch.org/whl/{cand}"]
                index_label = index_args[1]

            # Build package list for install attempt
            if torch_version:
                # pinned attempt
                pinned = True
                pkg_names = [
                    (
                        f"torch=={torch_version}+{cand}"
                        if cand != "cpu"
                        else f"torch=={torch_version}"
                    )
                ]
                pkg_names.append(
                    (
                        f"torchvision=={tv_ver}+{cand}"
                        if (tv_ver and cand != "cpu")
                        else (f"torchvision=={tv_ver}" if tv_ver else "torchvision")
                    )
                )
                pkg_names.append(
                    (
                        f"torchaudio=={ta_ver}+{cand}"
                        if (ta_ver and cand != "cpu")
                        else (f"torchaudio=={ta_ver}" if ta_ver else "torchaudio")
                    )
                )
            else:
                # unpinned: let pip choose the best wheel that matches the index & Python
                pinned = False
                pkg_names = ["torch", "torchvision", "torchaudio"]
                # if user gave specific tv/ta versions, preserve them (they override)
                if tv_ver:
                    pkg_names[1] = f"torchvision=={tv_ver}"
                if ta_ver:
                    pkg_names[2] = f"torchaudio=={ta_ver}"

            # First validate torch availability via pip download (quiet check)
            log(
                "info",
                f"Validating availability of torch on index {index_label} (candidate {cand})...",
            )
            tmpd = tempfile.mkdtemp(prefix="torch_validate_")
            try:
                # If pinned and cand != cpu, check pinned spec; otherwise check plain 'torch'
                check_spec = pkg_names[0] if pinned else "torch"
                cmd = (
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "download",
                        "--no-deps",
                        "--only-binary=:all:",
                        check_spec,
                        "-d",
                        tmpd,
                    ]
                    + index_args
                    + ["--extra-index-url", extra_index]
                )
                code, out, err = run(cmd, capture=True)
                files = os.listdir(tmpd)
                if code == 0 and any(f.endswith(".whl") for f in files):
                    log(
                        "ok",
                        f"Found wheel(s) for {check_spec} on {index_label}. Proceeding to install trio using that index.",
                    )
                    # Now install trio using package names (unpinned) but with the chosen index — pip will select matching wheel for Python
                    install_cmd = (
                        [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
                        + pkg_names
                        + index_args
                        + ["--extra-index-url", extra_index]
                    )
                    if no_deps:
                        install_cmd.append("--no-deps")
                    code2, out2, err2 = run(install_cmd, capture=True)
                    if code2 == 0:
                        log(
                            "ok",
                            f"Installed torch trio from index {index_label} (candidate {cand}) successfully.",
                        )
                        return True
                    else:
                        log(
                            "warn",
                            f"Install attempt failed on candidate {cand}. pip stdout/stderr: {out2} {err2}",
                        )
                        # try next candidate
                else:
                    log(
                        "warn",
                        f"No wheel found for {check_spec} on candidate {cand}. pip said: {out} {err}",
                    )
            finally:
                try:
                    shutil.rmtree(tmpd)
                except Exception:
                    pass

        log("error", "Could not find a suitable PyTorch trio wheel across candidates.")
        return False

        # --------------------- Installed-trio detection / compatibility helpers ---------------------

    def get_installed_torch_info(self) -> dict:
        """Return dict with installed versions (or None) for torch, torchvision, torchaudio and cuda reported by torch.
        Example return: {'torch': '2.7.0+cu118', 'torchvision': '0.22.0+cu118', 'torchaudio': None, 'torch_cuda': '11.8', 'cuda_available': True}
        """
        info = {
            "torch": None,
            "torchvision": None,
            "torchaudio": None,
            "torch_cuda": None,
            "cuda_available": False,
        }
        try:
            import importlib

            tmod = importlib.import_module("torch")
            info["torch"] = getattr(tmod, "__version__", None)
            # torch.version.cuda may be e.g. '11.8' or None
            info["torch_cuda"] = (
                getattr(tmod.version, "cuda", None)
                if hasattr(tmod, "version")
                else getattr(tmod, "cuda", None)
            )
            info["cuda_available"] = (
                getattr(tmod, "cuda", None)
                and getattr(tmod.cuda, "is_available", lambda: False)()
            )
        except Exception:
            pass
        try:
            import importlib

            tv = importlib.import_module("torchvision")
            info["torchvision"] = getattr(tv, "__version__", None)
        except Exception:
            pass
        try:
            import importlib

            ta = importlib.import_module("torchaudio")
            info["torchaudio"] = getattr(ta, "__version__", None)
        except Exception:
            pass
        return info

    def is_torch_trio_compatible(
        self, installed: dict, candidate_tag: Optional[str]
    ) -> Tuple[bool, List[str]]:
        """Return (is_compatible, reasons). Heuristics:
        - if torch not installed => not compatible.
        - if candidate_tag provided (e.g. 'cu118') check if installed torch version string contains that tag OR torch.version.cuda starts with that major/minor.
        - require at least torch present; torchvision/torchaudio are preferred but if missing we mark as incomplete.
        """
        reasons: List[str] = []
        # require torch at minimum
        tv = installed.get("torch")
        if not tv:
            reasons.append("torch not installed")
            return False, reasons

        # python-level compatibility (reuse existing check)
        py_ok, rng = check_package_python_support("torch")
        if rng and not py_ok:
            reasons.append(
                f"Installed Python incompatible with mapped torch range {rng[0]}.{rng[1]} - {rng[2]}.{rng[3]}"
            )
            # still we continue to check wheel/cuda matching

        # check candidate tag vs installed version string
        if candidate_tag and candidate_tag != "cpu":
            # installed version string may be '2.7.0+cu118' or '2.7.0'
            if candidate_tag in str(tv):
                # exact tag match present
                pass
            else:
                # check torch.version.cuda if available
                tcuda = installed.get("torch_cuda")
                if tcuda:
                    # e.g. tcuda='11.8' => tag 'cu118'
                    tag_from_tcuda = f"cu{tcuda.replace('.', '')}"
                    if tag_from_tcuda != candidate_tag:
                        reasons.append(
                            f"installed CUDA tag {tag_from_tcuda!r} != desired {candidate_tag!r}"
                        )
                else:
                    # no cuda info from installed torch, mark mismatch
                    reasons.append(
                        "installed torch has no cuda runtime info to verify against candidate tag"
                    )
        else:
            # candidate_tag == 'cpu' means CPU-only target — installed torch may be CPU or non-cuda
            # if installed torch string contains '+cu' it's GPU build; that's still OK but user may prefer CPU
            pass

        # check torchvision/torchaudio presence — if absent mark incomplete but still possibly ok
        if not installed.get("torchvision"):
            reasons.append("torchvision not installed")
        if not installed.get("torchaudio"):
            reasons.append("torchaudio not installed")

        # decide compatibility: if there are only 'missing optional' reasons (torchvision/torchaudio) we can still treat as compatible if torch matches tag.
        # If any reason is about mismatch or missing torch, incompatibility.
        critical_reasons = [
            r
            for r in reasons
            if "not installed" in r
            and not r.startswith("torchvision")
            and not r.startswith("torchaudio")
            or "mismatch" in r
            or "incompatible" in r
        ]
        if critical_reasons:
            return False, reasons
        # otherwise, treat as compatible (but caller can decide to reinstall to get all trio)
        return True, reasons

    def uninstall_torch_trio(self) -> None:
        """Uninstall existing torch/torchvision/torchaudio (best-effort)."""
        log(
            "info", "Uninstalling existing torch/torchvision/torchaudio (if present)..."
        )
        # -y to confirm
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "torch",
                "torchvision",
                "torchaudio",
            ],
            capture=True,
        )

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

        # --- auto-detect + install torch (if requested or auto-detect flag present) ---
        if getattr(self.args, "auto_detect_torch", False):
            # detect cuda runtime (robust)
            cuda_runtime = self.detect_cuda_runtime()
            tags = self.cuda_runtime_to_tags(cuda_runtime)
            log(
                "info",
                f"Auto-detected CUDA runtime: {cuda_runtime} -> trying tags: {tags}",
            )

            # Determine candidate tag we will try first (first in tags)
            candidate_tag = tags[0] if tags else None

            # Check existing installation
            installed_info = self.get_installed_torch_info()
            compat, reasons = self.is_torch_trio_compatible(
                installed_info, candidate_tag
            )
            if compat:
                log(
                    "ok",
                    f"Existing torch installation looks compatible: torch={installed_info.get('torch')} torchvision={installed_info.get('torchvision')} torchaudio={installed_info.get('torchaudio')}. Skipping auto-install.",
                )
                # Remove torch trio from queue (if present)
                queue = [
                    q
                    for q in queue
                    if q.name not in ("torch", "torchvision", "torchaudio")
                ]
            else:
                # Not compatible / incomplete
                log(
                    "warn",
                    f"Existing torch trio not fully compatible or incomplete: {', '.join(reasons) if reasons else 'unspecified reasons'}",
                )
                if getattr(self.args, "force_reinstall", False):
                    # user explicitly asked to force reinstall -> uninstall then try install
                    self.uninstall_torch_trio()
                    chosen_torch_ver = (
                        self.args.torch_version
                    )  # may be None; try_install handles None
                    installed = self.try_install_torch_trio(
                        chosen_torch_ver,
                        tags,
                        self.args.index_url,
                        self.args.extra_index_url,
                        no_deps=self.args.no_deps,
                    )
                    if installed:
                        queue = [
                            q
                            for q in queue
                            if q.name not in ("torch", "torchvision", "torchaudio")
                        ]
                    else:
                        log(
                            "warn",
                            "Auto-install of PyTorch trio failed after force-reinstall attempt. Will continue with the regular queue.",
                        )
                else:
                    # If user did not request force reinstall, try auto-install attempt (useful when nothing is installed)
                    chosen_torch_ver = self.args.torch_version
                    installed = self.try_install_torch_trio(
                        chosen_torch_ver,
                        tags,
                        self.args.index_url,
                        self.args.extra_index_url,
                        no_deps=self.args.no_deps,
                    )
                    if installed:
                        queue = [
                            q
                            for q in queue
                            if q.name not in ("torch", "torchvision", "torchaudio")
                        ]
                    else:
                        log(
                            "warn",
                            "Auto-install of PyTorch trio failed or not applicable. Will proceed with the regular queue (user can still provide --torch-version).",
                        )

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
    ap.add_argument(
        "--auto-detect-torch",
        action="store_true",
        help="Attempt to auto-detect CUDA/runtime and install matching torch/torchvision/torchaudio trio.",
    )
    ap.add_argument(
        "--force-reinstall",
        action="store_true",
        help="If set, uninstall and reinstall torch/torchvision/torchaudio when mismatch detected.",
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

    # Ensure rich is installed and importable (try separate install if needed).
    # This prevents Table/box NameError when the bulk install didn't make rich importable.
    rich_ok = ensure_rich_installed(no_deps=args.no_deps)
    if not rich_ok:
        log("warn", "Continuing with ANSI fallback logging — rich UI not available.")
    # Re-import rich objects for phase-2 (if available)
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

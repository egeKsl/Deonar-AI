#!/usr/bin/env python3
"""
setup_installer_enhanced.py

Ultra-enhanced two-phase installer (complete, robust, production-ready style):
- Phase-1: checks which non-machine deps are already present and installs only missing ones
- Phase-2: robust machine-dependent installer that auto-detects CUDA runtime (nvidia-smi/nvcc)
  and tries prioritized wheel tags (cuXXX) with CPU fallback
- Installs torch + torchvision + torchaudio trio in a safe way, preferring unpinned installs
  (let pip choose matching wheel for the environment). A --torch-version may be provided to
  force a specific base version but is not required.
- If any of the trio is missing, installer will attempt to install just-missing packages; use
  --force-reinstall to perform a full uninstall+reinstall of the trio.
- Guarantees `rich` importable early so phase-2 logging/tables work; safe fallback to ANSI
- Comprehensive, timestamped logs and a final detailed summary table showing (Installed / Skipped / Failed / AlreadyPresent)

Usage examples:
  python setup_installer_enhanced.py --auto-detect-torch
  python setup_installer_enhanced.py --torch-version 2.7.1 --force-reinstall

Design notes (short):
- We make careful pip "download" checks prior to installing heavy wheels (fast failure)
- We try indexes in order derived from detected CUDA runtime; always include CPU (PyPI) fallback
- Non-machine deps are validated by import or pip show to avoid reinstalling things.

"""

from __future__ import annotations
import argparse
import importlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# ---------------------------- Configuration knobs (editable) ----------------------------
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
    # Dev/testing (optional)
    "pytest>=7.4.0",
    "black>=23.9.1",
    "isort>=5.12.0",
    "flake8>=6.1.0",
]

# Python compatibility map (used only as advisory)
PYTHON_SUPPORT_MAP = {
    "numpy": (3, 8, 3, 12),
    "scipy": (3, 8, 3, 12),
    "torch": (3, 8, 3, 12),
    "torchvision": (3, 8, 3, 12),
    "torchaudio": (3, 8, 3, 12),
    "opencv": (3, 8, 3, 12),
    "av": (3, 8, 3, 12),
    "aiortc": (3, 8, 3, 12),
    "aiohttp": (3, 8, 3, 12),
    "Pillow": (3, 8, 3, 12),
    "boto3": (3, 8, 3, 12),
    "flask": (3, 8, 3, 12),
    "pyyaml": (3, 8, 3, 12),
    "rich": (3, 8, 3, 12),
    "tqdm": (3, 7, 3, 12),
    "cuda-python": (3, 8, 3, 12),
    "nvidia-ml-py": (3, 7, 3, 12),
}

# ---------------------------- Logging helpers -----------------------------------------


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# try to import rich early, but if not available we'll install it first
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    RICH = True
    CONSOLE = Console()

    def log(level: str, msg: str):
        emoji = {"info": "💡", "ok": "✅", "warn": "⚠️", "error": "❌"}.get(level, "")
        style = {"info": "cyan", "ok": "green", "warn": "yellow", "error": "red"}.get(
            level, ""
        )
        CONSOLE.print(f"[{_now()}] {emoji} ", end="")
        CONSOLE.print(msg, style=style)

except Exception:
    RICH = False

    ANSI = {
        "cyan": "\x1b[96m",
        "green": "\x1b[92m",
        "yellow": "\x1b[93m",
        "red": "\x1b[91m",
        "end": "\x1b[0m",
    }

    def log(level: str, msg: str):
        emoji = {"info": "[i]", "ok": "[+]", "warn": "[!]", "error": "[-]"}.get(
            level, ""
        )
        col = {"info": "cyan", "ok": "green", "warn": "yellow", "error": "red"}.get(
            level, ""
        )
        print(f"[{_now()}] {ANSI.get(col,'')}{emoji} {msg}{ANSI['end']}")


# ---------------------------- Utilities ----------------------------------------------


def run(
    cmd: List[str], capture: bool = True, env: dict | None = None
) -> Tuple[int, str, str]:
    """Run subprocess and return (code, stdout, stderr)."""
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
    for key, rng in PYTHON_SUPPORT_MAP.items():
        if key.lower() in pkg_name.lower():
            return version_in_range(version_tuple(), rng), rng
    return True, None


# ---------------------------- pkg present checks -------------------------------------


def pip_show(package: str) -> Optional[Dict[str, str]]:
    """Return pip show metadata dict if package present, else None."""
    code, out, err = run([sys.executable, "-m", "pip", "show", package])
    if code != 0 or not out.strip():
        return None
    data = {}
    for line in out.splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            data[k.strip()] = v.strip()
    return data


def importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def package_already_present(spec: str) -> bool:
    """Heuristic check: given a pip spec like 'numpy>=1.26.0' or 'rich>=13.6.0', determine if it's installed and satisfies version.
    We do a coarse check: try import first (common mapping) else pip show.
    This avoids unnecessary reinstallation.
    """
    token = spec.split()[0].split("=")[0].split(">")[0]
    token = token.strip()
    mapping = {
        "Pillow": "PIL",
        # add more special-cases here if needed
    }
    module = mapping.get(token, token)
    if importable(module):
        # optionally we could validate exact version using pkg_resources, but avoid heavy dependencies
        return True
    # fallback to pip show
    info = pip_show(token)
    return info is not None


# ---------------------------- ensure rich available ----------------------------------


def ensure_rich(no_deps: bool = False) -> bool:
    """Guarantee rich importable. Install it alone if needed.
    Returns True if rich importable afterwards.
    """
    try:
        importlib.import_module("rich")
        return True
    except Exception:
        pass
    log("info", "`rich` not importable — installing `rich` alone now...")
    cmd = [sys.executable, "-m", "pip", "install", "rich>=13.6.0"]
    if no_deps:
        cmd.append("--no-deps")
    code, out, err = run(cmd)
    if code != 0:
        log("error", f"Failed to install rich: {out} {err}")
        return False
    try:
        importlib.invalidate_caches()
        importlib.import_module("rich")
        # rebind rich console
        from rich.console import Console

        global CONSOLE, RICH

        CONSOLE = Console()
        RICH = True
        log("ok", "`rich` installed and usable.")
        return True
    except Exception as e:
        log("error", f"`rich` installed but not importable: {e}")
        return False


# ---------------------------- Installer class ---------------------------------------


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
        self.tempdir = tempfile.mkdtemp(prefix="setup_installer_")
        self.start = time.time()
        # summary tracking
        self.summary = {"installed": [], "skipped": [], "failed": [], "already": []}

    # --------------------- system detection ---------------------
    def detect_system(self) -> dict:
        cur_py = version_tuple()
        log("info", f"Current Python: {cur_py[0]}.{cur_py[1]}")
        n = self._detect_nvidia_smi_basic()
        if n:
            log(
                "ok",
                f"nvidia-smi found: GPUs={n.get('gpus')} driver={n.get('driver')} names={n.get('names')}",
            )
        else:
            log("warn", "nvidia-smi not found or returned non-zero; GPU info unknown")
        ff = self.detect_ffmpeg()
        if ff:
            log("ok", "ffmpeg present on PATH")
        else:
            log("warn", "ffmpeg not detected; some packages may require it installed")
        head = self.is_headless()
        if head:
            log(
                "warn",
                "Headless environment detected; will prefer opencv-python-headless",
            )
        else:
            log("info", "Display server looks available; GPU-capable OpenCV wheel ok")
        return {"nvidia": n, "ffmpeg": ff, "headless": head}

    def _detect_nvidia_smi_basic(self) -> Optional[dict]:
        code, out, err = run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,count",
                "--format=csv,noheader",
            ]
        )
        if code != 0 or not out.strip():
            return None
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        names = []
        driver = None
        total = 0
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 3:
                names.append(parts[0])
                driver = parts[1]
                try:
                    total += int(parts[2])
                except Exception:
                    total += 1
        return {"gpus": total, "names": names, "driver": driver}

    def detect_ffmpeg(self) -> bool:
        code, out, err = run(["ffmpeg", "-version"])
        return code == 0

    def is_headless(self) -> bool:
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                return False
            return True
        return False

    # --------------------- queue planning ---------------------
    def plan_queue(self, env: dict) -> List[PackageTask]:
        q: List[PackageTask] = []
        # look at torch_version flag: if provided we still prefer auto-detect if user requested auto
        if self.args.torch_version:
            torch_spec = (
                f"torch=={self.args.torch_version}+{self.args.cuda_tag}"
                if self.args.cuda_tag
                else f"torch=={self.args.torch_version}"
            )
            q.append(
                PackageTask(
                    name="torch",
                    wheel_spec=torch_spec,
                    install_args=[
                        "-i",
                        self.args.index_url,
                        "--extra-index-url",
                        self.args.extra_index_url,
                    ],
                    optional=False,
                    reason="PyTorch (user-pinned)",
                )
            )
            # append torchvision/torchaudio only if user supplied versions; otherwise trio will be handled by auto-install
            if self.args.torchvision_version:
                q.append(
                    PackageTask(
                        name="torchvision",
                        wheel_spec=f"torchvision=={self.args.torchvision_version}+{self.args.cuda_tag}",
                        install_args=[
                            "-i",
                            self.args.index_url,
                            "--extra-index-url",
                            self.args.extra_index_url,
                        ],
                        optional=True,
                    )
                )
            if self.args.torchaudio_version:
                q.append(
                    PackageTask(
                        name="torchaudio",
                        wheel_spec=f"torchaudio=={self.args.torchaudio_version}+{self.args.cuda_tag}",
                        install_args=[
                            "-i",
                            self.args.index_url,
                            "--extra-index-url",
                            self.args.extra_index_url,
                        ],
                        optional=True,
                    )
                )
        elif not self.args.auto_detect_torch:
            log(
                "warn",
                "PyTorch auto-install skipped (no --auto-detect-torch and no --torch-version provided)",
            )

        if self.args.install_cuda_python:
            q.append(
                PackageTask(
                    name="cuda-python",
                    optional=True,
                    reason="cuda-python: python bindings for CUDA runtime",
                )
            )
        if self.args.install_nvidia_ml:
            q.append(
                PackageTask(
                    name="nvidia-ml-py", optional=True, reason="pynvml / nvidia-ml-py"
                )
            )

        opencv_pkg = (
            "opencv-python-headless" if env.get("headless") else "opencv-python"
        )
        q.append(
            PackageTask(
                name=opencv_pkg,
                wheel_spec=f"{opencv_pkg}>={self.args.opencv_min}",
                optional=False,
            )
        )

        q.append(
            PackageTask(
                name="ffmpeg", optional=True, reason="system binary; validated only"
            )
        )
        # include non-machine checks as tasks for final reporting (we won't blindly reinstall these here)
        for spec in NON_MACHINE_DEPS:
            token = spec.split()[0].split("=")[0].split(">")[0]
            q.append(
                PackageTask(
                    name=token, wheel_spec=spec, optional=True, reason="non-machine dep"
                )
            )

        return q

    # -------------------- PyTorch auto-detection & helpers --------------------
    def detect_cuda_runtime(self) -> Optional[str]:
        # Attempt several robust strategies to obtain CUDA runtime like '12.8' or '13.0'
        queries = [
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            [
                "nvidia-smi",
                "--query-gpu=driver_version,cuda_version",
                "--format=csv,noheader",
            ],
        ]
        for q in queries:
            code, out, err = run(q)
            if code == 0 and out.strip():
                first = out.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",") if p.strip()]
                for p in reversed(parts):
                    if p and any(ch.isdigit() for ch in p):
                        if "." in p or p.isdigit():
                            return p
        # fallback to `nvidia-smi -q` parsing
        code, out, err = run(["nvidia-smi", "-q"])
        if code == 0 and out:
            for line in out.splitlines():
                if "CUDA Version" in line:
                    try:
                        return line.split(":")[-1].strip()
                    except Exception:
                        pass
        # fallback to nvcc
        code, out, err = run(["nvcc", "--version"])
        if code == 0 and out:
            for line in out.splitlines():
                if "release" in line:
                    try:
                        seg = line.split("release")[-1].split(",")[0].strip()
                        if seg:
                            return seg
                    except Exception:
                        pass
        return None

    def cuda_runtime_to_tags(self, cuda_runtime: Optional[str]) -> List[str]:
        # produce prioritized tag list and always include cpu fallback
        if not cuda_runtime:
            candidates = ["cu130", "cu128", "cu121", "cu118"]
        else:
            seg = cuda_runtime.split(".")
            try:
                major = int(seg[0])
                minor = int(seg[1]) if len(seg) > 1 else 0
                candidates = [f"cu{major}{minor}"]
                if minor >= 1:
                    candidates.append(f"cu{major}{minor-1}")
                if major >= 13:
                    candidates += ["cu130", "cu128", "cu121"]
                elif major == 12:
                    candidates += ["cu128", "cu121", "cu118"]
                elif major == 11:
                    candidates += ["cu118", "cu117", "cu116"]
                else:
                    candidates += ["cu118", "cu121"]
            except Exception:
                candidates = ["cu130", "cu128", "cu121", "cu118"]
        # dedupe and append cpu
        seen = []
        for c in candidates:
            if c not in seen:
                seen.append(c)
        seen.append("cpu")
        return seen

    def try_install_torch_trio(
        self, candidates: List[str], no_deps: bool = False
    ) -> bool:
        """Try candidates in order. We use unpinned package names so pip selects matching wheels for Python/runtime.
        If user supplied explicit torchvision/torchaudio versions they are respected.
        Returns True on success.
        """
        log(
            "info",
            f"Attempting PyTorch trio auto-install with candidates: {candidates}",
        )
        tv_ver = self.args.torchvision_version
        ta_ver = self.args.torchaudio_version

        for cand in candidates:
            # choose index
            if cand == "cpu":
                index_args = ["-i", self.args.extra_index_url]
                index_label = self.args.extra_index_url
            else:
                index_args = ["-i", f"https://download.pytorch.org/whl/{cand}"]
                index_label = index_args[1]

            # first quick validate - check that a wheel exists for torch (no heavy installs yet)
            check_spec = "torch"
            log(
                "info",
                f"Validating torch availability on index {index_label} (candidate {cand})...",
            )
            tmpd = tempfile.mkdtemp(prefix="torch_validate_")
            try:
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
                    + ["--extra-index-url", self.args.extra_index_url]
                )
                code, out, err = run(cmd)
                files = os.listdir(tmpd)
                if code == 0 and any(f.endswith(".whl") for f in files):
                    log(
                        "ok",
                        f"Found torch wheel(s) on {index_label}. Proceeding to install trio using that index.",
                    )
                    # build install names
                    pkg_names = ["torch"]
                    if tv_ver:
                        pkg_names.append(f"torchvision=={tv_ver}")
                    else:
                        pkg_names.append("torchvision")
                    if ta_ver:
                        pkg_names.append(f"torchaudio=={ta_ver}")
                    else:
                        pkg_names.append("torchaudio")

                    install_cmd = (
                        [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
                        + pkg_names
                        + index_args
                        + ["--extra-index-url", self.args.extra_index_url]
                    )
                    if no_deps:
                        install_cmd.append("--no-deps")
                    code2, out2, err2 = run(install_cmd)
                    if code2 == 0:
                        log(
                            "ok",
                            f"Installed torch trio from {index_label} (candidate {cand}) successfully.",
                        )
                        return True
                    else:
                        log(
                            "warn", f"Install failed on candidate {cand}: {out2} {err2}"
                        )
                else:
                    log(
                        "warn",
                        f"No wheel for torch on {index_label}. pip output: {out} {err}",
                    )
            finally:
                try:
                    shutil.rmtree(tmpd)
                except Exception:
                    pass

        log("error", "Could not install torch trio across all candidates.")
        return False

    # -------------------- installed trio detection/compat --------------------
    def get_installed_torch_info(self) -> dict:
        info = {
            "torch": None,
            "torchvision": None,
            "torchaudio": None,
            "torch_cuda": None,
            "cuda_available": False,
        }
        try:
            tmod = importlib.import_module("torch")
            info["torch"] = getattr(tmod, "__version__", None)
            # torch.version.cuda may exist
            try:
                info["torch_cuda"] = getattr(tmod, "version").cuda
            except Exception:
                info["torch_cuda"] = None
            try:
                info["cuda_available"] = tmod.cuda.is_available()
            except Exception:
                info["cuda_available"] = False
        except Exception:
            pass
        try:
            tv = importlib.import_module("torchvision")
            info["torchvision"] = getattr(tv, "__version__", None)
        except Exception:
            pass
        try:
            ta = importlib.import_module("torchaudio")
            info["torchaudio"] = getattr(ta, "__version__", None)
        except Exception:
            pass
        return info

    def trio_needs_install(
        self, candidate_tag: Optional[str]
    ) -> Tuple[bool, List[str]]:
        """Return (need_install, reasons). If anything missing or mismatched we return True.
        Heuristics: if torch missing -> True. If torch present but tag mismatch -> True. If torchvision/torchaudio missing -> True but optional.
        Caller can decide to force reinstall.
        """
        info = self.get_installed_torch_info()
        reasons = []
        if not info.get("torch"):
            reasons.append("torch not installed")
            return True, reasons
        # check tag
        if candidate_tag and candidate_tag != "cpu":
            tv = info.get("torch") or ""
            if candidate_tag not in tv:
                tcuda = info.get("torch_cuda")
                if tcuda:
                    tag_from_tcuda = f"cu{str(tcuda).replace('.','') }"
                    if tag_from_tcuda != candidate_tag:
                        reasons.append(
                            f"installed torch CUDA tag {tag_from_tcuda} != desired {candidate_tag}"
                        )
                else:
                    reasons.append("installed torch has no cuda metadata to verify tag")
        # missing extras
        if not info.get("torchvision"):
            reasons.append("torchvision missing")
        if not info.get("torchaudio"):
            reasons.append("torchaudio missing")
        # if critical reasons present (torch missing or mismatch), require install; if only missing optional libs, also return True but note as non-critical
        critical = [r for r in reasons if r.startswith("torch not") or "mismatch" in r]
        if critical or reasons:
            return True, reasons
        return False, []

    def uninstall_trio(self):
        log("info", "Uninstalling torch/torchvision/torchaudio (best-effort)...")
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
            ]
        )

    # -------------------- task execution --------------------
    def install_task(self, task: PackageTask) -> bool:
        """Generic installer for a PackageTask. Respects wheel_spec / install_args. Also checks already-present for non-machine deps."""
        # short-circuit: ffmpeg handled specially
        if task.name == "ffmpeg":
            ok = self.detect_ffmpeg()
            if ok:
                self.summary["already"].append("ffmpeg")
                log("ok", "ffmpeg available on PATH")
                return True
            else:
                log("warn", "ffmpeg not installed on system PATH")
                return False

        # non-machine dep check: if wheel_spec looks like 'name>=..' we do presence check
        if task.reason == "non-machine dep" or task.name in [
            t.split("==")[0] for t in NON_MACHINE_DEPS
        ]:
            # use heuristic check
            if package_already_present(task.wheel_spec or task.name):
                self.summary["already"].append(task.name)
                log("info", f"{task.name} already present; skipping install")
                return True
            # else fallthrough to install

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
        code, out, err = run(pip_cmd)
        if code == 0:
            self.summary["installed"].append(task.name)
            log("ok", f"Installed {task.name}")
            return True
        else:
            self.summary["failed"].append(task.name)
            log("error", f"Failed to install {task.name}: {out} {err}")
            return False

    # -------------------- execute flow --------------------
    def execute(self):
        env = self.detect_system()
        # make sure rich available for pretty UI
        ensure_rich(self.args.no_deps)
        # prepare queue
        queue = self.plan_queue(env)

        # --- PyTorch auto-detect + install logic ---
        if self.args.auto_detect_torch:
            cuda_rt = self.detect_cuda_runtime()
            tags = self.cuda_runtime_to_tags(cuda_rt)
            log("info", f"Auto-detected CUDA runtime: {cuda_rt} -> trying tags {tags}")
            candidate_tag = tags[0] if tags else None

            need_install, reasons = self.trio_needs_install(candidate_tag)
            if not need_install:
                # trio looks OK - add to summary as already
                info = self.get_installed_torch_info()
                self.summary["already"].append("torch")
                if info.get("torchvision"):
                    self.summary["already"].append("torchvision")
                if info.get("torchaudio"):
                    self.summary["already"].append("torchaudio")
                log("ok", f"Existing torch trio seems OK: {info}")
                # remove trio tasks from queue if present
                queue = [
                    q
                    for q in queue
                    if q.name not in ("torch", "torchvision", "torchaudio")
                ]
            else:
                log("warn", f"Torch trio needs install or fix: {reasons}")
                if self.args.force_reinstall:
                    self.uninstall_trio()
                    installed = self.try_install_torch_trio(
                        tags, no_deps=self.args.no_deps
                    )
                    if installed:
                        self.summary["installed"].extend(
                            ["torch", "torchvision", "torchaudio"]
                        )
                        queue = [
                            q
                            for q in queue
                            if q.name not in ("torch", "torchvision", "torchaudio")
                        ]
                    else:
                        log(
                            "error",
                            "Auto reinstall of trio failed; will fall back to normal queue",
                        )
                else:
                    # try to install only missing pieces (best-effort)
                    info = self.get_installed_torch_info()
                    missing = []
                    if not info.get("torch"):
                        missing.append("torch")
                    if not info.get("torchvision"):
                        missing.append("torchvision")
                    if not info.get("torchaudio"):
                        missing.append("torchaudio")
                    if missing:
                        # run the try_install on tags which installs trio (pip will skip already present ones or upgrade them)
                        installed = self.try_install_torch_trio(
                            tags, no_deps=self.args.no_deps
                        )
                        if installed:
                            self.summary["installed"].extend(missing)
                            queue = [
                                q
                                for q in queue
                                if q.name not in ("torch", "torchvision", "torchaudio")
                            ]
                        else:
                            log(
                                "warn",
                                "Auto-install attempt for missing trio parts failed; continuing with regular queue",
                            )

        # Show planned queue in a table — robust: filter out items already recorded in summary['already']
        visible_queue = [q for q in queue if q.name not in self.summary["already"]]
        if RICH:
            table = Table(title="Planned install queue", show_lines=True)
            table.add_column("#", style="bold")
            table.add_column("Package")
            table.add_column("Spec")
            table.add_column("Optional")
            table.add_column("Notes")
            for i, t in enumerate(visible_queue, 1):
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
            for i, t in enumerate(visible_queue, 1):
                log(
                    "info",
                    f"{i}. {t.name} spec={t.wheel_spec or '-'} optional={t.optional} notes={t.reason or '-'}",
                )

        if self.args.dry_run:
            log("warn", "Dry-run: not performing any installs. Exiting.")
            return

        # Install remaining tasks
        for t in visible_queue:
            ok = self.install_task(t)
            if not ok and not t.optional:
                log(
                    "error",
                    f"Required task {t.name} failed — aborting further non-optional installs",
                )
                break

        # Final summary
        self._print_summary()

    def _print_summary(self):
        total_time = time.time() - self.start
        if RICH:
            t = Table(title="Installation summary", show_lines=True)
            t.add_column("Status")
            t.add_column("Packages")
            t.add_row("Installed", ", ".join(self.summary["installed"]) or "-")
            t.add_row(
                "Already present / Skipped", ", ".join(self.summary["already"]) or "-"
            )
            t.add_row("Failed", ", ".join(self.summary["failed"]) or "-")
            t.add_row("Time(s)", f"{total_time:.1f}s")
            CONSOLE.print(t)
        else:
            log("info", f"Summary Installed: {self.summary['installed']}")
            log("info", f"Already/Skipped: {self.summary['already']}")
            log("info", f"Failed: {self.summary['failed']}")
            log("info", f"Total time: {total_time:.1f}s")

    def cleanup(self):
        try:
            shutil.rmtree(self.tempdir)
        except Exception:
            pass


# ---------------------------- CLI and main ------------------------------------------


def parse_args():
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
    return ap.parse_args()


def _signal_handler(sig, frame):
    log("error", "Interrupted by user. Exiting...")
    sys.exit(1)


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, _signal_handler)

    # Phase-1: install non-machine deps but only missing ones
    # ensure rich is installable early (so UI works)
    ensure_rich(args.no_deps)

    # check for missing non-machine deps & install only those
    missing = [s for s in NON_MACHINE_DEPS if not package_already_present(s)]
    if missing:
        log("info", f"Non-machine packages missing: {missing}")
        pip_cmd = [sys.executable, "-m", "pip", "install"] + missing
        if args.no_deps:
            pip_cmd.append("--no-deps")
        code, out, err = run(pip_cmd)
        if code == 0:
            log("ok", "Installed missing non-machine packages")
        else:
            log("warn", f"Some non-machine installs failed: {out} {err}")
    else:
        log("ok", "All non-machine dependencies already present")

    installer = Installer(args)
    try:
        installer.execute()
    finally:
        installer.cleanup()


if __name__ == "__main__":
    main()

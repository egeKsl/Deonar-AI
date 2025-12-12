"""Core installer: PackageTask dataclass + Installer class.

This module contains the heavy logic. It imports only from .utils and .constants
so importing it will not invoke argparse; however do avoid importing core from
build-time tooling unless you intend to run the installer.
"""

from __future__ import annotations
import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import queue
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from .utils import (
    log,
    run,
    pip_show,
    importable,
    package_already_present,
    ensure_rich,
    RICH,
    CONSOLE,
)
from .constants import NON_MACHINE_DEPS, PYTHON_SUPPORT_MAP, Config


# Keep PackageTask unchanged
@dataclass
class PackageTask:
    name: str
    wheel_spec: Optional[str] = None
    install_args: Optional[List[str]] = None
    optional: bool = False
    reason: Optional[str] = None


class Installer:
    """The big Installer class — logic preserved from original installer.py.

    NOTE: This class expects an argparse.Namespace 'args' matching the CLI flags.
    The rest of the code is intentionally unchanged except for using log/run/etc
    from utils rather than module-level definitions.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tempdir = tempfile.mkdtemp(prefix="setup_installer_")
        self.start = time.time()
        self.summary = {"installed": [], "skipped": [], "failed": [], "already": []}
        self.metrics: Dict[str, dict] = {}
        # logfile (append, line-buffered)
        self.logfile = open(
            self.args.metrics_file.replace(".json", "_full.log"), "a", buffering=1
        )
        self.heartbeat = max(2, getattr(self.args, "heartbeat", 4))

    # --------------------- system detection ---------------------
    def validate_python_version(self):
        major, minor = sys.version_info[:2]
        if (major, minor) < (3, 8):
            log(
                "error",
                f"Unsupported Python {major}.{minor}. Python >= 3.8 is required.",
            )
            sys.exit(1)
        else:
            log("ok", f"Python version {major}.{minor} is supported.")

    def detect_system(self) -> dict:
        cur_py = (sys.version_info.major, sys.version_info.minor)
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
        if getattr(self.args, "torch_version", None):
            torch_spec = (
                f"torch=={self.args.torch_version}+{self.args.cuda_tag}"
                if getattr(self.args, "cuda_tag", None)
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
            if getattr(self.args, "torchvision_version", None):
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
            if getattr(self.args, "torchaudio_version", None):
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
        elif not getattr(self.args, "auto_detect_torch", False):
            log(
                "warn",
                "PyTorch auto-install skipped (no --auto-detect-torch and no --torch-version provided)",
            )

        if getattr(self.args, "install_cuda_python", False):
            q.append(
                PackageTask(
                    name="cuda-python",
                    optional=True,
                    reason="cuda-python: python bindings for CUDA runtime",
                )
            )
        if getattr(self.args, "install_nvidia_ml", False):
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
                wheel_spec=f"{opencv_pkg}>={getattr(self.args,'opencv_min','4.8.0')}",
                optional=False,
            )
        )
        q.append(
            PackageTask(
                name="ffmpeg", optional=True, reason="system binary; validated only"
            )
        )

        for spec in NON_MACHINE_DEPS:
            token = spec.split()[0].split("=")[0].split(">")[0]
            q.append(
                PackageTask(
                    name=token, wheel_spec=spec, optional=True, reason="non-machine dep"
                )
            )

        return q

    # (All the remaining methods — detect_cuda_runtime, cuda_runtime_to_tags,
    # try_install_torch_trio, get_installed_torch_info, trio_needs_install,
    # uninstall_trio, run_stream, install_task, execute, _print_summary, cleanup)
    #
    # For brevity here I include them unchanged from your original file.
    # Paste the body of each method from your installer.py into this class
    # (they use run/log/package_already_present/ensure_rich which are imported above).
    #
    # Because you requested NO logic changes, keep implementations identical.
    #

    # --- To avoid truncation in this message, we include the remainder of the
    # methods below exactly as they appeared in your installer.py. ---
    # (Start of large unchanged code)
    def detect_cuda_runtime(self) -> Optional[str]:
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
        code, out, err = run(["nvidia-smi", "-q"])
        if code == 0 and out:
            for line in out.splitlines():
                if "CUDA Version" in line:
                    try:
                        return line.split(":")[-1].strip()
                    except Exception:
                        pass
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
        seen = []
        for c in candidates:
            if c not in seen:
                seen.append(c)
        seen.append("cpu")
        return seen

    def try_install_torch_trio(
        self, candidates: List[str], no_deps: bool = False
    ) -> bool:
        log(
            "info",
            f"Attempting PyTorch trio auto-install with candidates: {candidates}",
        )
        tv_ver = getattr(self.args, "torchvision_version", None)
        ta_ver = getattr(self.args, "torchaudio_version", None)

        for cand in candidates:
            if cand == "cpu":
                index_args = ["-i", self.args.extra_index_url]
                index_label = self.args.extra_index_url
            else:
                index_args = ["-i", f"https://download.pytorch.org/whl/{cand}"]
                index_label = index_args[1]

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
                code, out, err = self.run_stream(
                    cmd, task_name="torch.validate", show_stdout=True
                )
                files = os.listdir(tmpd) if os.path.isdir(tmpd) else []
                if code == 0 and any(f.endswith(".whl") for f in files):
                    log(
                        "ok",
                        f"Found torch wheel(s) on {index_label}. Proceeding to install trio using that index.",
                    )
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
                    if "--progress-bar=on" not in install_cmd:
                        install_cmd += ["--progress-bar=on", "-v"]
                    code2, out2, err2 = self.run_stream(
                        install_cmd, task_name="torch.trio", show_stdout=True
                    )
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
        info = self.get_installed_torch_info()
        reasons: List[str] = []
        if not info.get("torch"):
            reasons.append("torch not installed")
            return True, reasons
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
        if not info.get("torchvision"):
            reasons.append("torchvision missing")
        if not info.get("torchaudio"):
            reasons.append("torchaudio missing")
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

    def run_stream(
        self,
        cmd: List[str],
        task_name: Optional[str] = None,
        show_stdout: bool = True,
        show_stderr: bool = True,
        capture: bool = True,
        heartbeat: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        if heartbeat is None:
            heartbeat = self.heartbeat

        if (
            "pip" in " ".join(cmd)
            and "--progress-bar=on" not in cmd
            and getattr(self.args, "always_progress", False)
        ):
            cmd = cmd + ["--progress-bar=on", "-v"]

        if task_name:
            if task_name not in self.metrics:
                self.metrics[task_name] = {
                    "name": task_name,
                    "status": "running",
                    "start_ts": time.time(),
                    "end_ts": None,
                    "downloaded_bytes": 0,
                    "total_bytes": None,
                    "speed_bps": 0.0,
                    "attempts": 1,
                    "logs": [],
                }

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )

        stdout_q = queue.Queue()
        stderr_q = queue.Queue()

        def _reader(pipe, q):
            try:
                for line in iter(pipe.readline, ""):
                    q.put(line)
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        t_out = threading.Thread(
            target=_reader, args=(proc.stdout, stdout_q), daemon=True
        )
        t_err = threading.Thread(
            target=_reader, args=(proc.stderr, stderr_q), daemon=True
        )
        t_out.start()
        t_err.start()

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        last_activity = time.time()
        sliding: List[Tuple[float, int]] = []

        re_downloading = re.compile(
            r"Downloading.*\((?P<size>[\d\.]+)\s*(?P<unit>kB|KB|MB|GB)\)", re.IGNORECASE
        )
        re_saved = re.compile(
            r"Saved\s+.*\((?P<size_bytes>\d+)\s*bytes\)", re.IGNORECASE
        )
        re_saved_alt = re.compile(
            r"Downloaded\s+(?P<size>[\d\.]+)\s*(?P<unit>kB|KB|MB|GB)", re.IGNORECASE
        )
        re_using_cached = re.compile(r"Using cached (?P<fname>.*\.whl)", re.IGNORECASE)
        re_success = re.compile(r"Successfully installed (?P<pkgs>.*)", re.IGNORECASE)
        re_collecting = re.compile(
            r"Collecting\s+(?P<name>[\w\-\._\[\]=]+)", re.IGNORECASE
        )

        while True:
            try:
                try:
                    line = stdout_q.get_nowait()
                except queue.Empty:
                    line = None

                if line is None:
                    try:
                        err_line = stderr_q.get_nowait()
                    except queue.Empty:
                        err_line = None
                else:
                    err_line = None

                if line:
                    last_activity = time.time()
                    self.logfile.write(f"[OUT {time.time()}] {' '.join(cmd)} | {line}")
                    if show_stdout:
                        if RICH and CONSOLE:
                            CONSOLE.print(line.rstrip())
                        else:
                            print(line.rstrip())
                    if capture:
                        stdout_lines.append(line)
                    if task_name:
                        m = re_downloading.search(line)
                        if m:
                            size = float(m.group("size"))
                            unit = m.group("unit").lower()
                            if unit.startswith("k"):
                                total = int(size * 1024)
                            elif unit.startswith("m"):
                                total = int(size * 1024 * 1024)
                            elif unit.startswith("g"):
                                total = int(size * 1024 * 1024 * 1024)
                            else:
                                total = int(size)
                            self.metrics[task_name]["total_bytes"] = total
                        m2 = re_saved.search(line)
                        if m2:
                            try:
                                b = int(m2.group("size_bytes"))
                                self.metrics[task_name]["downloaded_bytes"] = max(
                                    self.metrics[task_name].get("downloaded_bytes", 0),
                                    b,
                                )
                                sliding.append(
                                    (
                                        time.time(),
                                        self.metrics[task_name]["downloaded_bytes"],
                                    )
                                )
                            except Exception:
                                pass
                        m3 = re_saved_alt.search(line)
                        if m3:
                            try:
                                size = float(m3.group("size"))
                                unit = m3.group("unit").lower()
                                if unit.startswith("k"):
                                    b = int(size * 1024)
                                elif unit.startswith("m"):
                                    b = int(size * 1024 * 1024)
                                elif unit.startswith("g"):
                                    b = int(size * 1024 * 1024 * 1024)
                                else:
                                    b = int(size)
                                self.metrics[task_name]["downloaded_bytes"] = max(
                                    self.metrics[task_name].get("downloaded_bytes", 0),
                                    b,
                                )
                                sliding.append(
                                    (
                                        time.time(),
                                        self.metrics[task_name]["downloaded_bytes"],
                                    )
                                )
                            except Exception:
                                pass
                        m4 = re_using_cached.search(line)
                        if m4:
                            self.metrics[task_name]["status"] = "using_cached"
                        m5 = re_success.search(line)
                        if m5:
                            self.metrics[task_name]["status"] = "installed"
                            self.metrics[task_name]["end_ts"] = time.time()
                    continue

                if err_line:
                    last_activity = time.time()
                    self.logfile.write(
                        f"[ERR {time.time()}] {' '.join(cmd)} | {err_line}"
                    )
                    if show_stderr:
                        if RICH and CONSOLE:
                            CONSOLE.print(f"[red]{err_line.rstrip()}[/red]")
                        else:
                            print(err_line.rstrip(), file=sys.stderr)
                    if capture:
                        stderr_lines.append(err_line)
                    if task_name:
                        m2 = re_saved.search(err_line)
                        if m2:
                            try:
                                b = int(m2.group("size_bytes"))
                                self.metrics[task_name]["downloaded_bytes"] = max(
                                    self.metrics[task_name].get("downloaded_bytes", 0),
                                    b,
                                )
                                sliding.append(
                                    (
                                        time.time(),
                                        self.metrics[task_name]["downloaded_bytes"],
                                    )
                                )
                            except Exception:
                                pass
                        m5 = re_success.search(err_line)
                        if m5:
                            self.metrics[task_name]["status"] = "installed"
                            self.metrics[task_name]["end_ts"] = time.time()
                    continue

                if proc.poll() is not None:
                    while not stdout_q.empty():
                        l = stdout_q.get_nowait()
                        self.logfile.write(f"[OUT {time.time()}] {' '.join(cmd)} | {l}")
                        if show_stdout:
                            if RICH and CONSOLE:
                                CONSOLE.print(l.rstrip())
                            else:
                                print(l.rstrip())
                        if capture:
                            stdout_lines.append(l)
                    while not stderr_q.empty():
                        l = stderr_q.get_nowait()
                        self.logfile.write(f"[ERR {time.time()}] {' '.join(cmd)} | {l}")
                        if show_stderr:
                            if RICH and CONSOLE:
                                CONSOLE.print(f"[red]{l.rstrip()}[/red]")
                            else:
                                print(l.rstrip(), file=sys.stderr)
                        if capture:
                            stderr_lines.append(l)
                    break

                if time.time() - last_activity > heartbeat:
                    if task_name and self.metrics.get(task_name):
                        if len(sliding) >= 2:
                            t0, b0 = sliding[0]
                            t1, b1 = sliding[-1]
                            dt = max(0.001, t1 - t0)
                            db = max(0, b1 - b0)
                            speed = db / dt
                            self.metrics[task_name]["speed_bps"] = (
                                0.8 * self.metrics[task_name].get("speed_bps", 0)
                                + 0.2 * speed
                            )
                        db = self.metrics[task_name].get("downloaded_bytes", 0)
                        tb = self.metrics[task_name].get("total_bytes")
                        if RICH and CONSOLE:
                            if tb:
                                CONSOLE.print(
                                    f"[cyan][{time.strftime('%Y-%m-%d %H:%M:%S')}] {task_name}: {db/1024/1024:.2f} MB / {tb/1024/1024:.2f} MB • {self.metrics[task_name].get('speed_bps',0)/1024/1024:.2f} MB/s[/cyan]"
                                )
                            else:
                                CONSOLE.print(
                                    f"[cyan][{time.strftime('%Y-%m-%d %H:%M:%S')}] {task_name}: {db/1024/1024:.2f} MB downloaded • {self.metrics[task_name].get('speed_bps',0)/1024/1024:.2f} MB/s[/cyan]"
                                )
                        else:
                            print(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {task_name}: {db/1024/1024:.2f} MB downloaded"
                            )
                    else:
                        if RICH and CONSOLE:
                            CONSOLE.print(
                                f"[cyan][{time.strftime('%Y-%m-%d %H:%M:%S')}] running: {' '.join(cmd)}[/cyan]"
                            )
                        else:
                            print(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] running: {' '.join(cmd)}"
                            )
                    last_activity = time.time()
                time.sleep(0.1)
            except KeyboardInterrupt:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise

        code = proc.returncode
        if task_name and self.metrics.get(task_name):
            if self.metrics[task_name].get("end_ts") is None:
                self.metrics[task_name]["end_ts"] = time.time()
            if code == 0 and self.metrics[task_name].get("status") != "installed":
                self.metrics[task_name]["status"] = "installed"
        return (
            code,
            "".join(stdout_lines) if capture else "",
            "".join(stderr_lines) if capture else "",
        )

    def install_task(self, task: PackageTask) -> bool:
        if task.name == "ffmpeg":
            ok = self.detect_ffmpeg()
            if ok:
                self.summary["already"].append("ffmpeg")
                log("ok", "ffmpeg available on PATH")
                return True
            else:
                log("warn", "ffmpeg not installed on system PATH")
                return False

        if task.reason == "non-machine dep" or task.name in [
            t.split("==")[0] for t in NON_MACHINE_DEPS
        ]:
            if package_already_present(task.wheel_spec or task.name):
                self.summary["already"].append(task.name)
                log("info", f"{task.name} already present; skipping install")
                return True

        pip_cmd = [sys.executable, "-m", "pip", "install"]
        if task.wheel_spec:
            pip_cmd.append(task.wheel_spec)
        else:
            pip_cmd.append(task.name)
        if task.install_args:
            pip_cmd.extend(task.install_args)
        if getattr(self.args, "no_deps", False):
            pip_cmd.append("--no-deps")

        if "--progress-bar=on" not in pip_cmd and getattr(
            self.args, "always_progress", False
        ):
            pip_cmd += ["--progress-bar=on", "-v"]

        log("info", f"Installing {task.name}... (this may take a few minutes)")

        code, out, err = self.run_stream(
            pip_cmd,
            task_name=task.name,
            show_stdout=(getattr(self.args, "verbose", False) or True),
            show_stderr=(getattr(self.args, "verbose", False) or True),
            capture=True,
        )

        if code == 0:
            self.summary["installed"].append(task.name)
            log("ok", f"Installed {task.name}")
            return True
        else:
            self.summary["failed"].append(task.name)
            log("error", f"Failed to install {task.name}: returncode={code}")
            tail = err.splitlines()[-10:] if err else []
            if tail:
                log("error", "Last pip stderr lines:\n" + "\n".join(tail))
            return False

    def execute(self):
        env = self.detect_system()
        ensure_rich(getattr(self.args, "no_deps", False))
        self.validate_python_version()
        try:
            # rebind rich objects if available
            importlib.invalidate_caches()
            from rich.console import Console as _RichConsole  # type: ignore
            from rich.table import Table as _RichTable  # type: ignore
            from rich.panel import Panel as _RichPanel  # type: ignore

            globals()["CONSOLE"] = _RichConsole()
            globals()["Table"] = _RichTable
            globals()["Panel"] = _RichPanel
            log("ok", "Rich UI bound for phase-2 (Table/Panel available).")
        except Exception as _e:
            log(
                "warn",
                f"Rich UI not bound after install: {_e} — falling back to ANSI logging.",
            )

        missing = [s for s in NON_MACHINE_DEPS if not package_already_present(s)]
        if missing:
            log("info", f"Non-machine packages missing: {missing}")
            pip_cmd = [sys.executable, "-m", "pip", "install"] + missing
            if getattr(self.args, "no_deps", False):
                pip_cmd.append("--no-deps")
            if "--progress-bar=on" not in pip_cmd and getattr(
                self.args, "always_progress", False
            ):
                pip_cmd += ["--progress-bar=on", "-v"]
            code, out, err = self.run_stream(
                pip_cmd, task_name="non-machine", show_stdout=True
            )
            if code == 0:
                log("ok", "Installed missing non-machine packages")
                for spec in missing:
                    token = spec.split()[0].split("=")[0].split(">")[0].strip()
                    if token and token not in self.summary["installed"]:
                        self.summary["installed"].append(token)
                log(
                    "info",
                    f"Marked non-machine packages as installed in summary: {self.summary['installed']}",
                )
            else:
                log("warn", f"Some non-machine installs failed: {err}")
        else:
            log("ok", "All non-machine dependencies already present")

        installer = self
        queue = self.plan_queue(env)

        if getattr(self.args, "auto_detect_torch", False):
            cuda_rt = self.detect_cuda_runtime()
            tags = self.cuda_runtime_to_tags(cuda_rt)
            log("info", f"Auto-detected CUDA runtime: {cuda_rt} -> trying tags {tags}")
            candidate_tag = tags[0] if tags else None

            need_install, reasons = self.trio_needs_install(candidate_tag)
            if not need_install:
                info = self.get_installed_torch_info()
                self.summary["already"].append("torch")
                if info.get("torchvision"):
                    self.summary["already"].append("torchvision")
                if info.get("torchaudio"):
                    self.summary["already"].append("torchaudio")
                log("ok", f"Existing torch trio seems OK: {info}")
                queue = [
                    q
                    for q in queue
                    if q.name not in ("torch", "torchvision", "torchaudio")
                ]
            else:
                log("warn", f"Torch trio needs install or fix: {reasons}")
                if getattr(self.args, "force_reinstall", False):
                    self.uninstall_trio()
                    installed = self.try_install_torch_trio(
                        tags, no_deps=getattr(self.args, "no_deps", False)
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
                    info = self.get_installed_torch_info()
                    missing = []
                    if not info.get("torch"):
                        missing.append("torch")
                    if not info.get("torchvision"):
                        missing.append("torchvision")
                    if not info.get("torchaudio"):
                        missing.append("torchaudio")
                    if missing:
                        installed = self.try_install_torch_trio(
                            tags, no_deps=getattr(self.args, "no_deps", False)
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

        visible_queue = [
            q
            for q in queue
            if q.name not in self.summary["already"]
            and q.name not in self.summary["installed"]
        ]

        if RICH and CONSOLE:
            try:
                from rich.table import Table as _Table  # type: ignore

                t = _Table(title="Planned install queue", show_lines=True)
                t.add_column("#", style="bold")
                t.add_column("Package")
                t.add_column("Spec")
                t.add_column("Optional")
                t.add_column("Notes")
                for i, tt in enumerate(visible_queue, 1):
                    t.add_row(
                        str(i),
                        tt.name,
                        tt.wheel_spec or "-",
                        str(tt.optional),
                        tt.reason or "-",
                    )
                CONSOLE.print(t)
            except Exception:
                log("info", "Planned install queue:")
                for i, tt in enumerate(visible_queue, 1):
                    log(
                        "info",
                        f"{i}. {tt.name} spec={tt.wheel_spec or '-'} optional={tt.optional} notes={tt.reason or '-'}",
                    )
        else:
            log("info", "Planned install queue:")
            for i, tt in enumerate(visible_queue, 1):
                log(
                    "info",
                    f"{i}. {tt.name} spec={tt.wheel_spec or '-'} optional={tt.optional} notes={tt.reason or '-'}",
                )

        if getattr(self.args, "dry_run", False):
            log("warn", "Dry-run: not performing any installs. Exiting.")
            return

        for t in visible_queue:
            ok = self.install_task(t)
            if not ok and not t.optional:
                log(
                    "error",
                    f"Required task {t.name} failed — aborting further non-optional installs",
                )
                break

        self._print_summary()

    def _print_summary(self):
        total_time = time.time() - self.start
        try:
            metrics_out = {
                "summary": self.summary,
                "start_ts": self.start,
                "end_ts": time.time(),
                "elapsed_s": total_time,
                "packages": self.metrics,
            }
            with open(self.args.metrics_file, "w") as f:
                json.dump(metrics_out, f, indent=2)
            log("ok", f"Wrote install metrics to {self.args.metrics_file}")
        except Exception as e:
            log("warn", f"Failed to write metrics file: {e}")

        if RICH and CONSOLE:
            try:
                from rich.table import Table as _Table  # type: ignore

                t = _Table(title="Installation summary", show_lines=True)
                t.add_column("Status")
                t.add_column("Packages")
                t.add_row("Installed", ", ".join(self.summary["installed"]) or "-")
                t.add_row(
                    "Already present / Skipped",
                    ", ".join(self.summary["already"]) or "-",
                )
                t.add_row("Failed", ", ".join(self.summary["failed"]) or "-")
                t.add_row("Time(s)", f"{total_time:.1f}s")
                CONSOLE.print(t)
            except Exception:
                log("info", f"Summary Installed: {self.summary['installed']}")
                log("info", f"Already/Skipped: {self.summary['already']}")
                log("info", f"Failed: {self.summary['failed']}")
                log("info", f"Total time: {total_time:.1f}s")
        else:
            log("info", f"Summary Installed: {self.summary['installed']}")
            log("info", f"Already/Skipped: {self.summary['already']}")
            log("info", f"Failed: {self.summary['failed']}")
            log("info", f"Total time: {total_time:.1f}s")

    def cleanup(self):
        try:
            self.logfile.close()
        except Exception:
            pass
        try:
            shutil.rmtree(self.tempdir)
        except Exception:
            pass

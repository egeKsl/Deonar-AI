"""Small, import-safe utility helpers used by core installer.

This module intentionally avoids argparse or heavy side-effects so importing
it during build-time is safe.
"""

from __future__ import annotations
import importlib
import json
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple, List

# Try to optionally bind 'rich' objects lazily; using module-level names is convenient
RICH = False
CONSOLE = None

try:
    # import only to test availability (importing rich here is fine: it's lightweight)
    importlib.import_module("rich")
    from rich.console import Console  # type: ignore

    CONSOLE = Console()
    RICH = True
except Exception:
    RICH = False
    CONSOLE = None


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(level: str, msg: str) -> None:
    """Simple logging shim used by core. Keeps ANSI fallback if rich not installed."""
    emoji = {"info": "💡", "ok": "✅", "warn": "⚠️", "error": "❌"}.get(level, "")
    style = {
        "info": "cyan",
        "ok": "green",
        "warn": "yellow",
        "error": "red",
    }.get(level, "")
    if RICH and CONSOLE:
        CONSOLE.print(f"[{_now()}] {emoji} ", end="")
        try:
            CONSOLE.print(msg, style=style)
        except Exception:
            CONSOLE.print(msg)
    else:
        ANSI = {
            "cyan": "\x1b[96m",
            "green": "\x1b[92m",
            "yellow": "\x1b[93m",
            "red": "\x1b[91m",
            "end": "\x1b[0m",
        }
        col = {"info": "cyan", "ok": "green", "warn": "yellow", "error": "red"}.get(
            level, ""
        )
        print(f"[{_now()}] {ANSI.get(col,'')}{emoji} {msg}{ANSI['end']}")


def run(
    cmd: List[str], capture: bool = True, env: dict | None = None
) -> Tuple[int, str, str]:
    """Run subprocess and return (returncode, stdout, stderr)."""
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


def pip_show(package: str) -> Optional[Dict[str, str]]:
    """Return pip show metadata dict if package present, else None."""
    code, out, err = run([sys.executable, "-m", "pip", "show", package])
    if code != 0 or not out.strip():
        return None
    data: Dict[str, str] = {}
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
    """Heuristic: given a pip spec like 'numpy>=1.26.0', decide if installed."""
    token = spec.split()[0].split("=")[0].split(">")[0].strip()
    mapping = {"Pillow": "PIL"}
    module = mapping.get(token, token)
    if importable(module):
        return True
    info = pip_show(token)
    return info is not None


def ensure_rich(no_deps: bool = False) -> bool:
    """Ensure 'rich' is importable — install it alone if missing."""
    global RICH, CONSOLE
    try:
        importlib.import_module("rich")
        if not CONSOLE:
            try:
                from rich.console import Console as _Console  # type: ignore

                CONSOLE = _Console()
            except Exception:
                pass
        RICH = True
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
        from rich.console import Console as _Console  # type: ignore

        CONSOLE = _Console()
        RICH = True
        log("ok", "`rich` installed and usable.")
        return True
    except Exception as e:
        log("error", f"`rich` installed but not importable: {e}")
        return False

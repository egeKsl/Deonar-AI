# src/utils/logger.py
"""
Lightweight non-blocking logger (Rich console + background JSONL file writer).

Design decisions:
 - Do NOT read .env at import time. Prefer src.config.LOGGING if present.
 - Console output matches the old v1 look & theme (banner, tags, timestamp).
 - File writes are offloaded to a background writer thread to avoid blocking hot paths.
 - Queue uses a drop-oldest policy when full to prevent blocking.
"""

from __future__ import annotations
import os
import json
import threading
import queue
import time
import sys
import traceback
from datetime import datetime
from typing import Any, Dict

# Rich console
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text


# ----------- Defaults (overridable from src.config.LOGGING) -----------
_DEFAULTS = {
    "LIVE": True,
    "LOG_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"),
    "LOG_FILE": None,  # computed from LOG_DIR if None
    "LOG_LEVEL": "INFO",
    "LOG_TRUNCATE_ON_START": False,
    "LOG_MAX_BYTES": 5 * 1024 * 1024,
    "LOG_BACKUP_COUNT": 3,
    # writer queue
    "QUEUE_MAX": 8192,
    # whether writer thread is daemon
    "WRITER_DAEMON": True,
    # whether to print banner when live
    "SHOW_BANNER": True,
}

_cfg_src = {}

CFG = {**_DEFAULTS, **_cfg_src}

LOG_DIR = CFG["LOG_DIR"]
LOG_FILE = CFG["LOG_FILE"] or os.path.join(LOG_DIR, "log.jsonl")
LIVE_MODE = bool(CFG["LIVE"])
LOG_TRUNCATE_ON_START = bool(CFG["LOG_TRUNCATE_ON_START"])
LOG_MAX_BYTES = int(CFG["LOG_MAX_BYTES"])
LOG_BACKUP_COUNT = int(CFG["LOG_BACKUP_COUNT"])
QUEUE_MAX = int(CFG["QUEUE_MAX"])
WRITER_DAEMON = bool(CFG["WRITER_DAEMON"])
SHOW_BANNER = bool(CFG.get("SHOW_BANNER", True))

os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- Rich theme & console (keeps v1 look) ----------------
_custom_theme = Theme(
    {
        "info": "bold cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "debug": "bold magenta",
        "tag": "bold bright_white on grey23",
        "time": "dim white",
        "banner": "bold bright_white",
        "watermark": "italic dim cyan",
    }
)
_console = Console(theme=_custom_theme, highlight=False)


# ------------------ Background writer thread ------------------
def _rotate_file_if_needed(path: str, max_bytes: int, backup_count: int) -> None:
    try:
        if not os.path.exists(path):
            return
        size = os.path.getsize(path)
        if size <= max_bytes:
            return
        # rotate: shift backups up, discard oldest
        for i in range(backup_count - 1, 0, -1):
            src = f"{path}.{i}"
            dst = f"{path}.{i+1}"
            if os.path.exists(src):
                try:
                    os.replace(src, dst)
                except Exception:
                    pass
        # current -> .1
        try:
            os.replace(path, f"{path}.1")
        except Exception:
            # final fallback: try copy+truncate
            try:
                with open(path, "rb") as r, open(f"{path}.1", "wb") as w:
                    w.write(r.read())
                open(path, "w", encoding="utf-8").close()
            except Exception:
                pass
    except Exception:
        # never raise from rotation
        pass


class _FileWriter(threading.Thread):
    def __init__(self, path: str, q: "queue.Queue[Dict[str, Any]]", max_bytes: int, backups: int, daemon: bool = True):
        super().__init__(daemon=daemon, name="LoggerFileWriter")
        self.path = path
        self.q = q
        self.max_bytes = int(max_bytes)
        self.backups = int(backups)
        self._stop = threading.Event()

    def close(self):
        self._stop.set()

    def run(self):
        # open once and append; check rotation when needed
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # rotate if needed before writing
                _rotate_file_if_needed(self.path, self.max_bytes, self.backups)
                line = json.dumps(item, ensure_ascii=False)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                # swallow exceptions to never crash writer thread
                try:
                    # best-effort fallback: sleep a bit
                    time.sleep(0.01)
                except Exception:
                    pass
            finally:
                try:
                    self.q.task_done()
                except Exception:
                    pass
        # flush remaining queue items on stop (best-effort)
        while True:
            try:
                item = self.q.get_nowait()
            except Exception:
                break
            try:
                _rotate_file_if_needed(self.path, self.max_bytes, self.backups)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception:
                pass
            finally:
                try:
                    self.q.task_done()
                except Exception:
                    pass


# ------------------ Logger (public API) ------------------
class Logger:
    def __init__(self, live: bool = LIVE_MODE, logfile: str = LOG_FILE):
        self.console = _console
        self.live = bool(live)
        self.logfile = logfile

        # writer queue and thread
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=QUEUE_MAX)
        # If truncation requested on start, clear file
        if LOG_TRUNCATE_ON_START and os.path.exists(self.logfile):
            try:
                open(self.logfile, "w", encoding="utf-8").close()
            except Exception:
                pass
        self._writer = _FileWriter(self.logfile, self._q, LOG_MAX_BYTES, LOG_BACKUP_COUNT, daemon=WRITER_DAEMON)
        self._writer.start()

        if self.live and SHOW_BANNER:
            self._print_banner()
            
    def configure(self, cfg: dict) -> None:
        """
        Reconfigure logger at runtime. Accepts keys similar to YAML logging section.
        Safe to call once at startup; safe to call repeatedly.
        Recognised keys (case-insensitive): live, show_banner, file/log_file,
        dir/log_dir, max_bytes, backup_count, queue_max, writer_daemon, truncate_on_start.
        """

        # We assign to module-level constants below — declare them up-front.
        global LOG_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT, QUEUE_MAX, WRITER_DAEMON, LOG_TRUNCATE_ON_START, SHOW_BANNER

        if not isinstance(cfg, dict):
            return

        lc = {k.lower(): v for k, v in cfg.items() if v is not None}

        # 1) simple toggles
        if "live" in lc:
            try:
                self.set_live(bool(lc["live"]))
            except Exception:
                pass

        if "show_banner" in lc:
            try:
                SHOW_BANNER = bool(lc["show_banner"])
            except Exception:
                pass

        # 2) dir / logfile path update
        new_dir = lc.get("dir") or lc.get("log_dir")
        if new_dir:
            try:
                os.makedirs(new_dir, exist_ok=True)
                LOG_DIR = new_dir
            except Exception:
                pass

        new_file = lc.get("file") or lc.get("log_file")
        if new_file:
            if not os.path.isabs(new_file) and os.path.sep not in new_file:
                new_file = os.path.join(LOG_DIR, new_file)
            try:
                self.logfile = new_file
            except Exception:
                pass

        # 3) rotation & queue params (optional)
        new_max_bytes = None
        if "max_bytes" in lc:
            try:
                new_max_bytes = int(lc["max_bytes"])
            except Exception:
                new_max_bytes = None

        new_backup_count = None
        if "backup_count" in lc:
            try:
                new_backup_count = int(lc["backup_count"])
            except Exception:
                new_backup_count = None

        new_queue_max = None
        if "queue_max" in lc:
            try:
                new_queue_max = int(lc["queue_max"])
            except Exception:
                new_queue_max = None

        new_writer_daemon = None
        if "writer_daemon" in lc:
            try:
                new_writer_daemon = bool(lc["writer_daemon"])
            except Exception:
                new_writer_daemon = None

        new_truncate = None
        if "truncate_on_start" in lc:
            try:
                new_truncate = bool(lc["truncate_on_start"])
            except Exception:
                new_truncate = None

        # 4) decide if we need to recreate queue+writer
        need_recreate = False
        try:
            old_qsize = getattr(self._q, "maxsize", None)
            if new_queue_max is not None and old_qsize != new_queue_max:
                need_recreate = True
            old_max_bytes = getattr(self._writer, "max_bytes", None)
            old_backups = getattr(self._writer, "backups", None)
            old_daemon = getattr(self._writer, "daemon", None)
            if new_max_bytes is not None and old_max_bytes is not None and new_max_bytes != old_max_bytes:
                need_recreate = True
            if new_backup_count is not None and old_backups is not None and new_backup_count != old_backups:
                need_recreate = True
            if new_writer_daemon is not None and old_daemon is not None and new_writer_daemon != old_daemon:
                need_recreate = True
        except Exception:
            need_recreate = True

        # 5) recreate if required (transfer queued items best-effort)
        if need_recreate:
            try:
                target_qsize = new_queue_max if new_queue_max is not None else getattr(self._q, "maxsize", QUEUE_MAX)
                new_q = queue.Queue(maxsize=target_qsize)

                # move items preserving "drop-oldest" preference
                try:
                    while True:
                        item = self._q.get_nowait()
                        try:
                            new_q.put_nowait(item)
                        except queue.Full:
                            # drop oldest from new_q then retry
                            try:
                                _ = new_q.get_nowait()
                                new_q.put_nowait(item)
                            except Exception:
                                pass
                        finally:
                            try:
                                self._q.task_done()
                            except Exception:
                                pass
                except queue.Empty:
                    pass

                # stop old writer
                try:
                    self._writer.close()
                    self._writer.join(timeout=1.0)
                except Exception:
                    pass

                # choose effective params
                effective_max = new_max_bytes if new_max_bytes is not None else getattr(self._writer, "max_bytes", LOG_MAX_BYTES)
                effective_backups = new_backup_count if new_backup_count is not None else getattr(self._writer, "backups", LOG_BACKUP_COUNT)
                effective_daemon = new_writer_daemon if new_writer_daemon is not None else getattr(self._writer, "daemon", WRITER_DAEMON)

                # create new writer and swap
                new_writer = _FileWriter(self.logfile, new_q, effective_max, effective_backups, daemon=effective_daemon)
                new_writer.start()

                # atomic-ish swap
                self._q = new_q
                self._writer = new_writer

                # update module-level constants for future sanity
                LOG_MAX_BYTES = effective_max
                LOG_BACKUP_COUNT = effective_backups
                QUEUE_MAX = target_qsize
                WRITER_DAEMON = effective_daemon

                # optionally truncate
                if new_truncate and os.path.exists(self.logfile):
                    try:
                        open(self.logfile, "w", encoding="utf-8").close()
                    except Exception:
                        pass

            except Exception:
                # if recreation fails, keep old writer silently
                pass
        else:
            # If not recreating, still update rotation globals if provided
            if new_max_bytes is not None:
                LOG_MAX_BYTES = new_max_bytes
            if new_backup_count is not None:
                LOG_BACKUP_COUNT = new_backup_count
            if new_truncate is not None:
                LOG_TRUNCATE_ON_START = new_truncate

                
    def get_current_config(self) -> Dict[str, Any]:
        """Return current active logger settings (for debugging)."""
        return {
            "live": self.live,
            "logfile": self.logfile,
            "queue_max": getattr(self._q, "maxsize", None),
            "log_max_bytes": LOG_MAX_BYTES,
            "backup_count": LOG_BACKUP_COUNT,
            "writer_daemon": WRITER_DAEMON,
            "show_banner": SHOW_BANNER,
        }


    # ---------- internal helpers ----------
    def _print_banner(self):
        goat_art = r"""
               (__)
               (oo) 
        /-------\/ 
       / |     ||
      *  ||----|| 
         ~~    ~~ 
        GOAT COUNTER SYSTEM
        """
        try:
            self.console.print(
                Panel.fit(
                    Text(goat_art, style="banner"),
                    border_style="cyan",
                    title="🐐 INITIATED",
                    subtitle="[watermark]Powered by Goat-Counter AI[/]",
                    padding=(1, 2),
                )
            )
            self.info("LOGGER", str(self.get_current_config()))
        except Exception:
            pass

    def _enqueue_json(self, entry: Dict[str, Any]) -> None:
        # non-blocking enqueue with drop-oldest policy to avoid blocking hot path
        try:
            self._q.put_nowait(entry)
        except queue.Full:
            try:
                # drop oldest
                _ = self._q.get_nowait()
                self._q.task_done()
            except Exception:
                pass
            try:
                self._q.put_nowait(entry)
            except Exception:
                # give up quietly
                pass

    def _write_entry(self, level: str, tag: str, message: str, emoji: str = "") -> Dict[str, Any]:
        ts = datetime.now().isoformat(timespec="seconds")
        entry = {"timestamp": ts, "level": level.upper(), "tag": tag, "emoji": emoji, "message": message}
        return entry

    def _print_rich(self, level: str, tag: str, message: str, emoji: str = ""):
        # timestamp like original v1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = Text()
        text.append(f"[{timestamp}] ", style="time")
        # style name must match theme keys
        style_key = level.lower() if level.lower() in ("info", "success", "warning", "error", "debug") else "info"
        text.append(f"[{level.upper()}] ", style=style_key)
        text.append(f"[{tag}] ", style="tag")
        text.append(f"{emoji} {message}", style="white")
        try:
            self.console.print(text, highlight=False)
        except Exception:
            # last-resort fallback
            try:
                print(f"{timestamp} [{level.upper()}] [{tag}] {emoji} {message}")
            except Exception:
                pass

    # ---------- core logging logic ----------
    def _log(self, level: str, tag: str, message: str, emoji: str = ""):
        # build json entry early
        entry = self._write_entry(level, tag, message, emoji)

        # console printing policy:
        # - live=True : print everything
        # - live=False: print only INFO and ERROR to avoid interfering with progress bars
        if self.live:
            try:
                self._print_rich(level, tag, message, emoji)
            except Exception:
                pass
        else:
            if level.lower() in ("info", "error"):
                try:
                    self._print_rich(level, tag, message, emoji)
                except Exception:
                    pass

        # always enqueue JSON line (even in live mode we keep file logs)
        try:
            self._enqueue_json(entry)
        except Exception:
            pass

    # ---------- public API (same signatures) ----------
    def info(self, tag: str, message: str):
        self._log("info", tag, message, "ℹ️")

    def success(self, tag: str, message: str):
        self._log("success", tag, message, "✅")

    def warn(self, tag: str, message: str):
        self._log("warning", tag, message, "⚠️")

    def error(self, tag: str, message: str):
        self._log("error", tag, message, "❌")

    def debug(self, tag: str, message: str, exc_info: bool = False):
        if exc_info:
            # Append traceback if an exception is active; otherwise add a note.
            if sys.exc_info()[0] is not None:
                message = f"{message}\n{traceback.format_exc()}"
            else:
                message = f"{message} (no active exception)"
        self._log("debug", tag, message, "🔍")

    def blank(self):
        if self.live:
            try:
                self.console.print("")
            except Exception:
                print("")
        else:
            # insert newline marker in file (enqueued)
            self._enqueue_json({"timestamp": datetime.now().isoformat(timespec="seconds"), "level": "", "tag": "", "emoji": "", "message": ""})

    # runtime helpers
    def set_live(self, live: bool):
        self.live = bool(live)
        if self.live and SHOW_BANNER:
            self._print_banner()

    def close(self, timeout: float = 2.0):
        """Stop background writer and flush queue (best-effort). Call at shutdown if desired."""
        try:
            self._writer.close()
            # wait for worker to finish a little
            self._writer.join(timeout)
        except Exception:
            pass


# singleton used throughout the project
log = Logger(live=LIVE_MODE, logfile=LOG_FILE)

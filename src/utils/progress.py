# src/utils/progress.py

import time
from rich.progress import (
    Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn
)
from rich.console import Console
from rich.live import Live

console = Console()

# Compact progress layout
progress = Progress(
    SpinnerColumn(style="cyan"),  # 🔄 active loader spinner
    TextColumn("[bold blue]{task.fields[prefix]}[/]"),
    BarColumn(bar_width=None, style="dim cyan", complete_style="green"),
    TextColumn("{task.completed}/{task.total}", justify="right", style="bold white"),
    TextColumn("src={task.fields[src_idx]}", style="dim cyan"),
    TextColumn("fps:{task.fields[proc_fps]:.1f}/{task.fields[avg_fps]:.1f}", style="yellow"),
    TextColumn("eff:{task.fields[eff_fps]:.1f}", style="magenta"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),   # ⏱️ remaining time
    console=console,
    expand=False,  # prevent over-stretching across terminal
)

task_id = None
live = None
_last_time, _last_idx, _ema = None, None, None


def init_progress(total: int, prefix="[WRITE]"):
    """Initialize progress bar."""
    global task_id, live, _last_time, _last_idx, _ema
    task_id = progress.add_task(
        "", total=total,
        prefix=prefix,
        src_idx=0,
        proc_fps=0.0,
        avg_fps=0.0,
        eff_fps=0.0,
    )
    _last_time, _last_idx, _ema = None, None, None
    live = Live(progress, console=console, refresh_per_second=10)
    live.start()


def progress_update(
    written_idx: int,
    total: int,
    t_start: float,
    src_idx: int | None = None,
    every: int = 30,
    prefix: str = "[WRITE]",
    eff_fps: float | None = None,
    ema_alpha: float = 0.30,
):
    """Update progress bar stats."""
    global _last_time, _last_idx, _ema

    if written_idx % max(1, every) != 0:
        return

    now = time.perf_counter()
    elapsed_total = max(1e-6, now - t_start)

    if _last_time is None or _last_idx is None:
        inst_rate = written_idx / elapsed_total
    else:
        dt = max(1e-6, now - _last_time)
        di = max(0, written_idx - _last_idx)
        inst_rate = di / dt

    if _ema is None:
        _ema = inst_rate
    else:
        _ema = ema_alpha * inst_rate + (1.0 - ema_alpha) * _ema

    _last_time, _last_idx = now, written_idx

    avg_rate = written_idx / elapsed_total
    progress.update(
        task_id,
        completed=written_idx,
        total=total,
        prefix=prefix,
        src_idx=(src_idx if src_idx is not None else "-"),
        proc_fps=_ema,
        avg_fps=avg_rate,
        eff_fps=(eff_fps if eff_fps is not None else 0.0),
    )


def stop_progress():
    """Stop progress bar and run log cleaner."""
    global live
    if live is not None:
        live.stop()
        live = None

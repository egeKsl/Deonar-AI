"""Constants and small config dataclass.

This file contains only data/constants and is safe to import at build time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

NON_MACHINE_DEPS: List[str] = [
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

PYTHON_SUPPORT_MAP: Dict[str, tuple] = {
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


@dataclass
class Config:
    """Basic defaults used by the installer (kept minimal here)."""

    # UI / metrics defaults
    metrics_file: str = "install_metrics.json"
    metrics_dir: str = "./logs"

    # UI tuning
    heartbeat: int = 4
    always_progress: bool = False

    # other defaults can be referenced in code, but keep config small here

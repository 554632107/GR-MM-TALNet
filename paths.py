# -*- coding: utf-8 -*-
"""
Load source-data and frozen text-encoder paths from environment variables.
The main training script reads prepared arrays from features/pack.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def data_path(name: str) -> Path:
    """Read one environment variable and check that the path exists."""
    value = os.environ.get(name, "").strip().strip('"')
    if not value:
        raise RuntimeError(
            f"还没有设置环境变量 {name}。"
        )
    path = Path(value)
    if not path.exists():
        raise RuntimeError(f"环境变量 {name} 指向的路径不存在：{path}")
    return path

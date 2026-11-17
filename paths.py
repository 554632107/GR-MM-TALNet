# -*- coding: utf-8 -*-
"""
Paths to source data. Preparation scripts read these environment variables only when opening files.
CA_XLSX is the in-hospital cardiac-arrest workbook. HR_XLSX is the high-risk control workbook.
CA_ECG and HR_ECG are ECG XML roots, with one subdirectory per patient id.
TEXT_DIR is the admission-note JSON directory. BERT_DIR is a frozen text encoder in Hugging Face format.
Training the main model does not read these files. It reads the arrays in features/pack.
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

# -*- coding: utf-8 -*-
"""
Read ECG XML and convert it to a 12-lead, 500 Hz, 10-second waveform with a 0.5–40 Hz bandpass.
Which recording is chosen, and how the 48-hour window is applied, is defined in extract_window_features.py.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, filtfilt, resample

FS = 500
SECONDS = 10
N_SAMPLES = FS * SECONDS  # 5000
BP_LOW, BP_HIGH, BP_ORDER = 0.5, 40.0, 4
STD_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def to_dt(s) -> datetime | None:
    if s is None or (isinstance(s, float) and np.isnan(s)) or pd.isna(s):
        return None
    if isinstance(s, datetime):
        return s
    s = str(s).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    for fmt in (
        "%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M", "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    ):
        try:
            return datetime.strptime(s.split(".")[0], fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).to_pydatetime()
    except Exception:
        return None


def patient_dir(root: Path, pid: str) -> Path | None:
    """Patient directory. Try the 8-digit id first, then the id without leading zeros."""
    for name in (pid, pid.lstrip("0") or "0"):
        p = root / name
        if p.is_dir():
            return p
    return None


def find_ecg_xmls(folder: Path) -> list[Path]:
    out = []
    if folder is None or not folder.is_dir():
        return out
    for dp, _, fns in os.walk(folder):
        for fn in fns:
            if fn.startswith("ECG") and fn.lower().endswith(".xml"):
                out.append(Path(dp) / fn)
    return out


def parse_ecg_time(xml_path: Path) -> datetime | None:
    try:
        with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4000)
    except Exception:
        head = xml_path.read_bytes()[:4000].decode("utf-8", errors="ignore")
    m = re.search(r'<low value="(\d{14})"', head)
    return to_dt(m.group(1)) if m else None


def parse_ecg_waveform(xml_path: Path) -> tuple[dict, float | None] | None:
    """Return {lead: 1-D array} and the sampling rate in Hz. For a repeated lead, keep the longer trace."""
    try:
        t = xml_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        t = xml_path.read_bytes().decode("utf-8", errors="ignore")
    m_inc = re.search(r'<increment value="([\d.]+)"\s+unit="s"', t)
    sr = round(1.0 / float(m_inc.group(1)), 1) if m_inc else None
    lead_map: dict[str, np.ndarray] = {}
    for s in re.findall(r"<sequence>(.*?)</sequence>", t, re.DOTALL):
        m_lead = re.search(r'code="MDC_ECG_LEAD_([A-Za-z0-9]+)"', s)
        m_dig = re.search(r"<digits[^>]*>(.*?)</digits>", s, re.DOTALL)
        if not (m_lead and m_dig):
            continue
        lead = m_lead.group(1).upper()
        try:
            samples = np.fromstring(m_dig.group(1).strip(), sep=" ", dtype=np.float64)
        except Exception:
            continue
        if lead not in lead_map or len(samples) > len(lead_map[lead]):
            lead_map[lead] = samples
    if not lead_map:
        return None
    return lead_map, sr


def to_500hz_10s(sig: np.ndarray, sr: float | None) -> np.ndarray:
    """Keep about 10 seconds and resample to 500 Hz, 5000 samples."""
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    if sr is None:
        sr = 1000.0 if n >= 8000 else 500.0
    n_keep = min(n, int(round(sr * SECONDS)))
    sig = sig[: max(n_keep, 1)]
    if abs(sr - 500) < 1 and len(sig) == N_SAMPLES:
        return sig.astype(np.float32)
    if abs(sr - 1000) < 1 and len(sig) >= 2 * N_SAMPLES:
        return decimate(sig[: 2 * N_SAMPLES], 2).astype(np.float32)[:N_SAMPLES]
    return resample(sig, N_SAMPLES).astype(np.float32)


def bandpass_one(sig: np.ndarray) -> np.ndarray:
    if np.std(sig) < 1e-8:
        return sig.astype(np.float32)
    nyq = FS / 2.0
    b, a = butter(BP_ORDER, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def waveform_to_array(lead_dict: dict, sr: float | None) -> np.ndarray:
    arr = np.zeros((12, N_SAMPLES), dtype=np.float32)
    for i, lead in enumerate(STD_LEADS):
        if lead not in lead_dict:
            continue
        x = to_500hz_10s(lead_dict[lead], sr)
        if len(x) < N_SAMPLES:
            arr[i, : len(x)] = x
        else:
            arr[i] = x[:N_SAMPLES]
        arr[i] = bandpass_one(arr[i])
    return arr

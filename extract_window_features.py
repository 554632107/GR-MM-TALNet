# -*- coding: utf-8 -*-
"""
Extract laboratory values, text, and ECG inside the 48-hour window. This script does not train a model.
The index time is 00:00 on the admission day. Controls are censored at 48 hours. Cases are censored at the earlier of 48 hours and the arrest time.
Each laboratory item keeps only the first and second measurements before censoring. Text uses that day's chief complaint and present illness, with arrest-related sentences removed.
ECG keeps the latest acceptable 10-second, 12-lead recording before censoring. Patients missing a modality stay in the cohort.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ROOT, data_path

OUT = ROOT / "features"
COHORT = ROOT / "cohort" / "cohort_index.csv"
CA_XLSX = None
HR_XLSX = None
TEXT_DIR = None
CA_ECG = None
HR_ECG = None

sys.path.insert(0, str(ROOT / "model"))
from prepare_ecg import (
    find_ecg_xmls,
    parse_ecg_time,
    parse_ecg_waveform,
    patient_dir,
    waveform_to_array,
)


def load_source_paths() -> None:
    """Ask the environment for their paths before reading."""
    global CA_XLSX, HR_XLSX, TEXT_DIR, CA_ECG, HR_ECG
    CA_XLSX = data_path("CA_XLSX")
    HR_XLSX = data_path("HR_XLSX")
    TEXT_DIR = data_path("TEXT_DIR")
    CA_ECG = data_path("CA_ECG")
    HR_ECG = data_path("HR_ECG")

SHEETS = ["入院时", "入院2天时", "入院3天时", "入院4天时", "入院5天时", "入院6天时", "入院7天时"]
LABS = [
    "白细胞计数", "血红蛋白", "血小板计数", "钾", "钠", "氯", "钙",
    "肌酐", "尿素", "血糖", "白蛋白", "丙氨酸氨基转移酶", "天冬氨酸氨基转移酶",
    "肌酸激酶-MB", "肌钙蛋白-T", "B型利钠肽", "D-二聚体", "全血CRP",
    "红细胞压积", "中性粒细胞绝对值", "淋巴细胞绝对值", "单核细胞绝对值",
    "平均红细胞体积", "红细胞分布宽度", "血小板压积",
    "总蛋白", "球蛋白", "总胆红素", "乳酸脱氢酶", "肌酸激酶",
    "凝血酶原时间", "活化部分凝血活酶时间", "血浆纤维蛋白原", "国际标准化比率",
    "尿酸", "肾小球滤过率", "磷", "镁",
    "总胆固醇", "甘油三酯", "高密度脂蛋白胆固醇", "低密度脂蛋白胆固醇",
    "血液酸碱度", "二氧化碳分压", "氧分压", "血氧饱和度", "血乳酸",
]
ANALYTES = LABS + ["NLR"]
COM12 = [
    "高血压", "冠心病", "心功能不全", "心脏病史", "糖尿病", "脑梗死",
    "脑血管病", "肿瘤", "吸烟史", "饮酒史", "降压药", "降糖药",
]
LEAK_RE = re.compile(
    r"心跳骤停|心脏骤停|心脏停搏|心跳停止|呼吸心跳|心肺复苏|胸外按压|电除颤|电复律|"
    r"室颤|心室颤动|室性颤动|自主循环|无灌注|ROSC|呼吸心跳停止|大动脉搏动消失"
)
MIN_CHARS = 4


def z8(v) -> str:
    text = str(v).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def clock_known(ts: pd.Timestamp) -> bool:
    """A timestamp at midnight is treated as date-only and cannot be shown to precede that day's censoring time."""
    return not (ts.hour == 0 and ts.minute == 0 and ts.second == 0)


def in_window(ts: pd.Timestamp, t0: pd.Timestamp, censor: pd.Timestamp) -> bool:
    if pd.isna(ts):
        return False
    if clock_known(ts):
        return t0 <= ts < censor
    day = ts.normalize()
    return t0.normalize() <= day < censor.normalize()


def split_sents(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；;])", text or "")
    return [p for p in parts if p.strip()]


def clean_note(chief: str, hpi: str) -> tuple[str, int, int]:
    """Return the cleaned text, the number of removed leak sentences, and whether the chief complaint was written as 0."""
    chief = re.sub(r"\s+", " ", str(chief or "")).strip()
    hpi = re.sub(r"\s+", " ", str(hpi or "")).strip()
    if chief.lower() in ("nan", "none"):
        chief = ""
    if hpi.lower() in ("nan", "none"):
        hpi = ""
    cc_zero = int(chief.strip("。").strip() in ("0", "0.0"))
    n_drop = 0
    kept = []
    blob = chief if chief.endswith(("。", "！", "？", "；", ";")) or not chief else chief + "。"
    blob = blob + hpi
    for sent in split_sents(blob):
        if LEAK_RE.search(sent):
            n_drop += 1
            continue
        kept.append(sent)
    body = re.sub(r"\s+", " ", "".join(kept)).strip()
    if cc_zero or len(body) < MIN_CHARS:
        return "", n_drop, cc_zero
    return body, n_drop, cc_zero


def is_dead_lead(arr: np.ndarray) -> tuple[bool, int, int]:
    """Mark a waveform unusable when at least 4 leads are flat or at most 1 lead is usable."""
    x = arr.astype(np.float64)
    lead_std = x.std(axis=1)
    lead_ptp = np.ptp(x, axis=1)
    dead = (lead_std < 1.0) | (lead_ptp < 5.0)
    weak = (~dead) & (lead_std < 20.0)
    good = ~(dead | weak)
    n_dead = int(dead.sum())
    n_good = int(good.sum())
    return (n_dead >= 4) or (n_good <= 1), n_dead, n_good


def read_cohort() -> pd.DataFrame:
    df = pd.read_csv(COHORT, dtype={"病案号": str}, encoding="utf-8-sig")
    df["病案号"] = df["病案号"].map(z8)
    df["t0"] = pd.to_datetime(df["索引住院日"]).dt.normalize()
    df["发生时间"] = pd.to_datetime(df["发生时间"], errors="coerce")
    horizon = df["t0"] + pd.Timedelta(hours=48)
    df["censor"] = horizon
    pos = df["标签"] == 1
    earlier = df.loc[pos, "发生时间"] < horizon.loc[pos]
    df.loc[pos & earlier, "censor"] = df.loc[pos & earlier, "发生时间"]
    return df


def read_lab_rows(path: Path, ids: set[str]) -> pd.DataFrame:
    want = ["病案号", "检查时间", "全血乳酸", *LABS]
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in SHEETS:
        if sheet not in xl.sheet_names:
            continue
        head = pd.read_excel(xl, sheet_name=sheet, nrows=0)
        have = {str(c).strip() for c in head.columns}
        use = [c for c in want if c in have]
        df = pd.read_excel(xl, sheet_name=sheet, usecols=use)
        df.columns = [str(c).strip() for c in df.columns]
        df["病案号"] = df["病案号"].map(z8)
        df = df[df["病案号"].isin(ids)].copy()
        df["来源表"] = sheet
        frames.append(df)
        print(f"  {path.name[:6]} {sheet}: {len(df)}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cell_value(row: pd.Series, col: str) -> float:
    if col == "血乳酸":
        value = pd.to_numeric(row.get("血乳酸"), errors="coerce")
        if pd.isna(value):
            value = pd.to_numeric(row.get("全血乳酸"), errors="coerce")
        return float(value) if pd.notna(value) else np.nan
    if col == "NLR":
        neu = pd.to_numeric(row.get("中性粒细胞绝对值"), errors="coerce")
        lym = pd.to_numeric(row.get("淋巴细胞绝对值"), errors="coerce")
        if pd.notna(neu) and pd.notna(lym) and float(lym) > 0:
            return float(neu) / float(lym)
        return np.nan
    value = pd.to_numeric(row.get(col), errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def rows_for_person(sub: pd.DataFrame, t0: pd.Timestamp, censor: pd.Timestamp) -> pd.DataFrame:
    sub = sub.copy()
    sub["检查时间"] = pd.to_datetime(sub["检查时间"], errors="coerce")
    sub = sub[sub["检查时间"].map(lambda ts: in_window(ts, t0, censor))]
    if sub.empty:
        return sub
    sub = sub.drop_duplicates(subset=["检查时间", * [c for c in LABS if c in sub.columns]], keep="first")
    sub["有钟点"] = sub["检查时间"].map(clock_known)
    known_days = set(sub.loc[sub["有钟点"], "检查时间"].dt.normalize())
    date_only_dup = (~sub["有钟点"]) & sub["检查时间"].dt.normalize().isin(known_days)
    sub = sub.loc[~date_only_dup].sort_values(["检查时间", "来源表"])
    return sub


def build_labs(cohort: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    grouped = {pid: g for pid, g in labs.groupby("病案号")} if len(labs) else {}
    rows = []
    late = 0
    for rec in cohort.itertuples(index=False):
        sub = rows_for_person(grouped.get(rec.病案号, labs.iloc[0:0]), rec.t0, rec.censor)
        out = {
            "病案号": rec.病案号,
            "标签": int(rec.标签),
            "t0": rec.t0,
            "censor": rec.censor,
            "n_panel": int(len(sub)),
            "n_first": 0,
            "n_second": 0,
        }
        for col in ANALYTES:
            times, values = [], []
            for _, row in sub.iterrows():
                value = cell_value(row, col)
                if pd.notna(value):
                    times.append(row["检查时间"])
                    values.append(value)
            first = values[0] if values else np.nan
            second = values[1] if len(values) > 1 else np.nan
            has_second = int(pd.notna(second))
            delta = float(second - first) if has_second and pd.notna(first) else 0.0
            out[f"{col}_1"] = first
            out[f"{col}_1时间"] = times[0] if times else pd.NaT
            out[f"{col}_2"] = second
            out[f"{col}_2时间"] = times[1] if len(times) > 1 else pd.NaT
            out[f"{col}_Δ"] = delta
            out[f"{col}_开关"] = has_second
            out["n_first"] += int(pd.notna(first))
            out["n_second"] += has_second
            if pd.notna(out[f"{col}_1时间"]) and out[f"{col}_1时间"] >= rec.censor:
                late += 1
            if pd.notna(out[f"{col}_2时间"]) and out[f"{col}_2时间"] >= rec.censor:
                late += 1
        rows.append(out)
    print(f"  晚于截止的化验格: {late}", flush=True)
    return pd.DataFrame(rows)


def build_text(cohort: pd.DataFrame) -> pd.DataFrame:
    wanted = set(cohort["病案号"])
    files: dict[str, list[Path]] = {pid: [] for pid in wanted}
    for path in TEXT_DIR.glob("*.json"):
        pid = z8(path.name.split("_")[0])
        if pid in files:
            files[pid].append(path)
    rows = []
    for rec in cohort.itertuples(index=False):
        chosen = None
        chosen_path = ""
        for path in files[rec.病案号]:
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            adm = pd.to_datetime(obj.get("入院时间"), errors="coerce")
            if pd.isna(adm):
                matched = re.search(r"入院记录(\d{8})", path.name)
                adm = pd.to_datetime(matched.group(1), format="%Y%m%d") if matched else pd.NaT
            if pd.isna(adm) or adm.normalize() != rec.t0:
                continue
            signed = pd.to_datetime(obj.get("记录时间"), errors="coerce")
            if chosen is None or (pd.notna(signed) and (pd.isna(chosen[0]) or signed < chosen[0])):
                chosen = (signed, obj)
                chosen_path = path.name
        if chosen is None:
            rows.append({
                "病案号": rec.病案号, "标签": int(rec.标签), "has_text": 0,
                "len_clean": 0, "n_leak_sent": 0, "cc_zero": 0,
                "记录晚于截止": 0, "来源文件": "", "叙述清洗": "", "原因": "索引住院日没有入院记录",
            })
            continue
        signed, obj = chosen
        body, n_leak, cc_zero = clean_note(obj.get("主诉", ""), obj.get("现病史", ""))
        late_note = int(pd.notna(signed) and signed >= rec.censor)
        rows.append({
            "病案号": rec.病案号,
            "标签": int(rec.标签),
            "has_text": int(len(body) >= MIN_CHARS),
            "len_clean": len(body),
            "n_leak_sent": n_leak,
            "cc_zero": cc_zero,
            "记录晚于截止": late_note,
            "来源文件": chosen_path,
            "叙述清洗": body,
            "原因": "可用" if body else ("主诉为0" if cc_zero else "正文不足4字"),
        })
    return pd.DataFrame(rows)


def build_ecg(cohort: pd.DataFrame) -> pd.DataFrame:
    npy_dir = OUT / "ecg_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, rec in enumerate(cohort.itertuples(index=False), start=1):
        root = CA_ECG if int(rec.标签) == 1 else HR_ECG
        xmls = find_ecg_xmls(patient_dir(root, rec.病案号))
        in_win = []
        for xml_path in xmls:
            dt = parse_ecg_time(xml_path)
            if dt is None:
                continue
            stamp = pd.Timestamp(dt)
            if rec.t0 <= stamp < rec.censor:
                in_win.append((xml_path, stamp))
        row = {
            "病案号": rec.病案号, "标签": int(rec.标签), "has_ecg": 0,
            "n_xml": len(xmls), "n_window": len(in_win), "pick_time": "",
            "n_leads": 0, "n_dead": 0, "n_good": 0, "原因": "",
        }
        if not in_win:
            row["原因"] = "截止前没有心电" if xmls else "没有心电文件"
            rows.append(row)
            continue
        xml_path, stamp = max(in_win, key=lambda item: item[1])
        row["pick_time"] = stamp.strftime("%Y-%m-%d %H:%M:%S")
        parsed = parse_ecg_waveform(xml_path)
        if parsed is None:
            row["原因"] = "波形读不出"
            rows.append(row)
            continue
        lead_dict, sr = parsed
        arr = waveform_to_array(lead_dict, sr)
        n_leads = int((np.abs(arr).sum(axis=1) > 0).sum())
        row["n_leads"] = n_leads
        if n_leads < 8:
            row["原因"] = f"导联不足({n_leads})"
            rows.append(row)
            continue
        dead, n_dead, n_good = is_dead_lead(arr)
        row["n_dead"] = n_dead
        row["n_good"] = n_good
        if dead:
            row["原因"] = "死导联，按缺失"
            rows.append(row)
            continue
        np.save(npy_dir / f"{rec.病案号}.npy", arr)
        row["has_ecg"] = 1
        row["原因"] = "可用"
        rows.append(row)
        if i % 100 == 0:
            print(f"  心电 {i}/{len(cohort)}", flush=True)
    return pd.DataFrame(rows)


def count_by_label(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for label, name in ((1, "阳性"), (0, "对照")):
        part = df[df["标签"] == label]
        out[name] = {"人数": int(len(part)), col: int(part[col].sum())}
    return out


def main() -> None:
    load_source_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = read_cohort()
    ids = set(cohort["病案号"])
    print(f"队列 {len(cohort)}", flush=True)
    print("读化验 ...", flush=True)
    pos_ids = set(cohort.loc[cohort["标签"] == 1, "病案号"])
    neg_ids = ids - pos_ids
    labs = pd.concat([
        read_lab_rows(CA_XLSX, pos_ids),
        read_lab_rows(HR_XLSX, neg_ids),
    ], ignore_index=True)
    lab_out = build_labs(cohort, labs)
    lab_out.to_csv(OUT / "labs_first_second.csv", index=False, encoding="utf-8-sig")

    print("读入院记录 ...", flush=True)
    text = build_text(cohort)
    text.to_csv(OUT / "text_admission.csv", index=False, encoding="utf-8-sig")

    print("读心电 ...", flush=True)
    ecg = build_ecg(cohort)
    ecg.to_csv(OUT / "ecg_pick.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n": int(len(cohort)),
        "rule": "t0=索引住院日00:00；截止=min(t0+48h, 骤停时间)；阳性对照同一套",
        "labs": {
            "有至少1项第一次": int((lab_out["n_first"] > 0).sum()),
            "有至少1项第二次": int((lab_out["n_second"] > 0).sum()),
            "第一次项数中位数": float(lab_out["n_first"].median()),
            "第二次项数中位数": float(lab_out["n_second"].median()),
            "阳性": {
                "有第一次": int(((lab_out["标签"] == 1) & (lab_out["n_first"] > 0)).sum()),
                "有第二次": int(((lab_out["标签"] == 1) & (lab_out["n_second"] > 0)).sum()),
            },
            "对照": {
                "有第一次": int(((lab_out["标签"] == 0) & (lab_out["n_first"] > 0)).sum()),
                "有第二次": int(((lab_out["标签"] == 0) & (lab_out["n_second"] > 0)).sum()),
            },
        },
        "text": {
            **count_by_label(text, "has_text"),
            "主诉写成0": int(text["cc_zero"].sum()),
            "删泄漏句": int(text["n_leak_sent"].sum()),
            "记录时间晚于截止": int(text["记录晚于截止"].sum()),
            "原因": {str(k): int(v) for k, v in text["原因"].value_counts().items()},
        },
        "ecg": {
            **count_by_label(ecg, "has_ecg"),
            "原因": {str(k): int(v) for k, v in ecg["原因"].value_counts().items()},
        },
        "retrained": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

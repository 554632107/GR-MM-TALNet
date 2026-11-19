# -*- coding: utf-8 -*-
"""
Pack the cohort into arrays that training can read directly. Row order matches cohort/cohort_index.csv.
Laboratory arrays store the first value, the second value, and a per-item switch, without imputation or standardization. The switch is 0 when there is no second value.
Context includes age, sex, clinical history, medication use, lifestyle factors, and four indicators of missing first laboratory measurements.
Text is a 768-dimensional [CLS] vector, or zeros when text is missing.
Store paths to usable ECG waveforms.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from paths import ROOT, data_path

BASE = ROOT
FEAT = BASE / "features"
PACK = FEAT / "pack"
COHORT = BASE / "cohort" / "cohort_index.csv"
CA_XLSX = None
HR_XLSX = None
BERT = None


def load_table_paths() -> None:
    """Load paths to the two Excel workbooks used for admission covariates."""
    global CA_XLSX, HR_XLSX
    if CA_XLSX is None:
        CA_XLSX = data_path("CA_XLSX")
        HR_XLSX = data_path("HR_XLSX")


def load_bert_path() -> None:
    """Load the path to the user-supplied frozen text encoder."""
    global BERT
    if BERT is None:
        BERT = data_path("BERT_DIR")

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
    "血液酸碱度", "二氧化碳分压", "氧分压", "血氧饱和度", "血乳酸", "NLR",
]
COM12 = [
    "高血压", "冠心病", "心功能不全", "心脏病史", "糖尿病", "脑梗死",
    "脑血管病", "肿瘤", "吸烟史", "饮酒史", "降压药", "降糖药",
]
COM_EXTRA = ["心血管用药", "神经系统疾病"]
MISS_FLAGS = ["B型利钠肽_缺失", "肌钙蛋白-T_缺失", "血乳酸_缺失", "血液酸碱度_缺失"]
CONTEXT = COM12 + ["年龄", "性别"] + COM_EXTRA + MISS_FLAGS
YES = {"是", "有", "1", "1.0", "true", "y"}


def z8(v) -> str:
    text = str(v).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def age_num(value) -> float:
    """Parse numeric ages directly and use the midpoint for age ranges."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in ("", "nan", "none", "nat"):
        return np.nan
    if "-" in text and not text.startswith("-"):
        left, right = text.split("-", 1)
        try:
            return (float(left) + float(right)) / 2
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def yesno(value) -> float:
    """Code yes, present, or 1 as 1, and everything else as 0. Blank and no are both 0."""
    return float(str(value).strip().lower() in YES)


def sex_num(value) -> float:
    return float(str(value).strip() in ("男", "M", "1"))


def read_sheet(path: Path, sheet: str, cols: list[str]) -> pd.DataFrame:
    head = pd.read_excel(path, sheet_name=sheet, nrows=0)
    have = {str(c).strip() for c in head.columns}
    use = [c for c in cols if c in have]
    df = pd.read_excel(path, sheet_name=sheet, usecols=use)
    df.columns = [str(c).strip() for c in df.columns]
    df["病案号"] = df["病案号"].map(z8)
    return df


def load_context(cohort: pd.DataFrame) -> pd.DataFrame:
    """Age and sex come from the index stay's baseline row. Comorbidities come from the selected admission laboratory row."""
    load_table_paths()
    ids = set(cohort["病案号"])
    base_cols = ["病案号", "性别", "年龄", "住院日期", "出生日期"]
    com_cols = ["病案号", "检查时间", *COM12, *COM_EXTRA]
    base = pd.concat([
        read_sheet(CA_XLSX, "基线", base_cols),
        read_sheet(HR_XLSX, "基线", base_cols),
    ], ignore_index=True)
    base = base[base["病案号"].isin(ids)].copy()
    base["住院日"] = pd.to_datetime(base["住院日期"], errors="coerce").dt.normalize()
    adm = pd.concat([
        read_sheet(CA_XLSX, "入院时", com_cols),
        read_sheet(HR_XLSX, "入院时", com_cols),
    ], ignore_index=True)
    adm = adm[adm["病案号"].isin(ids)].copy()
    adm["检查时间"] = pd.to_datetime(adm["检查时间"], errors="coerce").dt.floor("s")

    rows = []
    n_base_miss = n_com_miss = 0
    for rec in cohort.itertuples(index=False):
        stays = base[(base["病案号"] == rec.病案号) & (base["住院日"] == rec.t0)]
        if stays.empty:
            n_base_miss += 1
            age = np.nan
            sex = np.nan
        else:
            one = stays.iloc[0]
            age = age_num(one.get("年龄"))
            if np.isnan(age):
                born = pd.to_datetime(one.get("出生日期"), errors="coerce")
                if pd.notna(born) and pd.notna(rec.t0):
                    age = float(rec.t0.year - born.year)
            sex = sex_num(one.get("性别"))
        stamp = pd.Timestamp(rec.检查时间).floor("s") if pd.notna(rec.检查时间) else pd.NaT
        hit = adm[(adm["病案号"] == rec.病案号) & (adm["检查时间"] == stamp)]
        if hit.empty:
            n_com_miss += 1
            hit_row = None
        else:
            hit_row = hit.iloc[0]
        row = {"病案号": rec.病案号, "年龄": age, "性别": sex}
        for col in COM12 + COM_EXTRA:
            row[col] = 0.0 if hit_row is None else yesno(hit_row.get(col, 0))
        rows.append(row)
    print(f"  基线对不上索引住院日: {n_base_miss}  入院化验时间对不上: {n_com_miss}", flush=True)
    return pd.DataFrame(rows)


def pack_labs(cohort: pd.DataFrame, labs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aligned = cohort[["病案号"]].merge(labs, on="病案号", how="left")
    first = np.column_stack([
        pd.to_numeric(aligned[f"{c}_1"], errors="coerce").to_numpy(float) for c in LABS
    ])
    second = np.column_stack([
        pd.to_numeric(aligned[f"{c}_2"], errors="coerce").to_numpy(float) for c in LABS
    ])
    switch = np.column_stack([
        pd.to_numeric(aligned[f"{c}_开关"], errors="coerce").fillna(0).to_numpy(float) for c in LABS
    ]).astype(np.float32)
    # When the switch is 0, drop the second value so training does not treat it as a repeat measurement.
    second = np.where(switch > 0.5, second, np.nan)
    return first.astype(np.float32), second.astype(np.float32), switch


def pack_ecg(cohort: pd.DataFrame, ecg: pd.DataFrame) -> pd.DataFrame:
    aligned = cohort[["病案号", "标签"]].merge(ecg, on="病案号", how="left", suffixes=("", "_ecg"))
    src = []
    has = []
    for pid, flag in zip(aligned["病案号"], aligned["has_ecg"].fillna(0)):
        path = FEAT / "ecg_npy" / f"{pid}.npy"
        ok = int(flag) == 1 and path.exists()
        has.append(int(ok))
        src.append(str(path) if ok else "")
    return pd.DataFrame({
        "病案号": aligned["病案号"],
        "标签": aligned["标签"],
        "has_ecg": has,
        "src": src,
    })


def encode_text(texts: list[str], has: np.ndarray) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    load_bert_path()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  BERT {device}  有文本 {int(has.sum())}/{len(texts)}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(BERT))
    model = AutoModel.from_pretrained(str(BERT))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    cls = np.zeros((len(texts), 768), dtype=np.float32)
    keep = [i for i, flag in enumerate(has) if flag == 1]
    batch = 8
    with torch.no_grad():
        for start in range(0, len(keep), batch):
            idx = keep[start:start + batch]
            chunk = [texts[i] for i in idx]
            enc = tok(chunk, truncation=True, max_length=512, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
            for row, vec in zip(idx, out):
                cls[row] = vec
            done = min(start + batch, len(keep))
            if done % 80 == 0 or done == len(keep):
                print(f"  文本 {done}/{len(keep)}", flush=True)
    return cls


def main() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(COHORT, dtype={"病案号": str}, encoding="utf-8-sig")
    cohort["病案号"] = cohort["病案号"].map(z8)
    cohort["t0"] = pd.to_datetime(cohort["索引住院日"]).dt.normalize()
    cohort["检查时间"] = pd.to_datetime(cohort["检查时间"], errors="coerce")
    labs = pd.read_csv(FEAT / "labs_first_second.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    labs["病案号"] = labs["病案号"].map(z8)
    text = pd.read_csv(FEAT / "text_admission.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    text["病案号"] = text["病案号"].map(z8)
    ecg = pd.read_csv(FEAT / "ecg_pick.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    ecg["病案号"] = ecg["病案号"].map(z8)

    print("装化验 ...", flush=True)
    first, second, switch = pack_labs(cohort, labs)
    np.save(PACK / "labs_first.npy", first)
    np.save(PACK / "labs_second.npy", second)
    np.save(PACK / "labs_switch.npy", switch)

    print("装年龄、性别、合并症 ...", flush=True)
    ctx_df = load_context(cohort)
    aligned_labs = cohort[["病案号"]].merge(labs, on="病案号", how="left")
    for col, src in (
        ("B型利钠肽_缺失", "B型利钠肽_1"),
        ("肌钙蛋白-T_缺失", "肌钙蛋白-T_1"),
        ("血乳酸_缺失", "血乳酸_1"),
        ("血液酸碱度_缺失", "血液酸碱度_1"),
    ):
        ctx_df[col] = pd.to_numeric(aligned_labs[src], errors="coerce").isna().astype(float)
    context = ctx_df[CONTEXT].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    np.save(PACK / "context.npy", context)
    ctx_df.to_csv(PACK / "context.csv", index=False, encoding="utf-8-sig")

    print("装心电路径 ...", flush=True)
    ecg_out = pack_ecg(cohort, ecg)
    ecg_out.to_csv(PACK / "ecg_pick.csv", index=False, encoding="utf-8-sig")

    print("装文本向量 ...", flush=True)
    text_aligned = cohort[["病案号"]].merge(
        text[["病案号", "叙述清洗", "has_text"]], on="病案号", how="left",
    )
    texts = text_aligned["叙述清洗"].fillna("").astype(str).tolist()
    has = text_aligned["has_text"].fillna(0).astype(int).to_numpy()
    cls = encode_text(texts, has)
    np.save(PACK / "text_cls.npy", cls)
    pd.DataFrame({"病案号": cohort["病案号"], "标签": cohort["标签"], "has_text": has}).to_csv(
        PACK / "text_cls_ids.csv", index=False, encoding="utf-8-sig",
    )

    summary = {
        "n": int(len(cohort)),
        "row_order": "与 cohort/cohort_index.csv 相同",
        "labs_first": list(first.shape),
        "labs_second": list(second.shape),
        "labs_switch": list(switch.shape),
        "n_switch_people": int((switch.sum(axis=1) > 0).sum()),
        "context": CONTEXT,
        "context_shape": list(context.shape),
        "text_cls": list(cls.shape),
        "n_text": int(has.sum()),
        "n_ecg": int(ecg_out["has_ecg"].sum()),
        "bert": str(BERT),
        "note": "原始化验未填补、未标准化。没有第二次的格子是缺失，开关为 0。",
        "retrained": False,
    }
    (PACK / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (PACK / "columns.json").write_text(json.dumps({
        "labs": LABS,
        "context": CONTEXT,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

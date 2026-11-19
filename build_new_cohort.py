# -*- coding: utf-8 -*-
"""
Build the primary cohort using admission tabular-data completeness.
Retain patients graded good or excellent, then perform 1:3 nearest-neighbor
matching without replacement based on the completeness score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from paths import ROOT, data_path

OUT = ROOT / "cohort"

CORE18 = [
    "白细胞计数", "血红蛋白", "血小板计数", "钾", "钠", "氯", "钙", "肌酐", "尿素", "血糖",
    "白蛋白", "丙氨酸氨基转移酶", "天冬氨酸氨基转移酶", "肌酸激酶-MB", "肌钙蛋白-T",
    "B型利钠肽", "D-二聚体", "全血CRP",
]
COM12 = [
    "高血压", "冠心病", "心功能不全", "心脏病史", "糖尿病", "脑梗死",
    "脑血管病", "肿瘤", "吸烟史", "饮酒史", "降压药", "降糖药",
]
EMPTY = {"", "nan", "none", "nat", "-", "/", "无"}


def z8(v) -> str:
    text = str(v).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def is_filled(v) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() not in EMPTY


def n_filled(row: pd.Series, cols: list[str]) -> int:
    return int(sum(is_filled(row[c]) if c in row.index else False for c in cols))


def grade_of(n_lab: int, n_com: int, n_demo: int) -> tuple[float, str]:
    score = (n_lab / 18) * 70 + (n_com / 12) * 20 + (n_demo / 2) * 10
    if n_lab < 3:
        return score, "差"
    if score >= 85:
        return score, "优"
    if score >= 70:
        return score, "良"
    if score >= 50:
        return score, "中"
    return score, "差"


def read_sheet(path: Path, sheet: str, cols: list[str]) -> pd.DataFrame:
    head = pd.read_excel(path, sheet_name=sheet, nrows=0)
    have = {str(c).strip() for c in head.columns}
    use = [c for c in cols if c in have]
    df = pd.read_excel(path, sheet_name=sheet, usecols=use)
    df.columns = [str(c).strip() for c in df.columns]
    df["病案号"] = df["病案号"].map(z8)
    return df


def choose_lab(labs: pd.DataFrame) -> pd.Series | None:
    """When one stay has several rows, keep the earliest laboratory time."""
    if labs.empty:
        return None
    return labs.sort_values("检查时间").iloc[0]


def score_row(base_row: pd.Series, lab_row: pd.Series | None) -> dict:
    n_lab = n_filled(lab_row, CORE18) if lab_row is not None else 0
    n_com = n_filled(lab_row, COM12) if lab_row is not None else 0
    n_demo = int(is_filled(base_row.get("年龄"))) + int(is_filled(base_row.get("性别")))
    score, grade = grade_of(n_lab, n_com, n_demo)
    return {
        "n_lab18": n_lab,
        "n_com12": n_com,
        "n_demo2": n_demo,
        "score": score,
        "grade": grade,
        "入组": int(grade in ("优", "良")),
        "检查时间": None if lab_row is None else lab_row["检查时间"],
    }


def index_cases(base: pd.DataFrame, adm: pd.DataFrame, event: pd.DataFrame) -> pd.DataFrame:
    event = event.copy()
    event["发生时间"] = pd.to_datetime(event["发生时间"], errors="coerce")
    event = event.dropna(subset=["发生时间"]).sort_values("发生时间")
    event = event.drop_duplicates("病案号", keep="first")
    rows = []
    for pid, arrest in event.set_index("病案号")["发生时间"].items():
        stays = base[(base["病案号"] == pid) & base["住院日"].notna() & (base["住院日"] <= arrest.normalize())]
        if stays.empty:
            rows.append({"病案号": pid, "标签": 1, "发生时间": arrest, "索引住院日": pd.NaT, "原因": "骤停前没有住院日期"})
            continue
        index_day = stays["住院日"].max()
        base_row = stays[stays["住院日"] == index_day].iloc[0]
        labs = adm[(adm["病案号"] == pid) & adm["检查时间"].notna() & (adm["检查日"] == index_day) & (adm["检查时间"] < arrest)]
        lab_row = choose_lab(labs)
        rec = {
            "病案号": pid,
            "标签": 1,
            "发生时间": arrest,
            "索引住院日": index_day,
            "年龄": base_row.get("年龄"),
            "性别": base_row.get("性别"),
            "原因": "已评分" if lab_row is not None else "索引住院当天没有早于骤停的化验",
        }
        rec.update(score_row(base_row, lab_row))
        rows.append(rec)
    return pd.DataFrame(rows)


def index_controls(base: pd.DataFrame, adm: pd.DataFrame, blocked: set[str]) -> pd.DataFrame:
    pool = base[~base["病案号"].isin(blocked) & base["住院日"].notna()]
    rows = []
    for pid, stays in pool.groupby("病案号"):
        index_day = stays["住院日"].min()
        base_row = stays[stays["住院日"] == index_day].iloc[0]
        labs = adm[(adm["病案号"] == pid) & adm["检查时间"].notna() & (adm["检查日"] == index_day)]
        lab_row = choose_lab(labs)
        rec = {
            "病案号": pid,
            "标签": 0,
            "发生时间": pd.NaT,
            "索引住院日": index_day,
            "年龄": base_row.get("年龄"),
            "性别": base_row.get("性别"),
            "原因": "已评分" if lab_row is not None else "最早住院当天没有化验",
        }
        rec.update(score_row(base_row, lab_row))
        rows.append(rec)
    return pd.DataFrame(rows)


def match_1to3(pos: pd.DataFrame, cand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = pos.sort_values("病案号").reset_index(drop=True)
    cand = cand.sort_values("病案号").reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=len(cand), metric="euclidean")
    nn.fit(cand["score"].to_numpy(float).reshape(-1, 1))
    dists, idxs = nn.kneighbors(pos["score"].to_numpy(float).reshape(-1, 1))
    used, picked, pdists, pairs = set(), [], [], []
    for i in range(len(pos)):
        taken = 0
        for j, d in zip(idxs[i], dists[i]):
            j = int(j)
            if j in used:
                continue
            used.add(j)
            picked.append(j)
            pdists.append(float(d))
            pairs.append(i)
            taken += 1
            if taken == 3:
                break
    neg = cand.iloc[picked].copy().reset_index(drop=True)
    pos = pos.copy()
    pos["match_pair"] = np.arange(len(pos))
    pos["match_dist"] = 0.0
    neg["match_pair"] = pairs
    neg["match_dist"] = pdists
    return pos, neg


def main() -> None:
    CA_XLSX = data_path("CA_XLSX")
    HR_XLSX = data_path("HR_XLSX")
    OUT.mkdir(parents=True, exist_ok=True)
    base_cols = ["病案号", "性别", "年龄", "住院日期"]
    admit_cols = ["病案号", "检查时间", *CORE18, *COM12]
    ca_base = read_sheet(CA_XLSX, "基线", base_cols)
    hr_base = read_sheet(HR_XLSX, "基线", base_cols)
    ca_adm = read_sheet(CA_XLSX, "入院时", admit_cols)
    hr_adm = read_sheet(HR_XLSX, "入院时", admit_cols)
    event = read_sheet(CA_XLSX, "心脏骤停事件", ["病案号", "发生时间"])

    for df in (ca_base, hr_base):
        df["住院日"] = pd.to_datetime(df["住院日期"], errors="coerce").dt.normalize()
    for df in (ca_adm, hr_adm):
        df["检查时间"] = pd.to_datetime(df["检查时间"], errors="coerce")
        df["检查日"] = df["检查时间"].dt.normalize()

    blocked = set(ca_base["病案号"]) | set(ca_adm["病案号"]) | set(event["病案号"])
    cases = index_cases(ca_base, ca_adm, event)
    controls = index_controls(hr_base, hr_adm, blocked)
    pos = cases[cases["入组"] == 1].copy()
    cand = controls[controls["入组"] == 1].copy()
    matched_pos, matched_neg = match_1to3(pos, cand)
    cohort = pd.concat([matched_pos, matched_neg], ignore_index=True)
    cohort = cohort.sort_values(["标签", "病案号"], ascending=[False, True])

    keep = [
        "病案号", "标签", "索引住院日", "检查时间", "发生时间",
        "n_lab18", "n_com12", "n_demo2", "score", "grade", "match_pair", "match_dist",
    ]
    cohort[keep].to_csv(OUT / "cohort_index.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT / "case_index_stays.csv", index=False, encoding="utf-8-sig")
    controls.to_csv(OUT / "control_index_stays.csv", index=False, encoding="utf-8-sig")

    summary = {
        "index_saved_to": "cohort/cohort_index.csv",
        "index_rule": "阳性=骤停当天或之前最晚住院，化验早于骤停；对照=最早住院的当天化验",
        "cases": {
            "with_arrest_time": int(len(cases)),
            "eligible": int(len(pos)),
            "grade": {str(k): int(v) for k, v in cases["grade"].value_counts().items()},
            "reason": {str(k): int(v) for k, v in cases["原因"].value_counts().items()},
        },
        "controls": {
            "indexed": int(len(controls)),
            "eligible": int(len(cand)),
            "grade": {str(k): int(v) for k, v in controls["grade"].value_counts().items()},
        },
        "matched": {
            "n": int(len(cohort)),
            "pos": int((cohort["标签"] == 1).sum()),
            "neg": int((cohort["标签"] == 0).sum()),
            "grade_pos": {str(k): int(v) for k, v in matched_pos["grade"].value_counts().items()},
            "grade_neg": {str(k): int(v) for k, v in matched_neg["grade"].value_counts().items()},
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

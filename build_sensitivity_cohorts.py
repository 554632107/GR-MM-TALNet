# -*- coding: utf-8 -*-
"""
Rebuild the sensitivity cohorts at each admission-completeness threshold, with a separate 1:3 match at each threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COHORT = HERE / "cohort"
OUT = COHORT / "sensitivity"
THRESHOLDS = {
    "ge70": {"label": "不少于70分", "grades": ("优", "良"), "min_score": 70},
    "ge85": {"label": "不少于85分", "grades": ("优",), "min_score": 85},
    "ge50": {"label": "不少于50分", "grades": ("优", "良", "中"), "min_score": 50},
}


def match_1to3(pos: pd.DataFrame, cand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match 1:3 without replacement by score only. Ties keep the control with the smaller patient id."""
    pos = pos.sort_values("病案号").reset_index(drop=True)
    cand = cand.sort_values(["score", "病案号"]).reset_index(drop=True)
    scores = cand["score"].to_numpy(float)
    ids = cand["病案号"].to_numpy()
    used = np.zeros(len(cand), dtype=bool)
    picked, pdists, pairs = [], [], []
    for i, score in enumerate(pos["score"].to_numpy(float)):
        order = np.argsort(np.abs(scores - score), kind="mergesort")
        taken = 0
        for j in order:
            j = int(j)
            if used[j]:
                continue
            used[j] = True
            picked.append(j)
            pdists.append(float(abs(scores[j] - score)))
            pairs.append(i)
            taken += 1
            if taken == 3:
                break
        if taken < 3:
            raise RuntimeError(f"阳性 {ids[i] if False else pos.iloc[i]['病案号']} 只匹配到 {taken} 名对照")
    neg = cand.iloc[picked].copy().reset_index(drop=True)
    pos = pos.copy()
    pos["match_pair"] = np.arange(len(pos))
    pos["match_dist"] = 0.0
    neg["match_pair"] = pairs
    neg["match_dist"] = pdists
    return pos, neg


def eligible(df: pd.DataFrame, grades: tuple[str, ...]) -> pd.DataFrame:
    """Keep stays with a laboratory result on the index day and a grade inside this threshold."""
    return df[(df["原因"] == "已评分") & df["grade"].isin(grades)].copy()


def summarize(name: str, spec: dict, cohort: pd.DataFrame, pos_pool: int, neg_pool: int) -> dict:
    pos = cohort[cohort["标签"] == 1]
    neg = cohort[cohort["标签"] == 0]
    return {
        "name": name,
        "label": spec["label"],
        "min_score": spec["min_score"],
        "n": int(len(cohort)),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "pos_pool": pos_pool,
        "neg_pool": neg_pool,
        "grade_pos": {str(k): int(v) for k, v in pos["grade"].value_counts().items()},
        "grade_neg": {str(k): int(v) for k, v in neg["grade"].value_counts().items()},
        "score_pos_min": float(pos["score"].min()),
        "score_neg_min": float(neg["score"].min()),
        "match_dist_max": float(neg["match_dist"].max()),
        "match_dist_median": float(neg["match_dist"].median()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(COHORT / "case_index_stays.csv", dtype={"病案号": str})
    controls = pd.read_csv(COHORT / "control_index_stays.csv", dtype={"病案号": str})
    cases["病案号"] = cases["病案号"].str.zfill(8)
    controls["病案号"] = controls["病案号"].str.zfill(8)
    current = pd.read_csv(COHORT / "cohort_index.csv", dtype={"病案号": str})
    current["病案号"] = current["病案号"].str.zfill(8)
    keep = [
        "病案号", "标签", "索引住院日", "检查时间", "发生时间",
        "n_lab18", "n_com12", "n_demo2", "score", "grade", "match_pair", "match_dist",
    ]
    summary = {}
    for name, spec in THRESHOLDS.items():
        pos = eligible(cases, spec["grades"])
        cand = eligible(controls, spec["grades"])
        matched_pos, matched_neg = match_1to3(pos, cand)
        cohort = pd.concat([matched_pos, matched_neg], ignore_index=True)
        cohort = cohort.sort_values(["标签", "病案号"], ascending=[False, True])
        if name == "ge70":
            rebuild = cohort
            same = set(rebuild["病案号"]) == set(current["病案号"])
            cohort = current
            info = summarize(name, spec, cohort, len(pos), len(cand))
            info["rebuild_same_people_as_current_952"] = bool(same)
            info["rebuild_only_controls_swapped"] = int(len(set(rebuild["病案号"]) - set(current["病案号"])))
            info["saved_list"] = "cohort/cohort_index.csv"
            current.to_csv(OUT / "ge70_cohort.csv", index=False, encoding="utf-8-sig")
        else:
            info = summarize(name, spec, cohort, len(pos), len(cand))
            cohort[keep].to_csv(OUT / f"{name}_cohort.csv", index=False, encoding="utf-8-sig")
            info["saved_list"] = f"cohort/sensitivity/{name}_cohort.csv"
        summary[name] = info
    ids = {
        name: set(pd.read_csv(OUT / f"{name}_cohort.csv", dtype={"病案号": str})["病案号"].str.zfill(8))
        for name in THRESHOLDS
    }
    summary["overlap"] = {
        "ge85_inside_ge70": int(len(ids["ge85"] & ids["ge70"])),
        "ge70_inside_ge50": int(len(ids["ge70"] & ids["ge50"])),
        "ge85_inside_ge50": int(len(ids["ge85"] & ids["ge50"])),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

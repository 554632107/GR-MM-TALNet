# -*- coding: utf-8 -*-
"""
Extract window features for patients in the broader completeness cohort who are absent from the main cohort.
The stricter cohort is already inside the broader one. New files go to features/sensitivity_extra/ and do not change the main features.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import extract_window_features as ex

HERE = Path(__file__).resolve().parent
EXTRA = HERE / "features" / "sensitivity_extra"
GE50 = HERE / "cohort" / "sensitivity" / "ge50_cohort.csv"
GE85 = HERE / "cohort" / "sensitivity" / "ge85_cohort.csv"
OLD_LABS = HERE / "features" / "labs_first_second.csv"


def with_censor(path: Path) -> pd.DataFrame:
    """Use the same censoring rule as extract_window_features.read_cohort."""
    df = pd.read_csv(path, dtype={"病案号": str}, encoding="utf-8-sig")
    df["病案号"] = df["病案号"].map(ex.z8)
    df["t0"] = pd.to_datetime(df["索引住院日"]).dt.normalize()
    df["发生时间"] = pd.to_datetime(df["发生时间"], errors="coerce")
    horizon = df["t0"] + pd.Timedelta(hours=48)
    df["censor"] = horizon
    pos = df["标签"] == 1
    earlier = df.loc[pos, "发生时间"] < horizon.loc[pos]
    df.loc[pos & earlier, "censor"] = df.loc[pos & earlier, "发生时间"]
    return df


def main() -> None:
    ex.load_source_paths()
    EXTRA.mkdir(parents=True, exist_ok=True)
    ge50 = with_censor(GE50)
    ge85_ids = set(pd.read_csv(GE85, dtype={"病案号": str})["病案号"].map(ex.z8))
    if not ge85_ids.issubset(set(ge50["病案号"])):
        raise RuntimeError("严格档有人不在放宽档里，不能只抽放宽档")
    old = pd.read_csv(OLD_LABS, dtype={"病案号": str}, encoding="utf-8-sig")
    old_ids = set(old["病案号"].map(ex.z8))
    fresh = ge50[~ge50["病案号"].isin(old_ids)].copy()
    print(f"放宽档 {len(ge50)}  已有主队列特征 {len(ge50) - len(fresh)}  新抽 {len(fresh)}", flush=True)

    ex.OUT = EXTRA
    ids = set(fresh["病案号"])
    pos_ids = set(fresh.loc[fresh["标签"] == 1, "病案号"])
    print("读化验 ...", flush=True)
    labs = pd.concat([
        ex.read_lab_rows(ex.CA_XLSX, pos_ids),
        ex.read_lab_rows(ex.HR_XLSX, ids - pos_ids),
    ], ignore_index=True)
    lab_out = ex.build_labs(fresh, labs)
    lab_out.to_csv(EXTRA / "labs_first_second.csv", index=False, encoding="utf-8-sig")
    empty = int((lab_out["n_first"] == 0).sum())
    print(f"  新抽里一项第一次都没有的人: {empty}", flush=True)

    print("读入院记录 ...", flush=True)
    text = ex.build_text(fresh)
    text.to_csv(EXTRA / "text_admission.csv", index=False, encoding="utf-8-sig")
    print(f"  有文本 {int(text['has_text'].sum())}/{len(text)}", flush=True)

    print("读心电 ...", flush=True)
    ecg = ex.build_ecg(fresh)
    ecg.to_csv(EXTRA / "ecg_pick.csv", index=False, encoding="utf-8-sig")
    print(f"  有心电 {int(ecg['has_ecg'].sum())}/{len(ecg)}", flush=True)

    summary = {
        "n_new": int(len(fresh)),
        "n_pos": int((fresh["标签"] == 1).sum()),
        "n_reused_from_952": int(len(ge50) - len(fresh)),
        "n_first_zero": empty,
        "n_text": int(text["has_text"].sum()),
        "n_ecg": int(ecg["has_ecg"].sum()),
        "note": "只抽放宽档中不在主队列特征表里的人",
    }
    (EXTRA / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

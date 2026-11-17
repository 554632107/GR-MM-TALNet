# -*- coding: utf-8 -*-
"""
Pack the stricter and broader cohorts into the same training arrays as the main cohort.
Patients already in the main cohort reuse features/pack. The others use features/sensitivity_extra. features/pack is not overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import pack_for_train as pk

HERE = Path(__file__).resolve().parent
FEAT = HERE / "features"
OLD_PACK = FEAT / "pack"
EXTRA = FEAT / "sensitivity_extra"
COHORTS = {
    "ge85": HERE / "cohort" / "sensitivity" / "ge85_cohort.csv",
    "ge50": HERE / "cohort" / "sensitivity" / "ge50_cohort.csv",
}


def z8_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["病案号"] = out["病案号"].map(pk.z8)
    return out


def stack_by_id(cohort: pd.DataFrame, old: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per patient. Prefer features already present in the main cohort."""
    both = pd.concat([old, extra], ignore_index=True)
    both = both.drop_duplicates("病案号", keep="first")
    aligned = cohort[["病案号"]].merge(both, on="病案号", how="left", indicator=True)
    missing = int((aligned["_merge"] != "both").sum())
    if missing:
        raise RuntimeError(f"有 {missing} 人没有特征行")
    return aligned.drop(columns="_merge")


def main() -> None:
    old_labs = z8_frame(pd.read_csv(FEAT / "labs_first_second.csv", dtype={"病案号": str}, encoding="utf-8-sig"))
    extra_labs = z8_frame(pd.read_csv(EXTRA / "labs_first_second.csv", dtype={"病案号": str}, encoding="utf-8-sig"))
    old_text = z8_frame(pd.read_csv(FEAT / "text_admission.csv", dtype={"病案号": str}, encoding="utf-8-sig"))
    extra_text = z8_frame(pd.read_csv(EXTRA / "text_admission.csv", dtype={"病案号": str}, encoding="utf-8-sig"))
    old_ids = z8_frame(pd.read_csv(OLD_PACK / "text_cls_ids.csv", dtype={"病案号": str}, encoding="utf-8-sig"))
    old_cls = np.load(OLD_PACK / "text_cls.npy").astype(np.float32)
    if len(old_ids) != len(old_cls):
        raise RuntimeError("主队列的文本向量和病案号行数不一致")
    old_cls_map = {pid: old_cls[i] for i, pid in enumerate(old_ids["病案号"])}
    old_has = dict(zip(old_ids["病案号"], old_ids["has_text"].astype(int)))
    old_ecg = z8_frame(pd.read_csv(OLD_PACK / "ecg_pick.csv", dtype={"病案号": str}, encoding="utf-8-sig"))
    extra_ecg = z8_frame(pd.read_csv(EXTRA / "ecg_pick.csv", dtype={"病案号": str}, encoding="utf-8-sig"))

    for name, path in COHORTS.items():
        cohort = z8_frame(pd.read_csv(path, dtype={"病案号": str}, encoding="utf-8-sig"))
        cohort["t0"] = pd.to_datetime(cohort["索引住院日"]).dt.normalize()
        cohort["检查时间"] = pd.to_datetime(cohort["检查时间"], errors="coerce")
        out = FEAT / f"pack_{name}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"装 {name}  n={len(cohort)}", flush=True)

        labs = stack_by_id(cohort, old_labs, extra_labs)
        first, second, switch = pk.pack_labs(cohort, labs)
        np.save(out / "labs_first.npy", first)
        np.save(out / "labs_second.npy", second)
        np.save(out / "labs_switch.npy", switch)

        print("  年龄、性别、合并症 ...", flush=True)
        ctx_df = pk.load_context(cohort)
        for col, src in (
            ("B型利钠肽_缺失", "B型利钠肽_1"),
            ("肌钙蛋白-T_缺失", "肌钙蛋白-T_1"),
            ("血乳酸_缺失", "血乳酸_1"),
            ("血液酸碱度_缺失", "血液酸碱度_1"),
        ):
            ctx_df[col] = pd.to_numeric(labs[src], errors="coerce").isna().astype(float)
        context = ctx_df[pk.CONTEXT].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        np.save(out / "context.npy", context)
        ctx_df.to_csv(out / "context.csv", index=False, encoding="utf-8-sig")

        text = stack_by_id(cohort, old_text, extra_text)
        has = np.zeros(len(cohort), dtype=np.int32)
        texts = []
        cls = np.zeros((len(cohort), 768), dtype=np.float32)
        need = []
        for i, pid in enumerate(cohort["病案号"]):
            if pid in old_cls_map:
                cls[i] = old_cls_map[pid]
                has[i] = int(old_has[pid])
                texts.append("")
            else:
                body = str(text.loc[text["病案号"] == pid, "叙述清洗"].iloc[0])
                flag = int(text.loc[text["病案号"] == pid, "has_text"].iloc[0])
                has[i] = flag
                texts.append(body if flag == 1 else "")
                if flag == 1:
                    need.append(i)
        if need:
            print(f"  新文本向量 {len(need)}", flush=True)
            encoded = pk.encode_text(texts, has)
            for i in need:
                cls[i] = encoded[i]
        np.save(out / "text_cls.npy", cls)
        pd.DataFrame({
            "病案号": cohort["病案号"], "标签": cohort["标签"].astype(int), "has_text": has,
        }).to_csv(out / "text_cls_ids.csv", index=False, encoding="utf-8-sig")

        ecg_rows = []
        for pid, label in zip(cohort["病案号"], cohort["标签"].astype(int)):
            if pid in set(old_ecg["病案号"]):
                row = old_ecg.loc[old_ecg["病案号"] == pid].iloc[0]
                ecg_rows.append({
                    "病案号": pid, "标签": label,
                    "has_ecg": int(row["has_ecg"]), "src": str(row["src"]),
                })
                continue
            row = extra_ecg.loc[extra_ecg["病案号"] == pid].iloc[0]
            npy = EXTRA / "ecg_npy" / f"{pid}.npy"
            ok = int(row["has_ecg"]) == 1 and npy.exists()
            ecg_rows.append({
                "病案号": pid, "标签": label,
                "has_ecg": int(ok), "src": str(npy) if ok else "",
            })
        ecg_out = pd.DataFrame(ecg_rows)
        ecg_out.to_csv(out / "ecg_pick.csv", index=False, encoding="utf-8-sig")
        summary = {
            "cohort": name,
            "n": int(len(cohort)),
            "n_pos": int((cohort["标签"] == 1).sum()),
            "n_text": int(has.sum()),
            "n_ecg": int(ecg_out["has_ecg"].sum()),
            "n_switch_people": int((switch.sum(axis=1) > 0).sum()),
            "n_reused_text_vector": int(sum(pid in old_cls_map for pid in cohort["病案号"])),
            "note": "主队列中已有的人沿用原文本向量。新的人只用这一档的窗口特征。",
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "columns.json").write_text(
            json.dumps({"labs": pk.LABS, "context": pk.CONTEXT}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

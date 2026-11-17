# -*- coding: utf-8 -*-
"""
Mask text, ECG, or both on a trained GR-MM-TALNet without retraining.
Weights come from the matching fold in results_mm_talnet_gate_small. Masking changes availability only and leaves tabular inputs unchanged.
Late-fusion table probabilities come from LightGBM in train_table4_main.py. Text and ECG come from train_table3_unimodal.py.
Weights are 1.00/0.20/0.30. A masked modality has weight 0, and the remaining weights are renormalized.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
from train_mm_talnet import N_FOLDS, PACK, SEED, load_ecg, pack_fold
from network import MM_TALNetGateSmall
FULL = HERE / "results_mm_talnet_gate_small"
LATE = HERE / "results_table4_main" / "oof.csv"
OUT = HERE / "results_missing_robustness"
W_LATE = (1.0, 0.2, 0.3)


def delong_ci(y: np.ndarray, p_ref: np.ndarray, p_alt: np.ndarray) -> dict:
    """delta = alt − ref."""
    y = y.astype(bool)
    x1, z1 = p_ref[y], p_ref[~y]
    x2, z2 = p_alt[y], p_alt[~y]
    m, n = len(x1), len(z1)

    def v10(x, z):
        return np.array([(z < xi).mean() + 0.5 * (z == xi).mean() for xi in x])

    def v01(x, z):
        return np.array([(x > zj).mean() + 0.5 * (x == zj).mean() for zj in z])

    t1, t2 = v10(x1, z1), v10(x2, z2)
    b1, b2 = v01(x1, z1), v01(x2, z2)
    s = np.cov(np.vstack([t1, t2])) / m + np.cov(np.vstack([b1, b2])) / n
    var = float(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    diff = float(t2.mean() - t1.mean())
    se = float(np.sqrt(var)) if var > 0 else np.nan
    z = diff / se if var > 0 else np.nan
    p = float(2 * norm.sf(abs(z))) if var > 0 else np.nan
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auroc_ref": float(t1.mean()),
        "auroc_alt": float(t2.mean()),
        "delta": diff,
        "se": se,
        "ci95_low": diff - 1.96 * se if var > 0 else np.nan,
        "ci95_high": diff + 1.96 * se if var > 0 else np.nan,
        "p": p,
    }


def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "folds": [float(v) for v in a]}


def five_fold(y: np.ndarray, p: np.ndarray, folds) -> dict:
    aucs, aps = [], []
    for _tr, te in folds:
        aucs.append(float(roc_auc_score(y[te], p[te])))
        aps.append(float(average_precision_score(y[te], p[te])))
    return {
        "five_fold_auroc": mean_std(aucs),
        "five_fold_auprc": mean_std(aps),
        "oof_auroc": float(roc_auc_score(y, p)),
        "oof_auprc": float(average_precision_score(y, p)),
    }


def pair_change(y, p_ref, p_alt, mask: np.ndarray) -> dict:
    out = delong_ci(y[mask], p_ref[mask], p_alt[mask])
    gap = np.abs(p_alt[mask] - p_ref[mask])
    out["max_abs_diff"] = float(gap.max()) if len(gap) else None
    out["mean_abs_diff"] = float(gap.mean()) if len(gap) else None
    out["n_unchanged"] = int((gap <= 1e-6).sum()) if len(gap) else 0
    return out


def fuse_late(p_tab, p_txt, p_ecg, has_txt, has_ecg) -> np.ndarray:
    """Set a missing modality's weight to 0 and renormalize the remaining weights."""
    wt = np.where(has_txt & np.isfinite(p_txt), W_LATE[1], 0.0)
    we = np.where(has_ecg & np.isfinite(p_ecg), W_LATE[2], 0.0)
    pt = np.where(np.isfinite(p_txt), p_txt, 0.0)
    pe = np.where(np.isfinite(p_ecg), p_ecg, 0.0)
    return (W_LATE[0] * p_tab + wt * pt + we * pe) / (W_LATE[0] + wt + we)


def forward_patients(model, packed, txt, ix, has_txt, has_ecg, device):
    z1, z2, m1, m2, delta, ctx_z = packed
    model.eval()
    prob = np.zeros(len(ix), dtype=np.float64)
    tab = np.zeros(len(ix), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(ix), 64):
            sl = ix[start:start + 64]
            fuse, tab_logit, _gt, _ge = model(
                torch.from_numpy(z1[sl]).to(device),
                torch.from_numpy(z2[sl]).to(device),
                torch.from_numpy(m1[sl]).to(device),
                torch.from_numpy(m2[sl]).to(device),
                torch.from_numpy(delta[sl]).to(device),
                torch.from_numpy(ctx_z[sl]).to(device),
                torch.from_numpy(packed_switch[sl]).to(device),
                torch.from_numpy(packed_ecg[sl]).to(device),
                torch.from_numpy(has_ecg[sl]).to(device),
                torch.from_numpy(txt[sl]).to(device),
                torch.from_numpy(has_txt[sl]).to(device),
            )
            prob[start:start + len(sl)] = torch.sigmoid(fuse).cpu().numpy()
            tab[start:start + len(sl)] = torch.sigmoid(tab_logit).cpu().numpy()
    return prob, tab


packed_switch = None
packed_ecg = None


def run_full(device) -> dict:
    global packed_switch, packed_ecg
    ids_df = pd.read_csv(PACK / "text_cls_ids.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    ids_df["病案号"] = ids_df["病案号"].str.zfill(8)
    ids = ids_df["病案号"].tolist()
    y = ids_df["标签"].to_numpy(int)
    has_txt = ids_df["has_text"].to_numpy(np.float32)
    txt = np.load(PACK / "text_cls.npy").astype(np.float32)
    first = np.load(PACK / "labs_first.npy").astype(np.float32)
    later = np.load(PACK / "labs_second.npy").astype(np.float32)
    switch = np.load(PACK / "labs_switch.npy").astype(np.float32)
    ctx = np.load(PACK / "context.npy").astype(np.float32)
    ecg, has_ecg = load_ecg(ids, pd.read_csv(PACK / "ecg_pick.csv", dtype={"病案号": str}, encoding="utf-8-sig"))
    packed_switch, packed_ecg = switch, ecg
    saved = pd.read_csv(FULL / "oof.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    saved["病案号"] = saved["病案号"].str.zfill(8)
    saved = pd.DataFrame({"病案号": ids}).merge(saved, on="病案号", how="left")
    if saved["p_mm"].isna().any():
        raise RuntimeError("完整模型折外概率和名单对不齐")

    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    scenes = {
        "original": (has_txt, has_ecg),
        "mask_text": (np.zeros_like(has_txt), has_ecg),
        "mask_ecg": (has_txt, np.zeros_like(has_ecg)),
        "mask_both": (np.zeros_like(has_txt), np.zeros_like(has_ecg)),
    }
    pred = {name: np.zeros(len(y)) for name in scenes}
    tab = np.zeros(len(y))
    for fold, (tr, te) in enumerate(folds, 1):
        packed = pack_fold(first, later, switch, ctx, tr)
        model = MM_TALNetGateSmall(48, ctx.shape[1]).to(device)
        ckpt = torch.load(FULL / "weights" / f"fold{fold}.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        for name, (ht, he) in scenes.items():
            p, t = forward_patients(model, packed, txt, te, ht, he, device)
            pred[name][te] = p
            if name == "original":
                tab[te] = t
        print(f"  fold {fold} done", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gap = float(np.max(np.abs(pred["original"] - saved["p_mm"].to_numpy(float))))
    if gap > 1e-4:
        raise RuntimeError(f"原始预测和已保存折外概率最大差 {gap}")
    both_gap = float(np.max(np.abs(pred["mask_both"] - tab)))
    if both_gap > 1e-5:
        raise RuntimeError(f"两路都遮蔽后没有回到表格概率，最大差 {both_gap}")
    ht = has_txt > 0.5
    he = has_ecg > 0.5
    if float(np.max(np.abs(pred["mask_text"][~ht] - pred["original"][~ht]))) > 1e-5:
        raise RuntimeError("本来就没有叙述的人，遮蔽叙述后预测变了")
    if float(np.max(np.abs(pred["mask_ecg"][~he] - pred["original"][~he]))) > 1e-5:
        raise RuntimeError("本来就没有心电的人，遮蔽心电后预测变了")

    groups = {
        "all": np.ones(len(y), dtype=bool),
        "text_and_ecg": ht & he,
        "text_only": ht & ~he,
        "ecg_only": ~ht & he,
        "neither": ~ht & ~he,
        "has_text": ht,
        "has_ecg": he,
        "has_either": ht | he,
    }
    affected = {"mask_text": "has_text", "mask_ecg": "has_ecg", "mask_both": "has_either"}
    summary = {
        "note": "用各折已训练权重做推理。遮蔽只把可用性标成没有，不重训。",
        "replay_max_abs_diff": gap,
        "mask_both_equals_table_max_abs": both_gap,
        "scenes": {},
        "paired_vs_original": {},
        "real_availability": {},
    }
    for name, p in pred.items():
        summary["scenes"][name] = five_fold(y, p, folds)
    for name, group in affected.items():
        summary["paired_vs_original"][f"{name}_on_{group}"] = pair_change(
            y, pred["original"], pred[name], groups[group]
        )
        summary["paired_vs_original"][f"{name}_on_all"] = pair_change(
            y, pred["original"], pred[name], groups["all"]
        )
    for name, mask in groups.items():
        if name == "all":
            continue
        block = {"n": int(mask.sum()), "n_pos": int(y[mask].sum())}
        if len(np.unique(y[mask])) < 2:
            block["oof"] = None
        else:
            block["oof"] = {
                scene: {
                    "oof_auroc": float(roc_auc_score(y[mask], pred[scene][mask])),
                    "oof_auprc": float(average_precision_score(y[mask], pred[scene][mask])),
                }
                for scene in pred
            }
        summary["real_availability"][name] = block
    frame = pd.DataFrame({
        "病案号": ids,
        "标签": y,
        "has_text": ht.astype(int),
        "has_ecg": he.astype(int),
        "p_original": pred["original"],
        "p_mask_text": pred["mask_text"],
        "p_mask_ecg": pred["mask_ecg"],
        "p_mask_both": pred["mask_both"],
        "p_table": tab,
    })
    return {"summary": summary, "frame": frame, "y": y, "ids": ids, "folds_order": "text_cls_ids"}


def run_late(ids, y_full) -> dict:
    late = pd.read_csv(LATE, dtype={"病案号": str}, encoding="utf-8-sig")
    late["病案号"] = late["病案号"].str.zfill(8)
    base = pd.DataFrame({"病案号": ids})
    late = base.merge(late, on="病案号", how="left")
    if late["LightGBM"].isna().any():
        raise RuntimeError("LightGBM 折外概率有缺失，或与完整模型名单对不齐")
    y = late["标签"].to_numpy(int)
    if not np.array_equal(y, y_full):
        raise RuntimeError("两份标签不一致")
    has_txt = late["has_text"].to_numpy(int) == 1
    has_ecg = late["has_ecg"].to_numpy(int) == 1
    p_tab = late["LightGBM"].to_numpy(float)
    p_txt = late["冻结CLS+线性头"].to_numpy(float)
    p_ecg = late["浅层1D-CNN"].to_numpy(float)
    scenes = {
        "original": fuse_late(p_tab, p_txt, p_ecg, has_txt, has_ecg),
        "mask_text": fuse_late(p_tab, p_txt, p_ecg, np.zeros(len(y), dtype=bool), has_ecg),
        "mask_ecg": fuse_late(p_tab, p_txt, p_ecg, has_txt, np.zeros(len(y), dtype=bool)),
        "mask_both": fuse_late(p_tab, p_txt, p_ecg, np.zeros(len(y), dtype=bool), np.zeros(len(y), dtype=bool)),
    }
    # When both auxiliary weights are 0, fusion must equal the LightGBM probability.
    gap = float(np.max(np.abs(scenes["mask_both"] - p_tab)))
    if gap > 1e-8:
        raise RuntimeError(f"两路都遮蔽后没有回到 LightGBM 概率，最大差 {gap}")
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    ht, he = has_txt, has_ecg
    groups = {
        "all": np.ones(len(y), dtype=bool),
        "has_text": ht,
        "has_ecg": he,
        "has_either": ht | he,
        "text_and_ecg": ht & he,
        "text_only": ht & ~he,
        "ecg_only": ~ht & he,
        "neither": ~ht & ~he,
    }
    summary = {
        "note": "表格分是 train_table4_main.py 的 LightGBM 折外概率，叙述是冻结 CLS 线性头，心电是浅层 1D-CNN。权重 1.00/0.20/0.30。没有重训网络，只更换表格概率后重算融合。",
        "tabular": "LightGBM",
        "mask_both_equals_lightgbm_max_abs": gap,
        "scenes": {name: five_fold(y, p, folds) for name, p in scenes.items()},
        "paired_vs_original": {},
        "real_availability": {},
    }
    affected = {"mask_text": "has_text", "mask_ecg": "has_ecg", "mask_both": "has_either"}
    for name, group in affected.items():
        summary["paired_vs_original"][f"{name}_on_{group}"] = pair_change(y, scenes["original"], scenes[name], groups[group])
        summary["paired_vs_original"][f"{name}_on_all"] = pair_change(y, scenes["original"], scenes[name], groups["all"])
    for name, mask in groups.items():
        if name in {"all", "has_text", "has_ecg", "has_either"}:
            continue
        block = {"n": int(mask.sum()), "n_pos": int(y[mask].sum())}
        block["oof"] = {
            scene: {
                "oof_auroc": float(roc_auc_score(y[mask], scenes[scene][mask])),
                "oof_auprc": float(average_precision_score(y[mask], scenes[scene][mask])),
            }
            for scene in scenes
        } if len(np.unique(y[mask])) > 1 else None
        summary["real_availability"][name] = block
    return summary, scenes


def refresh_late() -> None:
    """Recompute late fusion only. Saved out-of-fold probabilities of the full model are left unchanged."""
    frame = pd.read_csv(OUT / "oof.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    frame["病案号"] = frame["病案号"].str.zfill(8)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    late_summary, late_scenes = run_late(frame["病案号"].tolist(), frame["标签"].to_numpy(int))
    for name, pred in late_scenes.items():
        frame[f"late_{name}"] = pred
    y = frame["标签"].to_numpy(int)
    ht = frame["has_text"].to_numpy(bool)
    he = frame["has_ecg"].to_numpy(bool)
    groups = {"all": np.ones(len(y), dtype=bool), "has_either": ht | he}
    cross = {}
    for scene in ("original", "mask_text", "mask_ecg", "mask_both"):
        for gname, mask in groups.items():
            cross[f"{scene}_on_{gname}"] = delong_ci(
                y[mask], frame[f"late_{scene}"].to_numpy(float)[mask], frame[f"p_{scene}"].to_numpy(float)[mask]
            )
    summary["late_fusion"] = late_summary
    summary["delong_full_minus_late"] = cross
    frame.to_csv(OUT / "oof.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"write {OUT}", flush=True)
    for name, block in late_summary["scenes"].items():
        auc, ap = block["five_fold_auroc"], block["five_fold_auprc"]
        print(
            f"{name}  5折 AUROC {auc['mean']:.3f} ± {auc['std']:.3f}"
            f"  AUPRC {ap['mean']:.3f} ± {ap['std']:.3f}  OOF {block['oof_auroc']:.3f}",
            flush=True,
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    full = run_full(device)
    late_summary, late_scenes = run_late(full["ids"], full["y"])
    frame = full["frame"]
    for name, p in late_scenes.items():
        frame[f"late_{name}"] = p
    cross = {}
    y = full["y"]
    ht = frame["has_text"].to_numpy(bool)
    he = frame["has_ecg"].to_numpy(bool)
    groups = {"all": np.ones(len(y), dtype=bool), "has_either": ht | he}
    for scene in ("original", "mask_text", "mask_ecg", "mask_both"):
        for gname, mask in groups.items():
            cross[f"{scene}_on_{gname}"] = delong_ci(
                y[mask], frame[f"late_{scene}"].to_numpy(float)[mask], frame[f"p_{scene}"].to_numpy(float)[mask]
            )
    summary = {
        "full_model": full["summary"],
        "late_fusion": late_summary,
        "delong_full_minus_late": cross,
    }
    frame.to_csv(OUT / "oof.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"write {OUT}", flush=True)


if __name__ == "__main__":
    import sys
    if "--late-only" in sys.argv:
        refresh_late()
    else:
        main()

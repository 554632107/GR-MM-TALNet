# -*- coding: utf-8 -*-
"""
Train shared-residual MM-TALNet and provide utilities used by GR-MM-TALNet.
Use stratified five-fold cross-validation with early stopping.
Fit preprocessing on the outer training fold, including its validation subset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
from network import MM_TALNet
from metrics import compute_metrics, find_f1_threshold, format_metrics
PACK = HERE / "features" / "pack"
OUT = HERE / "results_mm_talnet"
SEED = 42
N_FOLDS = 5
AUX_W = 0.35
BATCH = 16
LR = 5e-4
EPOCHS = 50
PATIENCE = 10
KEYS = [
    "auroc", "auprc", "accuracy", "balanced_acc",
    "sensitivity", "specificity", "precision", "f1", "threshold",
]


def fill_zscore(arr: np.ndarray, tr: np.ndarray) -> np.ndarray:
    med = np.nanmedian(arr[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    filled = np.where(np.isnan(arr), med, arr)
    mean = filled[tr].mean(axis=0)
    std = filled[tr].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((filled - mean) / std).astype(np.float32)


def pack_fold(first, second, switch, ctx, tr):
    """Without a second measurement, the single-time path uses the standardized first value. The delta is kept only when both raw values exist."""
    m_first = np.isnan(first).astype(np.float32)
    m_second = np.isnan(second).astype(np.float32)
    z_first = fill_zscore(first, tr)
    z_second = fill_zscore(second, tr)
    use_pair = switch > 0.5
    z_obs = np.where(use_pair, z_second, z_first).astype(np.float32)
    m_obs = np.where(use_pair, m_second, m_first).astype(np.float32)
    delta = np.where((m_first > 0.5) | (m_second > 0.5), 0.0, z_second - z_first).astype(np.float32)
    return z_first, z_obs, m_first, m_obs, delta, fill_zscore(ctx, tr)


def zscore_lead(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32).copy()
    for k in range(out.shape[0]):
        std = float(out[k].std())
        if std > 1e-6:
            out[k] = (out[k] - out[k].mean()) / (std + 1e-6)
    return out


def load_ecg(ids: list[str], pick: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    m = pd.DataFrame({"病案号": ids}).merge(pick, on="病案号", how="left")
    arr = np.zeros((len(ids), 12, 5000), dtype=np.float32)
    has = np.zeros(len(ids), dtype=np.float32)
    for i, (flag, src) in enumerate(zip(m["has_ecg"].fillna(0), m["src"].fillna(""))):
        path = Path(str(src))
        if int(flag) != 1 or not path.exists():
            continue
        x = np.load(path, allow_pickle=False)
        if x.shape != (12, 5000):
            continue
        arr[i] = zscore_lead(x)
        has[i] = 1.0
    return arr, has


def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "folds": [float(v) for v in a]}


def subset(y, p, mask, name: str) -> dict | None:
    if int(mask.sum()) < 8 or len(np.unique(y[mask])) < 2:
        print(f"    {name} 人数不足，跳过", flush=True)
        return None
    youden = compute_metrics(y[mask], p[mask])
    f1 = compute_metrics(y[mask], p[mask], threshold=find_f1_threshold(y[mask], p[mask]))
    print(f"    {name} n={int(mask.sum())}  OOF {format_metrics(youden)}", flush=True)
    return {
        "n": int(mask.sum()),
        "n_pos": int(y[mask].sum()),
        "youden": {k: float(youden[k]) for k in KEYS},
        "f1max": {k: float(f1[k]) for k in KEYS},
        "brier": float(brier_score_loss(y[mask], p[mask])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-switch", action="store_true")
    parser.add_argument("--text-npy", default="text_cls.npy")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    if args.out_dir:
        out = HERE / args.out_dir
    else:
        out = HERE / ("results_mm_talnet_noswitch" if args.no_switch else "results_mm_talnet")
    out.mkdir(parents=True, exist_ok=True)
    ids_df = pd.read_csv(PACK / "text_cls_ids.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    ids_df["病案号"] = ids_df["病案号"].str.zfill(8)
    ids = ids_df["病案号"].tolist()
    y = ids_df["标签"].to_numpy(int)
    has_txt = ids_df["has_text"].to_numpy(np.float32)
    txt = np.load(PACK / args.text_npy).astype(np.float32)
    if args.text_npy == "text_cls_nolab.npy":
        nolab = pd.read_csv(HERE / "features" / "text_nolab.csv", dtype={"病案号": str}, encoding="utf-8-sig")
        nolab["病案号"] = nolab["病案号"].str.zfill(8)
        aligned = ids_df[["病案号"]].merge(nolab[["病案号", "has_text_nolab"]], on="病案号", how="left")
        has_txt = aligned["has_text_nolab"].fillna(0).to_numpy(np.float32)
    first = np.load(PACK / "labs_first.npy").astype(np.float32)
    later = np.load(PACK / "labs_second.npy").astype(np.float32)
    switch = np.load(PACK / "labs_switch.npy").astype(np.float32)
    if args.no_switch:
        switch = np.zeros_like(switch)
    ctx = np.load(PACK / "context.npy").astype(np.float32)
    ecg, has_ecg = load_ecg(ids, pd.read_csv(PACK / "ecg_pick.csv", dtype={"病案号": str}, encoding="utf-8-sig"))
    if txt.shape[0] != len(y) or first.shape != (len(y), 48) or switch.shape[1] != 48:
        raise RuntimeError("打包数组和名单行数不一致")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device {device}  n={len(y)} pos={int(y.sum())}  "
        f"text={int(has_txt.sum())} ecg={int(has_ecg.sum())}  "
        f"有第二次化验的人={int((switch.sum(1) > 0).sum())}",
        flush=True,
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    oof_tab = np.zeros(len(y))
    folds_youden, folds_f1 = [], []
    n_param = n_ecg_head = 0

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        tr_i, va_i = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=y[tr])
        packed = pack_fold(first, later, switch, ctx, tr)

        def make_ds(ix):
            z1, z2, m1, m2, delta, ctx_z = packed
            return TensorDataset(
                torch.from_numpy(z1[ix]), torch.from_numpy(z2[ix]),
                torch.from_numpy(m1[ix]), torch.from_numpy(m2[ix]),
                torch.from_numpy(delta[ix]), torch.from_numpy(ctx_z[ix]),
                torch.from_numpy(switch[ix]),
                torch.from_numpy(ecg[ix]), torch.from_numpy(has_ecg[ix]),
                torch.from_numpy(txt[ix]), torch.from_numpy(has_txt[ix]),
                torch.from_numpy(y[ix].astype(np.float32)),
            )

        n_pos = max(int(y[tr].sum()), 1)
        pos_w = torch.tensor([(len(tr) - n_pos) / n_pos], device=device)
        model = MM_TALNet(48, ctx.shape[1]).to(device)
        n_param = sum(p.numel() for p in model.parameters())
        n_ecg_head = sum(p.numel() for p in model.ecg.head.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best_val, best_state, wait = -1.0, None, 0
        tr_dl = DataLoader(make_ds(tr_i), batch_size=BATCH, shuffle=True)
        va_dl = DataLoader(make_ds(va_i), batch_size=BATCH)
        te_dl = DataLoader(make_ds(te), batch_size=BATCH)
        for _epoch in range(EPOCHS):
            model.train()
            for batch in tr_dl:
                *xs, yb = [t.to(device) for t in batch]
                opt.zero_grad()
                fuse_logit, tab_logit = model(*xs)
                loss = loss_fn(fuse_logit, yb) + AUX_W * loss_fn(tab_logit, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            vp = []
            with torch.no_grad():
                for batch in va_dl:
                    *xs, _yb = [t.to(device) for t in batch]
                    logit, _tab = model(*xs)
                    vp.append(torch.sigmoid(logit).cpu().numpy())
            vp = np.concatenate(vp)
            val = float(roc_auc_score(y[va_i], vp)) if len(np.unique(y[va_i])) > 1 else 0.5
            if val > best_val:
                best_val, wait = val, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
            wdir = out / "weights"
            wdir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "fold": fold, "best_val": best_val}, wdir / f"fold{fold}.pt")
        model.eval()
        pred, pred_tab = [], []
        with torch.no_grad():
            for batch in te_dl:
                *xs, _yb = [t.to(device) for t in batch]
                fuse_logit, tab_logit = model(*xs)
                pred.append(torch.sigmoid(fuse_logit).cpu().numpy())
                pred_tab.append(torch.sigmoid(tab_logit).cpu().numpy())
        oof[te] = np.concatenate(pred)
        oof_tab[te] = np.concatenate(pred_tab)
        m_y = compute_metrics(y[te], oof[te])
        m_f = compute_metrics(y[te], oof[te], threshold=find_f1_threshold(y[te], oof[te]))
        folds_youden.append(m_y)
        folds_f1.append(m_f)
        print(f"  fold {fold}: {format_metrics(m_y)}  val {best_val:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pooled = compute_metrics(y, oof)
    pooled_f = compute_metrics(y, oof, threshold=find_f1_threshold(y, oof))
    tab = compute_metrics(y, oof_tab)
    has_txt_b = has_txt > 0.5
    has_ecg_b = has_ecg > 0.5
    subsets = {
        "all": subset(y, oof, np.ones(len(y), dtype=bool), "全体"),
        "has_text": subset(y, oof, has_txt_b, "有文本"),
        "has_ecg": subset(y, oof, has_ecg_b, "有心电"),
        "both": subset(y, oof, has_txt_b & has_ecg_b, "文本和心电都有"),
        "neither": subset(y, oof, (~has_txt_b) & (~has_ecg_b), "文本和心电都没有"),
        "tab_branch": subset(y, oof_tab, np.ones(len(y), dtype=bool), "表格分支"),
    }
    pd.DataFrame({
        "病案号": ids, "标签": y,
        "has_text": has_txt.astype(int), "has_ecg": has_ecg.astype(int),
        "p_mm": oof, "p_tab_branch": oof_tab,
    }).to_csv(out / "oof.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": "MM-TALNet，化验开关按项目" if not args.no_switch else "同一结构，化验开关全部关闭",
        "switch": "逐项" if not args.no_switch else "全部关闭",
        "cohort": "cohort/cohort_index.csv",
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_text": int(has_txt_b.sum()),
        "n_ecg": int(has_ecg_b.sum()),
        "n_both": int((has_txt_b & has_ecg_b).sum()),
        "n_neither": int((~has_txt_b & ~has_ecg_b).sum()),
        "n_people_with_any_second_lab": int((switch.sum(1) > 0).sum()),
        "n_param_including_unused_ecg_head": int(n_param),
        "n_param_excluding_unused_ecg_head": int(n_param - n_ecg_head),
        "aux_w": AUX_W,
        "seed": SEED,
        "folds": N_FOLDS,
        "five_fold_youden": {k: mean_std([r[k] for r in folds_youden]) for k in KEYS},
        "five_fold_f1max": {k: mean_std([r[k] for r in folds_f1]) for k in KEYS},
        "oof_youden": {k: float(pooled[k]) for k in KEYS},
        "oof_f1max": {k: float(pooled_f[k]) for k in KEYS},
        "oof_brier": float(brier_score_loss(y, oof)),
        "tab_branch_oof_auroc": float(tab["auroc"]),
        "subsets": subsets,
        "text_npy": args.text_npy,
        "note": "时间窗口和匹配规则见 README。预测概率在这一匹配队列内解释。"
        + (" 文本是删掉化验分句后重新编码的冻结 CLS。" if args.text_npy == "text_cls_nolab.npy" else ""),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    auc = summary["five_fold_youden"]["auroc"]
    print(
        f"5-fold AUROC {auc['mean']:.4f} ± {auc['std']:.4f}  "
        f"OOF {pooled['auroc']:.4f}  表格分支 OOF {tab['auroc']:.4f}",
        flush=True,
    )
    print(f"参数量 {n_param}（含未使用的心电分类头），去掉该头 {n_param - n_ecg_head}", flush=True)


if __name__ == "__main__":
    main()

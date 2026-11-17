# -*- coding: utf-8 -*-
"""
Train GR-MM-TALNet.
Final logit = tabular logit + has_text × text gate × text residual + has_ecg × ECG gate × ECG residual.
Folds, seed, loss, and text vectors match train_mm_talnet.py. Results are written to results_mm_talnet_gate_small/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
from train_mm_talnet import (
    AUX_W, BATCH, EPOCHS, LR, N_FOLDS, PACK, PATIENCE, SEED, load_ecg, pack_fold,
)
from network import MM_TALNetGateSmall

sys.path.insert(0, str(HERE / "model"))
from metrics import compute_metrics, find_f1_threshold
OUT = HERE / "results_mm_talnet_gate_small"
MM_OOF = HERE / "results_mm_talnet" / "oof.csv"


def delong(y: np.ndarray, p_ref: np.ndarray, p_alt: np.ndarray) -> dict:
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
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = float(2 * norm.sf(abs(z))) if var > 0 else np.nan
    return {"auroc_ref": float(t1.mean()), "auroc_alt": float(t2.mean()), "delta": diff, "p": p}


def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "folds": [float(v) for v in a]}


def gate_stats(mask: np.ndarray, g_txt: np.ndarray, g_ecg: np.ndarray) -> dict:
    if int(mask.sum()) == 0:
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "mean_gate_text": float(g_txt[mask].mean()),
        "mean_gate_ecg": float(g_ecg[mask].mean()),
        "min_gate_text": float(g_txt[mask].min()),
        "min_gate_ecg": float(g_ecg[mask].min()),
        "frac_text_below_0.5": float((g_txt[mask] < 0.5).mean()),
        "frac_ecg_below_0.5": float((g_ecg[mask] < 0.5).mean()),
    }


def check_neither_equals_table(device) -> None:
    """When both auxiliary modalities are absent, the output must equal the tabular logit."""
    torch.manual_seed(0)
    model = MM_TALNetGateSmall(8, 4, d_model=16, n_heads=2, n_layers=1).to(device)
    model.eval()
    b, n = 3, 8
    xs = [torch.randn(b, n, device=device) for _ in range(4)]
    delta = torch.randn(b, n, device=device)
    ctx = torch.randn(b, 4, device=device)
    switched = torch.zeros(b, n, device=device)
    ecg = torch.randn(b, 12, 64, device=device)
    txt = torch.randn(b, 768, device=device)
    has0 = torch.zeros(b, device=device)
    with torch.no_grad():
        fuse, tab, gt, ge = model(*xs, delta, ctx, switched, ecg, has0, txt, has0)
    gap = float((fuse - tab).abs().max())
    if gap > 1e-5 or float(gt.abs().max()) > 1e-6 or float(ge.abs().max()) > 1e-6:
        raise RuntimeError(f"两边都缺时输出偏离表格 logit {gap}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_neither_equals_table(device)
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

    print(f"device {device}  n={len(y)} pos={int(y.sum())}  text={int(has_txt.sum())} ecg={int(has_ecg.sum())}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    oof_tab = np.zeros(len(y))
    oof_gt = np.full(len(y), np.nan)
    oof_ge = np.full(len(y), np.nan)
    aucs, aps, f1s, vals = [], [], [], []
    n_param = 0

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        tr_i, va_i = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=y[tr])
        packed = pack_fold(first, later, switch, ctx, tr)

        def make_ds(ix, packed=packed):
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
        model = MM_TALNetGateSmall(48, ctx.shape[1]).to(device)
        n_param = sum(p.numel() for p in model.parameters())
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
                fuse, tab, _gt, _ge = model(*xs)
                loss = loss_fn(fuse, yb) + AUX_W * loss_fn(tab, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            vp = []
            with torch.no_grad():
                for batch in va_dl:
                    *xs, _yb = [t.to(device) for t in batch]
                    fuse, _tab, _gt, _ge = model(*xs)
                    vp.append(torch.sigmoid(fuse).cpu().numpy())
            val = float(roc_auc_score(y[va_i], np.concatenate(vp)))
            if val > best_val:
                best_val, wait = val, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        wdir = OUT / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "fold": fold, "best_val": best_val}, wdir / f"fold{fold}.pt")
        model.eval()
        pred, pred_tab, pred_gt, pred_ge = [], [], [], []
        with torch.no_grad():
            for batch in te_dl:
                *xs, _yb = [t.to(device) for t in batch]
                fuse, tab, gt, ge = model(*xs)
                pred.append(torch.sigmoid(fuse).cpu().numpy())
                pred_tab.append(torch.sigmoid(tab).cpu().numpy())
                pred_gt.append(gt.cpu().numpy())
                pred_ge.append(ge.cpu().numpy())
        oof[te] = np.concatenate(pred)
        oof_tab[te] = np.concatenate(pred_tab)
        oof_gt[te] = np.concatenate(pred_gt)
        oof_ge[te] = np.concatenate(pred_ge)
        aucs.append(float(roc_auc_score(y[te], oof[te])))
        aps.append(float(average_precision_score(y[te], oof[te])))
        f1s.append(float(compute_metrics(y[te], oof[te], threshold=find_f1_threshold(y[te], oof[te]))["f1"]))
        vals.append(best_val)
        print(f"  fold {fold}: AUROC={aucs[-1]:.4f} AUPRC={aps[-1]:.4f} F1={f1s[-1]:.4f} val={best_val:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ht = has_txt > 0.5
    he = has_ecg > 0.5
    neither = (~ht) & (~he)
    same = float(np.max(np.abs(oof[neither] - oof_tab[neither]))) if neither.any() else 0.0
    mm = pd.read_csv(MM_OOF, dtype={"病案号": str}, encoding="utf-8-sig")
    mm["病案号"] = mm["病案号"].str.zfill(8)
    aligned = pd.DataFrame({"病案号": ids}).merge(mm[["病案号", "p_mm"]], on="病案号", how="left")
    compared = delong(y, aligned["p_mm"].to_numpy(float), oof)
    summary = {
        "model": "小型门控残差。三个分支不变，共享残差拆成两路。",
        "text_npy": "text_cls.npy",
        "formula": "表格logit + 有文本×文本门×文本残差 + 有心电×心电门×心电残差",
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_param": int(n_param),
        "five_fold_auroc": mean_std(aucs),
        "five_fold_auprc": mean_std(aps),
        "five_fold_f1": mean_std(f1s),
        "val_auroc": [float(v) for v in vals],
        "oof_auroc": float(roc_auc_score(y, oof)),
        "oof_auprc": float(average_precision_score(y, oof)),
        "oof_brier": float(brier_score_loss(y, oof)),
        "tab_branch_oof_auroc": float(roc_auc_score(y, oof_tab)),
        "neither_max_abs_diff_vs_table": same,
        "gates": {
            "text_no_ecg": gate_stats(ht & ~he, oof_gt, oof_ge),
            "has_ecg": gate_stats(he, oof_gt, oof_ge),
            "neither": gate_stats(neither, oof_gt, oof_ge),
        },
        "delong_small_minus_proposed": compared,
    }
    pd.DataFrame({
        "病案号": ids, "标签": y,
        "has_text": has_txt.astype(int), "has_ecg": has_ecg.astype(int),
        "p_mm": oof, "p_tab_branch": oof_tab,
        "g_txt": oof_gt, "g_ecg": oof_ge,
    }).to_csv(OUT / "oof.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    auc = summary["five_fold_auroc"]
    print(
        f"5折 AUROC {auc['mean']:.4f} ± {auc['std']:.4f}  OOF {summary['oof_auroc']:.4f}"
        f"  相对提出方法 {compared['delta']:+.4f}  p={compared['p']:.4f}"
        f"  两者都没有的最大差 {same:.6f}",
        flush=True,
    )
    print(f"write {OUT}", flush=True)


if __name__ == "__main__":
    main()

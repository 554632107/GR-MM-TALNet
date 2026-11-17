# -*- coding: utf-8 -*-
"""
Train GR-MM-TALNet once on each of the stricter and broader admission-completeness cohorts.
Seed, folds, loss, and network match train_mm_talnet_gate_small.py.
The cohorts contain different patients, so folds are redrawn within each cohort. Results are saved separately and are not compared across cohorts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
from train_mm_talnet import (
    AUX_W, BATCH, EPOCHS, LR, N_FOLDS, PATIENCE, SEED, load_ecg, pack_fold,
)
from train_mm_talnet_gate_small import (
    check_neither_equals_table, gate_stats, mean_std,
)
from network import MM_TALNetGateSmall

sys.path.insert(0, str(HERE / "model"))
from metrics import compute_metrics, find_f1_threshold
RUNS = (
    ("ge85", "不少于85分", HERE / "features" / "pack_ge85", HERE / "results_mm_talnet_gate_small_ge85"),
    ("ge50", "不少于50分", HERE / "features" / "pack_ge50", HERE / "results_mm_talnet_gate_small_ge50"),
)


def train_one(name: str, label: str, pack: Path, out: Path, device: torch.device) -> None:
    """Stratify, early-stop, and write results for this cohort alone."""
    out.mkdir(parents=True, exist_ok=True)
    ids_df = pd.read_csv(pack / "text_cls_ids.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    ids_df["病案号"] = ids_df["病案号"].str.zfill(8)
    ids = ids_df["病案号"].tolist()
    y = ids_df["标签"].to_numpy(int)
    has_txt = ids_df["has_text"].to_numpy(np.float32)
    txt = np.load(pack / "text_cls.npy").astype(np.float32)
    first = np.load(pack / "labs_first.npy").astype(np.float32)
    later = np.load(pack / "labs_second.npy").astype(np.float32)
    switch = np.load(pack / "labs_switch.npy").astype(np.float32)
    ctx = np.load(pack / "context.npy").astype(np.float32)
    ecg, has_ecg = load_ecg(ids, pd.read_csv(pack / "ecg_pick.csv", dtype={"病案号": str}, encoding="utf-8-sig"))
    if txt.shape[0] != len(y) or first.shape[0] != len(y) or ctx.shape[1] != 20:
        raise RuntimeError(f"{name} 的特征行数对不上")

    print(f"{label} device {device}  n={len(y)} pos={int(y.sum())}  text={int(has_txt.sum())} ecg={int(has_ecg.sum())}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    oof_tab = np.zeros(len(y))
    oof_gt = np.full(len(y), np.nan)
    oof_ge = np.full(len(y), np.nan)
    aucs, aps, f1s, vals, test_n = [], [], [], [], []
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
        wdir = out / "weights"
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
        test_n.append(int(len(te)))
        print(f"  {name} fold {fold}: n={len(te)} AUROC={aucs[-1]:.4f} AUPRC={aps[-1]:.4f} F1={f1s[-1]:.4f} val={best_val:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ht = has_txt > 0.5
    he = has_ecg > 0.5
    neither = (~ht) & (~he)
    same = float(np.max(np.abs(oof[neither] - oof_tab[neither]))) if neither.any() else 0.0
    summary = {
        "model": "小型门控残差。敏感性队列单独分层训练。",
        "cohort": name,
        "threshold": label,
        "formula": "表格logit + 有文本×文本门×文本残差 + 有心电×心电门×心电残差",
        "folds": "种子42，分层5折，折内15%早停。折在本档内重划。",
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_param": int(n_param),
        "test_n": test_n,
        "n_text": int(has_txt.sum()),
        "n_ecg": int(has_ecg.sum()),
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
        "compare_across_thresholds": False,
    }
    pd.DataFrame({
        "病案号": ids, "标签": y,
        "has_text": has_txt.astype(int), "has_ecg": has_ecg.astype(int),
        "p_mm": oof, "p_tab_branch": oof_tab,
        "g_txt": oof_gt, "g_ecg": oof_ge,
    }).to_csv(out / "oof.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    auc = summary["five_fold_auroc"]
    print(
        f"{name} 5折 AUROC {auc['mean']:.4f} ± {auc['std']:.4f}  OOF {summary['oof_auroc']:.4f}"
        f"  两者都没有的最大差 {same:.6f}",
        flush=True,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_neither_equals_table(device)
    for name, label, pack, out in RUNS:
        train_one(name, label, pack, out, device)


if __name__ == "__main__":
    main()

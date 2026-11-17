# -*- coding: utf-8 -*-
"""
Modality ablations of GR-MM-TALNet. The full model reuses its saved result, and the other three settings are retrained.
Patients, seed 42, and 5 folds match the full model. A removed modality is off during both training and testing, and patients are not dropped.
Results are written to results_mm_talnet_gate_small_modality/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
from train_mm_talnet import (
    AUX_W, BATCH, EPOCHS, LR, N_FOLDS, PACK, PATIENCE, SEED, load_ecg, pack_fold,
)
from train_mm_talnet_gate_small import (
    check_neither_equals_table, delong, mean_std,
)
from network import MM_TALNetGateSmall

sys.path.insert(0, str(HERE / "model"))
from metrics import compute_metrics, find_f1_threshold
OUT = HERE / "results_mm_talnet_gate_small_modality"
FULL = HERE / "results_mm_talnet_gate_small" / "oof.csv"
RUNS = (
    ("tab", "表格", False, False),
    ("tab_text", "表格＋叙述", True, False),
    ("tab_ecg", "表格＋心电", False, True),
)


def train_one(name, use_txt, use_ecg, y, first, later, switch, ctx, txt, has_txt, ecg, has_ecg, folds, device):
    """Initialize and early-stop this setting on its own. Folds are supplied by the caller and shared by the three settings."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    flag_txt = has_txt if use_txt else np.zeros_like(has_txt)
    flag_ecg = has_ecg if use_ecg else np.zeros_like(has_ecg)
    txt_use = txt if use_txt else np.zeros_like(txt)
    ecg_use = ecg if use_ecg else np.zeros_like(ecg)
    oof = np.zeros(len(y))
    oof_tab = np.zeros(len(y))
    aucs, aps, f1s, vals, test_n = [], [], [], [], []
    for fold, (tr, te) in enumerate(folds, 1):
        tr_i, va_i = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=y[tr])
        packed = pack_fold(first, later, switch, ctx, tr)

        def make_ds(ix, packed=packed):
            z1, z2, m1, m2, delta, ctx_z = packed
            return TensorDataset(
                torch.from_numpy(z1[ix]), torch.from_numpy(z2[ix]),
                torch.from_numpy(m1[ix]), torch.from_numpy(m2[ix]),
                torch.from_numpy(delta[ix]), torch.from_numpy(ctx_z[ix]),
                torch.from_numpy(switch[ix]),
                torch.from_numpy(ecg_use[ix]), torch.from_numpy(flag_ecg[ix]),
                torch.from_numpy(txt_use[ix]), torch.from_numpy(flag_txt[ix]),
                torch.from_numpy(y[ix].astype(np.float32)),
            )

        n_pos = max(int(y[tr].sum()), 1)
        pos_w = torch.tensor([(len(tr) - n_pos) / n_pos], device=device)
        model = MM_TALNetGateSmall(48, ctx.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best_val, best_state, wait = -1.0, None, 0
        for _epoch in range(EPOCHS):
            model.train()
            for batch in DataLoader(make_ds(tr_i), batch_size=BATCH, shuffle=True):
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
                for batch in DataLoader(make_ds(va_i), batch_size=BATCH):
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
        wdir = OUT / "weights" / name
        wdir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "fold": fold, "best_val": best_val}, wdir / f"fold{fold}.pt")
        model.eval()
        pred, pred_tab = [], []
        with torch.no_grad():
            for batch in DataLoader(make_ds(te), batch_size=BATCH):
                *xs, _yb = [t.to(device) for t in batch]
                fuse, tab, _gt, _ge = model(*xs)
                pred.append(torch.sigmoid(fuse).cpu().numpy())
                pred_tab.append(torch.sigmoid(tab).cpu().numpy())
        oof[te] = np.concatenate(pred)
        oof_tab[te] = np.concatenate(pred_tab)
        aucs.append(float(roc_auc_score(y[te], oof[te])))
        aps.append(float(average_precision_score(y[te], oof[te])))
        f1s.append(float(compute_metrics(y[te], oof[te], threshold=find_f1_threshold(y[te], oof[te]))["f1"]))
        vals.append(best_val)
        test_n.append(int(len(te)))
        print(f"  {name} fold {fold}: n={len(te)} AUROC={aucs[-1]:.4f} AUPRC={aps[-1]:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gap = float(np.max(np.abs(oof - oof_tab))) if not (use_txt or use_ecg) else None
    return {
        "oof": oof,
        "five_fold_auroc": mean_std(aucs),
        "five_fold_auprc": mean_std(aps),
        "five_fold_f1": mean_std(f1s),
        "val_auroc": [float(v) for v in vals],
        "test_n": test_n,
        "oof_auroc": float(roc_auc_score(y, oof)),
        "oof_auprc": float(average_precision_score(y, oof)),
        "tab_max_abs_diff": gap,
    }


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
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    print(f"device {device}  n={len(y)} pos={int(y.sum())} text={int(has_txt.sum())} ecg={int(has_ecg.sum())}", flush=True)

    full = pd.read_csv(FULL, dtype={"病案号": str}, encoding="utf-8-sig")
    full["病案号"] = full["病案号"].str.zfill(8)
    aligned = pd.DataFrame({"病案号": ids}).merge(full[["病案号", "p_mm"]], on="病案号", how="left")
    if aligned["p_mm"].isna().any():
        raise RuntimeError("完整模型的折外预测对不齐")
    p_full = aligned["p_mm"].to_numpy(float)

    rows = {}
    pred = {"病案号": ids, "标签": y, "full": p_full}
    for name, label, use_txt, use_ecg in RUNS:
        print(label, flush=True)
        row = train_one(name, use_txt, use_ecg, y, first, later, switch, ctx, txt, has_txt, ecg, has_ecg, folds, device)
        pred[name] = row.pop("oof")
        row["label"] = label
        row["use_text"] = use_txt
        row["use_ecg"] = use_ecg
        rows[name] = row
        auc = row["five_fold_auroc"]
        print(f"{name} 5折 AUROC {auc['mean']:.4f} ± {auc['std']:.4f}  OOF {row['oof_auroc']:.4f}", flush=True)

    compare = {
        "text_minus_tab": delong(y, pred["tab"], pred["tab_text"]),
        "ecg_minus_tab": delong(y, pred["tab"], pred["tab_ecg"]),
        "full_minus_tab": delong(y, pred["tab"], p_full),
        "full_minus_tab_text": delong(y, pred["tab_text"], p_full),
        "full_minus_tab_ecg": delong(y, pred["tab_ecg"], p_full),
    }
    summary = {
        "model": "GR-MM-TALNet 模态贡献。少掉的模态对所有人关掉。",
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_text": int(has_txt.sum()),
        "n_ecg": int(has_ecg.sum()),
        "full_model": "results_mm_talnet_gate_small，没有重训",
        "runs": rows,
        "delong_alt_minus_ref": compare,
    }
    pd.DataFrame(pred).to_csv(OUT / "oof.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(compare, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

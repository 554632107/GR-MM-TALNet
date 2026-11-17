# -*- coding: utf-8 -*-
"""
Component ablations of GR-MM-TALNet. The full model is not retrained here, and each run changes one part.
The five removals are temporal pairing, adaptive gates, cross-modal attention, the tabular auxiliary loss, and the tabular-logit shortcut when an auxiliary modality is present.
Standardization still uses the whole outer training fold. Results are written to results_mm_talnet_gate_small_component/.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
from train_mm_talnet import (
    AUX_W, BATCH, EPOCHS, LR, N_FOLDS, PACK, PATIENCE, SEED, load_ecg, pack_fold,
)
from train_mm_talnet_gate_small import mean_std
from network import MM_TALNetGateSmall
OUT = HERE / "results_mm_talnet_gate_small_component"
FULL = HERE / "results_mm_talnet_gate_small" / "oof.csv"


class ComponentNet(MM_TALNetGateSmall):
    """Ablate one part of the full forward pass. The base mode matches MM_TALNetGateSmall."""

    def __init__(self, n_labs: int, n_ctx: int, mode: str = "base"):
        super().__init__(n_labs, n_ctx)
        self.mode = mode

    def forward(self, x_first, x_second, m_first, m_second, delta, ctx, switched, ecg, has_ecg, txt, has_txt):
        tab_logit, tab_h = self.tab(x_first, x_second, m_first, m_second, delta, ctx, switched)
        b = tab_h.size(0)
        _, feat = self.ecg(ecg)
        feat = feat * has_ecg.view(b, 1).float()
        txt_h = self.txt_proj(txt)
        ecg_h = self.ecg_proj(feat)
        txt_tok = txt_h + self.mod_emb.weight[1]
        ecg_tok = ecg_h + self.mod_emb.weight[2]
        if self.mode == "no_attn":
            # Skip three-token fusion. Residual heads read the modality tokens from before fusion.
            r_txt = self.head_txt(txt_tok).squeeze(-1)
            r_ecg = self.head_ecg(ecg_tok).squeeze(-1)
        else:
            tok = torch.stack([
                tab_h + self.mod_emb.weight[0],
                txt_tok,
                ecg_tok,
            ], dim=1)
            pad = torch.stack([
                torch.zeros(b, dtype=torch.bool, device=tab_h.device),
                ~has_txt.view(b).bool(),
                ~has_ecg.view(b).bool(),
            ], dim=1)
            fused = self.fuse(tok, src_key_padding_mask=pad)
            r_txt = self.head_txt(fused[:, 1]).squeeze(-1)
            r_ecg = self.head_ecg(fused[:, 2]).squeeze(-1)
        if self.mode == "fixed_gate":
            # When the modality is present, fix its gate at 1. A missing modality stays 0. Do not multiply the pre-fusion token.
            g_txt = has_txt.view(b).float()
            g_ecg = has_ecg.view(b).float()
        else:
            g_txt = torch.sigmoid(self.gate_txt(torch.cat([tab_h, txt_h], dim=-1)).squeeze(-1))
            g_ecg = torch.sigmoid(self.gate_ecg(torch.cat([tab_h, ecg_h], dim=-1)).squeeze(-1))
            g_txt = g_txt * has_txt.view(b).float()
            g_ecg = g_ecg * has_ecg.view(b).float()
        residual = g_txt * r_txt + g_ecg * r_ecg
        if self.mode == "no_shortcut":
            has_aux = (has_txt.view(b).float() + has_ecg.view(b).float()) > 0
            fuse = torch.where(has_aux, residual, tab_logit)
        else:
            fuse = tab_logit + residual
        return fuse, tab_logit, g_txt, g_ecg


def delong_ci(y: np.ndarray, p_ref: np.ndarray, p_alt: np.ndarray) -> dict:
    """delta = alt − ref, with a 95% interval."""
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
        "auroc_ref": float(t1.mean()),
        "auroc_alt": float(t2.mean()),
        "delta": diff,
        "se": se,
        "ci95_low": diff - 1.96 * se if var > 0 else np.nan,
        "ci95_high": diff + 1.96 * se if var > 0 else np.nan,
        "p": p,
    }


def check_modes(device) -> None:
    """Check the three variants with a changed forward pass before training."""
    torch.manual_seed(0)
    b, n = 4, 8
    xs = [torch.randn(b, n, device=device) for _ in range(4)]
    delta = torch.randn(b, n, device=device)
    ctx = torch.randn(b, 4, device=device)
    sw = torch.zeros(b, n, device=device)
    ecg = torch.randn(b, 12, 64, device=device)
    txt = torch.randn(b, 768, device=device)
    has_txt = torch.tensor([1, 1, 0, 0], device=device)
    has_ecg = torch.tensor([0, 1, 1, 0], device=device)
    fixed = ComponentNet(8, 4, "fixed_gate").to(device).eval()
    no_attn = ComponentNet(8, 4, "no_attn").to(device).eval()
    cut = ComponentNet(8, 4, "no_shortcut").to(device).eval()
    with torch.no_grad():
        _f, _t, gt, ge = fixed(*xs, delta, ctx, sw, ecg, has_ecg, txt, has_txt)
        if float((gt - has_txt.float()).abs().max()) > 1e-6 or float((ge - has_ecg.float()).abs().max()) > 1e-6:
            raise RuntimeError("固定门没有等于缺失标记")
        no_attn(*xs, delta, ctx, sw, ecg, has_ecg, txt, has_txt)
        fuse, tab, _gt, _ge = cut(*xs, delta, ctx, sw, ecg, has_ecg, txt, has_txt)
    neither = (has_txt + has_ecg) == 0
    if float((fuse[neither] - tab[neither]).abs().max()) > 1e-5:
        raise RuntimeError("两路都没有的人没有回退到表格 logit")
    aux = ~neither
    if float(fuse[aux].abs().max()) > 1e-5:
        raise RuntimeError("有辅助模态时，初始输出没有停在 0")


def train_one(name, mode, aux_w, zero_pair, y, first, later, switch, ctx, txt, has_txt, ecg, has_ecg, folds, device):
    """Initialize this variant on its own. zero_pair turns off only the pairing gate sent into the model; value selection still uses the original switch."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_switch = np.zeros_like(switch) if zero_pair else switch
    oof = np.zeros(len(y))
    aucs, aps, vals, test_n = [], [], [], []
    for fold, (tr, te) in enumerate(folds, 1):
        tr_i, va_i = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=y[tr])
        packed = pack_fold(first, later, switch, ctx, tr)

        def make_ds(ix, packed=packed):
            z1, z2, m1, m2, delta, ctx_z = packed
            return TensorDataset(
                torch.from_numpy(z1[ix]), torch.from_numpy(z2[ix]),
                torch.from_numpy(m1[ix]), torch.from_numpy(m2[ix]),
                torch.from_numpy(delta[ix]), torch.from_numpy(ctx_z[ix]),
                torch.from_numpy(model_switch[ix]),
                torch.from_numpy(ecg[ix]), torch.from_numpy(has_ecg[ix]),
                torch.from_numpy(txt[ix]), torch.from_numpy(has_txt[ix]),
                torch.from_numpy(y[ix].astype(np.float32)),
            )

        n_pos = max(int(y[tr].sum()), 1)
        pos_w = torch.tensor([(len(tr) - n_pos) / n_pos], device=device)
        model = ComponentNet(48, ctx.shape[1], mode).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best_val, best_state, wait = -1.0, None, 0
        for _epoch in range(EPOCHS):
            model.train()
            for batch in DataLoader(make_ds(tr_i), batch_size=BATCH, shuffle=True):
                *xs, yb = [t.to(device) for t in batch]
                opt.zero_grad()
                fuse, tab, _gt, _ge = model(*xs)
                loss = loss_fn(fuse, yb) + aux_w * loss_fn(tab, yb)
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
        pred = []
        with torch.no_grad():
            for batch in DataLoader(make_ds(te), batch_size=BATCH):
                *xs, _yb = [t.to(device) for t in batch]
                fuse, _tab, _gt, _ge = model(*xs)
                pred.append(torch.sigmoid(fuse).cpu().numpy())
        oof[te] = np.concatenate(pred)
        aucs.append(float(roc_auc_score(y[te], oof[te])))
        aps.append(float(average_precision_score(y[te], oof[te])))
        vals.append(best_val)
        test_n.append(int(len(te)))
        print(f"  {name} fold {fold}: n={len(te)} AUROC={aucs[-1]:.4f} AUPRC={aps[-1]:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "oof": oof,
        "five_fold_auroc": mean_std(aucs),
        "five_fold_auprc": mean_std(aps),
        "val_auroc": [float(v) for v in vals],
        "test_n": test_n,
        "oof_auroc": float(roc_auc_score(y, oof)),
        "oof_auprc": float(average_precision_score(y, oof)),
    }


def subset_metrics(name, y, p, mask, folds) -> dict:
    """Compute pooled and per-fold metrics on the given patients. Skip a fold that does not contain both classes."""
    aucs, aps = [], []
    for _fold, (_tr, te) in enumerate(folds, 1):
        keep = np.intersect1d(te, np.flatnonzero(mask))
        if len(np.unique(y[keep])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[keep], p[keep])))
        aps.append(float(average_precision_score(y[keep], p[keep])))
    return {
        "n": int(mask.sum()),
        "n_pos": int(y[mask].sum()),
        "five_fold_auroc": mean_std(aucs),
        "five_fold_auprc": mean_std(aps),
        "oof_auroc": float(roc_auc_score(y[mask], p[mask])),
        "oof_auprc": float(average_precision_score(y[mask], p[mask])),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_modes(device)
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
    aux = (has_txt > 0.5) | (has_ecg > 0.5)
    if int((~aux).sum()) != 126 or int(aux.sum()) != 826:
        raise RuntimeError(f"辅助模态人数不是 826/126：{int(aux.sum())}/{int((~aux).sum())}")
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    full = pd.read_csv(FULL, dtype={"病案号": str}, encoding="utf-8-sig")
    full["病案号"] = full["病案号"].str.zfill(8)
    aligned = pd.DataFrame({"病案号": ids}).merge(full[["病案号", "p_mm"]], on="病案号", how="left")
    if aligned["p_mm"].isna().any():
        raise RuntimeError("完整模型的折外预测对不齐")
    p_full = aligned["p_mm"].to_numpy(float)
    print(f"device {device}  n={len(y)} pos={int(y.sum())} aux={int(aux.sum())}", flush=True)

    jobs = [
        ("wo_pair", "w/o temporal-pair encoding", "base", AUX_W, True),
        ("wo_gate", "w/o adaptive gating", "fixed_gate", AUX_W, False),
        ("wo_attn", "w/o cross-modal attention", "no_attn", AUX_W, False),
        ("wo_aux", "w/o tabular auxiliary loss", "base", 0.0, False),
        ("wo_shortcut", "w/o tabular logit shortcut, with tabular fallback", "no_shortcut", AUX_W, False),
    ]
    rows, pred = {}, {"病案号": ids, "标签": y, "has_text": has_txt.astype(int), "has_ecg": has_ecg.astype(int), "full": p_full}
    for name, label, mode, aux_w, zero_pair in jobs:
        print(label, flush=True)
        row = train_one(name, mode, aux_w, zero_pair, y, first, later, switch, ctx, txt, has_txt, ecg, has_ecg, folds, device)
        pred[name] = row.pop("oof")
        row["label"] = label
        row["delong_variant_minus_full"] = delong_ci(y, p_full, pred[name])
        rows[name] = row
        d = row["delong_variant_minus_full"]
        auc = row["five_fold_auroc"]
        print(
            f"{name} 5折 AUROC {auc['mean']:.4f} ± {auc['std']:.4f}  OOF {row['oof_auroc']:.4f}"
            f"  delta {d['delta']:+.4f}  CI {d['ci95_low']:+.4f},{d['ci95_high']:+.4f}  p={d['p']:.4f}",
            flush=True,
        )

    both = {}
    for key in ("full", "wo_shortcut"):
        both[key] = subset_metrics(key, y, pred[key], aux, folds)
    both["delong_shortcut_minus_full"] = delong_ci(y[aux], p_full[aux], pred["wo_shortcut"][aux])
    summary = {
        "spec": "每次只改一处：时间配对、自适应门、跨模态注意力、表格辅助损失、表格 logit 捷径",
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_aux": int(aux.sum()),
        "n_neither": int((~aux).sum()),
        "full_model": "results_mm_talnet_gate_small，没有重训",
        "runs": rows,
        "aux_subset": both,
    }
    pd.DataFrame(pred).to_csv(OUT / "oof.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("write", OUT, flush=True)


if __name__ == "__main__":
    main()

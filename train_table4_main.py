# -*- coding: utf-8 -*-
"""
Train LightGBM, TabNet, and FT-Transformer using 116 tabular features.
Evaluate fixed-weight late fusion with saved text and ECG predictions.
Use the same patients and five-fold splits as the main model.
Save results to results_table4_main/.
"""

from __future__ import annotations

import json
import sys
import warnings
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
sys.path.insert(0, str(HERE / "model"))
from metrics import compute_metrics, find_f1_threshold
from models import FeatureTransformer
PACK = HERE / "features" / "pack"
OUT = HERE / "results_table4_main"
T3 = HERE / "results_table3_unimodal" / "oof.csv"
MM = HERE / "results_mm_talnet" / "oof.csv"
SEED = 42
N_FOLDS = 5

LAST68 = [
    "白细胞计数", "血红蛋白", "血小板计数", "钾", "钠", "氯", "钙", "肌酐", "尿素", "血糖",
    "白蛋白", "丙氨酸氨基转移酶", "天冬氨酸氨基转移酶", "肌酸激酶-MB", "肌钙蛋白-T", "B型利钠肽",
    "D-二聚体", "全血CRP",
    "高血压", "冠心病", "心功能不全", "心脏病史", "糖尿病", "脑梗死", "脑血管病", "肿瘤",
    "吸烟史", "饮酒史", "降压药", "降糖药", "年龄", "性别",
    "红细胞压积", "中性粒细胞绝对值", "淋巴细胞绝对值", "单核细胞绝对值", "平均红细胞体积",
    "红细胞分布宽度", "血小板压积", "总蛋白", "球蛋白", "总胆红素", "乳酸脱氢酶", "肌酸激酶",
    "凝血酶原时间", "活化部分凝血活酶时间", "血浆纤维蛋白原", "国际标准化比率", "尿酸",
    "肾小球滤过率", "磷", "镁", "总胆固醇", "甘油三酯", "高密度脂蛋白胆固醇", "低密度脂蛋白胆固醇",
    "血液酸碱度", "二氧化碳分压", "氧分压", "血氧饱和度", "血乳酸",
    "心血管用药", "神经系统疾病", "NLR",
    "B型利钠肽_缺失", "肌钙蛋白-T_缺失", "血乳酸_缺失", "血液酸碱度_缺失",
]


def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "folds": [float(v) for v in a]}


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


def score_row(name: str, y: np.ndarray, oof: np.ndarray, folds) -> dict:
    aucs, aps, f1s = [], [], []
    for _tr, te in folds:
        aucs.append(float(roc_auc_score(y[te], oof[te])))
        aps.append(float(average_precision_score(y[te], oof[te])))
        f1s.append(float(compute_metrics(y[te], oof[te], threshold=find_f1_threshold(y[te], oof[te]))["f1"]))
    return {
        "name": name,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "five_fold_auroc": mean_std(aucs),
        "five_fold_auprc": mean_std(aps),
        "five_fold_f1": mean_std(f1s),
        "oof_auroc": float(roc_auc_score(y, oof)),
        "oof_auprc": float(average_precision_score(y, oof)),
    }


def load_116() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Level at the second measurement plus the delta. Without a second measurement, keep the first level and leave the delta missing."""
    ids = pd.read_csv(PACK / "text_cls_ids.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    ids["病案号"] = ids["病案号"].str.zfill(8)
    y = ids["标签"].to_numpy(int)
    names = json.loads((PACK / "columns.json").read_text(encoding="utf-8"))
    labs = list(names["labs"])
    ctx_names = list(names["context"])
    first = np.load(PACK / "labs_first.npy").astype(np.float64)
    second = np.load(PACK / "labs_second.npy").astype(np.float64)
    switch = np.load(PACK / "labs_switch.npy").astype(np.float64)
    ctx = np.load(PACK / "context.npy").astype(np.float64)
    if first.shape[1] != 48 or ctx.shape[1] != 20:
        raise RuntimeError(f"化验 {first.shape[1]} 列、背景 {ctx.shape[1]} 列，不是 48+20")
    has2 = switch > 0.5
    last = np.where(has2, second, first)
    delta = np.where(has2 & np.isfinite(first) & np.isfinite(second), second - first, np.nan)
    lab_i = {name: i for i, name in enumerate(labs)}
    ctx_i = {name: i for i, name in enumerate(ctx_names)}
    cols = []
    blocks = []
    for name in LAST68:
        if name in lab_i:
            blocks.append(last[:, lab_i[name]])
        elif name in ctx_i:
            blocks.append(ctx[:, ctx_i[name]])
        else:
            raise RuntimeError(f"68 列水平里没有 {name}")
        cols.append(name)
    for name in labs:
        blocks.append(delta[:, lab_i[name]])
        cols.append(f"{name}_Δ")
    x = np.column_stack(blocks)
    if x.shape[1] != 116:
        raise RuntimeError(f"期望 116 列，实际 {x.shape[1]}")
    return ids, y, x, cols


def run_trees(x: np.ndarray, y: np.ndarray, folds) -> dict[str, np.ndarray]:
    from lightgbm import LGBMClassifier

    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(folds, 1):
        n_pos = max(int(y[tr].sum()), 1)
        clf = LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_child_samples=20, reg_lambda=1.0, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=(len(tr) - n_pos) / n_pos,
            random_state=SEED, verbosity=-1,
        )
        clf.fit(x[tr], y[tr])
        oof[te] = clf.predict_proba(x[te])[:, 1]
        print(f"LightGBM fold {fold}: AUROC={roc_auc_score(y[te], oof[te]):.4f}", flush=True)
    return {"LightGBM": oof}


def continuous_index(cols: list[str]) -> list[int]:
    """Standardize laboratory values, deltas, and age. History, medication, and missingness indicators stay 0/1."""
    skip = {"性别", "高血压", "冠心病", "心功能不全", "心脏病史", "糖尿病", "脑梗死", "脑血管病",
            "肿瘤", "吸烟史", "饮酒史", "降压药", "降糖药", "心血管用药", "神经系统疾病"}
    idx = []
    for i, name in enumerate(cols):
        if name.endswith("_Δ") or name == "年龄" or name not in skip and not name.endswith("_缺失"):
            idx.append(i)
    return idx


def fill_fold(x: np.ndarray, cols: list[str], tr: np.ndarray) -> np.ndarray:
    """Fill with the training-fold median and standardize continuous columns on that fold. Used only by TabNet and FT-Transformer."""
    med = np.nanmedian(x[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    filled = np.where(np.isnan(x), med, x)
    ci = continuous_index(cols)
    mean = filled[tr][:, ci].mean(axis=0)
    std = filled[tr][:, ci].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    filled[:, ci] = (filled[:, ci] - mean) / std
    return filled.astype(np.float32)


BASELINE_EPOCHS = 80
BASELINE_PATIENCE = 15


def run_tabnet(x: np.ndarray, cols: list[str], y: np.ndarray, folds) -> np.ndarray:
    from pytorch_tabnet.tab_model import TabNetClassifier

    warnings.filterwarnings("ignore")
    oof = np.zeros(len(y))
    print("TabNet", flush=True)
    for fold, (tr, te) in enumerate(folds, 1):
        filled = fill_fold(x, cols, tr)
        tr_i, va_i = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=y[tr])
        clf = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
            n_independent=1, n_shared=1,
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
            mask_type="sparsemax", seed=SEED, verbose=0,
            device_name="cuda" if torch.cuda.is_available() else "cpu",
        )
        clf.fit(
            filled[tr_i], y[tr_i],
            eval_set=[(filled[va_i], y[va_i])],
            eval_name=["val"], eval_metric=["auc"],
            max_epochs=BASELINE_EPOCHS, patience=BASELINE_PATIENCE, batch_size=32, virtual_batch_size=16,
            weights=1, drop_last=False,
        )
        oof[te] = clf.predict_proba(filled[te])[:, 1]
        print(f"  fold {fold}: AUROC={roc_auc_score(y[te], oof[te]):.4f}", flush=True)
    return oof


def focal_loss(logits, labels, gamma=2.0, alpha=0.75):
    labels = labels.view(-1).unsqueeze(1)
    logits = logits.view(-1).unsqueeze(1)
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (alpha_t * ((1 - p_t) ** gamma) * bce).mean()


def mixup(xb, yb, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(yb.size(0), device=yb.device)
    return lam * xb + (1 - lam) * xb[idx], lam * yb + (1 - lam) * yb[idx]


def run_ft(x: np.ndarray, cols: list[str], y: np.ndarray, folds, device) -> np.ndarray:
    oof = np.zeros(len(y))
    print("FT-Transformer", flush=True)
    for fold, (tr, te) in enumerate(folds, 1):
        filled = fill_fold(x, cols, tr)
        tr_i, va_i = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=y[tr])
        model = FeatureTransformer(n_features=filled.shape[1], d_model=32, n_heads=4).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BASELINE_EPOCHS, eta_min=1e-5)
        best, state, wait = -1.0, None, 0

        def batch(idx, shuffle):
            ds = TensorDataset(torch.from_numpy(filled[idx]), torch.from_numpy(y[idx].astype(np.float32)))
            return DataLoader(ds, batch_size=32, shuffle=shuffle)

        tr_dl, va_dl = batch(tr_i, True), batch(va_i, False)
        for _epoch in range(BASELINE_EPOCHS):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                xb, yb = mixup(xb, yb)
                opt.zero_grad()
                focal_loss(model(xb), yb).backward()
                opt.step()
            sch.step()
            model.eval()
            vp, vy = [], []
            with torch.no_grad():
                for xb, yb in va_dl:
                    vp.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy().tolist())
                    vy.extend(yb.numpy().tolist())
            val = float(roc_auc_score(vy, vp)) if len(np.unique(vy)) > 1 else 0.5
            if val > best:
                best, wait = val, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= BASELINE_PATIENCE:
                    break
        if state is not None:
            model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logit = model(torch.from_numpy(filled[te]).to(device))
            oof[te] = torch.sigmoid(logit).cpu().numpy()
        print(f"  fold {fold}: AUROC={roc_auc_score(y[te], oof[te]):.4f} val={best:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof


def fuse_one(w, p_tab, p_txt, p_ecg, has_txt, has_ecg) -> np.ndarray:
    """Drop weights of missing modalities and renormalize the remaining weights."""
    out = np.zeros(len(p_tab))
    for i in range(len(p_tab)):
        weights, scores = [], []
        if w[0] > 0:
            weights.append(w[0])
            scores.append(p_tab[i])
        if has_txt[i] and w[1] > 0 and np.isfinite(p_txt[i]):
            weights.append(w[1])
            scores.append(p_txt[i])
        if has_ecg[i] and w[2] > 0 and np.isfinite(p_ecg[i]):
            weights.append(w[2])
            scores.append(p_ecg[i])
        out[i] = float(np.dot(weights, scores) / np.sum(weights))
    return out


def best_grid(y, p_tab, p_txt, p_ecg, has_txt, has_ecg) -> dict:
    """Search weights again on the pooled out-of-fold scores. This is a sensitivity check and does not replace the three locked rows."""
    text_w = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    best = {"oof_auroc": -1.0}
    for wt in text_w:
        for we in text_w:
            p = fuse_one((1.0, wt, we), p_tab, p_txt, p_ecg, has_txt, has_ecg)
            a = float(roc_auc_score(y, p))
            if a > best["oof_auroc"]:
                best = {"w_table": 1.0, "w_text": wt, "w_ecg": we, "oof_auroc": a}
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ids, y, x, cols = load_116()
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    print(f"116 列表格  列 {x.shape[1]}  阳性 {int(y.sum())}  差值缺失 {int(np.isnan(x[:, 68:]).any(axis=1).sum())} 人至少一项", flush=True)
    summary = {
        "protocol": "第2次复查：有第二次时水平用第二次，否则留第一次；差值是第二次减第一次，没有第二次则为缺失。共116列。树不填补。TabNet 与 FT 用训练折中位数填补。晚期融合权重锁定。",
        "n_features": 116,
        "columns": cols,
        "models": {},
        "delong_alt_minus_lightgbm": {},
    }
    preds = {"病案号": ids["病案号"].to_numpy(), "标签": y}

    trees = run_trees(x, y, folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trees["TabNet"] = run_tabnet(x, cols, y, folds)
    trees["FT-Transformer"] = run_ft(x, cols, y, folds, device)

    t3 = pd.read_csv(T3, dtype={"病案号": str}, encoding="utf-8-sig")
    t3["病案号"] = t3["病案号"].str.zfill(8)
    t3 = ids[["病案号"]].merge(t3, on="病案号", how="left")
    has_txt = t3["has_text"].fillna(0).to_numpy(int) == 1
    has_ecg = t3["has_ecg"].fillna(0).to_numpy(int) == 1
    p_txt = t3["冻结CLS+线性头"].to_numpy(float)
    p_ecg = t3["浅层1D-CNN"].to_numpy(float)
    p_hist = trees["LightGBM"]
    fused = {
        "晚期表+文": fuse_one((1.0, 0.2, 0.0), p_hist, p_txt, p_ecg, has_txt, has_ecg),
        "晚期表+心电": fuse_one((1.0, 0.0, 0.5), p_hist, p_txt, p_ecg, has_txt, has_ecg),
        "晚期表+文+心电": fuse_one((1.0, 0.2, 0.3), p_hist, p_txt, p_ecg, has_txt, has_ecg),
    }
    summary["weight_search_on_pooled_oof"] = best_grid(y, p_hist, p_txt, p_ecg, has_txt, has_ecg)

    mm = pd.read_csv(MM, dtype={"病案号": str}, encoding="utf-8-sig")
    mm["病案号"] = mm["病案号"].str.zfill(8)
    mm = ids[["病案号"]].merge(mm, on="病案号", how="left")
    p_mm = mm["p_mm"].to_numpy(float)

    order = ["LightGBM", "晚期表+文", "晚期表+心电", "晚期表+文+心电", "TabNet", "FT-Transformer"]
    all_p = {**trees, **fused}
    for name in order:
        row = score_row(name, y, all_p[name], folds)
        summary["models"][name] = row
        preds[name] = all_p[name]
        auc, ap, f1 = row["five_fold_auroc"], row["five_fold_auprc"], row["five_fold_f1"]
        print(
            f"{name}  5折 AUROC {auc['mean']:.4f} ± {auc['std']:.4f}"
            f"  AUPRC {ap['mean']:.4f} ± {ap['std']:.4f}"
            f"  F1 {f1['mean']:.4f} ± {f1['std']:.4f}  OOF {row['oof_auroc']:.4f}",
            flush=True,
        )
        if name != "LightGBM":
            summary["delong_alt_minus_lightgbm"][name] = delong(y, p_hist, all_p[name])
    summary["delong_alt_minus_lightgbm"]["MM-TALNet"] = delong(y, p_hist, p_mm)
    summary["delong_alt_minus_mm"] = {
        "TabNet": delong(y, p_mm, trees["TabNet"]),
        "FT-Transformer": delong(y, p_mm, trees["FT-Transformer"]),
    }
    preds["冻结CLS+线性头"] = p_txt
    preds["浅层1D-CNN"] = p_ecg
    preds["has_text"] = has_txt.astype(int)
    preds["has_ecg"] = has_ecg.astype(int)

    pd.DataFrame(preds).to_csv(OUT / "oof.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"write {OUT}", flush=True)



if __name__ == "__main__":
    main()

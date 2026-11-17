# -*- coding: utf-8 -*-
"""
Train unimodal text and ECG models on patients who have that modality.
Outer folds match the main model: all patients, seed 42. Patients without the modality are excluded from that group's loss and metrics.
Text uses features/pack/text_cls.npy. ECG uses the extracted waveforms, and handcrafted features are read from the matching XML.
Results are written to results_table3_unimodal/ for late fusion.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
from train_mm_talnet import load_ecg
sys.path.insert(0, str(HERE / "model"))
from metrics import compute_metrics, find_f1_threshold
from network import ECGEncoder
from prepare_ecg import find_ecg_xmls, parse_ecg_time, patient_dir
from paths import data_path
# Machine intervals and axes from the XML. An interval of 0 is treated as missing. An axis of 0 is kept.
CODE_MAP = {
    "MDC_ECG_HEART_RATE": "心率",
    "MDC_ECG_TIME_PD_P": "P波时限",
    "MDC_ECG_TIME_PD_PR": "PR间期",
    "MDC_ECG_TIME_PD_QRS": "QRS时限",
    "MDC_ECG_TIME_PD_QT": "QT间期",
    "MDC_ECG_TIME_PD_QTc": "QTc间期",
    "MDC_ECG_ANGLE_P_FRONT": "P电轴",
    "MDC_ECG_ANGLE_QRS_FRONT": "QRS电轴",
    "MDC_ECG_ANGLE_T_FRONT": "T电轴",
}
XML_COLS = list(CODE_MAP.values()) + ["非正常搏动"]
ZERO_AS_MISS = {"P波时限", "PR间期", "QRS时限", "QT间期", "QTc间期"}
PQ_RE = re.compile(
    r'code="(MDC_ECG_(?:HEART_RATE|TIME_PD_[A-Za-z]+|ANGLE_[A-Z_]+))"[\s\S]{0,500}?'
    r'<value xsi:type="PQ" value="([^"]+)"',
)


def parse_meas(xml_path: Path) -> dict:
    out = {c: float("nan") for c in XML_COLS}
    try:
        text = xml_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = xml_path.read_bytes().decode("utf-8", errors="ignore")
    for code, raw in PQ_RE.findall(text):
        col = CODE_MAP.get(code)
        if not col:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if col in ZERO_AS_MISS and abs(v) < 1e-8:
            continue
        if col == "心率" and v <= 0:
            continue
        out[col] = v
    beats = re.findall(r'code="(MDC_ECG_BEAT_[A-Z0-9_]+)"', text)
    out["非正常搏动"] = float(sum(1 for b in beats if b != "MDC_ECG_BEAT_NORMAL"))
    return out

OUT = HERE / "results_table3_unimodal"
PACK = HERE / "features" / "pack"
SEED = 42


def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "folds": [float(v) for v in a]}


def subset_folds(y: np.ndarray, has: np.ndarray):
    """Split the outer folds on all patients, then keep only patients who have this modality."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for tr_all, te_all in skf.split(np.zeros(len(y)), y):
        tr = tr_all[has[tr_all]]
        te = te_all[has[te_all]]
        folds.append((tr, te))
    return folds


def score_fold(y: np.ndarray, p: np.ndarray) -> dict | None:
    if len(y) < 5 or len(np.unique(y)) < 2:
        return None
    base = compute_metrics(y, p)
    f1 = compute_metrics(y, p, threshold=find_f1_threshold(y, p))
    return {
        "auroc": float(base["auroc"]),
        "auprc": float(average_precision_score(y, p)),
        "f1": float(f1["f1"]),
    }


def pack_row(name: str, y: np.ndarray, oof: np.ndarray, has: np.ndarray, folds, fold_rows: list[dict]) -> dict:
    mask = has & np.isfinite(oof)
    pooled_auc = float(roc_auc_score(y[mask], oof[mask]))
    pooled_ap = float(average_precision_score(y[mask], oof[mask]))
    return {
        "name": name,
        "n": int(has.sum()),
        "n_pos": int(y[has].sum()),
        "five_fold_auroc": mean_std([r["auroc"] for r in fold_rows]),
        "five_fold_auprc": mean_std([r["auprc"] for r in fold_rows]),
        "five_fold_f1": mean_std([r["f1"] for r in fold_rows]),
        "oof_auroc": pooled_auc,
        "oof_auprc": pooled_ap,
        "test_n": [int(len(te)) for _tr, te in folds],
    }


def jieba_cut(texts: list[str]) -> list[str]:
    import jieba

    return [" ".join(jieba.lcut(t or "")) for t in texts]


def run_tfidf(name: str, docs: list[str], y: np.ndarray, has: np.ndarray, factory) -> tuple[np.ndarray, dict]:
    folds = subset_folds(y, has)
    oof = np.full(len(y), np.nan)
    rows = []
    print(name, flush=True)
    for fold, (tr, te) in enumerate(folds, 1):
        if len(tr) < 16 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            print(f"  fold {fold}: 类别不足，跳过", flush=True)
            continue
        pipe = factory()
        pipe.fit([docs[i] for i in tr], y[tr])
        oof[te] = pipe.predict_proba([docs[i] for i in te])[:, 1]
        row = score_fold(y[te], oof[te])
        rows.append(row)
        print(f"  fold {fold}: AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} n={len(te)}", flush=True)
    return oof, pack_row(name, y, oof, has, folds, rows)


def run_cls_head(name: str, cls_x: np.ndarray, y: np.ndarray, has: np.ndarray, head_factory, device) -> tuple[np.ndarray, dict]:
    folds = subset_folds(y, has)
    oof = np.full(len(y), np.nan)
    rows = []
    print(name, flush=True)
    for fold, (tr, te) in enumerate(folds, 1):
        if len(tr) < 20 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            print(f"  fold {fold}: 类别不足，跳过", flush=True)
            continue
        tr_i, va_i = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=y[tr])
        n_pos = max(int(y[tr_i].sum()), 1)
        pos_w = torch.tensor([(len(tr_i) - n_pos) / n_pos], device=device)
        model = head_factory().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, state, wait = -1.0, None, 0
        for _epoch in range(40):
            model.train()
            order = np.random.permutation(tr_i)
            for i0 in range(0, len(order), 64):
                ix = order[i0:i0 + 64]
                xb = torch.from_numpy(cls_x[ix]).to(device)
                yb = torch.from_numpy(y[ix].astype(np.float32)).to(device)
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(torch.from_numpy(cls_x[va_i]).to(device))).cpu().numpy()
            val = float(roc_auc_score(y[va_i], vp)) if len(np.unique(y[va_i])) > 1 else 0.5
            if val > best:
                best, wait = val, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 8:
                    break
        if state is not None:
            model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(model(torch.from_numpy(cls_x[te]).to(device))).cpu().numpy()
        row = score_fold(y[te], oof[te])
        rows.append(row)
        print(f"  fold {fold}: AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} val={best:.4f}", flush=True)
        del model
    return oof, pack_row(name, y, oof, has, folds, rows)


def run_hand(name: str, x: np.ndarray, y: np.ndarray, has: np.ndarray, factory) -> tuple[np.ndarray, dict]:
    folds = subset_folds(y, has)
    oof = np.full(len(y), np.nan)
    rows = []
    print(name, flush=True)
    for fold, (tr, te) in enumerate(folds, 1):
        if len(tr) < 16 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            print(f"  fold {fold}: 类别不足，跳过", flush=True)
            continue
        model = factory()
        model.fit(x[tr], y[tr])
        oof[te] = model.predict_proba(x[te])[:, 1]
        row = score_fold(y[te], oof[te])
        rows.append(row)
        print(f"  fold {fold}: AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} n={len(te)}", flush=True)
    return oof, pack_row(name, y, oof, has, folds, rows)


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(768, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 64), nn.GELU(), nn.Dropout(0.25), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class WiderECG(nn.Module):
    """One extra convolutional block beyond the main model's shallow ECG encoder, with 256 channels."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 64),
        )
        self.head = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Dropout(0.4), nn.Linear(16, 1))

    def forward(self, x):
        feat = self.net(x)
        return self.head(feat).squeeze(-1), feat


class ShallowECG(nn.Module):
    """Three-layer convolutional encoder, the same as ECGEncoder in model/network.py."""

    def __init__(self):
        super().__init__()
        self.encoder = ECGEncoder(out_dim=64)

    def forward(self, x):
        return self.encoder(x)


def run_cnn(name: str, waves: np.ndarray, y: np.ndarray, has: np.ndarray, factory, device) -> tuple[np.ndarray, dict]:
    folds = subset_folds(y, has)
    oof = np.full(len(y), np.nan)
    rows = []
    print(name, flush=True)
    for fold, (tr, te) in enumerate(folds, 1):
        if len(tr) < 16 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            print(f"  fold {fold}: 类别不足，跳过", flush=True)
            continue
        tr_i, va_i = train_test_split(tr, test_size=0.15, random_state=SEED, stratify=y[tr])
        n_pos = max(int(y[tr_i].sum()), 1)
        pos_w = torch.tensor([(len(tr_i) - n_pos) / n_pos], device=device)
        model = factory().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, state, wait = -1.0, None, 0
        for _epoch in range(40):
            model.train()
            order = np.random.permutation(tr_i)
            for i0 in range(0, len(order), 16):
                ix = order[i0:i0 + 16]
                xb = torch.from_numpy(waves[ix]).to(device)
                yb = torch.from_numpy(y[ix].astype(np.float32)).to(device)
                opt.zero_grad()
                logit, _feat = model(xb)
                loss_fn(logit, yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                logit, _feat = model(torch.from_numpy(waves[va_i]).to(device))
                vp = torch.sigmoid(logit).cpu().numpy()
            val = float(roc_auc_score(y[va_i], vp)) if len(np.unique(y[va_i])) > 1 else 0.5
            if val > best:
                best, wait = val, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 8:
                    break
        if state is not None:
            model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logit, _feat = model(torch.from_numpy(waves[te]).to(device))
            oof[te] = torch.sigmoid(logit).cpu().numpy()
        row = score_fold(y[te], oof[te])
        rows.append(row)
        print(f"  fold {fold}: AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} val={best:.4f} n={len(te)}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof, pack_row(name, y, oof, has, folds, rows)


def match_xml(pid: str, label: int, pick_time: str) -> Path | None:
    """Find the XML whose recording time matches the saved pick."""
    root = data_path("CA_ECG") if int(label) == 1 else data_path("HR_ECG")
    folder = patient_dir(root, pid)
    if folder is None or not pick_time:
        return None
    target = pd.Timestamp(pick_time)
    best = None
    best_gap = None
    for xml_path in find_ecg_xmls(folder):
        dt = parse_ecg_time(xml_path)
        if dt is None:
            continue
        gap = abs((pd.Timestamp(dt) - target).total_seconds())
        if best_gap is None or gap < best_gap:
            best, best_gap = xml_path, gap
    if best is None or best_gap is None or best_gap > 1:
        return None
    return best


def load_hand_meas(ids: pd.DataFrame) -> np.ndarray:
    """Read machine measurements for patients with ECG. Patients without ECG are entirely missing."""
    pick = pd.read_csv(HERE / "features" / "ecg_pick.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    pick["病案号"] = pick["病案号"].str.zfill(8)
    pick = ids[["病案号"]].merge(pick, on="病案号", how="left")
    rows = []
    n_hit = 0
    for rec in pick.itertuples(index=False):
        blank = {c: np.nan for c in XML_COLS}
        if int(rec.has_ecg) != 1:
            rows.append(blank)
            continue
        path = match_xml(rec.病案号, int(rec.标签), str(rec.pick_time))
        if path is None:
            rows.append(blank)
            continue
        rows.append(parse_meas(path))
        n_hit += 1
    print(f"手工心电测量读到 {n_hit} / {int(pick['has_ecg'].fillna(0).sum())}", flush=True)
    frame = pd.DataFrame(rows)
    frame.insert(0, "病案号", ids["病案号"].to_numpy())
    frame.to_csv(HERE / "features" / "ecg_xmlmeas_952.csv", index=False, encoding="utf-8-sig")
    return frame[XML_COLS].to_numpy(float)


def load_text():
    ids = pd.read_csv(PACK / "text_cls_ids.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    ids["病案号"] = ids["病案号"].str.zfill(8)
    text = pd.read_csv(HERE / "features" / "text_clean_v2.csv", dtype={"病案号": str}, encoding="utf-8-sig")
    text["病案号"] = text["病案号"].str.zfill(8)
    df = ids.merge(text[["病案号", "叙述清洗", "has_text"]], on="病案号", how="left", suffixes=("", "_file"))
    y = df["标签"].to_numpy(int)
    docs = df["叙述清洗"].fillna("").astype(str)
    has = (df["has_text"].fillna(0).to_numpy(int) == 1) & (docs.str.len().to_numpy() > 0)
    cls = np.load(PACK / "text_cls.npy").astype(np.float32)
    if cls.shape != (len(y), 768):
        raise RuntimeError(f"text_cls 形状 {cls.shape} 与名单 {len(y)} 不一致")
    return ids, y, docs.tolist(), has, cls


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ids, y, docs, has_txt, cls = load_text()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"文本 {int(has_txt.sum())} 人，阳性 {int(y[has_txt].sum())}，device {device}", flush=True)
    summary = {"protocol": "只在有该模态的人上训练和评价。外层折按全体患者划分，种子 42。", "models": {}}
    preds = {"病案号": ids["病案号"], "标签": y, "has_text": has_txt.astype(int)}

    def keep(name, oof, row):
        summary["models"][name] = row
        preds[name] = oof
        auc = row["five_fold_auroc"]
        ap = row["five_fold_auprc"]
        f1 = row["five_fold_f1"]
        print(
            f"{name}  5折 AUROC {auc['mean']:.4f} ± {auc['std']:.4f}"
            f"  AUPRC {ap['mean']:.4f} ± {ap['std']:.4f}"
            f"  F1 {f1['mean']:.4f} ± {f1['std']:.4f}  OOF {row['oof_auroc']:.4f}",
            flush=True,
        )

    words = jieba_cut(docs)
    oof, row = run_tfidf(
        "TF-IDF词+逻辑回归", words, y, has_txt,
        lambda: Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        ]),
    )
    keep("TF-IDF词+逻辑回归", oof, row)
    oof, row = run_tfidf(
        "TF-IDF字+逻辑回归", docs, y, has_txt,
        lambda: Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=30000, min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=SEED)),
        ]),
    )
    keep("TF-IDF字+逻辑回归", oof, row)
    oof, row = run_cls_head("冻结CLS+线性头", cls, y, has_txt, LinearHead, device)
    keep("冻结CLS+线性头", oof, row)
    oof, row = run_cls_head("冻结CLS+MLP头", cls, y, has_txt, MLPHead, device)
    keep("冻结CLS+MLP头", oof, row)

    waves, has_ecg = load_ecg(
        ids["病案号"].tolist(),
        pd.read_csv(PACK / "ecg_pick.csv", dtype={"病案号": str}, encoding="utf-8-sig"),
    )
    has_ecg = has_ecg > 0.5
    hand = load_hand_meas(ids)
    has_hand = has_ecg & np.isfinite(hand).any(axis=1)
    print(f"心电波形 {int(has_ecg.sum())} 人，手工测量 {int(has_hand.sum())} 人", flush=True)
    preds["has_ecg"] = has_ecg.astype(int)

    def make_lr():
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=SEED)),
        ])

    def make_hist():
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=100, max_leaf_nodes=20,
                l2_regularization=1.0, min_samples_leaf=20, random_state=SEED,
            )),
        ])

    oof, row = run_hand("手工心电+逻辑回归", hand, y, has_hand, make_lr)
    keep("手工心电+逻辑回归", oof, row)
    oof, row = run_hand("手工心电+HistGB", hand, y, has_hand, make_hist)
    keep("手工心电+HistGB", oof, row)
    oof, row = run_cnn("浅层1D-CNN", waves, y, has_ecg, ShallowECG, device)
    keep("浅层1D-CNN", oof, row)
    oof, row = run_cnn("通道加宽1D-CNN", waves, y, has_ecg, WiderECG, device)
    keep("通道加宽1D-CNN", oof, row)

    pd.DataFrame(preds).to_csv(OUT / "oof.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"write {OUT}", flush=True)


if __name__ == "__main__":
    main()

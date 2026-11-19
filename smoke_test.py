"""
Run five-fold smoke tests on synthetic arrays in an isolated temporary directory.
Synthetic scores are software checks, not estimates of clinical performance.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def generate(root: Path) -> None:
    rng = np.random.default_rng(42)
    pack = root / "features" / "pack"
    pack.mkdir(parents=True)
    n = 80
    ids = [f"SYN{i:05d}" for i in range(n)]
    y = np.arange(n) % 2
    patterns = (np.arange(n) // 2) % 4
    ht = (patterns & 1).astype(int)
    he = ((patterns >> 1) & 1).astype(int)
    first = rng.normal(size=(n, 48)).astype(np.float32)
    first[rng.random(first.shape) < 0.1] = np.nan
    switch = (rng.random((n, 48)) > 0.4).astype(np.float32)
    second = rng.normal(size=(n, 48)).astype(np.float32)
    second[switch == 0] = np.nan
    text = rng.normal(size=(n, 768)).astype(np.float32)
    text[ht == 0] = 0
    for name, arr in {
        "labs_first": first, "labs_second": second, "labs_switch": switch,
        "context": rng.normal(size=(n, 20)).astype(np.float32), "text_cls": text,
    }.items():
        np.save(pack / f"{name}.npy", arr)
    pd.DataFrame({"病案号": ids, "标签": y, "has_text": ht}).to_csv(
        pack / "text_cls_ids.csv", index=False, encoding="utf-8-sig",
    )
    rows = []
    for i, pid in enumerate(ids):
        path = pack / f"{pid}.npy"
        if he[i]:
            np.save(path, rng.normal(size=(12, 5000)).astype(np.float32))
        rows.append({"病案号": pid, "标签": y[i], "has_ecg": he[i], "src": str(path) if he[i] else ""})
    pd.DataFrame(rows).to_csv(pack / "ecg_pick.csv", index=False, encoding="utf-8-sig")
    import pack_for_train as schema
    (pack / "columns.json").write_text(
        json.dumps({"labs": schema.LABS, "context": schema.CONTEXT}, ensure_ascii=False), encoding="utf-8",
    )


def worker(root: Path) -> None:
    import torch

    torch.set_num_threads(2)
    sys.path.insert(0, str(root))
    base = importlib.import_module("train_mm_talnet")
    base.EPOCHS = 1
    base.PATIENCE = 1
    main = importlib.import_module("train_mm_talnet_gate_small")
    main.main()
    full = main.OUT
    summary = json.loads((full / "summary.json").read_text(encoding="utf-8"))
    for key in ("five_fold_auroc", "five_fold_auprc"):
        assert len(summary[key]["folds"]) == 5, key
    assert summary["delong_small_minus_proposed"] is None
    assert len(list((full / "weights").glob("fold*.pt"))) == 5
    frame = pd.read_csv(full / "oof.csv")
    assert len(frame) == 80 and np.isfinite(frame["p_mm"]).all()
    for name in ("train_mm_talnet_gate_small_modality", "train_mm_talnet_gate_small_component"):
        module = importlib.import_module(name)
        module.main()
        assert (module.OUT / "oof.csv").exists()
        assert (module.OUT / "summary.json").exists()
    missing = importlib.import_module("eval_missing_robustness")
    missing.OUT.mkdir(parents=True, exist_ok=True)
    missing.run_full(torch.device("cpu"))
    table = importlib.import_module("train_table4_main")
    table.BASELINE_EPOCHS = 1
    table.BASELINE_PATIENCE = 1
    # Synthetic upstream OOF fixtures test fusion wiring without source-note or XML extraction.
    ref = root / "results_mm_talnet"
    ref.mkdir()
    frame.to_csv(ref / "oof.csv", index=False, encoding="utf-8-sig")
    unimodal = root / "results_table3_unimodal"
    unimodal.mkdir()
    fixture = pd.read_csv(root / "features" / "pack" / "text_cls_ids.csv")
    fixture["has_ecg"] = pd.read_csv(root / "features" / "pack" / "ecg_pick.csv")["has_ecg"]
    rng = np.random.default_rng(123)
    for col, flag in (("冻结CLS+线性头", "has_text"), ("浅层1D-CNN", "has_ecg")):
        fixture[col] = np.where(fixture[flag] == 1, rng.uniform(0.1, 0.9, len(fixture)), np.nan)
    fixture.sample(frac=1, random_state=42).to_csv(unimodal / "oof.csv", index=False, encoding="utf-8-sig")
    table.main()
    predictions = pd.read_csv(table.OUT / "oof.csv")
    assert predictions["病案号"].tolist() == frame["病案号"].tolist()
    for col in ("LightGBM", "TabNet", "FT-Transformer", "晚期表+文", "晚期表+心电", "晚期表+文+心电"):
        assert np.isfinite(predictions[col]).all(), col
        assert predictions[col].between(0, 1).all(), col
    for col, mask in (
        ("晚期表+文", predictions["has_text"] == 0),
        ("晚期表+心电", predictions["has_ecg"] == 0),
        ("晚期表+文+心电", (predictions["has_text"] == 0) & (predictions["has_ecg"] == 0)),
    ):
        np.testing.assert_allclose(predictions.loc[mask, col], predictions.loc[mask, "LightGBM"])
    missing.LATE = table.OUT / "oof.csv"
    missing.run_late(frame["病案号"].tolist(), frame["标签"].to_numpy())
    print("PASS: three tabular baselines, shuffled-ID alignment, late fusion, and late-fusion masking.")
    print("PASS: five-fold training, checkpoints, OOF output, modality/component ablations, and missing-modality checks.")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
        return
    source = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="smoke_", dir=source) as tmp:
        root = Path(tmp)
        for path in source.glob("*.py"):
            shutil.copy2(path, root / path.name)
        shutil.copytree(source / "model", root / "model", ignore=shutil.ignore_patterns("__pycache__"))
        generate(root)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-B", str(root / "smoke_test.py"), "--worker", str(root)],
                       cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()

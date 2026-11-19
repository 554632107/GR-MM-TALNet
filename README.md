# GR-MM-TALNet

GR-MM-TALNet is a multimodal model for incomplete clinical inputs. The tabular branch always produces a prediction. When admission text, a 12-lead ECG, or both are available, cross-modal attention and modality-specific gated residuals adjust that prediction. When both are missing, the output falls back to the tabular branch.

The corresponding class is `MM_TALNetGateSmall`.

Table headers and the admission text records are in Chinese, so the code contains Chinese.

## What each file does

| File | Role |
|---|---|
| `model/network.py` | Networks. `MM_TALNetGateSmall` is GR-MM-TALNet; `MM_TALNet` is the shared-residual comparator |
| `model/metrics.py` | AUROC, AUPRC, threshold selection, and classification metrics |
| `model/models.py` | FT-Transformer used in the tabular baseline |
| `model/prepare_ecg.py` | Read ECG XML and convert it to 12×5000 |
| `train_mm_talnet_gate_small.py` | Train GR-MM-TALNet |
| `train_mm_talnet.py` | Train the shared-residual comparator, and provide folds, imputation, and ECG loading |
| `train_mm_talnet_gate_small_modality.py` | Modality ablation: retrain after removing text or ECG |
| `train_mm_talnet_gate_small_component.py` | Component ablation: change one part at a time |
| `eval_missing_robustness.py` | Mask available text or ECG at test time, without retraining |
| `train_table4_main.py` | LightGBM, TabNet, FT-Transformer, and late fusion |
| `train_table3_unimodal.py` | Text head and ECG models used by late fusion |
| `build_new_cohort.py` | 1:3 matching by admission tabular-data completeness |
| `extract_window_features.py` | Extract laboratory values, admission text, and ECG within an observation window of up to 48 hours |
| `pack_for_train.py` | Pack training arrays and extract 768-dimensional text vectors |
| `paths.py` | Source-data paths configured through environment variables |
| `smoke_test.py` | Run training and evaluation checks with synthetic data |

## Installation

```bash
pip install -r requirements.txt
```

Use a PyTorch build compatible with your system and GPU.

## Synthetic-data test

To check the training pipeline with synthetic inputs, run:

```bash
python smoke_test.py
```

The test generates 80 synthetic patients covering all four text/ECG availability patterns. It checks five-fold main-model training, checkpoint and out-of-fold (OOF) output, modality and component ablations, and missing-modality behavior. Neural networks train for one epoch per fold; LightGBM uses its regular settings.

It also trains LightGBM, TabNet, and FT-Transformer, then checks late fusion using shuffled synthetic text and ECG probability tables. These tables test patient alignment and fallback to LightGBM when auxiliary modalities are missing; they are not predictions from trained unimodal models.

Inputs and outputs are created in a temporary directory and removed afterward. Synthetic scores do not represent clinical performance. Source-data extraction and text encoding are outside the test scope.

## Train the main model

Place the following files in `features/pack/`. Row order must match. 

- `text_cls_ids.csv`: `病案号` (patient ID), `标签` (label), and `has_text` (text availability)
- `text_cls.npy`: 768 dimensions per patient. Zeros when text is missing
- `labs_first.npy`, `labs_second.npy`, `labs_switch.npy`: first value, second value, and per-item switch for 48 laboratory items
- `context.npy`: 20 admission-context columns
- `ecg_pick.csv`: `病案号`, `has_ecg` (ECG availability), and `src` (path to a 12×5000 waveform array)

After these files are in place, run from this directory:

```bash
python train_mm_talnet_gate_small.py
```

Results are written to `results_mm_talnet_gate_small/`.

The main model runs independently. If `results_mm_talnet/oof.csv` is available, a comparison with the shared-residual model is also computed after the main results are saved.

The seed is 42, with stratified 5-fold cross-validation and 15% of each training fold held out for early stopping. Training runs for at most 50 epochs and stops after 10 epochs without a gain in validation AUROC. Batch size is 16, learning rate is 5e-4, and weight decay is 1e-2. The loss is fusion BCE plus 0.35 times tabular BCE. Median imputation and standardization use the outer training fold, including the early-stopping validation subset. The outer test fold is excluded.

## Ablations and comparisons

After training the main model, run modality and component ablations with:

```bash
python train_mm_talnet_gate_small_modality.py
python train_mm_talnet_gate_small_component.py
```

To generate the shared-residual comparison and baseline results, run:

```bash
python train_mm_talnet.py
python train_table3_unimodal.py
python train_table4_main.py
python eval_missing_robustness.py
```

The tabular baseline script also requires `features/pack/columns.json`, containing the ordered laboratory and context feature names. The unimodal script uses extracted admission text and ECG XML files in addition to the prepared arrays. Late fusion uses LightGBM probabilities with saved text and ECG predictions. These comparison commands depend on the outputs of preceding scripts.

## Rebuild from source data

Point to the source Excel workbooks, ECG XML, and admission notes with environment variables.

PowerShell example:

```powershell
$env:CA_XLSX = "path to the arrest workbook.xlsx"
$env:HR_XLSX = "path to the control workbook.xlsx"
$env:CA_ECG = "ECG root for arrest patients"
$env:HR_ECG = "ECG root for control patients"
$env:TEXT_DIR = "admission-note JSON directory"
$env:BERT_DIR = "frozen text-encoder directory"
```

Then run in this order:

```bash
python build_new_cohort.py
python extract_window_features.py
python pack_for_train.py
```

The text encoder must be in Hugging Face format and emit a 768-dimensional `[CLS]` vector.

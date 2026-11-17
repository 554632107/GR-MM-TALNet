# GR-MM-TALNet

GR-MM-TALNet is a multimodal model for incomplete clinical inputs. The tabular branch always produces a prediction. When the admission note and the 12-lead ECG are available, cross-modal attention and modality-specific gated residuals adjust that prediction. When both are missing, the output falls back to the tabular branch.

The corresponding class is `MM_TALNetGateSmall`.

Table headers and the admission text records are in Chinese, so the code contains Chinese.

## What each file does

| File | Role |
|---|---|
| `model/network.py` | Networks. `MM_TALNetGateSmall` is GR-MM-TALNet; `MM_TALNet` is the shared-residual comparator |
| `model/metrics.py` | AUROC, AUPRC, Youden, and F1 |
| `model/models.py` | FT-Transformer used in the tabular baseline |
| `model/prepare_ecg.py` | Read ECG XML and convert it to 12×5000 |
| `train_mm_talnet_gate_small.py` | Train GR-MM-TALNet |
| `train_mm_talnet.py` | Train the shared-residual comparator, and provide folds, imputation, and ECG loading |
| `train_mm_talnet_gate_small_modality.py` | Modality ablation: retrain after removing text or ECG |
| `train_mm_talnet_gate_small_component.py` | Component ablation: change one part at a time |
| `eval_missing_robustness.py` | Mask available text or ECG at test time, without retraining |
| `train_mm_talnet_gate_small_sensitivity.py` | Stricter and broader admission-data completeness cohorts |
| `train_table4_main.py` | LightGBM, TabNet, FT-Transformer, and late fusion |
| `train_table3_unimodal.py` | Text head and ECG models used by late fusion |
| `build_new_cohort.py` | 1:3 matching by admission-laboratory completeness |
| `extract_window_features.py` | Extract laboratories, text, and ECG inside the 48-hour window |
| `pack_for_train.py` | Pack training arrays and extract 768-dimensional text vectors |
| `build_sensitivity_cohorts.py` | Completeness sensitivity cohorts |
| `extract_sensitivity_features.py` | Extract features for patients added in the sensitivity cohorts |
| `pack_sensitivity.py` | Pack the sensitivity cohorts into the same arrays |

## Arrays needed to train the main model

Place the following files in `features/pack/`. Row order must match. 

- `text_cls_ids.csv`: patient id, label, and whether text is present
- `text_cls.npy`: 768 dimensions per patient. Zeros when text is missing
- `labs_first.npy`, `labs_second.npy`, `labs_switch.npy`: first value, second value, and per-item switch for 48 laboratory items
- `context.npy`: 20 admission-context columns
- `ecg_pick.csv`: whether ECG is present, and the path to the 12×5000 waveform file

After these files are in place, run from this directory:

```bash
python train_mm_talnet_gate_small.py
```

Results are written to `results_mm_talnet_gate_small/`.

The seed is 42, with stratified 5-fold cross-validation and 15% of each training fold held out for early stopping. Training runs for at most 50 epochs and stops after 10 epochs without a gain in validation AUROC. Batch size is 16, learning rate is 5e-4, and weight decay is 1e-2. The loss is fusion BCE plus 0.35 times tabular BCE. Median imputation and standardization use only the current training fold.

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
python build_sensitivity_cohorts.py
python extract_sensitivity_features.py
python pack_sensitivity.py
```

The text encoder must be in Hugging Face format and emit a 768-dimensional `[CLS]` vector.

## Install

```bash
pip install -r requirements.txt
```

Install `torch` for your computer and GPU.

## Dependencies

See `requirements.txt`.

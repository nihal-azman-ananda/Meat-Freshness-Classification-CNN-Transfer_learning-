# Meat Freshness Classification using CNNs and Transfer Learning

A reproducible deep-learning project for **3-class meat freshness classification**:

- **Fresh**
- **Half-Fresh**
- **Spoiled**

This repository contains both a single-model ResNet-50 workflow and a multi-architecture benchmark protocol with saved metrics, plots, and confusion matrices.

---

## Project Contents

- `MeatFreshness_ResNet50.ipynb` — focused transfer-learning pipeline (ResNet-50) with early stopping, test metrics, and optional CAM visualization.
- `MeatFreshness_final_version.ipynb` — benchmark protocol with:
  - hold-out test split,
  - stratified K-fold cross-validation over multiple split seeds,
  - model zoo comparison,
  - saved confusion matrices and aggregate plots.
- `Meatfreshness_Benchmark.ipynb` — additional benchmarking notebook variant.
- `results/` — exported CSV metrics and visual artifacts.

---

## Dataset

Source dataset (Kaggle):  
<https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset>

### Expected directory structure

```text
Meat Freshness.v1-new-dataset.multiclass/
├── train/
│   ├── images/            # optional
│   └── *.jpg|*.jpeg|*.png|*.bmp|*.webp
└── valid/
    ├── images/            # optional
    └── *.jpg|*.jpeg|*.png|*.bmp|*.webp
```

### Label extraction rule

Labels are inferred from the filename prefix before `-`:
- `FRESH-xxx.jpg`
- `HALF-xxx.jpg`
- `SPOILED-xxx.jpg`

---

## Methodology Summary (from notebooks)

### 1) ResNet-50 protocol (`MeatFreshness_ResNet50.ipynb`)

- **Input size:** 416×416 (paper-aligned default).
- **Split strategy:** 70/30 train-test, with 20% of train reserved for validation.
- **Training:** up to 30 epochs, Adam (`lr=3e-4`), early stopping (`patience=5`, `min_delta=0.01`).
- **Evaluation:** accuracy, macro recall/sensitivity, macro F1, macro specificity, confusion matrix.

### 2) Multi-model benchmark (`MeatFreshness_final_version.ipynb`)

- **Outer hold-out test:** 20% (`OUTER_TEST_SEED=2023`).
- **Cross-validation:** 5-fold stratified CV across **2 split seeds** (`2023`, `42`).
- **Validation for early stopping:** 20% from each fold’s train partition.
- **Training defaults:** 30 epochs, `lr=3e-4`, `weight_decay=1e-4`, AMP enabled, optional weighted sampler.
- **Model zoo:**
  - efficientnet_b0
  - convnext_tiny
  - swin_t
  - resnet50
  - densenet121
  - mobilenet_v3_large
  - efficientnet_v2_s
  - vit_b_16

---

## Results (from `results/`)

### Cross-validation summary (mean ± std)
Source: `results/cv_benchmark_summary.csv`

| Model | Accuracy | Macro Recall | Macro F1 | Macro Specificity |
|---|---:|---:|---:|---:|
| efficientnet_v2_s | 0.9970 ± 0.0024 | 0.9972 ± 0.0022 | 0.9972 ± 0.0022 | 0.9984 ± 0.0013 |
| convnext_tiny | 0.9967 ± 0.0041 | 0.9970 ± 0.0037 | 0.9970 ± 0.0038 | 0.9983 ± 0.0021 |
| efficientnet_b0 | 0.9964 ± 0.0032 | 0.9967 ± 0.0030 | 0.9966 ± 0.0030 | 0.9982 ± 0.0017 |
| densenet121 | 0.9961 ± 0.0030 | 0.9965 ± 0.0027 | 0.9964 ± 0.0028 | 0.9980 ± 0.0015 |
| mobilenet_v3_large | 0.9959 ± 0.0040 | 0.9962 ± 0.0036 | 0.9962 ± 0.0036 | 0.9979 ± 0.0020 |
| resnet50 | 0.9948 ± 0.0042 | 0.9951 ± 0.0039 | 0.9951 ± 0.0038 | 0.9973 ± 0.0022 |
| swin_t | 0.9939 ± 0.0058 | 0.9944 ± 0.0053 | 0.9944 ± 0.0053 | 0.9968 ± 0.0030 |
| vit_b_16 | 0.9625 ± 0.0389 | 0.9652 ± 0.0373 | 0.9643 ± 0.0383 | 0.9805 ± 0.0204 |

### Hold-out test highlights
Source: `results/holdout_results.csv`

- **Best hold-out performer:** `efficientnet_v2_s` with **1.0000** accuracy and macro F1.
- Next top group (accuracy **0.9978**): `efficientnet_b0`, `swin_t`, `densenet121`.

### Visual outputs available

- CV confusion matrices: `results/confusion_matrices_cv/`
- Hold-out confusion matrices: `results/confusion_matrices_holdout/`
- Metric distribution boxplots: `results/boxplots/`

---

## Reproducibility

- Seeded runs with configurable deterministic mode.
- Consistent metric reporting (accuracy, macro recall/sensitivity, macro F1, macro specificity).
- Checkpoint and CSV artifact saving under versioned `results/` and `saved_models/` paths.

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn pillow matplotlib tqdm opencv-python
```

---

## How to Run

1. Download and extract the Kaggle dataset.
2. Open notebook of choice:
   - `MeatFreshness_ResNet50.ipynb` for a focused ResNet-50 experiment.
   - `MeatFreshness_final_version.ipynb` for full benchmark and result export.
3. Update dataset root in the config cell (`ROOT`).
4. Run all cells.

---

## License

This repository is distributed under the terms in `LICENSE`.

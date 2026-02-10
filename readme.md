# Meat Freshness Classification (Fresh / Half-Fresh / Spoiled)

This repository contains a reproducible image-classification workflow for predicting meat freshness into three classes:

- **Fresh**
- **Half-Fresh**
- **Spoiled**

The project includes:
- a focused **ResNet-50 transfer-learning notebook**, and
- a **multi-architecture benchmark notebook** with aggregated results in CSV format.

---

## Repository Structure

- `MeatFreshness_ResNet50.ipynb` — End-to-end transfer-learning pipeline for ResNet-50 (training + evaluation).  
- `Meatfreshness_Benchmark.ipynb` — Comparative benchmark across multiple architectures and seeds.  
- `results/benchmark_raw.csv` — Per-run raw metrics (model × seed).  
- `results/benchmark_summary.csv` — Mean ± std metrics aggregated by model.  
- `LICENSE` — Project license.

---

## Dataset

Kaggle dataset:  
<https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset>

### Expected folder layout

The notebooks support common layouts, including Roboflow-style `images/` subfolders:

```text
Meat Freshness.v1-new-dataset.multiclass/
├── train/
│   ├── images/ (optional)
│   └── *.jpg
└── valid/
    ├── images/ (optional)
    └── *.jpg
```

### Label inference

Class labels are inferred from the filename prefix before `-`.
Examples:
- `FRESH-123.jpg`
- `HALF-456.jpg`
- `SPOILED-789.jpg`

---

## ResNet-50 Pipeline Highlights

- **Backbone:** ImageNet-pretrained ResNet-50
- **Input preprocessing:** Resize to **416×416** + ImageNet normalization
- **Optimization:** Cross-entropy loss + Adam optimizer
- **Regularization/controls:** Early stopping on validation loss, fixed seeds, deterministic option
- **Evaluation:** Accuracy, macro recall/sensitivity, macro F1, macro specificity, confusion matrix

---

## Benchmark Results (from `results/benchmark_summary.csv`)

Metrics below are the **mean ± std over 5 seeds**.

| Model | Accuracy | Macro Recall | Macro F1 | Macro Specificity |
|---|---:|---:|---:|---:|
| efficientnet_b0 | 0.9994 ± 0.0008 | 0.9995 ± 0.0007 | 0.9995 ± 0.0007 | 0.9997 ± 0.0004 |
| densenet121 | 0.9979 ± 0.0013 | 0.9980 ± 0.0012 | 0.9981 ± 0.0012 | 0.9989 ± 0.0007 |
| efficientnet_v2_s | 0.9968 ± 0.0019 | 0.9970 ± 0.0017 | 0.9970 ± 0.0018 | 0.9983 ± 0.0010 |
| mobilenet_v3_large | 0.9965 ± 0.0038 | 0.9967 ± 0.0036 | 0.9966 ± 0.0038 | 0.9982 ± 0.0019 |
| convnext_tiny | 0.9962 ± 0.0022 | 0.9965 ± 0.0021 | 0.9964 ± 0.0021 | 0.9980 ± 0.0012 |
| swin_t | 0.9959 ± 0.0037 | 0.9958 ± 0.0041 | 0.9959 ± 0.0037 | 0.9979 ± 0.0019 |
| resnet50 | 0.9950 ± 0.0046 | 0.9954 ± 0.0041 | 0.9954 ± 0.0042 | 0.9974 ± 0.0024 |
| vit_b_16 | 0.9580 ± 0.0145 | 0.9617 ± 0.0132 | 0.9610 ± 0.0138 | 0.9783 ± 0.0073 |

> These values are read directly from `results/benchmark_summary.csv`.

---

## Quick Start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -U pip
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn pillow matplotlib tqdm opencv-python
```

### Run notebooks

1. Download and extract the dataset from Kaggle.
2. Open either notebook:
   - `MeatFreshness_ResNet50.ipynb` for the single-model pipeline, or
   - `Meatfreshness_Benchmark.ipynb` for multi-model comparison.
3. Set dataset root in the configuration cell (for example `cfg.ROOT = "path/to/dataset"`).
4. Run all cells.

---

## Reproducibility Notes

- Seed control is included (`set_seed(seed, deterministic=...)`).
- Multi-seed evaluation is used to report mean ± std.
- Benchmark CSV files are versioned in `results/` for traceability.


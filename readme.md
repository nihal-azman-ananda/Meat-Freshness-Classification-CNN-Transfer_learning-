# Meat Freshness Classification (Fresh / Half-Fresh / Spoiled) — ResNet-50 Transfer Learning

This repository contains a **reproducible, portfolio-ready** image classification pipeline that predicts meat freshness into three classes: **Fresh**, **Half-Fresh**, and **Spoiled**. The solution fine-tunes an **ImageNet-pretrained ResNet-50** using PyTorch and reports paper-style metrics (confusion matrix, macro sensitivity/specificity) with multi-seed benchmarking.

## Key Features
- **Model:** ResNet-50 (ImageNet pretrained) fine-tuned for 3-class classification
- **Preprocessing:** resize to **416×416** + ImageNet normalization
- **Training:** Adam optimizer + early stopping on validation loss
- **Evaluation:** confusion matrix, classification report, **macro sensitivity & specificity**
- **Reproducibility:** fixed seeds, optional deterministic mode, multi-seed mean ± std
- **Interpretability (optional):** Class Activation Maps (CAM) overlays

---

## Dataset
Kaggle dataset link:
https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset

### Expected folder layout
The notebook supports common folder layouts, including Roboflow style (`images/` subfolder):

```bash
Meat Freshness.v1-new-dataset.multiclass/
train/
(images/) *.jpg
valid/
(images/) *.jpg
```

**Label inference:** class name is read from the filename prefix before `-`  
Examples: `FRESH-123.jpg`, `HALF-456.jpg`, `SPOILED-789.jpg`.

---

## Method Overview
1. **Preprocessing**
   - Resize images to 416×416
   - Normalize using ImageNet mean/std
2. **Transfer Learning**
   - Initialize ResNet-50 with ImageNet weights
   - Replace final fully connected layer for 3 classes
3. **Optimization**
   - Cross entropy loss
   - Adam optimizer
   - Early stopping using validation loss (`patience`, `min_delta`)
4. **Evaluation**
   - Confusion matrix + precision/recall/F1 report
   - Macro **sensitivity (recall)** and macro **specificity** (one-vs-rest)
   - Multi-seed benchmarking (mean ± std)

---

## Results (from executed notebook)
### Single run (seed = 2023, stratified 70/30 split)
- **Test Accuracy:** **0.9897** (680 test images)
- **Macro Sensitivity (Recall):** **0.9909**
- **Macro Specificity:** **0.9947**

**Confusion Matrix (rows = true, cols = predicted)**

| True \ Pred | FRESH | HALF | SPOILED |
|---:|---:|---:|---:|
| **FRESH**   | 249 | 7   | 0   |
| **HALF**    | 0   | 237 | 0   |
| **SPOILED** | 0   | 0   | 187 |

Main error mode: **Fresh → Half-Fresh** (7 cases). Half-Fresh and Spoiled are otherwise classified cleanly in this split.

### Multi-seed benchmark (5 seeds)
Mean ± std across 5 runs:
- **Accuracy:** 0.9879 ± 0.0070
- **Macro Recall / Sensitivity:** 0.9890 ± 0.0062
- **Macro F1:** 0.9887 ± 0.0068
- **Macro Specificity:** 0.9938 ± 0.0034

> Note: Metrics depend on dataset organization, split, seed, and environment versions. This repository includes utilities to reproduce runs and report mean ± std.

---

## Repository Contents
- `MeatFreshness_ResNet50_Professional.ipynb` — end-to-end pipeline (data → training → evaluation → optional CAM)
- `saved_models/` — best checkpoints (created after training)

---

## Setup
### Option A: pip (recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -U pip
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn pillow matplotlib tqdm opencv-python

## How to Run

1. Download the dataset from Kaggle and extract it locally.
2. Open `MeatFreshness_ResNet50_Professional.ipynb`.
3. Set the dataset path in the config:
   ```python
   cfg.ROOT = "path/to/your/dataset/root"
4. Run all cells.

Reproducibility Notes

Seeds are controlled via set_seed(seed, deterministic=...).

Multi-seed evaluation is included to report mean ± std.

Optional duplicate leakage check (MD5) is included to detect exact duplicates across splits.

Interpretability (Optional)

The notebook includes Class Activation Maps (CAM) to visualize which regions of an image most influenced the predicted class. This is helpful for debugging and for demonstrating model reasoning in a portfolio context.
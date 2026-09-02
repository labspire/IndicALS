# ALS Speech Dataset – Technical Validation Experiments

This repository contains the official codebase for the technical validation experiments reported on the multi-lingual Indian ALS speech dataset. 

It provides end-to-end processing pipelines for subject-independent 5-fold cross-validation to evaluate binary classification between **ALS patients** and **Healthy Controls (HC)** across multiple speech tasks.

---

## 🧪 Speech Tasks & Experiments

The codebase is organized by distinct acoustic protocol tasks:

* **DDK**: Diadochokinetic speech tasks (`/experiments/DDK`)
* **SUV**: Sustained vowel tasks, excluding fricatives (`/experiments/SUV`)
* **SUF**: Sustained fricative tasks (`/experiments/SUF`)

---

## ⚙️ Experimental Methodology & Pipeline

1. **Preprocessing & Resampling**: Raw audio files are resampled to **8 kHz**.
2. **Speech Segmentation**: Task-specific segments are isolated using temporal annotations and segmented into **2-second chunks** with **50% overlap** (shorter chunks are zero-padded).
3. **Normalization**: Signal amplitudes undergo zero-mean / unit-variance normalization.
4. **Framing**: Framing is performed using a **20 ms window** with **50% overlap**.
5. **Feature Representation & Modeling**: Spatial-temporal sequence processing via a **1D ConvLSTM** network.
6. **Cross-Validation**: Subject-independent 5-fold cross-validation to prevent data leakage across subjects.
7. **Evaluation**: Chunk-level predictions are aggregated using **file-level majority voting** to yield final subject classifications.

---

## 🧠 Model Architecture

The binary classification model uses a custom 1D ConvLSTM architecture:

| Layer | Specifications / Configuration |
| :--- | :--- |
| **Conv1D (1)** | 256 filters, Kernel size: 120, ReLU, Batch Normalization, Max Pooling |
| **Conv1D (2)** | 30 filters, Kernel size: 20, ReLU, Max Pooling |
| **Recurrent** | 3 Stacked Unidirectional LSTM layers (150 hidden units each) |
| **Output** | Fully Connected (FC) layer with 2 output classes (ALS vs. HC) |

---

## 🏋️ Training Parameters

The default hyperparameter configuration used across experiments:

* **Optimizer**: Adam (`lr = 0.001`)
* **Batch Size**: `4`
* **Max Epochs**: `10`
* **Loss Function**: Cross-Entropy Loss
* **Validation Strategy**: 5-Fold Subject-Independent Cross-Validation

---

## 🔒 Data Access & Setup Guidelines


To run these experiments:
1. Request and obtain access through the official dataset authorization process.
2. Organise your local directory paths to point to:
   * **ALS** audio directory
   * **HC** audio directory
   * **ALS 5-Fold Split** spreadsheet (`ALS_5fold_split.xlsx`)
   * **HC 5-Fold Split** spreadsheet (`HC_5fold_split.xlsx`)


## 🚀 Running Experiments

Run the technical validation script for your desired task from the root directory:

```bash
# Run DDK experiment
python experiments/DDK/train_ddk.py

# Run Sustained Vowel (SUV) experiment
python experiments/SUV/train_suv.py

# Run Sustained Fricative (SUF) experiment
python experiments/SUF/train_suf.py

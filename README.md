# IndicALS: A Multi-Lingual Indian ALS Speech Dataset

This repository contains the official experimental codebase for the technical validation of the **IndicALS** dataset.

## 🧪 Experiments & Tasks

The technical validation consists of two primary classification setups across three acoustic speech protocols (`DDK`, `SUV`, and `SUF`):

1. **ALS vs. HC Classification (`/ALS vs HC Classification`)**


2. **ALS Severity Classification (`/Severity Classification`)**
 
---

## 📁 Repository Organization

```text
IndicALS/
├── ALS vs HC Classification/
│   ├── data/
│   │   ├── ALS_5fold_split.xlsx   # 5-fold cross-validation split for ALS subjects
│   │   ├── HC_5fold_split.xlsx    # 5-fold cross-validation split for HC subjects
│   │   └── README.md              # Dataset structure & annotation specifications
│   ├── experiments/               # Task-specific validation scripts (DDK, SUV, SUF)
│   ├── .gitignore
│   ├── README.md                  # Detailed binary classification documentation
│   └── requirements.txt           # Python dependencies
└── Severity Classification/
    ├── experiments/               # Severity grading scripts

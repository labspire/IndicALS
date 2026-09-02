# IndicALS Dataset & Reproduction Guide

This repository contains the official dataset structure and cross-validation splits to accurately replicate the baseline experiments and results reported in our paper.

The dataset includes raw speech audio recordings along with forced-alignment temporal annotations designed for automatic detection and severity classification of Amyotrophic Lateral Sclerosis (ALS) and motor speech disorders.


# 📁 Repository Structure

The dataset is organized hierarchically by group (`ALS` / `HC`), subject ID (e.g., `PTSPASPIRE001`), and acoustic task protocols (`DDK`, `IMG`, `SUF`, `SUV`):

```text
publication_data/
├── ALS/
│   ├── PTSPASPIRE001/
│   │   ├── DDK/
│   │   │   ├── PTSPASPIRE001_ZOOM_DDK_BDG_U1.wav
│   │   │   ├── PTSPASPIRE001_ZOOM_DDK_BDG_U1.txt
│   │   │   ├── PTSPASPIRE001_ZOOM_DDK_TA_U1.wav
│   │   │   └── ...
│   │   ├── SUF/
│   │   │   ├── PTSPASPIRE001_ZOOM_SUF_F_U1.wav
│   │   │   └── ...
│   │   └── SUV/
│   │       ├── PTSPASPIRE001_ZOOM_SUV_A_U1.wav
│   │       └── ...
│   └── PTSPASPIRE002/
│       └── ...
├── HC/
│   ├── PTSPASPIRE050/
│   │   ├── DDK/
│   │   └── ...
│   └── ...
├── ALS_5fold_split.xlsx
└── HC_5fold_split.xlsx

2. Annotation Files (.txt)
Every .wav file is paired with an identical-named .txt file containing forced-alignment start and end timestamps (in seconds) for speech segments:
1.869590	7.714921	S1
8.390262	13.221078	S2
13.997010	17.936029	S3


3. Subject-Independent 5-Fold Cross-Validation
Subject assignments are kept consistent across all task subfolders to prevent data leakage:

ALS_5fold_split.xlsx: Defines Train, Validation, and Test subject assignments across 5 folds for ALS patients.

HC_5fold_split.xlsx: Defines Train, Validation, and Test subject assignments across 5 folds for Healthy Controls.

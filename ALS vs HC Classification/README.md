# ALS Speech Dataset – Technical Validation Experiments

This repository contains the experimental code used for the technical validation of an ALS speech dataset.

## Experiments

The repository is organized into three speech-task experiments:

- DDK – diadochokinetic speech
- SUV – sustained vowel tasks, excluding fricatives
- SUF – sustained fricative tasks

The experiments perform subject-independent 5-fold cross-validation for binary ALS vs healthy control (HC) classification.

## Methodology

The experimental pipeline includes:

1. Audio resampling to 8 kHz.
2. Extraction of task-specific speech segments.
3. Segmentation into 2-second chunks with 50% overlap.
4. Zero-padding of shorter chunks.
5. Zero-mean/unit-variance normalization.
6. 20 ms frame length with 50% overlap.
7. ConvLSTM-based binary classification.
8. Subject-independent 5-fold cross-validation.
9. Chunk-level prediction followed by file-level majority voting.

## Model

The ConvLSTM architecture contains:

- Conv1D: 256 filters, kernel size 120
- ReLU, batch normalization, and max pooling
- Conv1D: 30 filters, kernel size 20
- ReLU and max pooling
- Three stacked unidirectional LSTM layers with 150 hidden units each
- Fully connected layer with 2 output classes

## Training

The experimental scripts use:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 4
- Maximum epochs: 10
- Loss: Cross-entropy
- Subject-independent 5-fold cross-validation

## Data

The clinical speech data and subject split spreadsheets are **not included in this repository** because they are not intended for public redistribution.

Obtain the data only through the authorized dataset/data-access procedure.

Expected local organization should be configured by the user in the experiment scripts or through command-line arguments/environment variables.

## Running an experiment

Example:

```bash
python experiments/DDK/train_ddk.py
python experiments/SUV/train_suv.py
python experiments/SUF/train_suf.py
```

Before running, configure the paths to:

- ALS audio data
- HC audio data
- ALS 5-fold subject split
- HC 5-fold subject split

Do not commit private server paths, patient data, split spreadsheets, model checkpoints, or logs containing sensitive information.

## Results

Aggregate results can be placed in `results/`. Raw experiment outputs and model checkpoints should normally remain outside version control.

## Reproducibility

For reproducibility, record the Python version, package versions, random seeds, hardware, and final training configuration used for each experiment.

## Important implementation note

Before publication of this repository, ensure that the implementation and manuscript describe the same training procedure. In particular, verify the manuscript's statements about L2 weight decay and early stopping against the final code.

# Computational Performance

> ⚠️ This page is a work in progress.

This page contains some example computational performance statistics for TemporAI tasks. It will be expanded we test more models, datasets, and hardware.



## Measurement details
- TemporAI version: `0.0.3`
- Last tested: `Nov 2023`
- Hardware:
    - CPU: `10-core Intel Core i9-10900K`
    - RAM: `64 GB`
    - GPU: `NVIDIA GeForce RTX 3090`

GPU used whenever supported by a model.



## Table of results
| Method | Dataset | Action(s) | Execution Time (seconds) |
|------|------|------|------------------------|
| Preprocessing : Imputation : ts_tabular_imputer (imputer=mice)  |  sine (n=300)  |  Fit and transform | 5.16 |
| Preprocessing : Imputation : ts_tabular_imputer (imputer=mean)  |  sine (n=300)  |  Fit and transform | 0.57 |
| Survival Analysis : dynamic_deephit (n_iter=100)  |  PBC (n=312)  |  Fit and predict | 3.56 |
| Treatment Effects : Temporal : crn_regressor (epochs=100)  |  Dummy dataset (n=300)  |  Fit | 327.57 |
| Prediction : One-off : nn_classifier (mode="transformer", n_iter=100)  |  Dummy dataset (n=300)  |  Fit and predict | 0.81 |

*More cases to be added...*

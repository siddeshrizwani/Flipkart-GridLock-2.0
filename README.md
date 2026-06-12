# GridLock 2.0 - Traffic Demand Forecast

## Overview

Traffic demand forecasting solution for the Flipkart GridLock 2.0 challenge.

**Final Offline Score:** 93.22

---

## Problem Statement

### Goal

Predict traffic demand for:

- Geohash
- Timestamp

### Metric

```python
score = max(0, 100 * r2_score(actual, predicted))
```

---

## Dataset

### train.csv

| Property | Value |
|-----------|---------|
| Rows | 77,299 |
| Day 48 | Full day |
| Day 49 | 00:00–02:00 |

### test.csv

| Property | Value |
|-----------|---------|
| Rows | 41,778 |
| Day | 49 |
| Time Range | 02:15–13:45 |

---

## Key Insight

Day 48 contains the complete demand curve for each geohash.

Therefore:

```text
Day 49 Demand
≈
Day 48 Profile
+
Residual Correction
```

---

## Solution Pipeline

```text
train.csv
    ↓
Feature Engineering
    ↓
SVD Denoised Profile
    ↓
Residual Modeling
    ↓
Ensemble Prediction
    ↓
submission.csv
```

### Components

- Geohash decoding
- KMeans clustering
- Neighbor aggregation
- SVD denoising
- Residual learning
- Ensemble blending

---

## Models

| Model | Weight |
|---------|---------|
| LightGBM | 0.40 |
| XGBoost | 0.18 |
| CatBoost | 0.12 |
| HistGBM | 0.11 |
| ExtraTrees | 0.09 |
| CatBoost-Categorical | 0.10 |

---

## Results

| Version | Score |
|----------|---------|
| Baseline | 83 |
| Day48 Features | 89.84 |
| Residual Learning | 90.97 |
| Full Ensemble | 91.27 |
| Final Model | **93.22** |

---

## Reproducibility

Only these files are used:

- train.csv
- test.csv

No external data is used during training.

---

## Run Commands

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost pygeohash

python build_submission.py
```

---

## Repository Structure

```text
build_submission.py
Traffic_Demand_Forecast.ipynb
offline_score.py
submission.csv
README.md
```

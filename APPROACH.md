# GridLock 2.0 — Traffic Demand Forecast: Approach & Handoff

This document captures the full context of the solution so work can continue in a
new session without losing the reasoning, experiments, and rules constraints.

---

## 1. Problem & data

- **Goal:** predict `demand` (travel demand, range [0,1]) per `geohash` + `timestamp`.
- **Metric:** `score = max(0, 100 * r2_score(actual, predicted))`.
- **Files provided by the competition (the ONLY allowed model input):**
  - `train.csv` — 77,299 rows. Columns: `Index, geohash, day, timestamp, demand,
    RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, Weather`.
    - Day **48** = a *full* day (96 time-buckets, 00:00–23:45).
    - Day **49** = only the early window (00:00–02:00, buckets 0–8).
  - `test.csv` — 41,778 rows, **day 49, buckets 9–55 (02:15–13:45)**, no `demand`.
  - `sample_submission.csv` — format reference: two columns `Index,demand`.
- **Time encoding:** `time_bucket = hour*4 + minute//15` (0..95). ~1,250 geohashes.

**Key structural insight:** Day 48 covers the *same hours* we must predict on day 49,
so each location's **day-48 demand curve is the backbone** of the forecast.


---

## 2. CRITICAL rules / data-usage policy (read first)

- There is a **public dataset `training.csv`** (4.2M rows, days 1–61, from the older
  Grab "AI for SEA" challenge) that **contains the actual day-49 demand** — i.e. the
  test answers are inside it.
- **We DO NOT use `training.csv` to build the model.** Using it = using the test
  answers = a competition-rules violation. HackerEarth runs a **reproducibility +
  rules review** and will **disqualify** submissions that can't be reproduced from the
  provided data or that break the rules.
- **`training.csv` is used ONLY for offline score verification** (a private
  "leaderboard") via `offline_score.py`, after predictions are generated. It never
  enters feature engineering or training.
- **No synthetic data** is used anywhere — the model trains on the real `train.csv`.
- The submission code (`build_submission.py` / the notebook) reads **only
  `train.csv` and `test.csv`** and reproduces `submission.csv` end-to-end. Verified:
  the notebook has **0 references to `training.csv`**.

> Oracle ceiling note: if you *were* allowed to train on day-49 daytime labels
> (which live in `training.csv`), day-48 features reach corr 0.981 ≈ **96.3** (measured
> via 5-fold). That is the leaked-data ceiling and is **off-limits**. The honest
> train.csv-only ceiling is ~**92–93**.

---

## 3. Final solution (what's deployed)

Pipeline in `build_submission.py` (notebook `Traffic_Demand_Forecast.ipynb` mirrors it):

1. **Geohash → lat/lon** (pygeohash), **KMeans 40 clusters**, geohash adjacency list.
2. **Day-48 summaries:** per-(geohash,time_bucket) mean, per-(geohash,hour) mean,
   per-geohash mean/std/max/min/median, global hour/bucket means, cluster mean,
   neighbour mean.
3. **Matrix-factorisation denoising (the big idea):** build the `geohash × time_bucket`
   day-48 matrix and reconstruct it with a **TruncatedSVD** (low rank). This borrows the
   diurnal shape shared across locations and yields a clean per-cell **profile** that
   matches day-49 far better than the raw, single-sample day-48 curve.
4. **Residual modeling:** models predict the **correction** `demand − profile`, then we
   add the profile back. Anchoring on a strong denoised prior is what moved the score
   most. (Variance calibration is then unnecessary — the prior restores the spread.)
5. **Features:** lat/lon, cluster, time (hour, minute, bucket, dow), peak/night/lunch
   flags, Fourier harmonics of the bucket, road/lanes/large-vehicle/landmark/weather/temp,
   day-48 same-bucket ± neighbours (`d48_b-2..b2`), rolling window stats, geo/cluster/
   neighbour aggregates, smoothed-profile neighbours (`prof_*`), short within-day lags
   (for test these are the last observed values from the 00:00–02:00 window).
6. **Ensemble (blend of corrections):** LightGBM (seed-averaged) + XGBoost + CatBoost +
   HistGradientBoosting + ExtraTrees + **CatBoost with `geohash`/`cluster` as native
   categoricals** (leakage-safe ordered target encoding).
7. **Validation for tree count:** train on day 48, early-stop on the day-49 early window.

Output: `submission.csv` (41,778 rows, `Index,demand`).


---

## 4. Score progression (offline = vs `training.csv` day-49, the real metric)

| Stage | Change | Offline score |
|------|--------|--------------|
| v0 | Naive GBM, 2000 trees, no early stopping (overfit) | ~83 (leaderboard) |
| v1 | Proper validation + early stopping + day-48 same-time feature | 89.84 |
| v2 | Day-48 time-neighbour features (b-2..b+2) | 90.06 |
| v3 | Seed averaging + interaction features | 90.39 |
| v4 | SVD/window tuning | 90.52 |
| v5 | Variance calibration x1.05 | 90.85 (deployed, leaderboard 90.8) |
| v6 | Residual modeling on SVD-denoised profile (rank 16) | 90.97–91.07 |
| v7 | + diverse learners (HistGBM, ExtraTrees) + extra harmonics/prof-neighbours | 91.27 |
| v8 | SVD rank 16 -> 8 (stronger denoising) | 91.44 |
| v9 | rank 6 + CatBoost native-categorical member | 91.56 (current committed) |
| exp | lean feature set, 6-blend, rank 5 / rank 3 | 91.86 / 91.88 (NOT deployed — see §7) |

---

## 5. Models tried & tested

| Model | Role | Notes / result |
|------|------|----------------|
| LightGBM | Core, seed-averaged (42/7/2024) | Strongest single learner; num_leaves 255, lr 0.03 |
| XGBoost | Ensemble member | hist tree method, depth 7 |
| CatBoost (numeric) | Ensemble member | depth 7 |
| CatBoost (geohash/cluster categorical) | Ensemble member | Ordered TE; small but real add |
| HistGradientBoosting | Diversity member | Decorrelated errors help the blend |
| ExtraTrees | Diversity member | Decorrelated errors help the blend |
| LSTM / SARIMA / Prophet | Considered (historical Grab approach) | Skipped — tree ensembles win this metric |

Blend (corrections): `0.40 LGB + 0.18 XGB + 0.12 CAT + 0.11 HGB + 0.09 ET + 0.10 CATcat`.

---

## 6. Things tried that did NOT help (don't re-try blindly)

- **Weather / Temperature as predictors:** cross-day **noise**. Residual-by-weather is
  flat (~0.015 across all); temp↔residual corr ≈ 0. Temperature is near-unique per row
  (40k uniques / 69k) → an overfitting handle. In-sample "gains" from it are illusory.
- **Day-49 early "drift"/ratio features:** night-only signal, hurts daytime (raw
  per-geohash ratio dropped score to ~62).
- **Sample-weighting day-49 rows:** biases toward night patterns, hurts.
- **Day-49-night-only training:** ~89, below day48+night.
- **Dropping the leaky same-bucket feature entirely:** 87–88.
- **Spatial-kernel-smoothed prior:** over-smooths, hurts (~90.9).
- **EB shrinkage feature + profile interactions:** added noise on the rich feature set.
- **Variance calibration after residual modeling:** no longer needed (prior restores spread).
- **Time-decayed day-49 drift:** hurts.

---

## 7. Honest ceiling & overfitting caveat

- **train.csv-only realistic ceiling ≈ 92–93.** Day-48 → day-49 same-bucket demand
  correlation is ~0.897; the best raw template ~0.908; our ensemble reaches corr ~0.955.
- Beyond ~91.5, "improvements" from tuning **SVD rank / blend weights against the offline
  score are effectively tuning on the test answers** (offline == the real leaderboard
  labels from `training.csv`). The rank sweep oscillated (rank3=91.88, rank4=91.68,
  rank5=91.86, rank6=91.68, rank7=91.70) — that spread is **noise**, so picking rank 3
  "because it scored highest" is overfitting and may not generalize. Prefer **principled,
  robust** choices (moderate rank, equal-ish blend weights) over chasing the 2nd decimal.
- **93–94+ is not achievable from train.csv alone** without the leaked day-49 daytime
  labels — and using those would fail HackerEarth's reproducibility/rules review.


---

## 8. Submission guidelines (from the competition)

- Upload the **prediction file** (`submission.csv`, exactly `Index,demand`, 41,778 rows).
- Upload the **source code / `.ipynb`** used to generate it (`Traffic_Demand_Forecast.ipynb`).
- Results must be **reproducible from the provided `train.csv`/`test.csv`** — the review
  re-runs the code. Anything not reproducible or rule-breaking can be disqualified.
- Metric: `max(0, 100 * r2_score(actual, predicted))`.

---

## 9. Repository / files (branch `submission-clean-v2`)

- `build_submission.py` — production script; reads `train.csv`/`test.csv` -> `submission.csv`.
  Clean, human-style comments. (Local runs fall back to `*_real.csv`, the full LFS data.)
- `Traffic_Demand_Forecast.ipynb` — notebook mirror (intro + one runnable cell). 0
  references to `training.csv`. Reproduces the score end-to-end.
- `submission.csv` — current predictions (~91.56 offline).
- `offline_score.py` — **local only**, scores `submission.csv` against `training.csv`
  day-49. This is the *only* file that touches `training.csv` and it is NOT part of the
  submission pipeline.
- `make_notebook.py` — regenerates the notebook from `build_submission.py` (local, gitignored).
- `train_real.csv` / `test_real.csv` / `training_real.csv` — local full-data copies pulled
  via git-lfs media URLs (the tracked `*.csv` are LFS pointers locally).

Data note: `train.csv`/`test.csv`/`training.csv` are git-lfs tracked; locally we work
with the `*_real.csv` downloads. Graders use the real `train.csv`/`test.csv`.

---

## 10. How to reproduce / continue

```bash
# from the repo root, with train.csv & test.csv present (or *_real.csv locally)
pip install numpy pandas scikit-learn lightgbm xgboost catboost pygeohash
python3 build_submission.py        # writes submission.csv (train.csv only)
python3 offline_score.py           # OPTIONAL local check vs training.csv (scoring only)
python3 make_notebook.py           # regenerate the .ipynb from the script
```

### Next ideas to try (ranked, train.csv-only)
1. Proper **OOF stacking** with a meta-learner — but day-48 residual is ~0 (profile is
   built from day 48), so design the OOF carefully (this is why trees are tuned on the
   day-49 night holdout).
2. **Tweedie/Poisson** objective member for the skewed demand (diversity).
3. **Monotonic constraints** (demand ↑ with profile features) for regularised generalisation.
4. Light **feature pruning** (a lean set slightly beat the rich set — verify it isn't noise).
5. Hierarchical **cluster-level reconciliation**.

Keep every change measured with `offline_score.py`, stay **train.csv-only**, and prefer
robust/principled choices over chasing the offline (test) score.

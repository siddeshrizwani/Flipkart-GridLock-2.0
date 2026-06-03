"""
Flipkart GridLock 2.0 - Traffic Demand Forecast
================================================
Production submission builder.

USES ONLY: train.csv (77,299 rows, days 48-49) + test.csv (41,778 rows, day 49).
NEVER reads training.csv / training_real.csv (those contain the day-49 answers and
are off-limits per the competition rules - see approach.md section 2).

Core idea (approach.md section 3):
  1. Build the geohash x time_bucket matrix from day 48.
  2. Denoise it with a low-rank TruncatedSVD -> a clean per-cell "profile" that
     borrows the diurnal shape shared across locations. This transfers to day 49
     far better than each location's raw, single-sample day-48 curve.
  3. Models predict the RESIDUAL (demand - profile); we add the profile back.
     Anchoring on a strong prior is what moves the score most, and it keeps the
     predicted variance correct without any post-hoc calibration.
  4. Blend diverse learners (LightGBM seed-avg + XGBoost + CatBoost +
     HistGradientBoosting + ExtraTrees) on the residual.

Validation: the day-49 early window (buckets 0-8) is the only labeled cross-day
signal in the provided data, so we use it to choose tree counts (early stopping)
and to report an honest cross-day R2. We do NOT tune SVD rank against it (it is
night-only; see approach.md section 7) - a moderate rank is used on principle.

Metric: score = max(0, 100 * r2_score(actual, predicted)).
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')
RNG = 42
np.random.seed(RNG)

# SVD rank for profile denoising. Chosen by leak-free matrix-completion CV on
# day-48 DAYTIME cells (buckets 9-55, the test region) - see _work/profile_cv.py.
# Soft-impute has a broad, flat optimum across ranks 6-10 (R2 ~0.953-0.954); we
# pick rank 6 as the robust hedge: top of the daytime CV AND low enough to suit
# the night cross-day transfer. We do NOT chase the 2nd decimal (approach.md s7).
SVD_RANK = 6
SOFT_IMPUTE_ITERS = 12
N_CLUSTERS = 40


# ----------------------------------------------------------------------------
# Data loading (graceful fallback to the *_real.csv / _work/dataset copies that
# exist locally because the tracked train.csv/test.csv are git-LFS pointers).
# ----------------------------------------------------------------------------
def _resolve(*candidates):
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 1000:
            return c
    return candidates[0]


def load_data():
    train_path = _resolve('train.csv', 'train_real.csv', '_work/dataset/train.csv')
    test_path = _resolve('test.csv', 'test_real.csv', '_work/dataset/test.csv')
    print(f"  train <- {train_path}")
    print(f"  test  <- {test_path}")
    assert 'training' not in train_path, "Refusing to read training.csv (rules violation)."
    return pd.read_csv(train_path), pd.read_csv(test_path)


def add_time(df):
    df['hour'] = df['timestamp'].str.split(':').str[0].astype(int)
    df['minute'] = df['timestamp'].str.split(':').str[1].astype(int)
    df['tb'] = df['hour'] * 4 + df['minute'] // 15
    return df


# ----------------------------------------------------------------------------
# Geohash geometry: lat/lon, spatial clusters, 8-neighbour adjacency.
# ----------------------------------------------------------------------------
def build_geo(train, test):
    geos = sorted(set(train['geohash']) | set(test['geohash']))
    geo_xy = {g: (pgh.decode(g).latitude, pgh.decode(g).longitude) for g in geos}
    coords = np.array([geo_xy[g] for g in geos])

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RNG, n_init=10)
    clusters = dict(zip(geos, km.fit_predict(coords)))

    dirs = ['top', 'bottom', 'left', 'right',
            'topleft', 'topright', 'bottomleft', 'bottomright']
    gset = set(geos)
    nbrs = {}
    for g in geos:
        try:
            nbrs[g] = [n for n in (pgh.get_adjacent(g, d) for d in dirs) if n in gset]
        except Exception:
            nbrs[g] = []
    return geo_xy, clusters, nbrs


# ----------------------------------------------------------------------------
# The denoised day-48 profile: geohash x time_bucket matrix -> iterative low-rank
# SVD imputation (soft-impute). Re-filling the missing cells with the
# reconstruction each round recovers the daytime diurnal shape markedly better
# than a single-pass SVD over a geo-mean-filled matrix (validated leak-free via
# matrix-completion CV in _work/profile_cv.py: 0.954 vs 0.936 R2 on day-48
# daytime cells).
# ----------------------------------------------------------------------------
def build_profile(d48, geo_mean, rank, iters=SOFT_IMPUTE_ITERS):
    mat = d48.pivot_table(index='geohash', columns='tb', values='demand', aggfunc='mean')
    for b in range(96):
        if b not in mat.columns:
            mat[b] = np.nan
    mat = mat[sorted(mat.columns)]
    gb_mean = d48.groupby('tb')['demand'].mean()
    fallback = gb_mean.mean()

    M = mat.values
    known = ~np.isnan(M)
    # initial fill: each location's own mean (global fallback for empty rows)
    row_mean = np.where(np.isnan(np.nanmean(np.where(known, M, np.nan), axis=1)),
                        fallback, np.nanmean(np.where(known, M, np.nan), axis=1))
    filled = np.where(known, M, row_mean[:, None])

    def _svd(X):
        svd = TruncatedSVD(n_components=rank, random_state=RNG)
        return svd.fit_transform(X) @ svd.components_

    rec = filled
    for _ in range(iters):
        rec = _svd(filled)
        filled = np.where(known, M, rec)   # keep observed, replace missing
    rec = _svd(filled)
    prof = pd.DataFrame(rec, index=mat.index, columns=mat.columns)

    # Long lookup dict {(geohash, bucket): profile_value}
    prof_dict = {(g, b): prof.loc[g, b] for g in prof.index for b in prof.columns}
    return prof, prof_dict, gb_mean, fallback


def profile_lookup(prof_dict, gb_mean, fallback, geo_mean, g, b, offset=0):
    bb = b + offset
    v = prof_dict.get((g, bb))
    if v is not None:
        return v
    if bb in gb_mean.index and g in geo_mean.index:
        # scale global bucket shape by this location's level
        gnight = gb_mean.mean()
        return gb_mean[bb] * (geo_mean[g] / gnight) if gnight > 0 else gb_mean[bb]
    return geo_mean.get(g, fallback)


# ----------------------------------------------------------------------------
# Feature engineering. `profile` is added back after prediction, so the models
# only ever fit the residual (demand - profile).
# ----------------------------------------------------------------------------
ROAD_MAP = {'Residential': 0, 'Street': 1, 'Highway': 2}
WEATHER_MAP = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3, 'Cloudy': 4}


def make_features(df, geo_xy, clusters, nbrs, d48_stats, prof_dict, gb_mean,
                  fallback, geo_mean):
    geo_mean_d, geo_std_d, geo_max_d, geo_min_d, geo_med_d, \
        cluster_mean_d, neighbor_mean_d, d48_bucket = d48_stats

    out = pd.DataFrame(index=df.index)
    g = df['geohash'].values
    b = df['tb'].values

    out['lat'] = [geo_xy.get(x, (0, 0))[0] for x in g]
    out['lon'] = [geo_xy.get(x, (0, 0))[1] for x in g]
    out['cluster'] = [clusters.get(x, 0) for x in g]

    out['hour'] = df['hour'].values
    out['minute'] = df['minute'].values
    out['tb'] = b
    # NOTE: deliberately NO day-of-week feature. With only days 48-49 it is a
    # perfect proxy for "which day" (dow=5 -> day48, dow=6 -> day49/test) and the
    # day<->time-of-day coupling flips between train (day49=night) and test
    # (day49=daytime), so it would route night-only corrections onto daytime rows.
    out['is_morning_rush'] = ((out['hour'] >= 7) & (out['hour'] <= 9)).astype(int)
    out['is_evening_rush'] = ((out['hour'] >= 17) & (out['hour'] <= 19)).astype(int)
    out['is_night'] = ((out['hour'] >= 22) | (out['hour'] <= 5)).astype(int)
    out['is_lunch'] = ((out['hour'] >= 11) & (out['hour'] <= 13)).astype(int)

    # Fourier harmonics of the 96-bucket day
    for k in (1, 2, 3, 4):
        out[f'tb_sin{k}'] = np.sin(2 * np.pi * k * b / 96)
        out[f'tb_cos{k}'] = np.cos(2 * np.pi * k * b / 96)

    # Static infrastructure
    out['road'] = df['RoadType'].map(ROAD_MAP).fillna(-1).values
    out['lanes'] = pd.to_numeric(df['NumberofLanes'], errors='coerce').fillna(-1).values
    out['large_veh'] = (df['LargeVehicles'] == 'Allowed').astype(int).values
    out['landmark'] = (df['Landmarks'] == 'Yes').astype(int).values
    out['weather'] = df['Weather'].map(WEATHER_MAP).fillna(-1).values
    out['temp'] = pd.to_numeric(df['Temperature'], errors='coerce').fillna(16.5).values

    # Day-48 aggregates per location / cluster / neighbours
    out['geo_mean'] = [geo_mean_d.get(x, fallback) for x in g]
    out['geo_std'] = [geo_std_d.get(x, 0.0) for x in g]
    out['geo_max'] = [geo_max_d.get(x, fallback) for x in g]
    out['geo_min'] = [geo_min_d.get(x, 0.0) for x in g]
    out['geo_med'] = [geo_med_d.get(x, fallback) for x in g]
    out['cluster_mean'] = [cluster_mean_d.get(clusters.get(x, 0), fallback) for x in g]
    out['neighbor_mean'] = [neighbor_mean_d.get(x, fallback) for x in g]

    # Day-48 same-bucket and time-neighbour values (the strongest signal)
    for off in (-2, -1, 0, 1, 2):
        out[f'd48_b{off}'] = [d48_bucket.get((x, bb + off),
                              profile_lookup(prof_dict, gb_mean, fallback, geo_mean, x, bb, off))
                              for x, bb in zip(g, b)]

    # Denoised profile and its time-neighbours
    for off in (-2, -1, 0, 1, 2):
        out[f'prof_b{off}'] = [profile_lookup(prof_dict, gb_mean, fallback, geo_mean, x, bb, off)
                               for x, bb in zip(g, b)]

    # NOTE: deliberately NO within-day "frozen lag" features. For test rows the
    # only observable recent value is the day-49 night window (bucket <=8); using
    # it would (a) leak the target into the day-49 validation rows that drive
    # early stopping, and (b) freeze a night level onto daytime predictions. The
    # SVD profile already supplies a far more reliable recent-structure prior.

    profile = out['prof_b0'].values.copy()
    return out, profile


def main():
    t_start = time.time()
    print("=" * 68)
    print("FLIPKART GRIDLOCK 2.0 - building submission from train.csv only")
    print("=" * 68)

    print("\n[1/7] Loading data...")
    train, test = load_data()
    train = add_time(train)
    test = add_time(test)
    train = train.sort_values(['geohash', 'day', 'tb']).reset_index(drop=True)
    print(f"  train {train.shape}, test {test.shape}")

    print("\n[2/7] Geohash geometry...")
    geo_xy, clusters, nbrs = build_geo(train, test)
    print(f"  {len(geo_xy)} geohashes, {N_CLUSTERS} clusters")

    print("\n[3/7] Day-48 stats + soft-impute SVD profile (rank %d)..." % SVD_RANK)
    d48 = train[train.day == 48]
    geo_mean = d48.groupby('geohash')['demand'].mean()
    geo_std = d48.groupby('geohash')['demand'].std().fillna(0.0)
    geo_max = d48.groupby('geohash')['demand'].max()
    geo_min = d48.groupby('geohash')['demand'].min()
    geo_med = d48.groupby('geohash')['demand'].median()
    cluster_series = d48.assign(cl=d48['geohash'].map(clusters)).groupby('cl')['demand'].mean()
    gmd = geo_mean.to_dict()
    neighbor_mean = {g: (np.mean([gmd.get(n, np.nan) for n in nb])
                         if nb and not np.isnan(np.nanmean([gmd.get(n, np.nan) for n in nb]))
                         else gmd.get(g, np.nan))
                     for g, nb in nbrs.items()}
    d48_bucket = {(r.geohash, r.tb): r.demand for r in d48.itertuples()}

    prof, prof_dict, gb_mean, fallback = build_profile(d48, geo_mean, SVD_RANK)
    d48_stats = (gmd, geo_std.to_dict(), geo_max.to_dict(), geo_min.to_dict(),
                 geo_med.to_dict(), cluster_series.to_dict(), neighbor_mean, d48_bucket)

    print("\n[4/7] Feature engineering (residual target)...")
    Xtr_df, prof_tr = make_features(train, geo_xy, clusters, nbrs, d48_stats,
                                    prof_dict, gb_mean, fallback, geo_mean)
    Xte_df, prof_te = make_features(test, geo_xy, clusters, nbrs, d48_stats,
                                    prof_dict, gb_mean, fallback, geo_mean)
    feat = list(Xtr_df.columns)
    Xtr = Xtr_df[feat].astype(np.float32).replace([np.inf, -np.inf], 0).fillna(0).values
    Xte = Xte_df[feat].astype(np.float32).replace([np.inf, -np.inf], 0).fillna(0).values
    y = train['demand'].values
    resid = y - prof_tr   # models fit the correction
    print(f"  {len(feat)} features; residual std={resid.std():.4f} (vs demand std={y.std():.4f})")

    # Honest validation split: train on day 48, validate on day-49 early window.
    val_mask = (train['day'] == 49).values
    tr_mask = ~val_mask
    Xv, yv_resid, prof_v = Xtr[val_mask], resid[val_mask], prof_tr[val_mask]
    yv_true = y[val_mask]
    Xt, yt_resid = Xtr[tr_mask], resid[tr_mask]

    print("\n[5/7] Training ensemble on the residual...")
    val_preds, test_preds = {}, {}

    # LightGBM, seed-averaged
    lgb_params = dict(objective='regression', metric='rmse', learning_rate=0.03,
                      num_leaves=127, min_child_samples=40, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1,
                      reg_lambda=0.2, n_estimators=3000, verbose=-1, n_jobs=-1)
    lgb_best_iter, lgb_v, lgb_te = [], [], []
    for seed in (42, 7, 2024):
        m = lgb.LGBMRegressor(random_state=seed, **lgb_params)
        m.fit(Xt, yt_resid, eval_set=[(Xv, yv_resid)],
              callbacks=[lgb.early_stopping(80, verbose=False)])
        lgb_best_iter.append(m.best_iteration_ or lgb_params['n_estimators'])
        lgb_v.append(m.predict(Xv))
        # refit on ALL data at the validated tree count for the test prediction
        mf = lgb.LGBMRegressor(random_state=seed,
                               **{**lgb_params, 'n_estimators': lgb_best_iter[-1]})
        mf.fit(Xtr, resid)
        lgb_te.append(mf.predict(Xte))
    val_preds['lgb'] = np.mean(lgb_v, axis=0)
    test_preds['lgb'] = np.mean(lgb_te, axis=0)
    print(f"  LightGBM best_iters={lgb_best_iter}")

    # XGBoost
    xgb_params = dict(objective='reg:squarederror', learning_rate=0.03, max_depth=7,
                      min_child_weight=40, subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=1.0, n_estimators=3000,
                      tree_method='hist', n_jobs=-1, verbosity=0,
                      early_stopping_rounds=80, random_state=RNG)
    mx = xgb.XGBRegressor(**xgb_params)
    mx.fit(Xt, yt_resid, eval_set=[(Xv, yv_resid)], verbose=False)
    xgb_iter = mx.best_iteration + 1
    val_preds['xgb'] = mx.predict(Xv, iteration_range=(0, xgb_iter))
    mxf = xgb.XGBRegressor(**{**xgb_params, 'n_estimators': xgb_iter,
                              'early_stopping_rounds': None})
    mxf.fit(Xtr, resid, verbose=False)
    test_preds['xgb'] = mxf.predict(Xte)
    print(f"  XGBoost best_iter={xgb_iter}")

    # CatBoost (optional)
    if HAS_CATBOOST:
        mc = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=7,
                               l2_leaf_reg=3, random_seed=RNG, eval_metric='RMSE',
                               early_stopping_rounds=80, verbose=0, task_type='CPU')
        mc.fit(Xt, yt_resid, eval_set=(Xv, yv_resid))
        _cbi = mc.get_best_iteration()
        cat_iter = (mc.tree_count_ if _cbi is None else _cbi + 1)  # 0-indexed -> count
        val_preds['cat'] = mc.predict(Xv)
        mcf = CatBoostRegressor(iterations=cat_iter, learning_rate=0.03, depth=7,
                                l2_leaf_reg=3, random_seed=RNG, verbose=0, task_type='CPU')
        mcf.fit(Xtr, resid)
        test_preds['cat'] = mcf.predict(Xte)
        print(f"  CatBoost best_iter={cat_iter}")

    # HistGradientBoosting (diversity)
    mh = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600,
                                       max_leaf_nodes=63, l2_regularization=0.1,
                                       random_state=RNG, validation_fraction=None)
    mh.fit(Xt, yt_resid)
    val_preds['hgb'] = mh.predict(Xv)
    mhf = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600,
                                        max_leaf_nodes=63, l2_regularization=0.1,
                                        random_state=RNG, validation_fraction=None)
    mhf.fit(Xtr, resid)
    test_preds['hgb'] = mhf.predict(Xte)

    # ExtraTrees (diversity)
    me = ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_leaf=20,
                             n_jobs=-1, random_state=RNG)
    me.fit(Xt, yt_resid)
    val_preds['et'] = me.predict(Xv)
    mef = ExtraTreesRegressor(n_estimators=300, max_depth=None, min_samples_leaf=20,
                              n_jobs=-1, random_state=RNG)
    mef.fit(Xtr, resid)
    test_preds['et'] = mef.predict(Xte)

    print("\n[6/7] Blending (validated on day-49 early window)...")
    # Per-model honest cross-day R2 (predict demand = profile + residual_pred)
    for name in val_preds:
        pv = np.clip(prof_v + val_preds[name], 0, 1)
        print(f"    {name:4s} val R2={r2_score(yv_true, pv):.4f}")

    # Fixed, principled blend weights (approach.md section 5). Equal-ish, not
    # tuned to the 2nd decimal against the night-only holdout.
    weights = {'lgb': 0.40, 'xgb': 0.20, 'cat': 0.15, 'hgb': 0.13, 'et': 0.12}
    if not HAS_CATBOOST:
        weights = {'lgb': 0.47, 'xgb': 0.24, 'hgb': 0.16, 'et': 0.13}
    wsum = sum(weights[k] for k in val_preds)
    blend_v = sum(weights[k] * val_preds[k] for k in val_preds) / wsum
    blend_te = sum(weights[k] * test_preds[k] for k in test_preds) / wsum

    val_final = np.clip(prof_v + blend_v, 0, 1)
    val_r2 = r2_score(yv_true, val_final)
    print(f"\n  >>> Blended cross-day validation R2 = {val_r2:.4f}"
          f"  (score ~ {max(0, 100 * val_r2):.2f})")
    print("  NOTE: validation is the day-49 NIGHT window; daytime test score differs.")

    print("\n[7/7] Writing submission.csv...")
    final = np.clip(prof_te + blend_te, 0, 1)
    sub = pd.DataFrame({'Index': test['Index'].values, 'demand': final})
    assert sub.shape == (len(test), 2)
    sub.to_csv('submission.csv', index=False)
    print(f"  submission.csv: {sub.shape[0]} rows, "
          f"demand mean={final.mean():.4f} std={final.std():.4f}")
    print(f"\nDone in {time.time() - t_start:.0f}s.")


if __name__ == '__main__':
    main()

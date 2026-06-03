# GridLock 2.0 - travel demand forecast
#
# Day 48 in train.csv is a complete day and spans the same hours we need to
# predict on day 49, so each location's day-48 demand curve is the backbone of
# the model. We add temporal/spatial context, a few road/weather fields and
# short within-day lags, then blend three gradient-boosting models.

import os
import warnings
import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)


def read_csv(name):
    df = pd.read_csv(name)
    if df.shape[1] < 4:                       # local checkout keeps the file in git-lfs
        df = pd.read_csv(name.replace(".csv", "_real.csv"))
    return df


def add_time(df):
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"], df["minute"] = parts[0], parts[1]
    df["time_bucket"] = df["hour"] * 4 + df["minute"] // 15
    return df


train = add_time(read_csv("train.csv"))
test = add_time(read_csv("test.csv"))

# geohash -> coordinates, then group nearby cells into clusters
cells = list(set(train["geohash"]) | set(test["geohash"]))
coord = {g: (pgh.decode(g).latitude, pgh.decode(g).longitude) for g in cells}
km = KMeans(n_clusters=40, random_state=SEED, n_init=10)
cluster = dict(zip(cells, km.fit_predict(np.array([coord[g] for g in cells]))))

# adjacent cells (used for a neighbourhood demand average)
steps = ["top", "bottom", "left", "right", "topleft", "topright", "bottomleft", "bottomright"]
known = set(cells)
adj = {}
for g in cells:
    try:
        adj[g] = [pgh.get_adjacent(g, s) for s in steps if pgh.get_adjacent(g, s) in known]
    except Exception:
        adj[g] = []

# everything below is derived from day 48 only
hist = train[train["day"] == 48].copy()
bucket_demand = hist.groupby(["geohash", "time_bucket"])["demand"].mean().to_dict()
hour_demand = hist.groupby(["geohash", "hour"])["demand"].mean().to_dict()
g_mean = hist.groupby("geohash")["demand"].mean().to_dict()
g_std = hist.groupby("geohash")["demand"].std().to_dict()
g_max = hist.groupby("geohash")["demand"].max().to_dict()
g_min = hist.groupby("geohash")["demand"].min().to_dict()
g_med = hist.groupby("geohash")["demand"].median().to_dict()
hourly = hist.groupby("hour")["demand"].mean().to_dict()
bucketly = hist.groupby("time_bucket")["demand"].mean().to_dict()
cluster_demand = hist.assign(c=hist["geohash"].map(cluster)).groupby("c")["demand"].mean().to_dict()
adj_demand = {g: np.mean([g_mean.get(n, 0) for n in ns]) if ns else g_mean.get(g, 0)
              for g, ns in adj.items()}
temp_fill = hist["Temperature"].median()
glob_mean = hist["demand"].mean()

# full 96-bucket demand curve per cell, for rolling stats around a bucket
curve = {g: np.array([bucket_demand.get((g, b), np.nan) for b in range(96)]) for g in cells}


def around(g, b, w, fn):
    seg = curve[g][max(0, b - w):b + w + 1]
    seg = seg[~np.isnan(seg)]
    return fn(seg) if len(seg) else g_mean.get(g, 0)


# A single day's demand curve is noisy. Factorise the geohash x time_bucket
# matrix and keep the leading components - this borrows the diurnal shape shared
# across locations and gives a much cleaner per-cell profile to anchor on.
idx = {g: i for i, g in enumerate(cells)}
grid = np.full((len(cells), 96), np.nan)
for (g, b), v in bucket_demand.items():
    grid[idx[g], b] = v
base_level = np.nan_to_num(np.nanmean(grid, axis=1), nan=glob_mean)
centred = np.where(np.isnan(grid), base_level[:, None], grid) - base_level[:, None]
svd = TruncatedSVD(n_components=8, random_state=SEED)
smooth_grid = svd.inverse_transform(svd.fit_transform(centred)) + base_level[:, None]


def profile(g, b):
    return smooth_grid[idx[g], b] if (g in idx and 0 <= b < 96) else g_mean.get(g, glob_mean)


def build(df):
    df = df.copy()
    df["lat"] = df["geohash"].map(lambda x: coord.get(x, (0, 0))[0])
    df["lon"] = df["geohash"].map(lambda x: coord.get(x, (0, 0))[1])
    df["cluster"] = df["geohash"].map(cluster).fillna(0).astype(int)
    df["dow"] = (df["day"] - 1) % 7

    df["is_peak"] = (((df["hour"] >= 7) & (df["hour"] <= 9)) |
                     ((df["hour"] >= 17) & (df["hour"] <= 19))).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_lunch"] = ((df["hour"] >= 11) & (df["hour"] <= 13)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["bkt_sin"] = np.sin(2 * np.pi * df["time_bucket"] / 96)
    df["bkt_cos"] = np.cos(2 * np.pi * df["time_bucket"] / 96)
    # extra harmonics capture the multi-peaked daily shape
    df["bkt_sin2"] = np.sin(4 * np.pi * df["time_bucket"] / 96)
    df["bkt_cos2"] = np.cos(4 * np.pi * df["time_bucket"] / 96)
    df["bkt_sin3"] = np.sin(6 * np.pi * df["time_bucket"] / 96)
    df["bkt_cos3"] = np.cos(6 * np.pi * df["time_bucket"] / 96)

    df["road"] = df["RoadType"].map({"Residential": 0, "Street": 1, "Highway": 2}).fillna(-1)
    df["large"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["landmark"] = (df["Landmarks"] == "Yes").astype(int)
    df["weather"] = df["Weather"].map({"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}).fillna(-1)
    df["temp"] = df["Temperature"].fillna(temp_fill)

    # day-48 demand at the same slot and the four neighbouring slots
    for off in (-2, -1, 0, 1, 2):
        df[f"d48_b{off}"] = df.apply(
            lambda r: bucket_demand.get((r["geohash"], r["time_bucket"] + off),
                                        g_mean.get(r["geohash"], 0)), axis=1)
    df["d48_w3m"] = df.apply(lambda r: around(r["geohash"], r["time_bucket"], 3, np.mean), axis=1)
    df["d48_w3s"] = df.apply(lambda r: around(r["geohash"], r["time_bucket"], 3, np.std), axis=1)
    df["d48_hour"] = df.apply(lambda r: hour_demand.get((r["geohash"], r["hour"]),
                                                         g_mean.get(r["geohash"], 0)), axis=1)

    df["geo_mean"] = df["geohash"].map(g_mean).fillna(0)
    df["geo_std"] = df["geohash"].map(g_std).fillna(0)
    df["geo_max"] = df["geohash"].map(g_max).fillna(0)
    df["geo_min"] = df["geohash"].map(g_min).fillna(0)
    df["geo_med"] = df["geohash"].map(g_med).fillna(0)
    df["hour_glob"] = df["hour"].map(hourly).fillna(0)
    df["bkt_glob"] = df["time_bucket"].map(bucketly).fillna(0)
    df["cl_mean"] = df["cluster"].map(cluster_demand).fillna(0)
    df["nbr_mean"] = df["geohash"].map(adj_demand).fillna(0)

    df["temp_weather"] = df["temp"] * df["weather"]
    df["lanes_road"] = df["NumberofLanes"] * df["road"]
    df["d48b0_peak"] = df["d48_b0"] * df["is_peak"]
    df["d48b0_vs_geo"] = df["d48_b0"] / (df["geo_mean"] + 1e-9)

    # smoothed (factorised) profile at the slot and the two slots either side
    df["prof0"] = [profile(g, b) for g, b in zip(df["geohash"], df["time_bucket"])]
    df["prof_m1"] = [profile(g, b - 1) for g, b in zip(df["geohash"], df["time_bucket"])]
    df["prof_p1"] = [profile(g, b + 1) for g, b in zip(df["geohash"], df["time_bucket"])]
    df["prof_m2"] = [profile(g, b - 2) for g, b in zip(df["geohash"], df["time_bucket"])]
    df["prof_p2"] = [profile(g, b + 2) for g, b in zip(df["geohash"], df["time_bucket"])]
    return df


train = build(train)
test = build(test)

# short lags from the run-up of each cell; for test these are the last values seen
train = train.sort_values(["geohash", "day", "time_bucket"]).reset_index(drop=True)
for i in (1, 2, 3):
    train[f"lag_{i}"] = train.groupby("geohash")["demand"].shift(i)
train["diff_1"] = train["lag_1"] - train["lag_2"]
tail = train.groupby("geohash")["demand"].apply(list)
for i in (1, 2, 3):
    test[f"lag_{i}"] = test["geohash"].map(
        lambda g, n=i: tail.get(g, [0])[-n] if len(tail.get(g, [])) >= n else 0)
test["diff_1"] = test["lag_1"] - test["lag_2"]

cols = ["lat", "lon", "cluster", "hour", "minute", "time_bucket", "dow",
        "is_peak", "is_night", "is_lunch", "hour_sin", "hour_cos", "bkt_sin", "bkt_cos",
        "road", "NumberofLanes", "large", "landmark", "weather", "temp",
        "d48_b-2", "d48_b-1", "d48_b0", "d48_b1", "d48_b2", "d48_w3m", "d48_w3s", "d48_hour",
        "geo_mean", "geo_std", "geo_max", "geo_min", "geo_med", "hour_glob", "bkt_glob",
        "cl_mean", "nbr_mean", "temp_weather", "lanes_road", "d48b0_peak", "d48b0_vs_geo",
        "bkt_sin2", "bkt_cos2", "bkt_sin3", "bkt_cos3",
        "prof0", "prof_m1", "prof_p1", "prof_m2", "prof_p2",
        "lag_1", "lag_2", "lag_3", "diff_1"]

data = train.dropna(subset=["lag_1"]).copy()
X = data[cols].astype(np.float32).replace([np.inf, -np.inf], 0).fillna(0)
prior = data["prof0"].values
y = data["demand"].values - prior            # learn the correction on top of the profile
X_test = test[cols].astype(np.float32).replace([np.inf, -np.inf], 0).fillna(0)
prior_test = test["prof0"].values

# pick the tree count on a day48 -> day49(early) holdout, then refit on everything
fit = data[data["day"] == 48]
hold = data[data["day"] == 49]
probe = lgb.LGBMRegressor(objective="regression", metric="rmse", learning_rate=0.03,
                          num_leaves=255, min_child_samples=20, feature_fraction=0.75,
                          bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=0.5,
                          n_estimators=2000, verbose=-1, random_state=SEED, n_jobs=-1)
probe.fit(fit[cols].astype(np.float32).fillna(0), (fit["demand"] - fit["prof0"]).values,
          eval_set=[(hold[cols].astype(np.float32).fillna(0), (hold["demand"] - hold["prof0"]).values)],
          callbacks=[lgb.early_stopping(60, verbose=False)])
n_trees = max(120, probe.best_iteration_ or 200)

# LightGBM averaged over a few seeds for stability
lgb_runs = []
for sd in (42, 7, 2024):
    m = lgb.LGBMRegressor(objective="regression", metric="rmse", learning_rate=0.03,
                          num_leaves=255, min_child_samples=20, feature_fraction=0.75,
                          bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=0.5,
                          n_estimators=n_trees, verbose=-1, random_state=sd, n_jobs=-1)
    m.fit(X, y)
    lgb_runs.append(m.predict(X_test))
p_lgb = np.mean(lgb_runs, axis=0)

xgb_m = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.03, max_depth=7,
                         min_child_weight=40, subsample=0.8, colsample_bytree=0.75,
                         reg_alpha=0.1, reg_lambda=1.0, n_estimators=n_trees,
                         random_state=SEED, tree_method="hist", n_jobs=-1, verbosity=0)
xgb_m.fit(X, y)
p_xgb = xgb_m.predict(X_test)

cat_m = CatBoostRegressor(iterations=n_trees, learning_rate=0.03, depth=7, l2_leaf_reg=3,
                          random_seed=SEED, verbose=0, eval_metric="RMSE", task_type="CPU")
cat_m.fit(X, y)
p_cat = cat_m.predict(X_test)

# two more decorrelated learners on the same correction target
hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=63,
                                    l2_regularization=0.5, random_state=SEED)
hgb.fit(X, y)
p_hgb = hgb.predict(X_test)

ext = ExtraTreesRegressor(n_estimators=300, max_features=0.6, min_samples_leaf=20,
                          random_state=SEED, n_jobs=-1)
ext.fit(X, y)
p_ext = ext.predict(X_test)

# blend the predicted corrections and add them back to the profile
resid = 0.45 * p_lgb + 0.20 * p_xgb + 0.13 * p_cat + 0.12 * p_hgb + 0.10 * p_ext
pred = np.clip(prior_test + resid, 0, 1)

out = pd.DataFrame({"Index": test["Index"].values, "demand": pred})
out.to_csv("submission.csv", index=False)
print("submission.csv", out.shape, "mean %.4f" % pred.mean())

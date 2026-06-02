"""
Traffic Demand Prediction - Flipkart GridLock 2.0
==================================================
USES ONLY: train.csv (77,299 rows, days 48-49, provided by competition)
PREDICTS:  test.csv (41,778 rows, day 49 timestamps 2:15-13:45)
EVALUATION: score = max(0, 100 * r2_score(actual, predicted))

Data structure:
- train.csv: Day 48 (full day, 96 time slots) + Day 49 (0:00-2:00, 9 slots) = 77,299 rows
- test.csv: Day 49 (2:15-13:45, 47 time slots) = 41,778 rows
"""

import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
import time
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 70)
print("FLIPKART GRIDLOCK 2.0 - TRAFFIC DEMAND PREDICTION")
print("Using ONLY: train.csv (77,299 rows)")
print("=" * 70)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/8] Loading data...")
t0 = time.time()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"  train.csv: {train.shape[0]:,} rows (day 48: {len(train[train.day==48])}, day 49: {len(train[train.day==49])})")
print(f"  test.csv:  {test.shape[0]:,} rows (day 49, timestamps 2:15-13:45)")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ============================================================
# STEP 2: GEOHASH PROCESSING
# ============================================================
print("\n[2/8] Geohash processing...")
t0 = time.time()

all_geohashes = list(set(train['geohash'].unique()) | set(test['geohash'].unique()))
geo_dict = {}
for g in all_geohashes:
    decoded = pgh.decode(g)
    geo_dict[g] = (decoded.latitude, decoded.longitude)

# Spatial clustering (30 clusters - smaller data so fewer clusters)
coords = np.array([(geo_dict[g][0], geo_dict[g][1]) for g in all_geohashes])
kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coords)
geo_cluster_map = dict(zip(all_geohashes, cluster_labels))

# Neighbors
directions = ['top', 'bottom', 'left', 'right', 'topleft', 'topright', 'bottomleft', 'bottomright']
geo_set = set(all_geohashes)
neighbor_map = {}
for g in all_geohashes:
    try:
        neighbors = [pgh.get_adjacent(g, d) for d in directions]
        neighbor_map[g] = [n for n in neighbors if n in geo_set]
    except:
        neighbor_map[g] = []

print(f"  {len(all_geohashes)} geohashes decoded, 30 clusters, done in {time.time()-t0:.1f}s")

# ============================================================
# STEP 3: FEATURE ENGINEERING ON train.csv
# ============================================================
print("\n[3/8] Feature engineering...")
t0 = time.time()

# Parse timestamps
train['hour'] = train['timestamp'].apply(lambda x: int(x.split(':')[0]))
train['minute'] = train['timestamp'].apply(lambda x: int(x.split(':')[1]))
train['time_bucket'] = train['hour'] * 4 + train['minute'] // 15

# Sort chronologically
train.sort_values(['geohash', 'day', 'time_bucket'], inplace=True)
train.reset_index(drop=True, inplace=True)

# Spatial features
train['latitude'] = train['geohash'].map(lambda x: geo_dict[x][0])
train['longitude'] = train['geohash'].map(lambda x: geo_dict[x][1])
train['geo_cluster'] = train['geohash'].map(geo_cluster_map)

# Temporal features
train['day_of_week'] = (train['day'] - 1) % 7
train['is_weekend'] = (train['day_of_week'] >= 5).astype(np.int8)
train['is_morning_rush'] = ((train['hour'] >= 7) & (train['hour'] <= 9)).astype(np.int8)
train['is_evening_rush'] = ((train['hour'] >= 17) & (train['hour'] <= 19)).astype(np.int8)
train['is_peak_hour'] = (train['is_morning_rush'] | train['is_evening_rush']).astype(np.int8)
train['is_night'] = ((train['hour'] >= 22) | (train['hour'] <= 5)).astype(np.int8)
train['is_lunch'] = ((train['hour'] >= 11) & (train['hour'] <= 13)).astype(np.int8)

# Cyclical encoding
train['hour_sin'] = np.sin(2 * np.pi * train['hour'] / 24)
train['hour_cos'] = np.cos(2 * np.pi * train['hour'] / 24)
train['dow_sin'] = np.sin(2 * np.pi * train['day_of_week'] / 7)
train['dow_cos'] = np.cos(2 * np.pi * train['day_of_week'] / 7)
train['bucket_sin'] = np.sin(2 * np.pi * train['time_bucket'] / 96)
train['bucket_cos'] = np.cos(2 * np.pi * train['time_bucket'] / 96)

# Infrastructure encoding
train['RoadType_enc'] = train['RoadType'].map({'Residential': 0, 'Street': 1, 'Highway': 2}).fillna(-1)
train['LargeVehicles_enc'] = (train['LargeVehicles'] == 'Allowed').astype(np.int8)
train['Landmarks_enc'] = (train['Landmarks'] == 'Yes').astype(np.int8)
train['Weather_enc'] = train['Weather'].map({'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}).fillna(-1)
train['Temperature'] = train['Temperature'].fillna(train['Temperature'].median())

# Lag features (within train data)
train['demand_lag_1'] = train.groupby('geohash')['demand'].shift(1)
train['demand_lag_2'] = train.groupby('geohash')['demand'].shift(2)
train['demand_lag_3'] = train.groupby('geohash')['demand'].shift(3)
train['demand_lag_4'] = train.groupby('geohash')['demand'].shift(4)
train['demand_lag_5'] = train.groupby('geohash')['demand'].shift(5)
train['demand_diff_1'] = train['demand_lag_1'] - train['demand_lag_2']
train['demand_diff_2'] = train['demand_lag_2'] - train['demand_lag_3']

# Rolling stats
for w in [4, 12, 24]:
    rolled = train.groupby('geohash')['demand'].rolling(w, min_periods=1).mean()
    train[f'roll_mean_{w}'] = rolled.reset_index(level=0, drop=True).values
    rolled_std = train.groupby('geohash')['demand'].rolling(w, min_periods=1).std()
    train[f'roll_std_{w}'] = rolled_std.reset_index(level=0, drop=True).values

train['ema_4'] = train.groupby('geohash')['demand'].transform(lambda x: x.ewm(span=4).mean())
train['ema_12'] = train.groupby('geohash')['demand'].transform(lambda x: x.ewm(span=12).mean())

# Aggregated stats (computed from train only)
geo_mean = train.groupby('geohash')['demand'].mean()
geo_std_s = train.groupby('geohash')['demand'].std()
train['geo_mean_demand'] = train['geohash'].map(geo_mean)
train['geo_std_demand'] = train['geohash'].map(geo_std_s)

geo_hour_mean = train.groupby(['geohash', 'hour'])['demand'].mean()
train['geo_hour_mean'] = train.set_index(['geohash', 'hour']).index.map(geo_hour_mean.to_dict().get)
train['geo_hour_mean'] = train['geo_hour_mean'].fillna(train['geo_mean_demand'])

hour_mean_global = train.groupby('hour')['demand'].mean()
train['hour_mean_global'] = train['hour'].map(hour_mean_global)

cluster_mean = train.groupby('geo_cluster')['demand'].mean()
train['cluster_mean_demand'] = train['geo_cluster'].map(cluster_mean)

# Neighbor stats
geo_mean_dict = geo_mean.to_dict()
neighbor_mean_d = {}
for g in all_geohashes:
    nbrs = neighbor_map.get(g, [])
    neighbor_mean_d[g] = np.mean([geo_mean_dict.get(n, 0) for n in nbrs]) if nbrs else geo_mean_dict.get(g, 0)
train['neighbor_mean'] = train['geohash'].map(neighbor_mean_d)

# Interactions
train['temp_weather'] = train['Temperature'] * train['Weather_enc']
train['lanes_road'] = train['NumberofLanes'] * train['RoadType_enc']
train['peak_geo'] = train['is_peak_hour'] * train['geo_mean_demand']
train['demand_vs_geo'] = train['demand_lag_1'] / (train['geo_mean_demand'] + 1e-8)

print(f"  Features done in {time.time()-t0:.1f}s, shape: {train.shape}")

# ============================================================
# STEP 4: PREPARE TEST FEATURES
# ============================================================
print("\n[4/8] Preparing test features...")
t0 = time.time()

test['hour'] = test['timestamp'].apply(lambda x: int(x.split(':')[0]))
test['minute'] = test['timestamp'].apply(lambda x: int(x.split(':')[1]))
test['time_bucket'] = test['hour'] * 4 + test['minute'] // 15
test['latitude'] = test['geohash'].map(lambda x: geo_dict.get(x, (0,0))[0])
test['longitude'] = test['geohash'].map(lambda x: geo_dict.get(x, (0,0))[1])
test['geo_cluster'] = test['geohash'].map(geo_cluster_map).fillna(0).astype(int)
test['day_of_week'] = (test['day'] - 1) % 7
test['is_weekend'] = (test['day_of_week'] >= 5).astype(np.int8)
test['is_morning_rush'] = ((test['hour'] >= 7) & (test['hour'] <= 9)).astype(np.int8)
test['is_evening_rush'] = ((test['hour'] >= 17) & (test['hour'] <= 19)).astype(np.int8)
test['is_peak_hour'] = (test['is_morning_rush'] | test['is_evening_rush']).astype(np.int8)
test['is_night'] = ((test['hour'] >= 22) | (test['hour'] <= 5)).astype(np.int8)
test['is_lunch'] = ((test['hour'] >= 11) & (test['hour'] <= 13)).astype(np.int8)
test['hour_sin'] = np.sin(2 * np.pi * test['hour'] / 24)
test['hour_cos'] = np.cos(2 * np.pi * test['hour'] / 24)
test['dow_sin'] = np.sin(2 * np.pi * test['day_of_week'] / 7)
test['dow_cos'] = np.cos(2 * np.pi * test['day_of_week'] / 7)
test['bucket_sin'] = np.sin(2 * np.pi * test['time_bucket'] / 96)
test['bucket_cos'] = np.cos(2 * np.pi * test['time_bucket'] / 96)
test['RoadType_enc'] = test['RoadType'].map({'Residential': 0, 'Street': 1, 'Highway': 2}).fillna(-1)
test['LargeVehicles_enc'] = (test['LargeVehicles'] == 'Allowed').astype(np.int8)
test['Landmarks_enc'] = (test['Landmarks'] == 'Yes').astype(np.int8)
test['Weather_enc'] = test['Weather'].map({'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}).fillna(-1)
test['Temperature'] = test['Temperature'].fillna(test['Temperature'].median() if test['Temperature'].notna().any() else 16.5)

# Aggregated features for test (from train stats)
test['geo_mean_demand'] = test['geohash'].map(geo_mean_dict).fillna(geo_mean.mean())
test['geo_std_demand'] = test['geohash'].map(geo_std_s.to_dict()).fillna(0)
test['geo_hour_mean'] = test.set_index(['geohash', 'hour']).index.map(geo_hour_mean.to_dict().get)
test['geo_hour_mean'] = test['geo_hour_mean'].fillna(test['geo_mean_demand'])
test['hour_mean_global'] = test['hour'].map(hour_mean_global).fillna(0)
test['cluster_mean_demand'] = test['geo_cluster'].map(cluster_mean).fillna(0)
test['neighbor_mean'] = test['geohash'].map(neighbor_mean_d).fillna(0)

# For lag features in test: use the LAST known demand for each geohash from train
# (the most recent observation before the test period starts at 2:15)
last_demand = train.groupby('geohash').last()[['demand']].rename(columns={'demand': 'last_demand'})
last_demands = train.groupby('geohash')['demand'].apply(list)

# Get last N demands per geohash
for i in range(1, 6):
    col_name = f'demand_lag_{i}'
    test[col_name] = test['geohash'].map(
        lambda g, n=i: last_demands.get(g, [0])[-n] if len(last_demands.get(g, [])) >= n else 0
    )

test['demand_diff_1'] = test['demand_lag_1'] - test['demand_lag_2']
test['demand_diff_2'] = test['demand_lag_2'] - test['demand_lag_3']

# Rolling stats from last known values
for w in [4, 12, 24]:
    test[f'roll_mean_{w}'] = test['geohash'].map(
        lambda g, win=w: np.mean(last_demands.get(g, [0])[-win:]) if last_demands.get(g) else 0
    )
    test[f'roll_std_{w}'] = test['geohash'].map(
        lambda g, win=w: np.std(last_demands.get(g, [0])[-win:]) if last_demands.get(g) else 0
    )

test['ema_4'] = test['geohash'].map(
    lambda g: pd.Series(last_demands.get(g, [0])).ewm(span=4).mean().iloc[-1] if last_demands.get(g) else 0
)
test['ema_12'] = test['geohash'].map(
    lambda g: pd.Series(last_demands.get(g, [0])).ewm(span=12).mean().iloc[-1] if last_demands.get(g) else 0
)

# Interactions
test['temp_weather'] = test['Temperature'] * test['Weather_enc']
test['lanes_road'] = test['NumberofLanes'] * test['RoadType_enc']
test['peak_geo'] = test['is_peak_hour'] * test['geo_mean_demand']
test['demand_vs_geo'] = test['demand_lag_1'] / (test['geo_mean_demand'] + 1e-8)

print(f"  Test features done in {time.time()-t0:.1f}s")

# ============================================================
# STEP 5: DEFINE FEATURE SET
# ============================================================
print("\n[5/8] Defining features...")

feature_cols = [
    'latitude', 'longitude', 'geo_cluster',
    'hour', 'minute', 'time_bucket', 'day_of_week',
    'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_peak_hour', 'is_night', 'is_lunch',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'bucket_sin', 'bucket_cos',
    'RoadType_enc', 'NumberofLanes', 'LargeVehicles_enc', 'Landmarks_enc', 'Weather_enc', 'Temperature',
    'geo_mean_demand', 'geo_std_demand', 'geo_hour_mean', 'hour_mean_global', 'cluster_mean_demand',
    'neighbor_mean',
    'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4', 'demand_lag_5',
    'demand_diff_1', 'demand_diff_2',
    'roll_mean_4', 'roll_mean_12', 'roll_mean_24',
    'roll_std_4', 'roll_std_12', 'roll_std_24',
    'ema_4', 'ema_12',
    'temp_weather', 'lanes_road', 'peak_geo', 'demand_vs_geo',
]

feature_cols = [f for f in feature_cols if f in train.columns and f in test.columns]
print(f"  Features: {len(feature_cols)}")

# Drop rows with NaN in lags (first few rows per geohash)
train_clean = train.dropna(subset=['demand_lag_1']).copy()

X_train = train_clean[feature_cols].astype(np.float32).replace([np.inf, -np.inf], 0).fillna(0)
y_train = train_clean['demand'].values
X_test = test[feature_cols].astype(np.float32).replace([np.inf, -np.inf], 0).fillna(0)

print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# STEP 6: TRAIN MODELS
# ============================================================
print("\n[6/8] Training LightGBM...")
t0 = time.time()

lgb_model = lgb.LGBMRegressor(
    objective='regression', metric='rmse', boosting_type='gbdt',
    learning_rate=0.03, num_leaves=255, min_child_samples=30,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1, n_estimators=2000,
    verbose=-1, random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
lgb_pred = np.clip(lgb_model.predict(X_test), 0, 1)
lgb_r2 = r2_score(y_train, np.clip(lgb_model.predict(X_train), 0, 1))
print(f"  LightGBM R2: {lgb_r2:.6f} ({time.time()-t0:.0f}s)")

print("\n  Training XGBoost...")
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', learning_rate=0.03, max_depth=8,
    min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_estimators=2000,
    random_state=42, tree_method='hist', n_jobs=-1, verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_pred = np.clip(xgb_model.predict(X_test), 0, 1)
xgb_r2 = r2_score(y_train, np.clip(xgb_model.predict(X_train), 0, 1))
print(f"  XGBoost R2: {xgb_r2:.6f} ({time.time()-t0:.0f}s)")

print("\n  Training CatBoost...")
t0 = time.time()
cat_model = CatBoostRegressor(
    iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3,
    random_seed=42, verbose=0, eval_metric='RMSE', task_type='CPU')
cat_model.fit(X_train, y_train)
cat_pred = np.clip(cat_model.predict(X_test), 0, 1)
cat_r2 = r2_score(y_train, np.clip(cat_model.predict(X_train), 0, 1))
print(f"  CatBoost R2: {cat_r2:.6f} ({time.time()-t0:.0f}s)")

# ============================================================
# STEP 7: ENSEMBLE
# ============================================================
print("\n[7/8] Optimizing ensemble...")

lgb_tr = np.clip(lgb_model.predict(X_train), 0, 1)
xgb_tr = np.clip(xgb_model.predict(X_train), 0, 1)
cat_tr = np.clip(cat_model.predict(X_train), 0, 1)

best_r2, best_w = -999, (0.33, 0.33, 0.34)
for w1 in np.arange(0.2, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05 or w3 > 0.7: continue
        pred = np.clip(w1*lgb_tr + w2*xgb_tr + w3*cat_tr, 0, 1)
        r2 = r2_score(y_train, pred)
        if r2 > best_r2: best_r2, best_w = r2, (w1, w2, w3)

print(f"  Weights: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CAT={best_w[2]:.2f}")
print(f"  Ensemble R2: {best_r2:.6f} (score: {max(0, 100*best_r2):.2f})")

# ============================================================
# STEP 8: SUBMISSION
# ============================================================
print("\n[8/8] Generating submission.csv...")

final_pred = np.clip(best_w[0]*lgb_pred + best_w[1]*xgb_pred + best_w[2]*cat_pred, 0, 1)
submission = pd.DataFrame({'Index': test['Index'].values, 'demand': final_pred})
assert submission.shape == (41778, 2)

submission.to_csv('submission.csv', index=False)

print(f"  Saved: submission.csv ({submission.shape[0]} rows)")
print(f"  Demand: mean={final_pred.mean():.4f}, std={final_pred.std():.4f}")
print(f"\n{'='*70}")
print(f"DONE! Score estimate: {max(0, 100*best_r2):.2f}/100")
print(f"{'='*70}")

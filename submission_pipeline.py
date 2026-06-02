"""
Traffic Demand Prediction - Flipkart GridLock 2.0
==================================================
Train on training.csv (4.2M rows, days 1-61) ONLY.
Predict test.csv (41,778 rows, day 49).
Evaluation: score = max(0, 100 * r2_score(actual, predicted))
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
print("Training on: training.csv (4.2M rows, days 1-61)")
print("Predicting:  test.csv (41,778 rows, day 49)")
print("=" * 70)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/9] Loading datasets...")
t0 = time.time()

train_df = pd.read_csv('/projects/sandbox/Flipkart-GridLock-2.0/training_real.csv')
train_df.rename(columns={'geohash6': 'geohash'}, inplace=True)
print(f"  training.csv: {train_df.shape[0]:,} rows, days {train_df.day.min()}-{train_df.day.max()}")

test_df = pd.read_csv('/projects/sandbox/Flipkart-GridLock-2.0/test_real.csv')
print(f"  test.csv: {test_df.shape[0]:,} rows, day {test_df.day.min()}")
print(f"  Loaded in {time.time()-t0:.1f}s")


# ============================================================
# STEP 2: GEOHASH PROCESSING
# ============================================================
print("\n[2/9] Geohash processing...")
t0 = time.time()

all_geohashes = list(set(train_df['geohash'].unique()) | set(test_df['geohash'].unique()))
print(f"  Total unique geohashes: {len(all_geohashes)}")

# Decode geohash to lat/lon
geo_dict = {}
for g in all_geohashes:
    decoded = pgh.decode(g)
    geo_dict[g] = (decoded.latitude, decoded.longitude)

# Spatial clustering (50 clusters)
coords = np.array([(geo_dict[g][0], geo_dict[g][1]) for g in all_geohashes])
kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coords)
geo_cluster_map = dict(zip(all_geohashes, cluster_labels))

# Neighbor mapping
directions = ['top', 'bottom', 'left', 'right', 'topleft', 'topright', 'bottomleft', 'bottomright']
neighbor_map = {}
geo_set = set(all_geohashes)
for g in all_geohashes:
    try:
        neighbors = [pgh.get_adjacent(g, d) for d in directions]
        neighbor_map[g] = [n for n in neighbors if n in geo_set]
    except:
        neighbor_map[g] = []

print(f"  Done in {time.time()-t0:.1f}s")


# ============================================================
# STEP 3: TEMPORAL + SPATIAL FEATURES ON training.csv
# ============================================================
print("\n[3/9] Feature engineering on training.csv...")
t0 = time.time()

# Parse timestamp
train_df['hour'] = train_df['timestamp'].apply(lambda x: int(x.split(':')[0]))
train_df['minute'] = train_df['timestamp'].apply(lambda x: int(x.split(':')[1]))
train_df['time_bucket'] = train_df['hour'] * 4 + train_df['minute'] // 15

# Spatial
train_df['latitude'] = train_df['geohash'].map(lambda x: geo_dict.get(x, (0, 0))[0])
train_df['longitude'] = train_df['geohash'].map(lambda x: geo_dict.get(x, (0, 0))[1])
train_df['geo_cluster'] = train_df['geohash'].map(geo_cluster_map).fillna(-1).astype(int)

# Temporal
train_df['day_of_week'] = (train_df['day'] - 1) % 7
train_df['is_weekend'] = (train_df['day_of_week'] >= 5).astype(np.int8)
train_df['is_morning_rush'] = ((train_df['hour'] >= 7) & (train_df['hour'] <= 9)).astype(np.int8)
train_df['is_evening_rush'] = ((train_df['hour'] >= 17) & (train_df['hour'] <= 19)).astype(np.int8)
train_df['is_peak_hour'] = (train_df['is_morning_rush'] | train_df['is_evening_rush']).astype(np.int8)
train_df['is_night'] = ((train_df['hour'] >= 22) | (train_df['hour'] <= 5)).astype(np.int8)
train_df['is_lunch'] = ((train_df['hour'] >= 11) & (train_df['hour'] <= 13)).astype(np.int8)

# Cyclical
train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour'] / 24)
train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour'] / 24)
train_df['dow_sin'] = np.sin(2 * np.pi * train_df['day_of_week'] / 7)
train_df['dow_cos'] = np.cos(2 * np.pi * train_df['day_of_week'] / 7)
train_df['bucket_sin'] = np.sin(2 * np.pi * train_df['time_bucket'] / 96)
train_df['bucket_cos'] = np.cos(2 * np.pi * train_df['time_bucket'] / 96)

# Sort for lag computation
train_df.sort_values(['geohash', 'day', 'time_bucket'], inplace=True)
train_df.reset_index(drop=True, inplace=True)

print(f"  Basic features done in {time.time()-t0:.1f}s")


# ============================================================
# STEP 4: LAG + ROLLING FEATURES
# ============================================================
print("\n[4/9] Lag + rolling features...")
t0 = time.time()

# Lag features
train_df['demand_lag_1'] = train_df.groupby('geohash')['demand'].shift(1)
train_df['demand_lag_2'] = train_df.groupby('geohash')['demand'].shift(2)
train_df['demand_lag_3'] = train_df.groupby('geohash')['demand'].shift(3)
train_df['demand_lag_4'] = train_df.groupby('geohash')['demand'].shift(4)
train_df['demand_lag_5'] = train_df.groupby('geohash')['demand'].shift(5)

# Same time previous day (96 buckets/day)
train_df['demand_prev_day'] = train_df.groupby('geohash')['demand'].shift(96)
# Same time 2 days ago
train_df['demand_prev_2day'] = train_df.groupby('geohash')['demand'].shift(192)
# Same time previous week
train_df['demand_prev_week'] = train_df.groupby('geohash')['demand'].shift(672)

# Demand momentum
train_df['demand_diff_1'] = train_df['demand_lag_1'] - train_df['demand_lag_2']
train_df['demand_diff_2'] = train_df['demand_lag_2'] - train_df['demand_lag_3']

print(f"  Lags done in {time.time()-t0:.1f}s")
t1 = time.time()

# Rolling statistics
for w in [4, 12, 24, 96]:
    rolled = train_df.groupby('geohash')['demand'].rolling(w, min_periods=1).mean()
    train_df[f'roll_mean_{w}'] = rolled.reset_index(level=0, drop=True).values
    rolled_std = train_df.groupby('geohash')['demand'].rolling(w, min_periods=1).std()
    train_df[f'roll_std_{w}'] = rolled_std.reset_index(level=0, drop=True).values

for w in [12, 96]:
    rolled_max = train_df.groupby('geohash')['demand'].rolling(w, min_periods=1).max()
    train_df[f'roll_max_{w}'] = rolled_max.reset_index(level=0, drop=True).values
    rolled_min = train_df.groupby('geohash')['demand'].rolling(w, min_periods=1).min()
    train_df[f'roll_min_{w}'] = rolled_min.reset_index(level=0, drop=True).values

# EMA
train_df['ema_4'] = train_df.groupby('geohash')['demand'].transform(lambda x: x.ewm(span=4).mean())
train_df['ema_12'] = train_df.groupby('geohash')['demand'].transform(lambda x: x.ewm(span=12).mean())
train_df['ema_96'] = train_df.groupby('geohash')['demand'].transform(lambda x: x.ewm(span=96).mean())

print(f"  Rolling done in {time.time()-t1:.1f}s")


# ============================================================
# STEP 5: AGGREGATED SPATIAL/TEMPORAL STATISTICS
# ============================================================
print("\n[5/9] Aggregated statistics...")
t0 = time.time()

# Mean demand per geohash
geo_mean = train_df.groupby('geohash')['demand'].mean()
train_df['geo_mean_demand'] = train_df['geohash'].map(geo_mean)

geo_std = train_df.groupby('geohash')['demand'].std()
train_df['geo_std_demand'] = train_df['geohash'].map(geo_std)

# Mean demand per geohash + hour
geo_hour_mean = train_df.groupby(['geohash', 'hour'])['demand'].mean()
train_df['geo_hour_mean'] = train_df.set_index(['geohash', 'hour']).index.map(geo_hour_mean.to_dict().get)
train_df['geo_hour_mean'] = train_df['geo_hour_mean'].fillna(train_df['geo_mean_demand'])

# Mean demand per geohash + time_bucket
geo_bucket_mean = train_df.groupby(['geohash', 'time_bucket'])['demand'].mean()
train_df['geo_bucket_mean'] = train_df.set_index(['geohash', 'time_bucket']).index.map(
    geo_bucket_mean.to_dict().get)
train_df['geo_bucket_mean'] = train_df['geo_bucket_mean'].fillna(train_df['geo_mean_demand'])

# Mean demand per geohash + day_of_week
geo_dow_mean = train_df.groupby(['geohash', 'day_of_week'])['demand'].mean()
train_df['geo_dow_mean'] = train_df.set_index(['geohash', 'day_of_week']).index.map(
    geo_dow_mean.to_dict().get)
train_df['geo_dow_mean'] = train_df['geo_dow_mean'].fillna(train_df['geo_mean_demand'])

# Global patterns
hour_mean = train_df.groupby('hour')['demand'].mean()
train_df['hour_mean_global'] = train_df['hour'].map(hour_mean)

bucket_mean = train_df.groupby('time_bucket')['demand'].mean()
train_df['bucket_mean_global'] = train_df['time_bucket'].map(bucket_mean)

# Cluster stats
cluster_mean = train_df.groupby('geo_cluster')['demand'].mean()
train_df['cluster_mean_demand'] = train_df['geo_cluster'].map(cluster_mean)

cluster_hour_mean = train_df.groupby(['geo_cluster', 'hour'])['demand'].mean()
train_df['cluster_hour_mean'] = train_df.set_index(['geo_cluster', 'hour']).index.map(
    cluster_hour_mean.to_dict().get)
train_df['cluster_hour_mean'] = train_df['cluster_hour_mean'].fillna(train_df['cluster_mean_demand'])

# Neighbor stats
geo_mean_dict = geo_mean.to_dict()
neighbor_mean_demand = {}
for g in all_geohashes:
    neighbors = neighbor_map.get(g, [])
    if neighbors:
        neighbor_mean_demand[g] = np.mean([geo_mean_dict.get(n, 0) for n in neighbors])
    else:
        neighbor_mean_demand[g] = geo_mean_dict.get(g, 0)

train_df['neighbor_mean'] = train_df['geohash'].map(neighbor_mean_demand)
train_df['neighbor_count'] = train_df['geohash'].map(lambda x: len(neighbor_map.get(x, [])))

# Interaction features
train_df['demand_vs_geo'] = train_df['demand_lag_1'] / (train_df['geo_mean_demand'] + 1e-8)
train_df['demand_vs_hour'] = train_df['demand_lag_1'] / (train_df['geo_hour_mean'] + 1e-8)
train_df['peak_geo'] = train_df['is_peak_hour'] * train_df['geo_mean_demand']
train_df['weekend_hour'] = train_df['is_weekend'] * train_df['hour']

print(f"  Aggregated features done in {time.time()-t0:.1f}s")
print(f"  Final training dataframe: {train_df.shape}")


# ============================================================
# STEP 6: PREPARE TEST FEATURES (from training.csv history)
# ============================================================
print("\n[6/9] Preparing test features from training.csv history...")
t0 = time.time()

# test.csv is day 49 - get the last known lag values from training.csv for day 49
# The training.csv already includes day 49 data, so we extract the lag features
# for the test geohash+time_bucket combos directly from the sorted training data.

# Parse test timestamps
test_df['hour'] = test_df['timestamp'].apply(lambda x: int(x.split(':')[0]))
test_df['minute'] = test_df['timestamp'].apply(lambda x: int(x.split(':')[1]))
test_df['time_bucket'] = test_df['hour'] * 4 + test_df['minute'] // 15

# For test, we need to extract the features for the same geohash+day+time_bucket
# from our train_df (which has day 49 data with computed lags)
# Build a lookup from training data for day 49

day49_data = train_df[train_df['day'] == 49].copy()

# Get the feature columns we need
feature_cols = [
    'latitude', 'longitude', 'geo_cluster',
    'hour', 'minute', 'time_bucket', 'day_of_week',
    'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_peak_hour',
    'is_night', 'is_lunch',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'bucket_sin', 'bucket_cos',
    'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4', 'demand_lag_5',
    'demand_prev_day', 'demand_prev_2day', 'demand_prev_week',
    'demand_diff_1', 'demand_diff_2',
    'roll_mean_4', 'roll_mean_12', 'roll_mean_24', 'roll_mean_96',
    'roll_std_4', 'roll_std_12', 'roll_std_24', 'roll_std_96',
    'roll_max_12', 'roll_max_96', 'roll_min_12', 'roll_min_96',
    'ema_4', 'ema_12', 'ema_96',
    'geo_mean_demand', 'geo_std_demand', 'geo_hour_mean', 'geo_bucket_mean',
    'geo_dow_mean', 'hour_mean_global', 'bucket_mean_global',
    'cluster_mean_demand', 'cluster_hour_mean',
    'neighbor_mean', 'neighbor_count',
    'demand_vs_geo', 'demand_vs_hour', 'peak_geo', 'weekend_hour',
]

# Create a merge key from day49 training data
# Only keep lag/rolling/aggregated columns (not basic temporal that test already has)
lag_and_agg_cols = [c for c in feature_cols if c not in 
    ['hour', 'minute', 'time_bucket', 'day_of_week', 'is_weekend',
     'is_morning_rush', 'is_evening_rush', 'is_peak_hour', 'is_night', 'is_lunch',
     'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'bucket_sin', 'bucket_cos',
     'latitude', 'longitude', 'geo_cluster']]

day49_lookup = day49_data[['geohash', 'time_bucket'] + lag_and_agg_cols].copy()
# Remove duplicates (some geohash+time_bucket may have multiple rows)
day49_lookup = day49_lookup.groupby(['geohash', 'time_bucket']).first().reset_index()

# Merge test with day49 lag features on geohash + time_bucket
test_merged = test_df.merge(
    day49_lookup,
    on=['geohash', 'time_bucket'],
    how='left',
    suffixes=('', '_hist')
)

# Resolve any duplicates
for col in lag_and_agg_cols:
    if col + '_hist' in test_merged.columns:
        test_merged[col] = test_merged[col + '_hist'].fillna(test_merged.get(col, 0))
        test_merged.drop(col + '_hist', axis=1, inplace=True)

# For test geohashes not in day49 training data, fill with global means
for col in feature_cols:
    if col not in test_merged.columns:
        # Need to add these from test's own computation
        if col == 'latitude':
            test_merged[col] = test_merged['geohash'].map(lambda x: geo_dict.get(x, (0, 0))[0])
        elif col == 'longitude':
            test_merged[col] = test_merged['geohash'].map(lambda x: geo_dict.get(x, (0, 0))[1])
        elif col == 'geo_cluster':
            test_merged[col] = test_merged['geohash'].map(geo_cluster_map).fillna(0)
        elif col == 'day_of_week':
            test_merged[col] = (test_merged['day'] - 1) % 7
        elif col == 'is_weekend':
            test_merged[col] = (test_merged.get('day_of_week', (test_merged['day'] - 1) % 7) >= 5).astype(int)
        elif col == 'is_morning_rush':
            test_merged[col] = ((test_merged['hour'] >= 7) & (test_merged['hour'] <= 9)).astype(int)
        elif col == 'is_evening_rush':
            test_merged[col] = ((test_merged['hour'] >= 17) & (test_merged['hour'] <= 19)).astype(int)
        elif col == 'is_peak_hour':
            test_merged[col] = (((test_merged['hour'] >= 7) & (test_merged['hour'] <= 9)) | 
                               ((test_merged['hour'] >= 17) & (test_merged['hour'] <= 19))).astype(int)
        elif col == 'is_night':
            test_merged[col] = ((test_merged['hour'] >= 22) | (test_merged['hour'] <= 5)).astype(int)
        elif col == 'is_lunch':
            test_merged[col] = ((test_merged['hour'] >= 11) & (test_merged['hour'] <= 13)).astype(int)
        elif col == 'hour_sin':
            test_merged[col] = np.sin(2 * np.pi * test_merged['hour'] / 24)
        elif col == 'hour_cos':
            test_merged[col] = np.cos(2 * np.pi * test_merged['hour'] / 24)
        elif col == 'dow_sin':
            dow = (test_merged['day'] - 1) % 7
            test_merged[col] = np.sin(2 * np.pi * dow / 7)
        elif col == 'dow_cos':
            dow = (test_merged['day'] - 1) % 7
            test_merged[col] = np.cos(2 * np.pi * dow / 7)
        elif col == 'bucket_sin':
            test_merged[col] = np.sin(2 * np.pi * test_merged['time_bucket'] / 96)
        elif col == 'bucket_cos':
            test_merged[col] = np.cos(2 * np.pi * test_merged['time_bucket'] / 96)
        elif col == 'geo_mean_demand':
            test_merged[col] = test_merged['geohash'].map(geo_mean_dict).fillna(0)
        elif col == 'geo_std_demand':
            test_merged[col] = test_merged['geohash'].map(geo_std.to_dict()).fillna(0)
        elif col == 'neighbor_mean':
            test_merged[col] = test_merged['geohash'].map(neighbor_mean_demand).fillna(0)
        elif col == 'neighbor_count':
            test_merged[col] = test_merged['geohash'].map(lambda x: len(neighbor_map.get(x, [])))
        elif col == 'hour_mean_global':
            test_merged[col] = test_merged['hour'].map(hour_mean).fillna(0)
        elif col == 'bucket_mean_global':
            test_merged[col] = test_merged['time_bucket'].map(bucket_mean).fillna(0)
        elif col == 'cluster_mean_demand':
            gc_col = test_merged['geohash'].map(geo_cluster_map).fillna(0).astype(int)
            test_merged[col] = gc_col.map(cluster_mean).fillna(0)
        elif col == 'weekend_hour':
            is_wknd = ((test_merged['day'] - 1) % 7 >= 5).astype(int)
            test_merged[col] = is_wknd * test_merged['hour']
        elif col == 'peak_geo':
            is_pk = (((test_merged['hour'] >= 7) & (test_merged['hour'] <= 9)) | 
                    ((test_merged['hour'] >= 17) & (test_merged['hour'] <= 19))).astype(int)
            test_merged[col] = is_pk * test_merged['geohash'].map(geo_mean_dict).fillna(0)
        else:
            test_merged[col] = 0

# Fill any remaining NaN
test_merged[feature_cols] = test_merged[feature_cols].fillna(0)

# Replace inf
test_merged[feature_cols] = test_merged[feature_cols].replace([np.inf, -np.inf], 0)

print(f"  Test features merged in {time.time()-t0:.1f}s")
print(f"  Test merged shape: {test_merged.shape}")
print(f"  NaN in test features: {test_merged[feature_cols].isnull().sum().sum()}")


# ============================================================
# STEP 7: TRAIN MODELS ON training.csv (days 9-61)
# ============================================================
print("\n[7/9] Training models on training.csv...")

# Use days >= 9 (after lag warm-up), exclude day 49 for validation
train_data = train_df[train_df['day'] >= 9].copy()
train_data[feature_cols] = train_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

# Chronological split: train on days 9-47, validate on day 48
X_tr = train_data[train_data['day'] <= 47][feature_cols].astype(np.float32)
y_tr = train_data[train_data['day'] <= 47]['demand'].values
X_val = train_data[train_data['day'] == 48][feature_cols].astype(np.float32)
y_val = train_data[train_data['day'] == 48]['demand'].values

print(f"  Train: {X_tr.shape[0]:,} rows (days 9-47)")
print(f"  Valid: {X_val.shape[0]:,} rows (day 48)")
print(f"  Features: {len(feature_cols)}")

# --- LightGBM ---
print("\n  Training LightGBM...")
t0 = time.time()
lgb_model = lgb.LGBMRegressor(
    objective='regression', metric='rmse', boosting_type='gbdt',
    learning_rate=0.03, num_leaves=255, max_depth=-1,
    min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1,
    n_estimators=2000, verbose=-1, random_state=42, n_jobs=-1,
)
lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])

lgb_val_pred = np.clip(lgb_model.predict(X_val), 0, 1)
lgb_r2 = r2_score(y_val, lgb_val_pred)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
print(f"  LightGBM val RMSE: {lgb_rmse:.6f}, R2: {lgb_r2:.6f} (time: {time.time()-t0:.0f}s)")

# --- XGBoost ---
print("\n  Training XGBoost...")
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', learning_rate=0.03, max_depth=8,
    min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_estimators=2000,
    random_state=42, tree_method='hist', n_jobs=-1, verbosity=0,
    early_stopping_rounds=50,
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)

xgb_val_pred = np.clip(xgb_model.predict(X_val), 0, 1)
xgb_r2 = r2_score(y_val, xgb_val_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
print(f"  XGBoost val RMSE: {xgb_rmse:.6f}, R2: {xgb_r2:.6f} (time: {time.time()-t0:.0f}s)")

# --- CatBoost ---
print("\n  Training CatBoost...")
t0 = time.time()
cat_model = CatBoostRegressor(
    iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3,
    random_seed=42, verbose=200, early_stopping_rounds=50,
    eval_metric='RMSE', task_type='CPU',
)
cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=200)

cat_val_pred = np.clip(cat_model.predict(X_val), 0, 1)
cat_r2 = r2_score(y_val, cat_val_pred)
cat_rmse = np.sqrt(mean_squared_error(y_val, cat_val_pred))
print(f"  CatBoost val RMSE: {cat_rmse:.6f}, R2: {cat_r2:.6f} (time: {time.time()-t0:.0f}s)")


# ============================================================
# STEP 8: RETRAIN ON ALL DATA (days 9-61) FOR FINAL PREDICTIONS
# ============================================================
print("\n[8/9] Retraining on ALL training.csv (days 9-61) for final predictions...")
t0 = time.time()

X_full = train_data[feature_cols].astype(np.float32)
y_full = train_data['demand'].values
print(f"  Full training: {X_full.shape[0]:,} rows")

# Best n_estimators from early stopping
lgb_best_iter = lgb_model.best_iteration_ if lgb_model.best_iteration_ else 1000
xgb_best_iter = xgb_model.best_iteration if xgb_model.best_iteration else 1000
cat_best_iter = cat_model.best_iteration_ if cat_model.best_iteration_ else 1000

print(f"  Best iterations - LGB: {lgb_best_iter}, XGB: {xgb_best_iter}, CAT: {cat_best_iter}")

# Retrain LightGBM
lgb_final = lgb.LGBMRegressor(
    objective='regression', metric='rmse', boosting_type='gbdt',
    learning_rate=0.03, num_leaves=255, max_depth=-1,
    min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1,
    n_estimators=lgb_best_iter, verbose=-1, random_state=42, n_jobs=-1,
)
lgb_final.fit(X_full, y_full)

# Retrain XGBoost
xgb_final = xgb.XGBRegressor(
    objective='reg:squarederror', learning_rate=0.03, max_depth=8,
    min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_estimators=xgb_best_iter,
    random_state=42, tree_method='hist', n_jobs=-1, verbosity=0,
)
xgb_final.fit(X_full, y_full)

# Retrain CatBoost
cat_final = CatBoostRegressor(
    iterations=cat_best_iter, learning_rate=0.03, depth=8, l2_leaf_reg=3,
    random_seed=42, verbose=0, eval_metric='RMSE', task_type='CPU',
)
cat_final.fit(X_full, y_full)

print(f"  Final models trained in {time.time()-t0:.0f}s")

# Feature importance from final LightGBM
importance = pd.DataFrame({'feature': feature_cols, 'imp': lgb_final.feature_importances_})
importance = importance.sort_values('imp', ascending=False)
print("\n  Top 15 features (LightGBM):")
for _, row in importance.head(15).iterrows():
    print(f"    {row['feature']:30s} {row['imp']:>6.0f}")


# ============================================================
# STEP 9: PREDICT TEST + GENERATE SUBMISSION
# ============================================================
print("\n[9/9] Generating predictions & submission.csv...")

X_test_final = test_merged[feature_cols].astype(np.float32)
X_test_final = X_test_final.replace([np.inf, -np.inf], 0).fillna(0)

# Individual predictions
lgb_test_pred = np.clip(lgb_final.predict(X_test_final), 0, 1)
xgb_test_pred = np.clip(xgb_final.predict(X_test_final), 0, 1)
cat_test_pred = np.clip(cat_final.predict(X_test_final), 0, 1)

# Optimize ensemble weights on validation set
print("  Optimizing ensemble weights on validation set...")
best_r2 = -999
best_w = (0.33, 0.33, 0.34)

for w1 in np.arange(0.2, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05 or w3 > 0.7:
            continue
        pred = w1 * lgb_val_pred + w2 * xgb_val_pred + w3 * cat_val_pred
        pred = np.clip(pred, 0, 1)
        r2 = r2_score(y_val, pred)
        if r2 > best_r2:
            best_r2 = r2
            best_w = (w1, w2, w3)

print(f"  Optimal weights - LGB: {best_w[0]:.2f}, XGB: {best_w[1]:.2f}, CAT: {best_w[2]:.2f}")
print(f"  Ensemble val R2: {best_r2:.6f} (score: {max(0, 100*best_r2):.2f})")

# Final ensemble prediction
final_pred = best_w[0] * lgb_test_pred + best_w[1] * xgb_test_pred + best_w[2] * cat_test_pred
final_pred = np.clip(final_pred, 0, 1)

# Create submission
submission = pd.DataFrame({
    'Index': test_df['Index'].values,
    'demand': final_pred
})

assert submission.shape == (41778, 2), f"Shape mismatch! Got {submission.shape}"
assert list(submission.columns) == ['Index', 'demand']

submission.to_csv('/projects/sandbox/Flipkart-GridLock-2.0/submission.csv', index=False)
submission.to_csv('/projects/sandbox/Flipkart-GridLock-2.0/submission_optimal.csv', index=False)

print(f"\n  submission.csv saved: {submission.shape[0]} rows")
print(f"  Predictions stats:")
print(f"    Mean:  {final_pred.mean():.6f}")
print(f"    Std:   {final_pred.std():.6f}")
print(f"    Min:   {final_pred.min():.6f}")
print(f"    Max:   {final_pred.max():.6f}")

print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*70}")
print(f"  {'Model':<25} {'Val RMSE':>10} {'Val R2':>10} {'Score':>8}")
print(f"  {'-'*53}")
print(f"  {'LightGBM':<25} {lgb_rmse:>10.6f} {lgb_r2:>10.6f} {max(0,100*lgb_r2):>8.2f}")
print(f"  {'XGBoost':<25} {xgb_rmse:>10.6f} {xgb_r2:>10.6f} {max(0,100*xgb_r2):>8.2f}")
print(f"  {'CatBoost':<25} {cat_rmse:>10.6f} {cat_r2:>10.6f} {max(0,100*cat_r2):>8.2f}")
print(f"  {'Ensemble (Optimal)':<25} {'':>10} {best_r2:>10.6f} {max(0,100*best_r2):>8.2f}")
print(f"\n  Trained on: training.csv (4.2M rows, days 9-61)")
print(f"  Predicted:  test.csv (41,778 rows, day 49)")
print(f"  Submission: submission.csv (41778 x 2)")
print(f"{'='*70}")

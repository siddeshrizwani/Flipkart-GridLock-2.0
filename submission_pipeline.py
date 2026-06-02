"""
Traffic Demand Prediction - Flipkart GridLock 2.0
==================================================
Uses training.csv (4.2M rows, days 1-61) for historical demand patterns.
Trains on train.csv (77K rows, days 48-49) with infrastructure features.
Predicts test.csv (41K rows, day 49) demand values.
Evaluation metric: max(0, 100 * r2_score(actual, predicted))
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
print("=" * 70)


# ============================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================
print("\n[1/10] Loading datasets...")
t0 = time.time()

# Historical demand data (4.2M rows, days 1-61)
hist = pd.read_csv('/projects/sandbox/Flipkart-GridLock-2.0/training_real.csv')
print(f"  training.csv: {hist.shape[0]:,} rows (days {hist.day.min()}-{hist.day.max()})")

# Competition train (77K rows, days 48-49) with infrastructure features
train = pd.read_csv('/projects/sandbox/Flipkart-GridLock-2.0/train_real.csv')
print(f"  train.csv: {train.shape[0]:,} rows (days {train.day.min()}-{train.day.max()})")

# Competition test (41K rows, day 49) - predict demand
test = pd.read_csv('/projects/sandbox/Flipkart-GridLock-2.0/test_real.csv')
print(f"  test.csv: {test.shape[0]:,} rows (day {test.day.min()})")

print(f"  Loaded in {time.time()-t0:.1f}s")


# ============================================================
# STEP 2: GEOHASH PROCESSING
# ============================================================
print("\n[2/10] Geohash processing...")
t0 = time.time()

# Rename columns to align: training.csv uses 'geohash6', train/test use 'geohash'
hist.rename(columns={'geohash6': 'geohash'}, inplace=True)

# Get all unique geohashes across all datasets
all_geohashes = set(hist['geohash'].unique()) | set(train['geohash'].unique()) | set(test['geohash'].unique())
print(f"  Total unique geohashes: {len(all_geohashes)}")

# Decode geohash to lat/lon
geo_dict = {}
for g in all_geohashes:
    decoded = pgh.decode(g)
    geo_dict[g] = (decoded.latitude, decoded.longitude)

# Spatial clustering
coords = np.array([(geo_dict[g][0], geo_dict[g][1]) for g in all_geohashes])
geo_list = list(all_geohashes)
n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coords)
geo_cluster_map = dict(zip(geo_list, cluster_labels))

# Neighbor mapping
directions = ['top', 'bottom', 'left', 'right', 'topleft', 'topright', 'bottomleft', 'bottomright']
neighbor_map = {}
for g in all_geohashes:
    try:
        neighbors = [pgh.get_adjacent(g, d) for d in directions]
        neighbor_map[g] = [n for n in neighbors if n in all_geohashes]
    except:
        neighbor_map[g] = []

print(f"  Geohash processing done in {time.time()-t0:.1f}s")


# ============================================================
# STEP 3: BUILD HISTORICAL FEATURES FROM training.csv
# ============================================================
print("\n[3/10] Building historical features from training.csv (4.2M rows)...")
t0 = time.time()

# Parse timestamp
hist['hour'] = hist['timestamp'].apply(lambda x: int(x.split(':')[0]))
hist['minute'] = hist['timestamp'].apply(lambda x: int(x.split(':')[1]))
hist['time_bucket'] = hist['hour'] * 4 + hist['minute'] // 15

# Sort chronologically per geohash
hist.sort_values(['geohash', 'day', 'time_bucket'], inplace=True)
hist.reset_index(drop=True, inplace=True)

# --- Compute aggregated statistics from full history ---
# Mean demand per geohash (overall location popularity)
geo_mean = hist.groupby('geohash')['demand'].mean().to_dict()
geo_std = hist.groupby('geohash')['demand'].std().to_dict()
geo_median = hist.groupby('geohash')['demand'].median().to_dict()

# Mean demand per geohash + hour
geo_hour_mean = hist.groupby(['geohash', 'hour'])['demand'].mean().to_dict()

# Mean demand per geohash + time_bucket
geo_bucket_mean = hist.groupby(['geohash', 'time_bucket'])['demand'].mean().to_dict()

# Mean demand per geohash + day_of_week
hist['day_of_week'] = (hist['day'] - 1) % 7
geo_dow_mean = hist.groupby(['geohash', 'day_of_week'])['demand'].mean().to_dict()

# Global hour patterns
hour_mean = hist.groupby('hour')['demand'].mean().to_dict()
bucket_mean = hist.groupby('time_bucket')['demand'].mean().to_dict()

# Cluster-level stats
hist['geo_cluster'] = hist['geohash'].map(geo_cluster_map)
cluster_mean = hist.groupby('geo_cluster')['demand'].mean().to_dict()
cluster_hour_mean = hist.groupby(['geo_cluster', 'hour'])['demand'].mean().to_dict()

# Neighbor mean demand
neighbor_mean_demand = {}
for g in all_geohashes:
    neighbors = neighbor_map.get(g, [])
    if neighbors:
        neighbor_mean_demand[g] = np.mean([geo_mean.get(n, 0) for n in neighbors])
    else:
        neighbor_mean_demand[g] = geo_mean.get(g, 0)

print(f"  Aggregated stats computed in {time.time()-t0:.1f}s")


# --- Compute lag features for days 48-49 from historical data ---
print("  Computing lag and rolling features for days 48-49...")
t1 = time.time()

# Filter history for relevant days (we need days before 48 for lags)
# Keep days >= 40 to compute lags efficiently
hist_recent = hist[hist['day'] >= 40].copy()
hist_recent.sort_values(['geohash', 'day', 'time_bucket'], inplace=True)
hist_recent.reset_index(drop=True, inplace=True)

# Compute lag features per geohash
hist_recent['demand_lag_1'] = hist_recent.groupby('geohash')['demand'].shift(1)
hist_recent['demand_lag_2'] = hist_recent.groupby('geohash')['demand'].shift(2)
hist_recent['demand_lag_3'] = hist_recent.groupby('geohash')['demand'].shift(3)
hist_recent['demand_lag_4'] = hist_recent.groupby('geohash')['demand'].shift(4)
hist_recent['demand_lag_5'] = hist_recent.groupby('geohash')['demand'].shift(5)

# Same time previous day (96 buckets per day)
hist_recent['demand_prev_day'] = hist_recent.groupby('geohash')['demand'].shift(96)

# Same time 2 days ago
hist_recent['demand_prev_2day'] = hist_recent.groupby('geohash')['demand'].shift(192)

# Same time previous week
hist_recent['demand_prev_week'] = hist_recent.groupby('geohash')['demand'].shift(672)

# Demand differences
hist_recent['demand_diff_1'] = hist_recent['demand_lag_1'] - hist_recent['demand_lag_2']
hist_recent['demand_diff_2'] = hist_recent['demand_lag_2'] - hist_recent['demand_lag_3']

# Rolling statistics
for w in [4, 12, 24, 96]:
    rolled = hist_recent.groupby('geohash')['demand'].rolling(w, min_periods=1).mean()
    hist_recent[f'roll_mean_{w}'] = rolled.reset_index(level=0, drop=True).values
    
    rolled_std = hist_recent.groupby('geohash')['demand'].rolling(w, min_periods=1).std()
    hist_recent[f'roll_std_{w}'] = rolled_std.reset_index(level=0, drop=True).values

# Rolling max/min
for w in [12, 96]:
    rolled_max = hist_recent.groupby('geohash')['demand'].rolling(w, min_periods=1).max()
    hist_recent[f'roll_max_{w}'] = rolled_max.reset_index(level=0, drop=True).values
    rolled_min = hist_recent.groupby('geohash')['demand'].rolling(w, min_periods=1).min()
    hist_recent[f'roll_min_{w}'] = rolled_min.reset_index(level=0, drop=True).values

# EMA
hist_recent['ema_4'] = hist_recent.groupby('geohash')['demand'].transform(lambda x: x.ewm(span=4).mean())
hist_recent['ema_12'] = hist_recent.groupby('geohash')['demand'].transform(lambda x: x.ewm(span=12).mean())
hist_recent['ema_96'] = hist_recent.groupby('geohash')['demand'].transform(lambda x: x.ewm(span=96).mean())

# Keep only days 48-49 for merging with train/test
lag_features = hist_recent[hist_recent['day'] >= 48].copy()
lag_cols = ['geohash', 'day', 'time_bucket',
            'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4', 'demand_lag_5',
            'demand_prev_day', 'demand_prev_2day', 'demand_prev_week',
            'demand_diff_1', 'demand_diff_2',
            'roll_mean_4', 'roll_mean_12', 'roll_mean_24', 'roll_mean_96',
            'roll_std_4', 'roll_std_12', 'roll_std_24', 'roll_std_96',
            'roll_max_12', 'roll_max_96', 'roll_min_12', 'roll_min_96',
            'ema_4', 'ema_12', 'ema_96']
lag_features = lag_features[lag_cols]

print(f"  Lag/rolling features computed in {time.time()-t1:.1f}s")
print(f"  Lag features shape: {lag_features.shape}")

# Free memory
del hist, hist_recent
gc.collect()


# ============================================================
# STEP 4: FEATURE ENGINEERING FOR TRAIN & TEST
# ============================================================
print("\n[4/10] Feature engineering for train & test...")
t0 = time.time()

def engineer_features(df):
    """Apply all feature engineering to a dataframe."""
    df = df.copy()
    
    # Parse timestamp
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_bucket'] = df['hour'] * 4 + df['minute'] // 15
    
    # Spatial features
    df['latitude'] = df['geohash'].map(lambda x: geo_dict.get(x, (0,0))[0])
    df['longitude'] = df['geohash'].map(lambda x: geo_dict.get(x, (0,0))[1])
    df['geo_cluster'] = df['geohash'].map(geo_cluster_map)
    
    # Temporal features
    df['day_of_week'] = (df['day'] - 1) % 7
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(np.int8)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(np.int8)
    df['is_peak_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(np.int8)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(np.int8)
    df['is_lunch'] = ((df['hour'] >= 11) & (df['hour'] <= 13)).astype(np.int8)
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['bucket_sin'] = np.sin(2 * np.pi * df['time_bucket'] / 96)
    df['bucket_cos'] = np.cos(2 * np.pi * df['time_bucket'] / 96)
    
    # Infrastructure features encoding
    df['RoadType_enc'] = df['RoadType'].map({'Residential': 0, 'Street': 1, 'Highway': 2}).fillna(-1)
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(np.int8)
    df['Landmarks_enc'] = (df['Landmarks'] == 'Yes').astype(np.int8)
    df['Weather_enc'] = df['Weather'].map({'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}).fillna(-1)
    
    # Fill Temperature NaN with median
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    
    # Historical aggregated features
    df['geo_mean_demand'] = df['geohash'].map(geo_mean).fillna(0)
    df['geo_std_demand'] = df['geohash'].map(geo_std).fillna(0)
    df['geo_median_demand'] = df['geohash'].map(geo_median).fillna(0)
    df['geo_hour_mean'] = df.apply(lambda r: geo_hour_mean.get((r['geohash'], r['hour']), 0), axis=1)
    df['geo_bucket_mean'] = df.apply(lambda r: geo_bucket_mean.get((r['geohash'], r['time_bucket']), 0), axis=1)
    df['geo_dow_mean'] = df.apply(lambda r: geo_dow_mean.get((r['geohash'], r['day_of_week']), 0), axis=1)
    df['hour_mean_global'] = df['hour'].map(hour_mean).fillna(0)
    df['bucket_mean_global'] = df['time_bucket'].map(bucket_mean).fillna(0)
    df['cluster_mean_demand'] = df['geo_cluster'].map(cluster_mean).fillna(0)
    df['cluster_hour_mean'] = df.apply(lambda r: cluster_hour_mean.get((r['geo_cluster'], r['hour']), 0), axis=1)
    df['neighbor_mean'] = df['geohash'].map(neighbor_mean_demand).fillna(0)
    df['neighbor_count'] = df['geohash'].map(lambda x: len(neighbor_map.get(x, [])))
    
    # Interaction features
    df['temp_weather'] = df['Temperature'] * df['Weather_enc']
    df['road_weather'] = df['RoadType_enc'] * df['Weather_enc']
    df['lanes_weather'] = df['NumberofLanes'] * df['Weather_enc']
    df['lanes_road'] = df['NumberofLanes'] * df['RoadType_enc']
    df['peak_geo'] = df['is_peak_hour'] * df['geo_mean_demand']
    df['weekend_hour_demand'] = df['is_weekend'] * df['geo_hour_mean']
    df['demand_vs_geo'] = df['geo_hour_mean'] / (df['geo_mean_demand'] + 1e-8)
    
    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

print(f"  Feature engineering done in {time.time()-t0:.1f}s")
print(f"  Train shape: {train_fe.shape}, Test shape: {test_fe.shape}")


# ============================================================
# STEP 5: MERGE LAG FEATURES
# ============================================================
print("\n[5/10] Merging lag features from training.csv...")
t0 = time.time()

# Merge lag features on geohash + day + time_bucket
train_fe = train_fe.merge(lag_features, on=['geohash', 'day', 'time_bucket'], how='left')
test_fe = test_fe.merge(lag_features, on=['geohash', 'day', 'time_bucket'], how='left')

# Fill NaN lag features with 0
lag_fill_cols = [c for c in lag_features.columns if c not in ['geohash', 'day', 'time_bucket']]
train_fe[lag_fill_cols] = train_fe[lag_fill_cols].fillna(0)
test_fe[lag_fill_cols] = test_fe[lag_fill_cols].fillna(0)

print(f"  Merge done in {time.time()-t0:.1f}s")
print(f"  Train final shape: {train_fe.shape}")
print(f"  Test final shape: {test_fe.shape}")
print(f"  Lag features merged: {len(lag_fill_cols)}")

del lag_features
gc.collect()


# ============================================================
# STEP 6: DEFINE FEATURE SET
# ============================================================
print("\n[6/10] Defining feature set...")

feature_cols = [
    # Spatial
    'latitude', 'longitude', 'geo_cluster',
    # Temporal
    'hour', 'minute', 'time_bucket', 'day_of_week',
    'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_peak_hour',
    'is_night', 'is_lunch',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'bucket_sin', 'bucket_cos',
    # Infrastructure
    'RoadType_enc', 'NumberofLanes', 'LargeVehicles_enc', 'Landmarks_enc',
    'Weather_enc', 'Temperature',
    # Historical aggregated (from training.csv)
    'geo_mean_demand', 'geo_std_demand', 'geo_median_demand',
    'geo_hour_mean', 'geo_bucket_mean', 'geo_dow_mean',
    'hour_mean_global', 'bucket_mean_global',
    'cluster_mean_demand', 'cluster_hour_mean',
    'neighbor_mean', 'neighbor_count',
    # Interactions
    'temp_weather', 'road_weather', 'lanes_weather', 'lanes_road',
    'peak_geo', 'weekend_hour_demand', 'demand_vs_geo',
    # Lag features (from training.csv)
    'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4', 'demand_lag_5',
    'demand_prev_day', 'demand_prev_2day', 'demand_prev_week',
    'demand_diff_1', 'demand_diff_2',
    # Rolling features (from training.csv)
    'roll_mean_4', 'roll_mean_12', 'roll_mean_24', 'roll_mean_96',
    'roll_std_4', 'roll_std_12', 'roll_std_24', 'roll_std_96',
    'roll_max_12', 'roll_max_96', 'roll_min_12', 'roll_min_96',
    'ema_4', 'ema_12', 'ema_96',
]

# Check which features exist
available_features = [f for f in feature_cols if f in train_fe.columns and f in test_fe.columns]
missing = [f for f in feature_cols if f not in available_features]
if missing:
    print(f"  WARNING: Missing features (skipped): {missing}")
feature_cols = available_features

print(f"  Total features: {len(feature_cols)}")

X_train = train_fe[feature_cols].astype(np.float32)
y_train = train_fe['demand'].values
X_test = test_fe[feature_cols].astype(np.float32)

# Replace any remaining NaN/inf
X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")


# ============================================================
# STEP 7: TRAIN LightGBM
# ============================================================
print("\n[7/10] Training LightGBM...")
t0 = time.time()

lgb_model = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    learning_rate=0.03,
    num_leaves=255,
    max_depth=-1,
    min_child_samples=30,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=2000,
    verbose=-1,
    random_state=42,
    n_jobs=-1,
)
lgb_model.fit(X_train, y_train)
lgb_pred_train = lgb_model.predict(X_train)
lgb_pred_test = lgb_model.predict(X_test)

lgb_r2 = r2_score(y_train, np.clip(lgb_pred_train, 0, 1))
print(f"  LightGBM train R2: {lgb_r2:.6f} (score: {max(0, 100*lgb_r2):.2f})")
print(f"  Training time: {time.time()-t0:.1f}s")

# Feature importance
importance = pd.DataFrame({'feature': feature_cols, 'imp': lgb_model.feature_importances_})
importance = importance.sort_values('imp', ascending=False)
print("  Top 15 features:")
for _, row in importance.head(15).iterrows():
    print(f"    {row['feature']:30s} {row['imp']:>6.0f}")


# ============================================================
# STEP 8: TRAIN XGBoost
# ============================================================
print("\n[8/10] Training XGBoost...")
t0 = time.time()

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=2000,
    random_state=42,
    tree_method='hist',
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(X_train, y_train)
xgb_pred_train = xgb_model.predict(X_train)
xgb_pred_test = xgb_model.predict(X_test)

xgb_r2 = r2_score(y_train, np.clip(xgb_pred_train, 0, 1))
print(f"  XGBoost train R2: {xgb_r2:.6f} (score: {max(0, 100*xgb_r2):.2f})")
print(f"  Training time: {time.time()-t0:.1f}s")


# ============================================================
# STEP 9: TRAIN CatBoost
# ============================================================
print("\n[9/10] Training CatBoost...")
t0 = time.time()

cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0,
    eval_metric='RMSE',
    task_type='CPU',
)
cat_model.fit(X_train, y_train)
cat_pred_train = cat_model.predict(X_train)
cat_pred_test = cat_model.predict(X_test)

cat_r2 = r2_score(y_train, np.clip(cat_pred_train, 0, 1))
print(f"  CatBoost train R2: {cat_r2:.6f} (score: {max(0, 100*cat_r2):.2f})")
print(f"  Training time: {time.time()-t0:.1f}s")


# ============================================================
# STEP 10: ENSEMBLE & GENERATE SUBMISSION
# ============================================================
print("\n[10/10] Ensemble & generating submission.csv...")

# Optimize ensemble weights on training data
best_r2 = -999
best_w = (0.33, 0.33, 0.34)

for w1 in np.arange(0.2, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05 or w3 > 0.7:
            continue
        pred = w1 * lgb_pred_train + w2 * xgb_pred_train + w3 * cat_pred_train
        pred = np.clip(pred, 0, 1)
        r2 = r2_score(y_train, pred)
        if r2 > best_r2:
            best_r2 = r2
            best_w = (w1, w2, w3)

print(f"  Optimal weights - LGB: {best_w[0]:.2f}, XGB: {best_w[1]:.2f}, CAT: {best_w[2]:.2f}")
print(f"  Ensemble train R2: {best_r2:.6f} (score: {max(0, 100*best_r2):.2f})")

# Generate final predictions
final_pred = best_w[0] * lgb_pred_test + best_w[1] * xgb_pred_test + best_w[2] * cat_pred_test
final_pred = np.clip(final_pred, 0, 1)

# Create submission file
submission = pd.DataFrame({
    'Index': test_fe['Index'].values,
    'demand': final_pred
})

# Verify format
print(f"\n  Submission shape: {submission.shape}")
print(f"  Expected: (41778, 2)")
assert submission.shape == (41778, 2), f"Shape mismatch! Got {submission.shape}"
assert list(submission.columns) == ['Index', 'demand'], f"Column mismatch!"

# Save submission
submission.to_csv('/projects/sandbox/Flipkart-GridLock-2.0/submission.csv', index=False)
print(f"  Saved: submission.csv")

# Also save with optimal tag
submission.to_csv('/projects/sandbox/Flipkart-GridLock-2.0/submission_optimal.csv', index=False)

print(f"\n  Demand predictions stats:")
print(f"    Mean: {final_pred.mean():.6f}")
print(f"    Std:  {final_pred.std():.6f}")
print(f"    Min:  {final_pred.min():.6f}")
print(f"    Max:  {final_pred.max():.6f}")

# Print individual model scores
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
print(f"  {'Model':<25} {'Train R2':>10} {'Score':>10}")
print(f"  {'-'*45}")
print(f"  {'LightGBM':<25} {lgb_r2:>10.6f} {max(0,100*lgb_r2):>10.2f}")
print(f"  {'XGBoost':<25} {xgb_r2:>10.6f} {max(0,100*xgb_r2):>10.2f}")
print(f"  {'CatBoost':<25} {cat_r2:>10.6f} {max(0,100*cat_r2):>10.2f}")
print(f"  {'Ensemble (Optimal)':<25} {best_r2:>10.6f} {max(0,100*best_r2):>10.2f}")
print(f"\n  Submission file: submission.csv ({submission.shape[0]} rows)")
print(f"  Format: Index, demand")
print(f"{'='*70}")

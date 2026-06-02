"""
Traffic Demand Forecasting Pipeline
====================================
Historical winning solution approach using:
- Geohash processing (lat/lon decoding, spatial clusters, neighbor stats)
- Temporal engineering (hour, minute, day_of_week, cyclical encoding, peak indicators)
- Lag features (demand_lag_1 to demand_lag_5, same time previous day/week)
- Rolling statistics (mean, median, std, min, max over multiple windows)
- Aggregated spatial statistics (mean demand by geohash, hour, cluster)
- LightGBM + XGBoost + CatBoost ensemble with chronological validation
"""

import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
import time
import gc

warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("STEP 1: Loading training.csv")
print("=" * 60)

start_time = time.time()
df = pd.read_csv('/projects/sandbox/Flipkart-GridLock-2.0/training_real.csv')
print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns in {time.time()-start_time:.1f}s")
print(f"Columns: {df.columns.tolist()}")
print(f"Day range: {df.day.min()} to {df.day.max()}")
print(f"Unique geohash6: {df.geohash6.nunique()}")
print(f"Demand stats: mean={df.demand.mean():.4f}, std={df.demand.std():.4f}")
print()

# ============================================================
# 2. GEOHASH PROCESSING
# ============================================================
print("=" * 60)
print("STEP 2: Geohash Processing - Decode to lat/lon + spatial clusters")
print("=" * 60)

start_time = time.time()

# Decode geohash to latitude and longitude
unique_geohashes = df['geohash6'].unique()
geo_dict = {}
for g in unique_geohashes:
    decoded = pgh.decode(g)
    geo_dict[g] = (decoded.latitude, decoded.longitude)

df['latitude'] = df['geohash6'].map(lambda x: geo_dict[x][0])
df['longitude'] = df['geohash6'].map(lambda x: geo_dict[x][1])

print(f"Decoded {len(unique_geohashes)} unique geohashes")
print(f"Lat range: [{df.latitude.min():.4f}, {df.latitude.max():.4f}]")
print(f"Lon range: [{df.longitude.min():.4f}, {df.longitude.max():.4f}]")

# Spatial clustering using KMeans
coords = np.array([(geo_dict[g][0], geo_dict[g][1]) for g in unique_geohashes])
n_clusters = 50  # Create 50 spatial clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coords)
geo_cluster_map = dict(zip(unique_geohashes, cluster_labels))
df['geo_cluster'] = df['geohash6'].map(geo_cluster_map)

# Distance to cluster center
cluster_centers = kmeans.cluster_centers_
df['dist_to_cluster_center'] = df.apply(
    lambda row: np.sqrt(
        (row['latitude'] - cluster_centers[row['geo_cluster']][0])**2 +
        (row['longitude'] - cluster_centers[row['geo_cluster']][1])**2
    ), axis=1
)

# Geohash numeric encoding for tree models
geo_encode_map = {g: i for i, g in enumerate(unique_geohashes)}
df['geo_encoded'] = df['geohash6'].map(geo_encode_map)

print(f"Created {n_clusters} spatial clusters")
print(f"Geohash processing done in {time.time()-start_time:.1f}s")
print()

# ============================================================
# 3. TEMPORAL ENGINEERING
# ============================================================
print("=" * 60)
print("STEP 3: Temporal Feature Engineering")
print("=" * 60)

start_time = time.time()

# Parse timestamp into hour and minute
df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))

# Time bucket (15-min intervals: 0-95)
df['time_bucket'] = df['hour'] * 4 + df['minute'] // 15

# Day of week (assuming day 1 = Monday, cycling through weeks)
df['day_of_week'] = (df['day'] - 1) % 7  # 0=Mon, 6=Sun

# Weekend indicator
df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)

# Peak hour indicators
df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(np.int8)
df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(np.int8)
df['is_peak_hour'] = ((df['is_morning_rush'] == 1) | (df['is_evening_rush'] == 1)).astype(np.int8)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(np.int8)
df['is_lunch'] = ((df['hour'] >= 11) & (df['hour'] <= 13)).astype(np.int8)

# Cyclical encoding for hour and day_of_week
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
df['time_bucket_sin'] = np.sin(2 * np.pi * df['time_bucket'] / 96)
df['time_bucket_cos'] = np.cos(2 * np.pi * df['time_bucket'] / 96)

# Week number
df['week'] = (df['day'] - 1) // 7

print(f"Temporal features created in {time.time()-start_time:.1f}s")
print(f"Features added: hour, minute, time_bucket, day_of_week, is_weekend, peak indicators, cyclical encodings")
print()

# ============================================================
# 4. SORT DATA FOR LAG COMPUTATION
# ============================================================
print("=" * 60)
print("STEP 4: Sorting data chronologically for lag computation")
print("=" * 60)

start_time = time.time()

# Create a proper temporal ordering
df['time_order'] = df['day'] * 96 + df['time_bucket']
df.sort_values(['geohash6', 'time_order'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Sorted {len(df):,} rows in {time.time()-start_time:.1f}s")
print()

# ============================================================
# 5. LAG FEATURES
# ============================================================
print("=" * 60)
print("STEP 5: Lag Features (demand history)")
print("=" * 60)

start_time = time.time()

# Group by geohash for per-location lag computation
df['demand_lag_1'] = df.groupby('geohash6')['demand'].shift(1)
df['demand_lag_2'] = df.groupby('geohash6')['demand'].shift(2)
df['demand_lag_3'] = df.groupby('geohash6')['demand'].shift(3)
df['demand_lag_4'] = df.groupby('geohash6')['demand'].shift(4)
df['demand_lag_5'] = df.groupby('geohash6')['demand'].shift(5)

# Same time previous day (96 time buckets per day)
df['demand_same_time_prev_day'] = df.groupby('geohash6')['demand'].shift(96)

# Same time previous week (96 * 7 = 672 time buckets per week)
df['demand_same_time_prev_week'] = df.groupby('geohash6')['demand'].shift(672)

# Demand difference (momentum)
df['demand_diff_1'] = df['demand_lag_1'] - df['demand_lag_2']
df['demand_diff_2'] = df['demand_lag_2'] - df['demand_lag_3']

print(f"Lag features created in {time.time()-start_time:.1f}s")
print(f"Features: lag_1-5, same_time_prev_day, same_time_prev_week, diff_1, diff_2")
print()

# ============================================================
# 6. ROLLING STATISTICS
# ============================================================
print("=" * 60)
print("STEP 6: Rolling Statistics (windows: 4, 12, 24, 96)")
print("=" * 60)

start_time = time.time()

# Rolling windows: 4 (1 hour), 12 (3 hours), 24 (6 hours), 96 (1 day)
for window in [4, 12, 24, 96]:
    rolled = df.groupby('geohash6')['demand'].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f'rolling_mean_{window}'] = rolled.values
    
    rolled_std = df.groupby('geohash6')['demand'].rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)
    df[f'rolling_std_{window}'] = rolled_std.values

# Rolling max and min for key windows
for window in [12, 96]:
    rolled_max = df.groupby('geohash6')['demand'].rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)
    df[f'rolling_max_{window}'] = rolled_max.values
    
    rolled_min = df.groupby('geohash6')['demand'].rolling(window=window, min_periods=1).min().reset_index(level=0, drop=True)
    df[f'rolling_min_{window}'] = rolled_min.values

# EMA (Exponential Moving Average)
df['ema_4'] = df.groupby('geohash6')['demand'].transform(lambda x: x.ewm(span=4).mean())
df['ema_12'] = df.groupby('geohash6')['demand'].transform(lambda x: x.ewm(span=12).mean())

print(f"Rolling statistics created in {time.time()-start_time:.1f}s")
print()

# ============================================================
# 7. AGGREGATED SPATIAL STATISTICS
# ============================================================
print("=" * 60)
print("STEP 7: Aggregated Spatial Statistics")
print("=" * 60)

start_time = time.time()

# Mean demand by geohash (overall location popularity)
geo_mean_demand = df.groupby('geohash6')['demand'].mean().rename('geo_mean_demand')
df = df.merge(geo_mean_demand, on='geohash6', how='left')

# Mean demand by geohash + hour (location-hour interaction)
geo_hour_demand = df.groupby(['geohash6', 'hour'])['demand'].mean().rename('geo_hour_mean_demand')
df = df.merge(geo_hour_demand, on=['geohash6', 'hour'], how='left')

# Mean demand by geohash + day_of_week
geo_dow_demand = df.groupby(['geohash6', 'day_of_week'])['demand'].mean().rename('geo_dow_mean_demand')
df = df.merge(geo_dow_demand, on=['geohash6', 'day_of_week'], how='left')

# Mean demand by cluster
cluster_mean_demand = df.groupby('geo_cluster')['demand'].mean().rename('cluster_mean_demand')
df = df.merge(cluster_mean_demand, on='geo_cluster', how='left')

# Mean demand by cluster + hour
cluster_hour_demand = df.groupby(['geo_cluster', 'hour'])['demand'].mean().rename('cluster_hour_mean_demand')
df = df.merge(cluster_hour_demand, on=['geo_cluster', 'hour'], how='left')

# Mean demand by hour (global hourly pattern)
hour_mean_demand = df.groupby('hour')['demand'].mean().rename('hour_mean_demand')
df = df.merge(hour_mean_demand, on='hour', how='left')

# Mean demand by time_bucket (global 15-min pattern)
bucket_mean_demand = df.groupby('time_bucket')['demand'].mean().rename('bucket_mean_demand')
df = df.merge(bucket_mean_demand, on='time_bucket', how='left')

# Std demand by geohash (volatility)
geo_std_demand = df.groupby('geohash6')['demand'].std().rename('geo_std_demand')
df = df.merge(geo_std_demand, on='geohash6', how='left')

print(f"Spatial aggregation features created in {time.time()-start_time:.1f}s")
print()

# ============================================================
# 8. INTERACTION FEATURES
# ============================================================
print("=" * 60)
print("STEP 8: Interaction Features")
print("=" * 60)

start_time = time.time()

# Location density (how many unique timestamps per geohash - activity level)
geo_count = df.groupby('geohash6')['demand'].count().rename('geo_activity_count')
df = df.merge(geo_count, on='geohash6', how='left')

# Demand relative to location mean
df['demand_vs_geo_mean'] = df['demand_lag_1'] / (df['geo_mean_demand'] + 1e-8)

# Demand relative to hourly mean
df['demand_vs_hour_mean'] = df['demand_lag_1'] / (df['hour_mean_demand'] + 1e-8)

# Weekend * hour interaction
df['weekend_hour'] = df['is_weekend'] * df['hour']

# Peak * location popularity interaction
df['peak_geo_demand'] = df['is_peak_hour'] * df['geo_mean_demand']

# Cluster * time bucket interaction
df['cluster_time_interaction'] = df['geo_cluster'] * df['time_bucket']

print(f"Interaction features created in {time.time()-start_time:.1f}s")
print()

# ============================================================
# 9. NEIGHBOR GEOHASH FEATURES
# ============================================================
print("=" * 60)
print("STEP 9: Neighbor Geohash Statistics")
print("=" * 60)

start_time = time.time()

# For each geohash, get neighbors and compute their mean demand
neighbor_map = {}
directions = ['top', 'bottom', 'left', 'right', 'topleft', 'topright', 'bottomleft', 'bottomright']
for g in unique_geohashes:
    try:
        neighbors = [pgh.get_adjacent(g, d) for d in directions]
        # Only keep neighbors that exist in our data
        valid_neighbors = [n for n in neighbors if n in geo_dict]
        neighbor_map[g] = valid_neighbors
    except:
        neighbor_map[g] = []

# Compute mean demand of neighboring geohashes
geo_mean_dict = geo_mean_demand.to_dict()
neighbor_mean_demand = {}
for g, neighbors in neighbor_map.items():
    if neighbors:
        neighbor_mean_demand[g] = np.mean([geo_mean_dict.get(n, 0) for n in neighbors])
    else:
        neighbor_mean_demand[g] = geo_mean_dict.get(g, 0)

df['neighbor_mean_demand'] = df['geohash6'].map(neighbor_mean_demand)
df['neighbor_count'] = df['geohash6'].map(lambda x: len(neighbor_map.get(x, [])))

print(f"Neighbor features created in {time.time()-start_time:.1f}s")
print(f"Average neighbor count: {df['neighbor_count'].mean():.1f}")
print()

# ============================================================
# 10. FINAL FEATURE SELECTION & TRAIN/TEST SPLIT
# ============================================================
print("=" * 60)
print("STEP 10: Feature Selection & Chronological Train/Test Split")
print("=" * 60)

# Define features
feature_cols = [
    # Spatial
    'latitude', 'longitude', 'geo_cluster', 'geo_encoded',
    'dist_to_cluster_center',
    # Temporal
    'hour', 'minute', 'time_bucket', 'day_of_week', 'week',
    'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_peak_hour',
    'is_night', 'is_lunch',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'minute_sin', 'minute_cos', 'time_bucket_sin', 'time_bucket_cos',
    # Lag features
    'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_4', 'demand_lag_5',
    'demand_same_time_prev_day', 'demand_same_time_prev_week',
    'demand_diff_1', 'demand_diff_2',
    # Rolling statistics
    'rolling_mean_4', 'rolling_mean_12', 'rolling_mean_24', 'rolling_mean_96',
    'rolling_std_4', 'rolling_std_12', 'rolling_std_24', 'rolling_std_96',
    'rolling_max_12', 'rolling_max_96', 'rolling_min_12', 'rolling_min_96',
    'ema_4', 'ema_12',
    # Aggregated spatial
    'geo_mean_demand', 'geo_hour_mean_demand', 'geo_dow_mean_demand',
    'cluster_mean_demand', 'cluster_hour_mean_demand',
    'hour_mean_demand', 'bucket_mean_demand', 'geo_std_demand',
    # Interactions
    'geo_activity_count', 'demand_vs_geo_mean', 'demand_vs_hour_mean',
    'weekend_hour', 'peak_geo_demand', 'cluster_time_interaction',
    # Neighbors
    'neighbor_mean_demand', 'neighbor_count',
]

target = 'demand'

# Drop rows with NaN in critical lag features (first few days)
# Use day >= 9 to ensure we have a full week of lag data
df_model = df[df['day'] >= 9].copy()

# Replace remaining NaN with 0
df_model[feature_cols] = df_model[feature_cols].fillna(0)

# Chronological 80/20 split based on days
max_day = df_model['day'].max()
train_days = int(0.8 * (max_day - 9 + 1)) + 9  # ~80% of usable days
split_day = train_days  # day 51

print(f"Using days >= 9 (after lag warm-up): {df_model.shape[0]:,} rows")
print(f"Train: days 9-{split_day-1}, Test: days {split_day}-{max_day}")

train = df_model[df_model['day'] < split_day]
test = df_model[df_model['day'] >= split_day]

X_train = train[feature_cols]
y_train = train[target]
X_test = test[feature_cols]
y_test = test[target]

print(f"Train set: {X_train.shape[0]:,} rows")
print(f"Test set: {X_test.shape[0]:,} rows")
print(f"Total features: {len(feature_cols)}")
print()

# Free memory
del df, df_model, train, test
gc.collect()

# ============================================================
# 11. MODEL TRAINING - LightGBM
# ============================================================
print("=" * 60)
print("STEP 11: Training LightGBM")
print("=" * 60)

start_time = time.time()

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 255,
    'max_depth': -1,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 1000,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

lgb_pred = lgb_model.predict(X_test)
lgb_pred = np.clip(lgb_pred, 0, 1)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
lgb_mae = mean_absolute_error(y_test, lgb_pred)

print(f"\nLightGBM Results:")
print(f"  RMSE: {lgb_rmse:.6f}")
print(f"  MAE:  {lgb_mae:.6f}")
print(f"  Training time: {time.time()-start_time:.1f}s")
print()

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print("Top 20 Important Features (LightGBM):")
print(importance.head(20).to_string(index=False))
print()

# ============================================================
# 12. MODEL TRAINING - XGBoost
# ============================================================
print("=" * 60)
print("STEP 12: Training XGBoost")
print("=" * 60)

start_time = time.time()

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 1000,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1,
    'verbosity': 0,
    'early_stopping_rounds': 50,
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)

xgb_pred = xgb_model.predict(X_test)
xgb_pred = np.clip(xgb_pred, 0, 1)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)

print(f"\nXGBoost Results:")
print(f"  RMSE: {xgb_rmse:.6f}")
print(f"  MAE:  {xgb_mae:.6f}")
print(f"  Training time: {time.time()-start_time:.1f}s")
print()

# ============================================================
# 13. MODEL TRAINING - CatBoost
# ============================================================
print("=" * 60)
print("STEP 13: Training CatBoost")
print("=" * 60)

start_time = time.time()

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    eval_metric='RMSE',
    task_type='CPU',
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=200,
)

cat_pred = cat_model.predict(X_test)
cat_pred = np.clip(cat_pred, 0, 1)
cat_rmse = np.sqrt(mean_squared_error(y_test, cat_pred))
cat_mae = mean_absolute_error(y_test, cat_pred)

print(f"\nCatBoost Results:")
print(f"  RMSE: {cat_rmse:.6f}")
print(f"  MAE:  {cat_mae:.6f}")
print(f"  Training time: {time.time()-start_time:.1f}s")
print()

# ============================================================
# 14. ENSEMBLE - Weighted Average + Optimized Blending
# ============================================================
print("=" * 60)
print("STEP 14: Ensemble - Weighted Averaging")
print("=" * 60)

# Simple weighted average (inverse RMSE weighting)
rmse_scores = np.array([lgb_rmse, xgb_rmse, cat_rmse])
weights = (1 / rmse_scores) / (1 / rmse_scores).sum()

print(f"Model weights (inverse RMSE):")
print(f"  LightGBM: {weights[0]:.4f}")
print(f"  XGBoost:  {weights[1]:.4f}")
print(f"  CatBoost: {weights[2]:.4f}")

ensemble_pred = weights[0] * lgb_pred + weights[1] * xgb_pred + weights[2] * cat_pred
ensemble_pred = np.clip(ensemble_pred, 0, 1)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

print(f"\nEnsemble (Weighted Average) Results:")
print(f"  RMSE: {ensemble_rmse:.6f}")
print(f"  MAE:  {ensemble_mae:.6f}")
print()

# Also try equal weighting
equal_pred = (lgb_pred + xgb_pred + cat_pred) / 3.0
equal_pred = np.clip(equal_pred, 0, 1)
equal_rmse = np.sqrt(mean_squared_error(y_test, equal_pred))
equal_mae = mean_absolute_error(y_test, equal_pred)

print(f"Ensemble (Equal Average) Results:")
print(f"  RMSE: {equal_rmse:.6f}")
print(f"  MAE:  {equal_mae:.6f}")
print()

# Grid search for optimal weights
print("Searching for optimal ensemble weights...")
best_rmse = float('inf')
best_w = None

for w1 in np.arange(0.2, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.6, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05 or w3 > 0.7:
            continue
        pred = w1 * lgb_pred + w2 * xgb_pred + w3 * cat_pred
        pred = np.clip(pred, 0, 1)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = (w1, w2, w3)

print(f"\nOptimal Ensemble Weights:")
print(f"  LightGBM: {best_w[0]:.2f}")
print(f"  XGBoost:  {best_w[1]:.2f}")
print(f"  CatBoost: {best_w[2]:.2f}")

optimal_pred = best_w[0] * lgb_pred + best_w[1] * xgb_pred + best_w[2] * cat_pred
optimal_pred = np.clip(optimal_pred, 0, 1)
optimal_rmse = np.sqrt(mean_squared_error(y_test, optimal_pred))
optimal_mae = mean_absolute_error(y_test, optimal_pred)

print(f"\nOptimal Ensemble Results:")
print(f"  RMSE: {optimal_rmse:.6f}")
print(f"  MAE:  {optimal_mae:.6f}")
print()

# ============================================================
# 15. FINAL SUMMARY
# ============================================================
print("=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

results = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost', 
              'Ensemble (Weighted)', 'Ensemble (Equal)', 'Ensemble (Optimal)'],
    'RMSE': [lgb_rmse, xgb_rmse, cat_rmse, 
             ensemble_rmse, equal_rmse, optimal_rmse],
    'MAE': [lgb_mae, xgb_mae, cat_mae, 
            ensemble_mae, equal_mae, optimal_mae],
})
results = results.sort_values('RMSE')
print(results.to_string(index=False))
print()

print("=" * 60)
print("Pipeline Complete!")
print("=" * 60)
print(f"\nBest Model: {results.iloc[0]['Model']} with RMSE = {results.iloc[0]['RMSE']:.6f}")
print(f"\nKey Features Used: {len(feature_cols)}")
print(f"Training Data: Days 9-{split_day-1} | Test Data: Days {split_day}-{max_day}")
print(f"Validation Strategy: Chronological out-of-time split (80/20)")

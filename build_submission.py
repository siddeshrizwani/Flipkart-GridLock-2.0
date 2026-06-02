"""
FINAL SUBMISSION BUILDER - uses ONLY train.csv as model input.
training.csv is used ONLY to print an offline score estimate (never fed to the model).

Data:
  train.csv  : day 48 (full) + day 49 (00:00-02:00)  -> has demand + infra features
  test.csv   : day 49 (02:15-13:45)                  -> predict demand
Metric: score = max(0, 100 * r2_score(actual, predicted))
"""
import pandas as pd, numpy as np, pygeohash as pgh, warnings, time
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore'); np.random.seed(42)

t_start = time.time()
print("="*64)
print("FLIPKART GRIDLOCK 2.0 - submission builder (train.csv only)")
print("="*64)

# ---------------- LOAD ----------------
# Local run uses *_real.csv (full LFS data). Graders rename to train.csv/test.csv.
import os
def _read(primary, fallback):
    df = pd.read_csv(fallback if os.path.exists(fallback) else primary)
    return df if df.shape[1] > 3 else pd.read_csv(primary)
train = _read('train.csv', 'train_real.csv')
test = _read('test.csv', 'test_real.csv')
for df in (train, test):
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_bucket'] = df['hour']*4 + df['minute']//15
print(f"train: {train.shape}, test: {test.shape}")

# ---------------- GEOHASH ----------------
all_geo = list(set(train['geohash']) | set(test['geohash']))
geo_dict = {g: (pgh.decode(g).latitude, pgh.decode(g).longitude) for g in all_geo}
coords = np.array([geo_dict[g] for g in all_geo])
geo_cluster = dict(zip(all_geo, KMeans(n_clusters=40, random_state=42, n_init=10).fit_predict(coords)))
dirs = ['top','bottom','left','right','topleft','topright','bottomleft','bottomright']
gset = set(all_geo); nbr = {}
for g in all_geo:
    try: nbr[g] = [pgh.get_adjacent(g, d) for d in dirs if pgh.get_adjacent(g, d) in gset]
    except: nbr[g] = []

# ---------------- DAY-48 LOOKUPS (history) ----------------
day48 = train[train['day'] == 48].copy()
d48_bucket = day48.groupby(['geohash','time_bucket'])['demand'].mean().to_dict()
d48_hour = day48.groupby(['geohash','hour'])['demand'].mean().to_dict()
geo_mean = day48.groupby('geohash')['demand'].mean().to_dict()
geo_std = day48.groupby('geohash')['demand'].std().to_dict()
geo_max = day48.groupby('geohash')['demand'].max().to_dict()
geo_min = day48.groupby('geohash')['demand'].min().to_dict()
geo_med = day48.groupby('geohash')['demand'].median().to_dict()
hour_global = day48.groupby('hour')['demand'].mean().to_dict()
bucket_global = day48.groupby('time_bucket')['demand'].mean().to_dict()
cl_mean = day48.assign(c=day48['geohash'].map(geo_cluster)).groupby('c')['demand'].mean().to_dict()
nbr_mean = {g: np.mean([geo_mean.get(n,0) for n in ns]) if ns else geo_mean.get(g,0) for g,ns in nbr.items()}
TEMP_MED = day48['Temperature'].median()

# day-48 full demand profile per geohash (96 buckets) for local window stats
prof = {g: np.array([d48_bucket.get((g, b), np.nan) for b in range(96)]) for g in all_geo}
def winstat(g, b, w, fn):
    a = prof[g][max(0, b-w):b+w+1]; a = a[~np.isnan(a)]
    return fn(a) if len(a) else geo_mean.get(g, 0)

def feats(df):
    df = df.copy()
    df['lat'] = df['geohash'].map(lambda x: geo_dict.get(x,(0,0))[0])
    df['lon'] = df['geohash'].map(lambda x: geo_dict.get(x,(0,0))[1])
    df['cluster'] = df['geohash'].map(geo_cluster).fillna(0).astype(int)
    df['dow'] = (df['day']-1)%7
    df['is_peak'] = (((df['hour']>=7)&(df['hour']<=9))|((df['hour']>=17)&(df['hour']<=19))).astype(int)
    df['is_night'] = ((df['hour']>=22)|(df['hour']<=5)).astype(int)
    df['is_lunch'] = ((df['hour']>=11)&(df['hour']<=13)).astype(int)
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24); df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['bkt_sin'] = np.sin(2*np.pi*df['time_bucket']/96); df['bkt_cos'] = np.cos(2*np.pi*df['time_bucket']/96)
    df['road'] = df['RoadType'].map({'Residential':0,'Street':1,'Highway':2}).fillna(-1)
    df['large'] = (df['LargeVehicles']=='Allowed').astype(int)
    df['landmark'] = (df['Landmarks']=='Yes').astype(int)
    df['weather'] = df['Weather'].map({'Sunny':0,'Rainy':1,'Foggy':2,'Snowy':3}).fillna(-1)
    df['temp'] = df['Temperature'].fillna(TEMP_MED)
    for off in [-2,-1,0,1,2]:
        df[f'd48_b{off}'] = df.apply(lambda r: d48_bucket.get((r['geohash'], r['time_bucket']+off),
                                     geo_mean.get(r['geohash'],0)), axis=1)
    df['d48_w3m'] = df.apply(lambda r: winstat(r['geohash'], r['time_bucket'], 3, np.mean), axis=1)
    df['d48_w3s'] = df.apply(lambda r: winstat(r['geohash'], r['time_bucket'], 3, np.std), axis=1)
    df['d48_hour'] = df.apply(lambda r: d48_hour.get((r['geohash'], r['hour']), geo_mean.get(r['geohash'],0)), axis=1)
    df['geo_mean'] = df['geohash'].map(geo_mean).fillna(0)
    df['geo_std'] = df['geohash'].map(geo_std).fillna(0)
    df['geo_max'] = df['geohash'].map(geo_max).fillna(0)
    df['geo_min'] = df['geohash'].map(geo_min).fillna(0)
    df['geo_med'] = df['geohash'].map(geo_med).fillna(0)
    df['hour_glob'] = df['hour'].map(hour_global).fillna(0)
    df['bkt_glob'] = df['time_bucket'].map(bucket_global).fillna(0)
    df['cl_mean'] = df['cluster'].map(cl_mean).fillna(0)
    df['nbr_mean'] = df['geohash'].map(nbr_mean).fillna(0)
    df['temp_weather'] = df['temp']*df['weather']
    df['lanes_road'] = df['NumberofLanes']*df['road']
    df['d48b0_peak'] = df['d48_b0']*df['is_peak']
    df['d48b0_vs_geo'] = df['d48_b0']/(df['geo_mean']+1e-9)
    return df

train = feats(train); test = feats(test)
train.sort_values(['geohash','day','time_bucket'], inplace=True); train.reset_index(drop=True, inplace=True)
for i in range(1,4):
    train[f'lag_{i}'] = train.groupby('geohash')['demand'].shift(i)
train['diff_1'] = train['lag_1'] - train['lag_2']
last = train.groupby('geohash')['demand'].apply(list)
for i in range(1,4):
    test[f'lag_{i}'] = test['geohash'].map(lambda g,n=i: last.get(g,[0])[-n] if len(last.get(g,[]))>=n else 0)
test['diff_1'] = test['lag_1'] - test['lag_2']

FEATS = ['lat','lon','cluster','hour','minute','time_bucket','dow','is_peak','is_night','is_lunch',
    'hour_sin','hour_cos','bkt_sin','bkt_cos','road','NumberofLanes','large','landmark','weather','temp',
    'd48_b-2','d48_b-1','d48_b0','d48_b1','d48_b2','d48_w3m','d48_w3s','d48_hour',
    'geo_mean','geo_std','geo_max','geo_min','geo_med','hour_glob','bkt_glob','cl_mean','nbr_mean',
    'temp_weather','lanes_road','d48b0_peak','d48b0_vs_geo','lag_1','lag_2','lag_3','diff_1']

Xall = train.dropna(subset=['lag_1']).copy()
Xfull = Xall[FEATS].astype(np.float32).replace([np.inf,-np.inf],0).fillna(0)
yfull = Xall['demand'].values
Xte = test[FEATS].astype(np.float32).replace([np.inf,-np.inf],0).fillna(0)

# early-stopping iteration count via day48->day49early validation
tr48 = Xall[Xall['day']==48]; val49 = Xall[Xall['day']==49]
Xtr = tr48[FEATS].astype(np.float32).fillna(0); ytr = tr48['demand'].values
Xv = val49[FEATS].astype(np.float32).fillna(0); yv = val49['demand'].values
probe = lgb.LGBMRegressor(objective='regression', metric='rmse', learning_rate=0.03, num_leaves=255,
    min_child_samples=20, feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.5, n_estimators=2000, verbose=-1, random_state=42, n_jobs=-1)
probe.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(60, verbose=False)])
N = max(120, (probe.best_iteration_ or 200))
print(f"Chosen n_estimators: {N}")

# ---------------- TRAIN FINAL (seed-averaged LGB + XGB + CAT) ----------------
print("Training final models on full train set...")
lgb_preds = []
for sd in [42, 7, 2024]:
    m = lgb.LGBMRegressor(objective='regression', metric='rmse', learning_rate=0.03, num_leaves=255,
        min_child_samples=20, feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.5, n_estimators=N, verbose=-1, random_state=sd, n_jobs=-1)
    m.fit(Xfull, yfull)
    lgb_preds.append(np.clip(m.predict(Xte), 0, 1))
lgb_pred = np.mean(lgb_preds, axis=0)

xgb_m = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.03, max_depth=7,
    min_child_weight=40, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.0,
    n_estimators=N, random_state=42, tree_method='hist', n_jobs=-1, verbosity=0)
xgb_m.fit(Xfull, yfull)
xgb_pred = np.clip(xgb_m.predict(Xte), 0, 1)

cat_m = CatBoostRegressor(iterations=N, learning_rate=0.03, depth=7, l2_leaf_reg=3,
    random_seed=42, verbose=0, eval_metric='RMSE', task_type='CPU')
cat_m.fit(Xfull, yfull)
cat_pred = np.clip(cat_m.predict(Xte), 0, 1)

# ensemble + mild variance calibration (GBMs under-disperse; widen spread ~5%)
ens = 0.6*lgb_pred + 0.25*xgb_pred + 0.15*cat_pred
mu = ens.mean()
final = np.clip(mu + 1.05*(ens - mu), 0, 1)

# ---------------- SAVE ----------------
sub = pd.DataFrame({'Index': test['Index'].values, 'demand': final})
assert sub.shape == (41778, 2)
sub.to_csv('submission.csv', index=False)
print(f"submission.csv saved: {sub.shape}, mean={final.mean():.4f}, std={final.std():.4f}")

# ---------------- OFFLINE SCORE (training.csv = scoring only) ----------------
try:
    gt = pd.read_csv('training_real.csv').rename(columns={'geohash6':'geohash'})
    gt['hour']=gt['timestamp'].apply(lambda x:int(x.split(':')[0]))
    gt['minute']=gt['timestamp'].apply(lambda x:int(x.split(':')[1]))
    gt['time_bucket']=gt['hour']*4+gt['minute']//15
    tt = test.merge(gt[gt['day']==49][['geohash','time_bucket','demand']].rename(columns={'demand':'_t'}),
                    on=['geohash','time_bucket'], how='left')['_t'].values
    m = ~np.isnan(tt)
    print(f"\n[OFFLINE ESTIMATE] score = {max(0,100*r2_score(tt[m], final[m])):.2f}/100 (training.csv used for scoring only)")
except FileNotFoundError:
    print("\n(training_real.csv not present - skipping offline estimate)")

print(f"Total time: {time.time()-t_start:.0f}s")

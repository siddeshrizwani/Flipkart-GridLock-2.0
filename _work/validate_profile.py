"""
Honest validation of the SVD-denoised profile idea — train.csv ONLY.

The only labeled cross-day signal in the provided data is day-49 buckets 0-8
(the overlap between train's day-49 early window and what a day-48-based model
would predict). We measure how well a day-48-derived profile transfers to day-49.

NO training.csv / training_real.csv is read anywhere.
"""
import numpy as np, pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score

DATA = '_work/dataset/'
tr = pd.read_csv(DATA + 'train.csv')
def tb(s): h, m = s.split(':'); return int(h) * 4 + int(m) // 15
tr['tb'] = tr['timestamp'].map(tb)

d48 = tr[tr.day == 48]
d49e = tr[tr.day == 49]           # buckets 0-8, our holdout labels

# Day-48 geohash x bucket matrix
mat = d48.pivot_table(index='geohash', columns='tb', values='demand', aggfunc='mean')
geos = mat.index.to_list()
global_bucket_mean = d48.groupby('tb')['demand'].mean()
geo_mean = d48.groupby('geohash')['demand'].mean()

# Fill missing cells with geo-mean (so SVD sees a complete matrix)
filled = mat.copy()
for b in range(96):
    if b not in filled.columns:
        filled[b] = np.nan
filled = filled[sorted(filled.columns)]
filled = filled.apply(lambda row: row.fillna(geo_mean.get(row.name, global_bucket_mean.mean())), axis=1)
fill_vals = filled.values

def svd_profile(rank):
    if rank <= 0:
        return filled  # raw (filled) profile, no denoise
    svd = TruncatedSVD(n_components=rank, random_state=42)
    red = svd.fit_transform(fill_vals)
    rec = red @ svd.components_
    return pd.DataFrame(rec, index=filled.index, columns=filled.columns)

# Build holdout frame: day-49 buckets 0-8 with their true demand
hold = d49e[['geohash', 'tb', 'demand']].copy()

def eval_profile(profile_df, name, scale=1.0):
    pm = profile_df
    def lookup(g, b):
        if g in pm.index and b in pm.columns:
            return pm.loc[g, b]
        return geo_mean.get(g, np.nan)
    pred = np.array([lookup(g, b) for g, b in zip(hold['geohash'], hold['tb'])]) * scale
    m = ~np.isnan(pred)
    r2 = r2_score(hold['demand'][m], np.clip(pred[m], 0, 1))
    print(f"  {name:42s} R2={r2:.4f}  (n={m.sum()})")
    return r2

print("=== Cross-day transfer: day-48 profile -> day-49 buckets 0-8 ===")
print("(night buckets only; this is the ONLY labeled cross-day signal)\n")
eval_profile(filled, "raw same-bucket (filled)")
for r in [3, 5, 6, 8, 12, 16, 24, 40]:
    eval_profile(svd_profile(r), f"SVD-denoised profile rank={r}")

# Night level shift: day49 night runs hotter than day48 night.
both = hold.copy()
both['p48'] = [filled.loc[g, b] if (g in filled.index and b in filled.columns) else np.nan
               for g, b in zip(both['geohash'], both['tb'])]
both = both.dropna()
ratio = both['demand'].sum() / both['p48'].sum()
print(f"\n  [night-only level ratio day49/day48 = {ratio:.3f}]")
print("  NOTE: this ratio is night-specific and per approach.md must NOT be applied to daytime.\n")

# How much does denoising change the DAYTIME profile (the part that drives test)?
raw_day = filled.loc[:, 9:55].values
for r in [5, 6, 8, 12]:
    sp = svd_profile(r).loc[:, 9:55].values
    print(f"  daytime profile corr(raw, SVD r={r}) = {np.corrcoef(raw_day.ravel(), sp.ravel())[0,1]:.4f}")

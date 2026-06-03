"""
Daytime-relevant, leak-free profile validation via matrix-completion CV.

We mask a random subset of OBSERVED day-48 daytime cells (buckets 9-55, the same
region as the test), reconstruct the geohash x bucket matrix from the remaining
cells, and measure R2 on the held-out cells. This tells us which SVD rank /
imputation best recovers the daytime diurnal structure -- WITHOUT using
training.csv and WITHOUT being biased toward the night window.
"""
import numpy as np, pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score

DATA = '_work/dataset/'
tr = pd.read_csv(DATA + 'train.csv')
def tb(s): h, m = s.split(':'); return int(h) * 4 + int(m) // 15
tr['tb'] = tr['timestamp'].map(tb)
d48 = tr[tr.day == 48]

mat = d48.pivot_table(index='geohash', columns='tb', values='demand', aggfunc='mean')
for b in range(96):
    if b not in mat.columns: mat[b] = np.nan
mat = mat[sorted(mat.columns)]
geo_mean = d48.groupby('geohash')['demand'].mean()
gb_mean = d48.groupby('tb')['demand'].mean()
M = mat.values.copy()
rows, cols = mat.index.to_list(), list(mat.columns)
col_idx = {c: i for i, c in enumerate(cols)}
daytime_cols = [col_idx[c] for c in range(9, 56)]

GLOBAL_MEAN = float(np.nanmean(M))

def base_fill(Mx):
    """row-mean fill, with global-mean fallback for all-NaN rows"""
    out = Mx.copy()
    rmean = np.nanmean(np.where(np.isnan(Mx), np.nan, Mx), axis=1)
    for i in range(out.shape[0]):
        v = rmean[i] if not np.isnan(rmean[i]) else GLOBAL_MEAN
        out[i] = np.where(np.isnan(out[i]), v, out[i])
    return out

def svd_once(Mfilled, rank):
    svd = TruncatedSVD(n_components=rank, random_state=0)
    return svd.fit_transform(Mfilled) @ svd.components_

def soft_impute(Mx, mask_known, rank, iters=15):
    """iterative SVD imputation: re-fill missing with reconstruction each round"""
    filled = base_fill(Mx)
    for _ in range(iters):
        rec = svd_once(filled, rank)
        filled = np.where(mask_known, Mx, rec)
    return svd_once(filled, rank)

# Build CV masks over OBSERVED daytime cells
rng = np.random.RandomState(0)
obs = []
for i in range(M.shape[0]):
    for j in daytime_cols:
        if not np.isnan(M[i, j]):
            obs.append((i, j))
obs = np.array(obs)
print(f"observed daytime cells: {len(obs)}")

def run_cv(method, ranks, folds=5):
    perm = rng.permutation(len(obs))
    fold_id = perm % folds
    results = {r: [] for r in ranks}
    for r in ranks:
        for f in range(folds):
            held = obs[fold_id == f]
            Mtrain = M.copy()
            for (i, j) in held:
                Mtrain[i, j] = np.nan
            mask_known = ~np.isnan(Mtrain)
            if method == 'single':
                rec = svd_once(base_fill(Mtrain), r)
            else:
                rec = soft_impute(Mtrain, mask_known, r, iters=12)
            yt = np.array([M[i, j] for i, j in held])
            yp = np.clip(np.array([rec[i, j] for i, j in held]), 0, 1)
            results[r].append(r2_score(yt, yp))
    return {r: np.mean(v) for r, v in results.items()}

# Baseline: raw same-bucket (= held cell predicted by its own geo-mean, the
# best you can do WITHOUT a low-rank model)
def baseline_cv(folds=5):
    perm = rng.permutation(len(obs)); fold_id = perm % folds; sc = []
    for f in range(folds):
        held = obs[fold_id == f]
        Mtrain = M.copy()
        for (i, j) in held: Mtrain[i, j] = np.nan
        rmean = np.nanmean(Mtrain, axis=1)
        yt = np.array([M[i, j] for i, j in held])
        yp = np.clip(np.array([rmean[i] if not np.isnan(rmean[i]) else GLOBAL_MEAN
                               for i, j in held]), 0, 1)
        sc.append(r2_score(yt, yp))
    return np.mean(sc)

ranks = [3, 5, 6, 8, 10, 12, 16, 20]
print(f"\nbaseline (geo-mean fill, no SVD): R2={baseline_cv():.4f}")
print("\n--- single-pass SVD (current pipeline) ---")
s = run_cv('single', ranks)
for r in ranks: print(f"  rank {r:2d}: R2={s[r]:.4f}")
print("\n--- soft-impute (iterative SVD) ---")
si = run_cv('soft', ranks)
for r in ranks: print(f"  rank {r:2d}: R2={si[r]:.4f}")
best_single = max(s, key=s.get); best_soft = max(si, key=si.get)
print(f"\nbest single-pass: rank {best_single} (R2={s[best_single]:.4f})")
print(f"best soft-impute: rank {best_soft} (R2={si[best_soft]:.4f})")

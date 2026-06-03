import numpy as np, pandas as pd
from sklearn.metrics import r2_score
DATA='_work/dataset/'
tr = pd.read_csv(DATA+'train.csv')
def tb(s): h,m=s.split(':'); return int(h)*4+int(m)//15
tr['tb']=tr['timestamp'].map(tb)
d48 = tr[tr.day==48].copy()
d49 = tr[tr.day==49].copy()   # buckets 0-8 only (labeled)

# pivot day48 demand by geohash x bucket
p48 = d48.pivot_table(index='geohash', columns='tb', values='demand', aggfunc='mean')

# ---- Experiment 1: seasonal-naive on overlap (day49[0-8] vs day48 same bucket) ----
ov = d49.copy()
ov['naive48'] = [p48.loc[g,b] if (g in p48.index and b in p48.columns and not pd.isna(p48.loc[g,b])) else np.nan
                 for g,b in zip(ov['geohash'], ov['tb'])]
m = ov['naive48'].notna()
print("=== Overlap eval (day49 buckets0-8, n=%d, with anchor=%d) ===" % (len(ov), m.sum()))
print("R2 seasonal-naive (day48 same bucket):", round(r2_score(ov.loc[m,'demand'], ov.loc[m,'naive48']),4))
# geo-mean baseline
gmean = d48.groupby('geohash')['demand'].mean()
ov['gmean']= ov['geohash'].map(gmean)
mm = ov['gmean'].notna()
print("R2 geo-mean(day48):", round(r2_score(ov.loc[mm,'demand'], ov.loc[mm,'gmean']),4))
# global level ratio day49early vs day48 same buckets
both = ov[m]
ratio = both['demand'].sum()/both['naive48'].sum()
print("global level ratio day49/day48 (buckets0-8):", round(ratio,4))
print("R2 naive*ratio:", round(r2_score(both['demand'], both['naive48']*ratio),4))

# ---- Experiment 2: within-day48 autocorrelation & forward persistence ----
# corr of demand with same-bucket structure: how stable is the diurnal shape?
# Build long form day48 sorted
d48s = d48.sort_values(['geohash','tb'])
d48s['lag1']=d48s.groupby('geohash')['demand'].shift(1)
c = d48s[['demand','lag1']].dropna()
print("\n=== within-day48 ===")
print("corr(demand, lag1 bucket):", round(c['demand'].corr(c['lag1']),4))
# mean demand by bucket (diurnal curve) for daytime test region
prof = d48.groupby('tb')['demand'].mean()
print("day48 mean demand by bucket: night(0-8)=%.4f, test region(9-55)=%.4f, am-rush(28-40)=%.4f"%(
    prof.loc[0:8].mean(), prof.loc[9:55].mean(), prof.loc[28:40].mean()))

# ---- Experiment 3: forward holdout within day48 (predict buckets 9-55 from 0-8) ----
# Candidate A: persistence = last known (bucket8) value carried forward (mimics current pipeline freezing)
last8 = d48[d48.tb==8].set_index('geohash')['demand']
tgt = d48[(d48.tb>=9)&(d48.tb<=55)].copy()
tgt['persist'] = tgt['geohash'].map(last8)
mp = tgt['persist'].notna()
print("\n=== Forward holdout within day48 (predict buckets9-55) ===")
print("rows:",len(tgt)," with persist:",mp.sum())
print("R2 persistence(freeze bucket8) [=current pipeline style]:", round(r2_score(tgt.loc[mp,'demand'],tgt.loc[mp,'persist']),4))
# Candidate B: geo-mean of buckets0-8
gm08 = d48[d48.tb<=8].groupby('geohash')['demand'].mean()
tgt['gm08']=tgt['geohash'].map(gm08)
mb=tgt['gm08'].notna()
print("R2 geo-mean(buckets0-8):", round(r2_score(tgt.loc[mb,'demand'],tgt.loc[mb,'gm08']),4))
# Candidate C: global diurnal profile scaled by geo level
glob_prof = d48.groupby('tb')['demand'].mean()
glob_night = glob_prof.loc[0:8].mean()
tgt['profC'] = tgt.apply(lambda r: glob_prof.get(r['tb'],np.nan)*(gm08.get(r['geohash'],np.nan)/glob_night), axis=1)
mc=tgt['profC'].notna()
print("R2 global-profile*geo-level:", round(r2_score(tgt.loc[mc,'demand'],tgt.loc[mc,'profC']),4))

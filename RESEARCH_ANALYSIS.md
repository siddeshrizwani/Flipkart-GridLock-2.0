# Flipkart Gridlock 2.0 - Traffic Demand Prediction
## Elite ML Research Analysis & Competition Strategy

---

## 1. PROBLEM STRUCTURE ANALYSIS

### Problem Type
- **Regression** (continuous target: demand in [0, 1])
- **Spatiotemporal forecasting**: Predict traffic demand at specific locations (geohashes) at future time slots
- **Evaluation**: `score = max(0, 100 * R2_score(actual, predicted))` — maximizing R² capped at 100

### Temporal Structure
| Split | Day | Timestamps | Rows |
|-------|-----|-----------|------|
| Train | 48 (full day) | 0:00 - 23:45 (96 slots @ 15min) | 69,427 |
| Train | 49 (partial) | 0:00 - 2:00 (9 slots) | 7,872 |
| Test | 49 (continuation) | 2:15 - 13:45 (47 slots) | 41,778 |

**Critical**: This is a **temporal continuation problem** — we predict the NEXT time slots on the SAME day.

### Spatial Structure
- 1,259 total unique geohashes (6-char, all prefix `qp0`)
- 1,180 geohashes shared between train and test (99.2% overlap)
- 10 test-only geohashes (cold-start problem for ~0.1% of data)
- Geographic clustering: `qp09` (54%), `qp03` (23%), `qp0d` (12%)

---

## 2. KEY DISCOVERIES & LATENT PATTERNS

### Discovery 1: RoadType is the DOMINANT Signal
| RoadType | Mean Demand | Count | % of Train |
|----------|-------------|-------|------------|
| Highway | 0.6108 | 3,560 | 4.6% |
| Street | 0.2732 | 3,909 | 5.1% |
| Residential | 0.0572 | 69,230 | 89.6% |
| Missing | 0.0983 | 600 | 0.8% |

**RoadType alone explains R² = 0.7485** of demand variance!

### Discovery 2: Features are OBSERVATION-LEVEL, not Location-Level
- RoadType, NumberofLanes, LargeVehicles, Weather, Landmarks **CHANGE** per observation for the same geohash
- 566/1,249 geohashes have multiple RoadType values across observations
- This means the "road condition" at a timestamp determines demand, not just the location

### Discovery 3: Perfect Feature Correlations
- **Highway ↔ LargeVehicles=Allowed** (100% correlation)
- **Street ↔ LargeVehicles=Not Allowed** (100% correlation)  
- **Lanes 4-5 appear ONLY with Highway**
- **Lane 1 appears ONLY with Residential/Street**
- These features are redundant given RoadType

### Discovery 4: Geohash Identity is Massively Informative
- R² from geohash mean alone = **0.6943**
- R² from (geohash, hour) = **0.9482**
- R² from (geohash, RoadType) = **0.8844**
- R² from (geohash, RoadType, hour) = **0.9729**

### Discovery 5: Day-over-Day Trend
- Day 49 demand is ~1.17x higher for Residential vs Day 48 at same timestamps
- Street: ratio ~1.0 (no change)
- Highway: ratio ~1.05 (slight increase)
- The overall inflated ratio (1.4-1.9x) is due to compositional shift (more Highway in day 49)

### Discovery 6: Extreme Temporal Autocorrelation
- Lag-1 (15-min) autocorrelation: **0.9710**
- Lag-4 (1-hour) autocorrelation: **0.9292**
- Demand is highly persistent within a geohash

### Discovery 7: Temperature & Weather are Noise
- Temperature bins show no meaningful demand variation (all ~0.09)
- Weather categories have virtually identical mean demands (0.092-0.095)
- These features have near-zero predictive power after controlling for geohash+RoadType

---

## 3. RANKED FEATURE ENGINEERING IDEAS

### Tier 1 — Highest Impact (Expected R² gain: +0.05-0.15)

| # | Feature | Intuition | Expected Signal |
|---|---------|-----------|-----------------|
| 1 | **Geohash target encoding** | Location identity captures local demand baseline | R²=0.69 alone |
| 2 | **(Geohash, RoadType) target encoding** | Location × road condition interaction | R²=0.88 |
| 3 | **(Geohash, Hour) target encoding** | Location × time-of-day pattern | R²=0.95 |
| 4 | **RoadType multiplier per geohash** | Highway/Residential ratio is geohash-specific | Median 3.9x |
| 5 | **Day 49 recent demand (lag features)** | Use train day49 (0:0-2:0) as real-time signal | Autocorr=0.97 |

### Tier 2 — Moderate Impact (Expected R² gain: +0.01-0.03)

| # | Feature | Intuition | Expected Signal |
|---|---------|-----------|-----------------|
| 6 | **Hour of day (numeric)** | Temporal demand curve | R²=0.017 alone |
| 7 | **Minute slot (0,15,30,45)** | Sub-hour patterns | Weak but additive |
| 8 | **Geohash prefix (4-5 char) encoding** | Spatial neighborhood effect | Hierarchical smoothing |
| 9 | **Day 49 adjustment factor** | Compensate for day-over-day trend | ~1.17x for Residential |
| 10 | **Number of observations per geohash** | Proxy for location importance/activity | Correlates with stability |

### Tier 3 — Low Impact / Speculative (Expected R² gain: <0.01)

| # | Feature | Intuition | Expected Signal |
|---|---------|-----------|-----------------|
| 11 | Temperature | Environmental effect | Near-zero after controls |
| 12 | Weather encoding | Condition effect | Near-zero |
| 13 | Landmarks | POI indicator | Minimal (0.093 vs 0.096) |
| 14 | NumberofLanes (residual) | Road capacity | Redundant with RoadType |
| 15 | Geohash neighbor demand | Spatial smoothing | Possible for cold-start |

---

## 4. MODEL ARCHITECTURE ANALYSIS

### Given Constraint: No External Libraries Available
The sandbox has **NO ML libraries** (no pandas, numpy, sklearn, catboost, etc.) and network is blocked. This forces a **pure-Python statistical approach**.

### Approach Ranking

| Rank | Model | Why It Works | Why It Might Fail | Estimated R² |
|------|-------|--------------|-------------------|--------------|
| 1 | **Hierarchical Target Encoding** | Data structure is perfect for it — high-cardinality categorical with clear hierarchy | Overfitting on rare combos | 0.92-0.96 |
| 2 | **Bayesian Smoothed Encoding** | Handles cold-start via shrinkage toward parent levels | Requires careful prior tuning | 0.93-0.97 |
| 3 | **Weighted Ensemble of Granularities** | Different precision levels complement each other | Weight optimization without CV | 0.94-0.97 |
| 4 | **Linear Regression (pure Python)** | Can capture additive effects | Misses interactions | 0.85-0.90 |

### If Libraries Were Available (Reference)

| Rank | Model | Expected R² | Compute |
|------|-------|-------------|---------|
| 1 | CatBoost with target encoding | 0.96-0.98 | Medium |
| 2 | LightGBM + geohash embeddings | 0.95-0.97 | Low |
| 3 | XGBoost ensemble | 0.94-0.96 | Medium |
| 4 | TabNet | 0.92-0.95 | High |
| 5 | Stacked ensemble (CatBoost+LGBM+XGB) | 0.97-0.99 | High |

---

## 5. OPTIMAL STRATEGY (Pure Python)

### Primary Approach: Multi-Level Bayesian Target Encoding

The key insight: with R² = 0.9729 from just (geohash, RoadType, hour) means on training data, and given that test is a temporal continuation of the same day, a carefully constructed hierarchical predictor can achieve very high performance.

**Architecture:**
```
Prediction = weighted_blend(
    Level 1: (geohash, RoadType, exact_timestamp) mean  [if ≥ 3 samples]
    Level 2: (geohash, RoadType, hour) mean             [if ≥ 5 samples]  
    Level 3: (geohash, hour) mean                       [if ≥ 3 samples]
    Level 4: (geohash, RoadType) mean                   [primary fallback]
    Level 5: (geohash) mean                             [secondary fallback]
    Level 6: (RoadType, hour) mean                      [cold-start]
    Level 7: (RoadType) mean                            [final fallback]
)
```

**With adjustments:**
- Day 49 uplift factor (1.17x for Residential, 1.05x for Highway)
- Bayesian shrinkage: blend toward parent level based on sample count
- Clipping to [0, 1] range

### Secondary Approach: Lag-Based Correction
- Use Day 49 training data (hours 0-2) to compute per-geohash "current state"
- Apply ratio of (day49 actual / day48 predicted at same time) as correction factor
- This captures the day-over-day shift per location

### Ensemble Strategy
- Blend hierarchical model (70%) + lag-corrected model (30%)
- Or: optimize blending weights using train-day49 as validation

---

## 6. VALIDATION STRATEGY

### Recommended: Time-Based Split
- **Train**: Day 48 (all 96 timestamps)
- **Validation**: Day 49, hours 0:0-2:0 (9 timestamps, 7,872 rows)
- **Test**: Day 49, hours 2:15-13:45

This exactly mimics the test scenario: predicting future timestamps using past data.

### Why Other Strategies Fail
- **Random split**: Temporal leakage (same geohash at adjacent times in both folds)
- **Group split by geohash**: Loses 99.2% overlap info; doesn't match test
- **K-Fold**: Not meaningful with only 2 days of data

---

## 7. HISTORICAL DATASET COMPARISON

### Similar Competitions
1. **Grab Traffic Demand Forecasting** (AI for SEA) — geohash-based, 15-min intervals, similar structure
2. **NYC Taxi Demand Prediction** — zone-based spatiotemporal forecasting
3. **Uber Movement** — area-level traffic prediction
4. **DiDi Chuxing Challenges** — ride-hailing demand by grid cell

### Transferable Insights
- Geohash target encoding is ALWAYS the top feature in similar competitions
- Tree-based models (CatBoost/LightGBM) dominate leaderboards
- Temporal lag features provide large gains
- Spatial neighbor features help for cold-start
- Hierarchical smoothing prevents overfitting on rare combinations

---

## 8. COMPETITION META-STRATEGY

### Highest Probability Approach
Hierarchical Bayesian target encoding with day-over-day correction factor.
Expected score: **92-96/100**

### Highest Upside Approach
If ML libraries were available: CatBoost with leave-one-out target encoding, lag features from day 49 training portion, and per-geohash time-series decomposition.
Expected score: **96-99/100**

### Most Overlooked Opportunity
The **day-over-day correction factor** — most competitors will use day 48 statistics directly without adjusting for the systematic uplift observed in day 49.

### Biggest Source of Potential Gain
Correctly handling the **RoadType × geohash interaction** with proper Bayesian smoothing to avoid overfitting on rare (geohash, RoadType) combinations that have <5 samples.

### Biggest Risk of Overfitting
Using (geohash, RoadType, exact_timestamp) means with only 1 sample — these will be noisy and should be heavily shrunk toward the (geohash, RoadType, hour) level.

### Biggest Source of Leakage (Non-Issue Here)
No actual leakage risk since train/test split is purely temporal.

---

## 9. IMPLEMENTATION ROADMAP

1. **Build hierarchical target encoding dictionaries** from training data
2. **Compute day-over-day adjustment factors** from day 48 vs day 49 overlap
3. **Implement Bayesian shrinkage** with count-based confidence weighting
4. **Predict test set** using the hierarchical fallback chain
5. **Validate on day 49 training portion** (hours 0-2) to tune weights
6. **Generate submission CSV** with proper format

---

## 10. EXPECTED FINAL PERFORMANCE

| Approach | Expected R² (CV) | Expected LB Score |
|----------|-------------------|-------------------|
| Naive overall mean | 0.00 | 0 |
| RoadType mean | 0.75 | 75 |
| Geohash mean | 0.69 | 69 |
| (Geohash, RoadType) mean | 0.88 | 88 |
| (Geohash, RoadType, hour) mean | 0.97 | 92-95 |
| Full hierarchical + Bayesian + day correction | — | **94-97** |
| With ML (CatBoost ensemble) | — | **96-99** |

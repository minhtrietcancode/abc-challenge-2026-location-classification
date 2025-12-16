# ABC2026 Challenge - Location Recognition Project Brief

## Challenge Overview

**Competition:** ABC2026 Sozolab Challenge - Activity & Location Recognition in Care Facilities
**Task:** Predict room location based on Bluetooth Low Energy (BLE) beacon RSSI signals
**Organizer Contact:** Christina Garcia

---

## Problem Context

### Data Collection Setup

1. **User ID 90 (Data Collector):** Caregiver carrying a mobile phone
   - Phone app continuously records RSSI values from nearby BLE beacons
   - Logs beacon signals as they move through 5th floor of care facility

2. **User ID 97 (Labeler):** Person who annotates User 90's location
   - Records which room User 90 was in during specific time ranges
   - Creates ground truth labels with start/end timestamps

3. **Infrastructure:** 25 BLE beacons installed throughout 5th floor
   - Each beacon continuously transmits signals
   - Phone detects signals and records strength (RSSI)
   - Different beacon patterns = different locations

---

## Dataset Characteristics

### 1. Raw BLE Data (Input Features)

**Format:**
```csv
user_id, timestamp, name, mac_address, RSSI, power
90, 2023-04-10T10:22:55.589+0900, , FD:07:0E:D5:28:AE, -75, -2147483648
90, 2023-04-10T10:22:55.595+0900, , FD:07:0E:D5:28:AE, -75, -2147483648
```

**Key Characteristics:**
- **Timestamp precision:** Millisecond level (format: `YYYY-MM-DDTHH:MM:SS.mmm+TZTZ`)
- **Total records:** ~1.67M BLE readings (after cleaning)
- **Time range:** 2023-04-10 13:00:00 to 2023-04-13 17:29:59 (~3.5 days)
- **Number of beacons:** 25 unique beacons (filtered from hundreds of MAC addresses)
- **RSSI values:** Negative dBm (closer to 0 = stronger signal = closer proximity)
- **Asynchronous recording:** NOT fixed frame rate - beacons detected whenever signal received
- **Expected records per second:** 50-200 records (varies based on beacon proximity)
- **Beacons detected per second:** 5-15 unique beacons typically
- **Multiple readings per beacon:** Same beacon can be detected 3-10 times per second

**Data Sparsity Issue:**
- At any single millisecond timestamp: only 3-5 beacons detected (not all 25)
- RSSI values fluctuate significantly due to signal interference
- **Aggregation is essential** to get stable feature representation

### 2. Location Labels (Target Variable)

**Format:**
```csv
started_at, finished_at, room, floor
2023-04-10 14:21:46+09:00, 2023-04-10 14:21:50+09:00, kitchen, 5th
```

**Key Characteristics:**
- **Labeled records:** 451 location labels (after cleaning)
- **Labeled BLE data:** ~1.10M records (66% of total BLE data)
- **Unlabeled data:** ~0.57M records (34%) - gaps where labeler wasn't tracking
- **Target classes:** Multiple rooms (kitchen, cafeteria, nurse station, hallway, room numbers like 523, etc.)
- **Class imbalance:** Likely present (some rooms visited more than others)

### 3. Cleaned Dataset Structure

**File:** `labelled_ble_data.csv`
```csv
timestamp, mac address, RSSI, room
2023-04-10 14:21:46+09:00, 6, -93, kitchen
```

**Columns:**
- `timestamp`: Detection time (timezone-aware)
- `mac address`: Beacon ID (1-25)
- `RSSI`: Signal strength in dBm
- `room`: Target label

---

## Testing & Submission Format

### Information from Supervisor

1. **Submission Format:** Submit **both** prediction results file AND complete code
   - Results file: `predictions.csv` with room labels
   - Code: Complete pipeline for reproducibility
   - **Do NOT** submit trained model (usually)

2. **Test Data Format:** Most likely **Format A (raw individual records)**
   - Test data will be raw BLE records (millisecond-level timestamps)
   - **We must handle aggregation** as part of our pipeline
   - Organizers will verify results by re-running our code

3. **Time Window Selection:** Part of our proposed methodology
   - **No fixed requirement** - we choose optimal window size
   - Must experiment and justify our choice
   - This is where we can differentiate from other teams

### Questions Sent to Organizer (Awaiting Response)

- Confirm test data is raw Format A?
- Code submission requirements (Python script, notebook, Docker)?
- Evaluation metric (Accuracy, F1-Score, Weighted F1)?
- Any restrictions on libraries or preprocessing?

---

## Evaluation Metrics

**Competition Scoring:**
1. **Activity classification:** Macro F1 score (primary) - 35%
2. **Location classification:** Macro F1 score (primary) - 35%
3. **Relevance of Method and Paper Quality** - 30%

### Understanding Macro F1 Score

**Macro F1 = Unweighted average of per-class F1 scores**

```python
# Example with 3 rooms:
kitchen:    F1 = 0.95  (500 samples)
hallway:    F1 = 0.90  (300 samples)  
room_523:   F1 = 0.60  (50 samples)

# Macro F1 (EQUAL weight for each class):
Macro F1 = (0.95 + 0.90 + 0.60) / 3 = 0.817
```

**Critical Implications:**
- ⚠️ **Rare classes matter as much as common classes**
- ⚠️ If model ignores minority classes → HUGE score penalty
- ⚠️ Must handle class imbalance carefully
- ⚠️ Cannot just optimize for overall accuracy

**This is why `class_weight='balanced'` is CRITICAL for our approach**

---

## Our Final Proposed Approach

### Phase 1: Baseline Implementation (START HERE)

This is our initial approach based on thorough analysis and research on BLE fingerprinting:

#### **Step 1: Time Window Aggregation**

**Choice:** 2-second windows (primary), with experiments on 1s, 3s, 5s

**Rationale:**
- Balance between signal stability and temporal resolution
- Enough samples per window for stable statistics (6-10 readings per beacon)
- Captures multiple readings from same beacon
- Person unlikely to change rooms within 2 seconds
- More stable std calculation compared to 1-second windows

**Implementation:**
```python
df['window'] = df['timestamp'].dt.floor('2S')
```

**Alternative windows to test:** 1s, 3s, 5s, 10s

#### **Step 2: Feature Engineering**

**For each 2-second window, create features for all 25 beacons:**

**Proposed features per beacon (4 statistics):**
- `Beacon_X_mean`: Average RSSI value in the window (proximity indicator)
- `Beacon_X_std`: Standard deviation of RSSI (signal stability indicator)
- `Beacon_X_count`: Number of detections in the window (visibility indicator)
- `Beacon_X_max`: Maximum RSSI value in the window (peak signal strength)

**Missing beacon handling:** 
- Fill with `0` for mean, std, count when beacon not detected
- Fill with `-100` for max (representing very weak/no signal)
- Rationale: Clearly distinguishes "not detected" from "weak signal"

**Total features:** 25 beacons × 4 statistics = **100 features**

**Why we DON'T use skewness/kurtosis:**
- Too few samples per beacon in 2-second window (6-10 samples)
- These statistics need 30+ samples to be reliable
- Would introduce noise rather than signal

**Example aggregation:**
```python
def engineer_features(window_df):
    features = {}
    for beacon_id in range(1, 26):
        beacon_data = window_df[window_df['mac address'] == beacon_id]
        if len(beacon_data) > 0:
            features[f'beacon_{beacon_id}_mean'] = beacon_data['RSSI'].mean()
            features[f'beacon_{beacon_id}_std'] = beacon_data['RSSI'].std()
            features[f'beacon_{beacon_id}_count'] = len(beacon_data)
            features[f'beacon_{beacon_id}_max'] = beacon_data['RSSI'].max()
        else:
            features[f'beacon_{beacon_id}_mean'] = 0
            features[f'beacon_{beacon_id}_std'] = 0
            features[f'beacon_{beacon_id}_count'] = 0
            features[f'beacon_{beacon_id}_max'] = -100
    return features
```

#### **Step 3: Model Selection - Gradient Boosting**

**Primary Model: LightGBM (Light Gradient Boosting Machine)**

**Why Gradient Boosting over kNN?**

✅ **Advantages:**
1. **Handles beacon combinations naturally** - learns "if Beacon_1 strong AND Beacon_2 strong → Room A"
2. **No spatial ambiguity issue** - unlike kNN, understands that Beacon_1 and Beacon_23 are different
3. **Handles continuous features excellently** - RSSI values are continuous, trees split on optimal thresholds
4. **Feature importance rankings** - can see which beacons/statistics matter most
5. **Built-in class imbalance handling** - with `class_weight='balanced'`
6. **Fast training and prediction** - especially LightGBM on large datasets
7. **No feature scaling needed** - unlike kNN which is sensitive to scale

**Why LightGBM specifically:**
- ⚡ Fastest training for our 1.1M records dataset
- 💾 Lower memory usage than XGBoost
- 🎯 Same accuracy as XGBoost (typically within 1%)
- 📊 Excellent visualization tools
- 🌳 Leaf-wise tree growth (faster, more accurate than level-wise)

**Alternative:** XGBoost (slightly slower but very stable, can use if LightGBM has issues)

**Model Configuration:**
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=500,           # Number of trees
    learning_rate=0.05,         # Conservative learning rate
    max_depth=7,                # Tree depth
    num_leaves=50,              # Leaf-wise growth parameter
    min_child_samples=5,        # Allow small leaf nodes (helps rare classes)
    subsample=0.8,              # Row sampling
    colsample_bytree=0.8,       # Feature sampling
    class_weight='balanced',    # ← CRITICAL: Handles class imbalance for Macro F1
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)
```

**Why `class_weight='balanced'` is CRITICAL:**
- Competition uses Macro F1 score (equal weight per class)
- Automatically penalizes mistakes on minority classes more
- Forces model to care about rare rooms as much as common rooms
- Example: room with 50 samples gets 10x weight vs room with 500 samples

#### **Step 4: Model Training and Validation**

**Cross-Validation Strategy:**
```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report

# Use TimeSeriesSplit (respects temporal ordering)
tscv = TimeSeriesSplit(n_splits=5)

# Evaluate with Macro F1 (same as competition metric)
macro_f1_scores = []
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    macro_f1_scores.append(macro_f1)
    
    # Check per-class performance
    print(classification_report(y_val, y_pred))

print(f"Average Macro F1: {np.mean(macro_f1_scores):.3f} (±{np.std(macro_f1_scores):.3f})")
```

**Expected Performance:**
- Target: 80%+ Macro F1 score
- If below 75%, proceed to Phase 2 optimization

#### **Step 5: Prediction on Test Data**
```python
# Test data (raw BLE records) 
→ Aggregate by 2-second windows 
→ Engineer same 100 features 
→ LightGBM prediction per window
→ Assign predicted label to ALL frames within that window
```

**Output format:**
```csv
timestamp (original millisecond), predicted_room
2023-04-13 18:00:00.001, kitchen
2023-04-13 18:00:00.105, kitchen
2023-04-13 18:00:00.203, kitchen
...
(all frames in same 2-second window get same label)
```

---

### Phase 2: Performance Optimization (IF NEEDED)

If baseline Macro F1 < 80%, try these improvements in order:

#### **Option 2A: Hyperparameter Tuning**

Focus on parameters that help minority classes and prevent overfitting:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9, 11],
    'num_leaves': [31, 50, 70, 100],
    'min_child_samples': [3, 5, 10, 20],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=5, 
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

#### **Option 2B: Additional Feature Engineering**

Add spatial and contextual features:

```python
# 1. Dominant beacon pattern
features['dominant_beacon_id'] = beacon_with_max_RSSI
features['dominant_beacon_strength'] = max_RSSI

# 2. Number of active beacons
features['n_active_beacons'] = sum(1 for count in beacon_counts if count > 0)

# 3. Regional signal strength (define zones based on floor map!)
# Example: if beacons 1-8 are east wing, 9-17 middle, 18-25 west wing
features['zone_east_strength'] = sum(RSSI for beacons 1-8)
features['zone_middle_strength'] = sum(RSSI for beacons 9-17)
features['zone_west_strength'] = sum(RSSI for beacons 18-25)

# 4. Signal ratios (relative positioning)
total_strength = abs(sum(all_RSSI_values))
if total_strength > 0:
    features['zone_east_ratio'] = abs(zone_east) / total_strength
    features['zone_middle_ratio'] = abs(zone_middle) / total_strength
    features['zone_west_ratio'] = abs(zone_west) / total_strength

# 5. Beacon activation binary vector
for beacon_id in range(1, 26):
    features[f'beacon_{beacon_id}_active'] = 1 if count > 0 else 0

# 6. Additional statistics (if needed)
features['beacon_X_median'] = median(RSSI)
features['beacon_X_range'] = max(RSSI) - min(RSSI)
```

**Note:** Start simple (100 features), add these only if needed. More features = risk of overfitting.

#### **Option 2C: Test Different Window Sizes**

```python
window_sizes = [1, 2, 3, 5, 10]  # seconds

results = {}
for window_size in window_sizes:
    # Aggregate data
    df['window'] = df['timestamp'].dt.floor(f'{window_size}S')
    
    # Engineer features
    features = engineer_features(df)
    
    # Train and evaluate
    model.fit(X_train, y_train)
    macro_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    results[window_size] = macro_f1
    print(f"Window={window_size}s: Macro F1={macro_f1:.3f}")

# Pick best window size
best_window = max(results, key=results.get)
```

**Expected findings:**
- 1s: Might be too noisy (unstable std)
- 2-3s: Likely sweet spot
- 5s+: Might lose temporal resolution

#### **Option 2D: Model Ensemble**

Combine multiple models for better predictions:

```python
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

ensemble = VotingClassifier([
    ('lgb', lgb.LGBMClassifier(class_weight='balanced', ...)),
    ('xgb', XGBClassifier(class_weight='balanced', ...)),
    ('rf', RandomForestClassifier(class_weight='balanced', ...))
], voting='soft')  # Use probability averaging

ensemble.fit(X_train, y_train)
```

**Benefits:**
- Reduces variance (more stable predictions)
- Combines strengths of different algorithms
- Typically +2-5% improvement

**Costs:**
- 3x slower training and prediction
- More complex pipeline

#### **Option 2E: Temporal Smoothing (Post-processing)**

Apply moving window to smooth predictions:

```python
from scipy.stats import mode

def smooth_predictions(predictions, window=5):
    """
    Smooth predictions using moving window mode.
    Person unlikely to teleport between rooms.
    """
    smoothed = []
    for i in range(len(predictions)):
        window_preds = predictions[max(0, i-window):i+window+1]
        most_common = mode(window_preds)[0]
        smoothed.append(most_common)
    return smoothed

# Apply after prediction
y_pred_raw = model.predict(X_test)
y_pred_smoothed = smooth_predictions(y_pred_raw, window=5)
```

**Benefits:**
- Reduces noise in predictions
- Respects physical constraints (can't teleport)
- Simple post-processing, no retraining needed

**Tune window size:** Test 3, 5, 7, 10 time steps

#### **Option 2F: Alternative Gradient Boosting Algorithms**

If LightGBM doesn't perform well:

```python
# Option 1: XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    class_weight='balanced',
    tree_method='hist',  # Faster for large datasets
    random_state=42
)

# Option 2: CatBoost (best for categorical features)
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    auto_class_weights='Balanced',
    random_state=42,
    verbose=False
)
```

---

### Phase 3: Advanced Techniques (ONLY IF STILL STRUGGLING)

#### **Option 3A: Hierarchical Classification**

Two-stage classification for better spatial structure:

```python
# Stage 1: Coarse localization (predict zone/area)
# Example zones: East Wing, West Wing, Central Area
zone_model = lgb.LGBMClassifier(class_weight='balanced')
zone_model.fit(X_train, y_zone_train)

predicted_zone = zone_model.predict(X_test)

# Stage 2: Fine localization (predict room within zone)
zone_specific_models = {}
for zone in zones:
    mask = y_zone_train == zone
    zone_specific_models[zone] = lgb.LGBMClassifier(class_weight='balanced')
    zone_specific_models[zone].fit(X_train[mask], y_train[mask])

# Predict room based on zone
final_predictions = []
for i, zone in enumerate(predicted_zone):
    room = zone_specific_models[zone].predict(X_test[i:i+1])
    final_predictions.append(room)
```

**Benefits:**
- Reduces confusion between distant rooms
- Each zone model focuses on subset of classes
- Can help minority classes within each zone

**Challenges:**
- More complex pipeline
- Risk of cascade errors (wrong zone → wrong room)
- Need to define zones (manual or clustering)

#### **Option 3B: RSSI Signal Preprocessing**

Apply Kalman Filter to reduce noise before aggregation:

```python
from pykalman import KalmanFilter

def apply_kalman_filter(df):
    """
    Apply Kalman filter per beacon to smooth RSSI signals.
    Reduces multipath interference and signal fluctuations.
    """
    filtered_dfs = []
    for beacon_id in range(1, 26):
        beacon_df = df[df['mac address'] == beacon_id].copy()
        if len(beacon_df) > 0:
            beacon_df = beacon_df.sort_values('timestamp')
            kf = KalmanFilter(
                initial_state_mean=beacon_df['RSSI'].iloc[0],
                n_dim_obs=1
            )
            state_means, _ = kf.filter(beacon_df['RSSI'].values)
            beacon_df['RSSI_filtered'] = state_means
            filtered_dfs.append(beacon_df)
    return pd.concat(filtered_dfs)

# Apply before aggregation
df_filtered = apply_kalman_filter(df)
# Then proceed with normal pipeline
```

**Benefits:**
- Reduces RSSI noise and fluctuations
- More stable mean/std calculations
- Research shows significant improvement

**When to use:**
- If signal quality is very noisy
- If std values are very high
- After trying simpler approaches first

#### **Option 3C: Synthetic Data Augmentation (USE WITH CAUTION)**

**Note:** We discussed a paper on relabeling for class imbalance, but this approach is RISKY for BLE fingerprinting because:
- Each location has unique physical beacon signature
- Relabeling samples from different locations creates fake patterns
- Risk of data leakage and unrealistic training data
- `class_weight='balanced'` should handle imbalance sufficiently

**If you still want to try synthetic data:**

```python
from imblearn.over_sampling import SMOTE

# ONLY use if:
# 1. Baseline + Phase 2 optimizations still give low Macro F1
# 2. Specific minority classes are completely failing
# 3. You validate carefully with stratified CV

# Apply SMOTE carefully
smote = SMOTE(
    sampling_strategy='auto',  # Only oversample minority classes
    k_neighbors=3,  # Conservative: only use 3 nearest neighbors
    random_state=42
)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# CRITICAL: Use stratified CV to avoid leakage
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='f1_macro')
```

**Validation is CRITICAL:**
- Compare Macro F1 with vs without SMOTE
- Check if minority class F1 actually improves
- Watch for overfitting (train-test gap)
- If validation score increases but test score decreases → overfitting

**Our recommendation:** Try everything in Phase 1 and Phase 2 first before considering synthetic data.

---

## Known Issues & Limitations (Original kNN Approach)

### Problem: Spatial Ambiguity in Distance Metric

**Scenario:**
- **Room A:** Near beacons 1, 2, 3 (RSSI: -70, -75, -80, others: 0)
- **Room B:** Near beacons 23, 24, 25 (RSSI: -70, -75, -80, others: 0)

**Issue with standard Euclidean distance:**
```python
# Feature vectors:
Room_A = [−70, −75, −80, 0, 0, ..., 0, 0, 0]  # Strong signals from beacons 1,2,3
Room_B = [0, 0, 0, 0, 0, ..., −70, −75, −80]  # Strong signals from beacons 23,24,25

# Euclidean distance:
distance = sqrt((−70−0)² + (−75−0)² + (−80−0)² + ... + (0−(−70))² + (0−(−75))² + (0−(−80))²)
        = sqrt(2×(4900 + 5625 + 6400))
        ≈ same distance!
```

**Problem:** Two physically distant rooms appear similar because:
- Same number of beacons detected
- Similar RSSI magnitudes
- Different beacon IDs don't matter to Euclidean distance

**This causes kNN to misclassify rooms that are far apart but have similar signal patterns**

**Why Gradient Boosting Solves This:**
- Tree models learn: "if beacon_1 strong → East wing" vs "if beacon_23 strong → West wing"
- Beacon identity (which beacon) matters, not just signal strength
- No distance metric confusion

---

## Alternative Solutions Previously Considered

### Option 1: Use Tree-Based Models Instead ✅ CHOSEN

**Models:** Random Forest, XGBoost, LightGBM, CatBoost

**Advantages:**
- ✅ Can learn "if Beacon_1 is strong AND Beacon_2 is strong → Room A"
- ✅ Naturally handles feature interactions
- ✅ Learns which beacon combinations matter
- ✅ Handles class imbalance with class weights
- ✅ Feature importance rankings

**Status:** **Selected as primary approach (LightGBM)**

### Option 2: Hybrid Approach (Decision Tree + kNN)

**Concept:** Two-stage classification

**Stage 1 - Coarse localization (Decision Tree):**
```python
# Use decision tree to split into regions/zones
if Beacon_1 > threshold OR Beacon_2 > threshold OR Beacon_3 > threshold:
    zone = "Zone_A"  # East wing
elif Beacon_23 > threshold OR Beacon_24 > threshold OR Beacon_25 > threshold:
    zone = "Zone_B"  # West wing
...
```

**Stage 2 - Fine localization (kNN within zone):**
```python
# Apply kNN only within predicted zone
# Reduces confusion between distant rooms
knn_model = train_knn_per_zone(zone="Zone_A")
room = knn_model.predict(features)
```

**Advantages:**
- ✅ Combines strengths of both algorithms
- ✅ Decision tree handles spatial structure
- ✅ kNN handles fine-grained RSSI patterns
- ✅ Reduces search space for kNN

**Challenges:**
- ⚠️ More complex pipeline
- ⚠️ Need to define zones (manual or learned)
- ⚠️ Risk of cascade errors (wrong zone → wrong room)

**Status:** Available in Phase 3 as "Hierarchical Classification" but using Gradient Boosting instead of kNN

### Option 3: Weighted Distance Metric for kNN

**Concept:** Modify distance calculation to emphasize detected beacons

**Standard Euclidean:**
```python
distance = sqrt(sum((x_i - y_i)² for all i))
```

**Weighted by detection count:**
```python
# Give more weight to beacons that were actually detected
w_i = count_i  # or sqrt(count_i) or log(1 + count_i)
distance = sqrt(sum(w_i × (x_i - y_i)² for all i))
```

**Weighted by signal strength:**
```python
# Stronger signals (closer to 0) get more weight
w_i = 1 / (1 + abs(RSSI_i)) if RSSI_i != 0 else 0
distance = sqrt(sum(w_i × (x_i - y_i)² for all i))
```

**Advantages:**
- ✅ Still uses kNN framework
- ✅ Can emphasize important beacons
- ✅ Reduces impact of zero-values

**Challenges:**
- ⚠️ Need to determine optimal weights
- ⚠️ May overfit if not cross-validated properly
- ⚠️ Still doesn't fully solve spatial ambiguity

**Status:** Not chosen - Gradient Boosting is superior

### Option 4: Feature Engineering Enhancements

**Add spatial features:**
```python
# Dominant beacon (strongest signal)
features['dominant_beacon_id'] = beacon_with_max_RSSI
features['dominant_beacon_strength'] = max_RSSI

# Beacon activation pattern (binary)
features['beacons_detected_binary'] = [1 if count > 0 else 0 for each beacon]

# Regional signal strength
features['zone_A_total_strength'] = sum(RSSI for beacons 1-8)
features['zone_B_total_strength'] = sum(RSSI for beacons 9-16)
features['zone_C_total_strength'] = sum(RSSI for beacons 17-25)

# Signal ratios
features['east_west_ratio'] = zone_A_strength / zone_B_strength
```

**Advantages:**
- ✅ Captures spatial structure explicitly
- ✅ Works with any classifier
- ✅ Interpretable features

**Status:** Available in Phase 2 as "Additional Feature Engineering"

---

## Implementation Checklist

### Phase 1: Baseline (Must Complete First)
- [ ] Load and preprocess data
- [ ] Implement 2-second time window aggregation
- [ ] Engineer 100 features (mean, std, count, max for 25 beacons)
- [ ] Implement LightGBM with `class_weight='balanced'`
- [ ] Setup TimeSeriesSplit cross-validation
- [ ] Evaluate with Macro F1 score
- [ ] Analyze per-class performance with classification_report
- [ ] Document results and feature importance

**Target: 80%+ Macro F1**

### Phase 2: Optimization (If Needed)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Test alternative window sizes (1s, 3s, 5s)
- [ ] Add spatial features (dominant beacon, zones)
- [ ] Implement temporal smoothing
- [ ] Try ensemble models
- [ ] Compare XGBoost vs LightGBM

**Target: 85%+ Macro F1**

### Phase 3: Advanced (Last Resort)
- [ ] Implement hierarchical classification
- [ ] Apply Kalman filtering
- [ ] Consider synthetic data (with careful validation)
- [ ] Final ensemble with best models

**Target: 90%+ Macro F1**

### Final Submission
- [ ] Train final model on all training data
- [ ] Generate predictions on test data
- [ ] Apply temporal smoothing if beneficial
- [ ] Create clean, documented code
- [ ] Write methodology report
- [ ] Package everything for submission

---

## Data Files Reference

- **Raw BLE data:** `Dataset/BLE Data/userbleid_90_*.csv`
- **Location labels:** `Dataset/5f_label_loc_train.csv`
- **Cleaned BLE data:** `cleaned_dataset/cleaned_ble_data.csv`
- **Labeled BLE data:** `cleaned_dataset/labelled_ble_data.csv` ← Main training data
- **Floor map:** `Dataset/5th floor map.png`

---

## Key Constraints & Requirements

- **Must handle raw test data** - aggregation is our responsibility
- **Must submit both predictions AND code** - reproducibility required
- **Time window is our choice** - must justify decision (we chose 2 seconds)
- **Must handle 34% unlabeled training data** - expected behavior
- **Must handle class imbalance** - some rooms visited more than others (using `class_weight='balanced'`)
- **Must handle missing beacons** - not all 25 detected every second (handled with 0 and -100 fill values)
- **Must optimize for Macro F1** - equal weight per class regardless of sample size

---

## Summary: Our Strategy

### Core Approach
1. **2-second time windows** (balance stability and resolution)
2. **100 features** (25 beacons × 4 statistics: mean, std, count, max)
3. **LightGBM** with `class_weight='balanced'` (handles continuous features and class imbalance)
4. **Macro F1 evaluation** (matches competition metric)

### Key Decisions Explained
- **Why 2 seconds?** More stable statistics than 1s, better than arbitrary choice
- **Why 4 statistics?** Mean (proximity), std (stability), count (visibility), max (peak signal)
- **Why not skewness?** Too few samples per window (need 30+, we have 6-10)
- **Why LightGBM?** Fast, handles continuous features, learns beacon combinations, built-in class weighting
- **Why not kNN?** Spatial ambiguity problem - can't distinguish beacon identity

### If Baseline Isn't Enough
1. Try different window sizes
2. Add spatial features (zones, dominant beacon)
3. Tune hyperparameters
4. Apply temporal smoothing
5. Use ensemble models
6. Consider hierarchical classification

### Success Criteria
- **Minimum:** 80% Macro F1 (competitive)
- **Target:** 85% Macro F1 (strong submission)
- **Stretch:** 90% Macro F1 (top tier)

---

## Questions for Discussion

1. **Should we start with 2-second or still test 1-second first?**
   - Recommendation: Start with 2s (more stable), then test 1s, 3s, 5s

2. **Should we use LightGBM or XGBoost for baseline?**
   - Recommendation: LightGBM (faster for experimentation, same accuracy)

3. **How much hyperparameter tuning before adding features?**
   - Recommendation: Quick baseline → test windows → tune → add features

4. **When should we try ensemble?**
   - Recommendation: Only if single model < 80% Macro F1

5. **Should we implement Kalman filtering?**
   - Recommendation: Only if signal quality is very poor and baseline fails

---

## Research References

Based on literature review on BLE indoor localization:
- RSSI-based fingerprinting with ML can achieve 94-96% accuracy
- Kalman filtering effective at reducing RSSI noise
- Gradient boosting methods (Random Forest, XGBoost) proven successful
- Class imbalance best handled with class weighting rather than synthetic data
- 1-5 second time windows commonly used in practice

---

## Next Steps

1. ✅ Finalize approach (COMPLETED - using 2s windows + LightGBM)
2. ⏳ Implement baseline pipeline
3. ⏳ Validate with TimeSeriesSplit and Macro F1
4. ⏳ Analyze feature importance
5. ⏳ Test different window sizes
6. ⏳ Optimize if needed (Phase 2)
7. ⏳ Prepare final submission with documentation
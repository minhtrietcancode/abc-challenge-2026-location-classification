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

## Current Proposed Approach

### Step 1: Time Window Aggregation

**Choice:** 1-second windows (to be validated through experiments)

**Rationale:**
- Balance between signal stability and temporal resolution
- Enough samples per window (50-200 records)
- Captures multiple readings from same beacon
- Person unlikely to change rooms within 1 second

**Alternatives to test:** 2s, 5s, 10s windows

### Step 2: Feature Engineering

**For each 1-second window, create features for all 25 beacons:**
```python
# Raw data in 1 second:
# timestamp                    | beacon | RSSI
# 2023-04-10 10:22:55.001     | 6      | -85
# 2023-04-10 10:22:55.105     | 6      | -86
# 2023-04-10 10:22:55.203     | 4      | -78
# 2023-04-10 10:22:55.310     | 4      | -79
# (no detection for other beacons)

# Aggregated features per window:
# timestamp           | Beacon_1_mean | Beacon_1_std | Beacon_1_count | ... | Beacon_25_mean | Beacon_25_std | Beacon_25_count
# 2023-04-10 10:22:55 | 0             | 0            | 0              | ... | 0              | 0             | 0
#                     | (Beacon 1 not detected)                           | (Beacon 25 not detected)
```

**Proposed features per beacon:**
- `Beacon_X_mean`: Average RSSI value in the window
- `Beacon_X_std`: Standard deviation of RSSI (signal stability)
- `Beacon_X_count`: Number of detections in the window
- Additional possibilities: `min`, `max`, `median`, `range`

**Missing beacon handling:** 
- Fill with `0` for undetected beacons
- Rationale: 0 means "not detected" and contributes nothing to distance in kNN

**Total features:** 25 beacons × 3 statistics = 75 features (minimum)

### Step 3: Model Selection - k-Nearest Neighbors (kNN)

**Why kNN?**

✅ **Advantages:**
1. **Handles numerical features naturally** - maintains continuous nature of RSSI values
2. **Zero-value friendly** - undetected beacons (RSSI=0) contribute 0 to Euclidean distance
3. **No assumption about data distribution** - works with irregular beacon patterns
4. **Naturally handles class imbalance** - with appropriate k, minority classes still found
5. **Interpretable** - can inspect nearest neighbors to understand predictions
6. **No training required** - instance-based learning

**Key hyperparameters to tune:**
- `k` (number of neighbors): Test k=3, 5, 7, 9, 11, etc.
- `distance metric`: Euclidean (default), Manhattan, Minkowski
- `weights`: uniform vs. distance-weighted

### Step 4: Prediction on Test Data
```python
# Test data (raw BLE records) 
→ Aggregate by 1-second windows 
→ Engineer same 75 features 
→ kNN prediction per window
→ Assign predicted label to ALL frames within that window
```

**Output format:**
```csv
timestamp (original millisecond), predicted_room
2023-04-13 18:00:00.001, kitchen
2023-04-13 18:00:00.105, kitchen
2023-04-13 18:00:00.203, kitchen
...
(all frames in same second get same label)
```

---

## Known Issues & Limitations with kNN Approach

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
        = sqrt(4900 + 5625 + 6400 + ... + 4900 + 5625 + 6400)
        = sqrt(2×(4900 + 5625 + 6400))
        ≈ same distance!
```

**Problem:** Two physically distant rooms appear similar because:
- Same number of beacons detected
- Similar RSSI magnitudes
- Different beacon IDs don't matter to Euclidean distance

**This causes kNN to misclassify rooms that are far apart but have similar signal patterns**

---

## Potential Solutions to Explore

### Option 1: Use Tree-Based Models Instead

**Models:** Random Forest, XGBoost, LightGBM, CatBoost

**Advantages:**
- ✅ Can learn "if Beacon_1 is strong AND Beacon_2 is strong → Room A"
- ✅ Naturally handles feature interactions
- ✅ Learns which beacon combinations matter
- ✅ Handles class imbalance with class weights
- ✅ Feature importance rankings

**Concerns:**
- ⚠️ May treat continuous RSSI as categorical (binning)
- ⚠️ Need to verify they handle 0-values appropriately

**TODO:** Test if tree models treat 0 as "not detected" vs. "weak signal at 0 dBm"

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

**Feature-based weighting:**
```python
# Learn optimal weights through cross-validation
from sklearn.neighbors import KNeighborsClassifier

# Use learned feature weights
knn = KNeighborsClassifier(n_neighbors=5, metric='wminkowski', metric_params={'w': learned_weights})
```

**Advantages:**
- ✅ Still uses kNN framework
- ✅ Can emphasize important beacons
- ✅ Reduces impact of zero-values

**Challenges:**
- ⚠️ Need to determine optimal weights
- ⚠️ May overfit if not cross-validated properly

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

---

## Questions for Discussion

1. **Which approach should we prioritize?**
   - Pure kNN with weighted distance?
   - Tree-based models (Random Forest/XGBoost)?
   - Hybrid Decision Tree + kNN?
   - Enhanced feature engineering + simple model?

2. **How to handle the spatial ambiguity issue?**
   - Is weighted distance sufficient?
   - Do we need regional/zonal features?
   - Should we encode beacon topology explicitly?

3. **Feature engineering priorities:**
   - Which statistics are most informative? (mean, std, count, max, min?)
   - Should we add beacon activation patterns?
   - Do we need regional aggregations?

4. **Validation strategy:**
   - How to split data to test spatial generalization?
   - Cross-validation by time? By room? By spatial location?

5. **Time window experiments:**
   - Test 1s, 2s, 5s, 10s - which performs best?
   - Trade-off between stability and temporal resolution?

---

## Next Steps

1. ✅ Wait for organizer response on test data format
2. ⏳ Implement baseline kNN with 1-second windows
3. ⏳ Experiment with different time windows (1s, 2s, 5s, 10s)
4. ⏳ Test alternative models (Random Forest, XGBoost)
5. ⏳ Evaluate spatial ambiguity issue on actual data
6. ⏳ Implement and compare proposed solutions
7. ⏳ Document choice justifications for final report

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
- **Time window is our choice** - must justify decision
- **Must handle 34% unlabeled training data** - expected behavior
- **Must handle class imbalance** - some rooms visited more than others
- **Must handle missing beacons** - not all 25 detected every second
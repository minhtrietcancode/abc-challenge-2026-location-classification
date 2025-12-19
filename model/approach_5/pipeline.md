# Relabeling-Based Data Augmentation Pipeline for Indoor Localization

## Overview
This pipeline implements the relabeling method from the paper to address data imbalance in BLE beacon-based indoor localization.

---

## STEP 1: Data Preparation and Minority Class Identification

### 1.1 Load Raw Data
**Input:** Raw beacon data with columns:
- `timestamp`: datetime
- `mac_address`: string (MAC address of detected beacon)
- `rssi`: integer (signal strength)
- `location`: string/integer (room label)

### 1.2 Pivot to Beacon Matrix
Transform data from long format to wide format:

**Before:**
| timestamp           | mac_address       | rssi | location |
|---------------------|-------------------|------|----------|
| 2023-04-10 10:22:55 | FD:07:0E:D5:28:AE | -75  | 508      |
| 2023-04-10 10:22:55 | D2:1C:25:72:FB:E3 | -62  | 508      |

**After:**
| timestamp           | beacon_1 | beacon_2 | ... | beacon_25 | location |
|---------------------|----------|----------|-----|-----------|----------|
| 2023-04-10 10:22:55 | -75      | -62      | ... | 0         | 508      |

**Implementation notes:**
- Map each MAC address to beacon_1, beacon_2, ..., beacon_25
- Fill with 0 if beacon not detected at that timestamp
- Keep one row per timestamp

### 1.3 Identify Minority Classes
**Action:** Check value counts of `location` column in training data

**Definition of minority class:**
- Rooms with significantly fewer samples than median/mean
- In the paper: Room 508 (50 samples) and Room 516 (small samples) were minority classes
- Threshold: You can use < 10% of max class samples or manual inspection

**Output:** List of minority class rooms (e.g., [508, 516])

---

## STEP 2: Define 6-Beacon Surrounding Vectors

### 2.1 Define Beacon Layout
Based on the facility floor plan (Figure 8 in paper), define the 6 surrounding beacons for each room:

**Format:** `f_room = [fl, sl, s, f, sr, fr]`
- `fl`: front-left beacon
- `sl`: side-left beacon
- `s`: source beacon (at the room itself)
- `f`: front beacon
- `sr`: side-right beacon
- `fr`: front-right beacon

**Example:**
- Room 508: [7, 8, 9, 1, 10, 2] - all 6 beacons present
- Room 516: [None, 13, 15, 1, 16, 2] - incomplete (missing front-left beacon)
- Room 520: [6, 19, 20, 8, 21, 9] - complete
- Room 522: [7, 20, 22, 9, 23, 10] - complete

### 2.2 Classify Rooms by Matching Type
**Full Matching:**
- Rooms with all 6 beacons present (no None/null values)
- Example: [520, 522, 521, 511, ...]

**Partial Matching:**
- All rooms including those with incomplete beacons
- Example: [520, 522, 521, 511, 516, 515, ...]

**Implementation note:**
- Prefer full matching for better results
- Use partial matching only when full matching doesn't yield good results
- Replace `None` with 0 in beacon columns for partial matching

---

## STEP 3: Signal Pattern Matching (KL Divergence)

For each minority class room, find the best matched majority class room.

### 3.1 Create Sub-dataframes

**For minority class (e.g., Room 508):**
- Filter training data where location = 508
- Extract only the 6 surrounding beacon columns defined in beacon_layout[508]
- Result: pattern_min with shape (50 rows, 6 beacon columns)

Example structure:
| beacon_7 | beacon_8 | beacon_9 | beacon_1 | beacon_10 | beacon_2 |
|----------|----------|----------|----------|-----------|----------|
| -65      | -55      | -72      | -75      | -80       | -80      |
| -63      | -57      | -70      | -73      | -78       | -82      |
| ...      | ...      | ...      | ...      | ...       | ...      |

**For each candidate majority class (e.g., Room 520):**
- Filter training data where location = 520
- Extract only Room 520's 6 surrounding beacon columns from beacon_layout[520]
- Result: pattern_520 with shape (1000 rows, 6 beacon columns)

Example structure:
| beacon_6 | beacon_19 | beacon_20 | beacon_8 | beacon_21 | beacon_9 |
|----------|-----------|-----------|----------|-----------|----------|
| -60      | -62       | -68       | -58      | -75       | -70      |
| -62      | -64       | -66       | -56      | -73       | -72      |
| ...      | ...       | ...       | ...      | ...       | ...      |

### 3.2 Downsample Majority Class
**Purpose:** Ensure fair comparison between minority and majority patterns

**Process:**
- Get the number of samples in minority class (e.g., 50)
- Randomly sample the same number from majority class pattern
- Use random sampling without replacement
- Set random seed for reproducibility

**Result:** Both pattern_min and pattern_majority_downsampled have same number of rows (50, 6)

### 3.3 Calculate KL Divergence
**Process for each beacon column pair:**

1. Take column 1 from minority pattern (e.g., beacon_7 values)
2. Take column 1 from majority pattern (e.g., beacon_6 values)
3. Convert RSSI values to probability distributions:
   - Create histogram bins (e.g., from -100 to 0 dBm)
   - Count values in each bin
   - Normalize to sum to 1.0
4. Calculate KL divergence: D_KL(P || Q) = Σ P(i) × log(P(i)/Q(i))
   - P = minority distribution
   - Q = majority distribution
5. Repeat for all 6 beacon columns
6. Sum all 6 KL values to get total divergence score

**Example calculation:**
- Beacon column 1: KL = 0.35
- Beacon column 2: KL = 0.42
- Beacon column 3: KL = 0.28
- Beacon column 4: KL = 0.31
- Beacon column 5: KL = 0.45
- Beacon column 6: KL = 0.38
- **Total KL for Room 520** = 2.19

### 3.4 Find Best Match
**Process:**
1. Calculate KL divergence for minority class vs. ALL candidate majority rooms
2. Store results: {room_id: kl_score}
   - Room 520: KL = 2.19
   - Room 522: KL = 1.85 ← **LOWEST (BEST MATCH)**
   - Room 511: KL = 2.67
   - Room 521: KL = 2.34
3. Select room with minimum KL divergence as best match

**Output:** 
- Best matched room for each minority class
- Example: Room 508 → Room 522 (KL = 1.85)

---

## STEP 4: Relabeling and Data Augmentation

### 4.1 Determine Number of Records to Relabel
**Goal:** Balance the minority class

**Strategy options:** 
- **Option 1:** Add same number of samples as current minority class size (doubles it)
  - If minority has 50 samples, relabel 50 records from matched majority class
  - Result: minority class has 100 samples total
  
- **Option 2:** Use a target ratio
  - Set target count (e.g., 200 samples)
  - Relabel enough to reach that target
  - If minority has 50, relabel 150 records

**Paper's approach:** They doubled the minority class (Option 1)

### 4.2 Sample Records from Matched Majority Class
**Process:**
1. Identify matched majority room (e.g., Room 522 for minority Room 508)
2. Determine number of records to relabel (e.g., 50 records)
3. Randomly sample that many records from the matched majority room
4. Use random sampling without replacement
5. Set random seed for reproducibility

**Important:** Sample from the **full** matched majority class data (not the downsampled version used for KL calculation)

Example:
- Matched room: Room 522 (has 800 records)
- Minority room: Room 508 (has 50 records)
- Sample 50 random records from Room 522's 800 records

### 4.3 Relabel the Sampled Records
**Process:**
1. Take the sampled records from matched majority class
2. Keep all beacon columns (beacon_1 to beacon_25) with their original RSSI values
3. **Only change the location label** from majority class to minority class

Example:
| timestamp           | beacon_1 | beacon_2 | ... | beacon_25 | location |
|---------------------|----------|----------|-----|-----------|----------|
| 2023-04-10 09:15:20 | -75      | -80      | ... | -90       | 522      |
| 2023-04-10 09:15:21 | -73      | -82      | ... | -88       | 522      |

**After relabeling (522 → 508):**
| timestamp           | beacon_1 | beacon_2 | ... | beacon_25 | location |
|---------------------|----------|----------|-----|-----------|----------|
| 2023-04-10 09:15:20 | -75      | -80      | ... | -90       | **508**  |
| 2023-04-10 09:15:21 | -73      | -82      | ... | -88       | **508**  |

**Critical:** Only the location label changes; all RSSI values remain identical

### 4.4 Concatenate to Training Data
**Process:**
1. Take original training data
2. Append the relabeled records
3. Create augmented training dataset

**Before augmentation:**
- Room 508: 50 records
- Room 522: 800 records
- Total: 850 records

**After augmentation:**
- Room 508: 50 (original) + 50 (relabeled from 522) = **100 records**
- Room 522: 800 records (unchanged)
- Total: 900 records

**Note:** The augmented dataset now has better class balance

---

## STEP 5: Windowing and Feature Extraction

### 5.1 Apply Time-Based Windowing
**Window specification:**
- **Window size: 1-2 seconds** (specific to your dataset)
  - Paper used 45 seconds, but your data requires shorter windows
  - Start with 1 second, can increase to 2 seconds if needed
  - Choose based on data density and sampling rate
- **Overlap: 0 seconds** (non-overlapping)
- Apply to augmented training data

**Important note about window size:**
- Different datasets have different optimal window sizes
- Paper's facility: 45 seconds worked due to slower movement patterns
- Your competition data: 1-2 seconds is more appropriate
- Window size depends on beacon detection frequency and data characteristics

**Process:**
1. Group consecutive records within each 1-2 second window
2. Each window becomes one sample for the model

### 5.2 Extract Statistical Features for Each Beacon
For each of the 25 beacons, calculate these 5 statistics per window:
1. **Mean RSSI:** Average signal strength
2. **Standard deviation:** Signal variability
3. **Minimum RSSI:** Weakest signal
4. **Maximum RSSI:** Strongest signal
5. **Count:** Number of detections (how many times beacon was detected)

**Result:** 25 beacons × 5 features = **125 features per window**

### 5.3 Extract Temporal Features (Try later, we will start with 125 features above first)
From the timestamp, extract:
1. **Hour:** Hour of day (0-23)
2. **Minute:** Minute of hour (0-59)
3. **Microsecond:** Sub-second precision

**Result:** **3 temporal features**

### 5.4 Create Final Feature Matrix (Try later, we will start with 125 features above first)
**Total features per window:** 125 (statistical) + 3 (temporal) = **128 features**

**Final training data structure:**
| beacon_1_mean | beacon_1_std | ... | beacon_25_count | hour | minute | microsecond | location |
|---------------|--------------|-----|-----------------|------|--------|-------------|----------|
| -72.5         | 5.2          | ... | 15              | 10   | 22     | 55000       | 508      |
| -68.3         | 4.8          | ... | 18              | 10   | 23     | 40000       | 520      |
| ...           | ...          | ... | ...             | ...  | ...    | ...         | ...      |

---

## STEP 6: Model Training and Evaluation

### 6.1 Train Random Forest Classifier
**Model:** Random Forest
**Why Random Forest:**
- Handles imbalanced data well
- No need for feature scaling
- Can capture non-linear relationships
- Proven effective in indoor localization

**Training:**
- Input: Feature matrix (128 features) (Try later, we will start with 125 features above first)
- Output: Room location labels
- Use augmented training data from Step 5

### 6.2 Evaluate on Test Data
**Test data preparation:**
1. Apply same preprocessing as training data:
   - Pivot to beacon matrix
   - Apply windowing (1-2 seconds)
   - Extract same 128 features
2. **Do NOT apply any augmentation to test data**

**Prediction and Label Assignment:**
1. For each window in test data, model predicts a room location
2. **Assign the predicted window label to ALL records within that window**
   - Example: If a 2-second window (from 10:22:55 to 10:22:57) is predicted as Room 508
   - Then ALL frame-level records within that 2-second window get assigned location = 508
3. This converts window-level predictions back to frame-level predictions

**Evaluation metrics:**
1. **Macro F1-Score:** PRIMARY metric for competition ranking (equal weight to all classes)
2. **Per-Class F1-Score:** Especially for minority classes (Room 508, 516)
3. **Precision:** How accurate predictions are
4. **Recall:** How many actual samples were found

**Why Macro F1-Score:**
- Competition uses macro F1 for ranking
- Treats all classes equally (unlike weighted F1 which favors majority classes)
- Better metric for imbalanced datasets when all classes matter equally

### 6.3 Compare with Baseline
**Baseline:** Model trained on original data (no augmentation)

**Comparison points:**
1. Minority class F1-score improvement
2. **Overall Macro F1-score improvement** (competition metric)
3. Whether minority classes can now be detected (0% → >0%)

**Expected results (adapted from paper):**
- Target class F1-score improvement: 27% to 40%
- Overall macro F1-score improvement: expected similar gains
- Minority classes (Room 508, 516) become detectable

---

## Summary of Pipeline

1. **Data Prep** → Identify minority classes, pivot to beacon matrix
2. **Define Layout** → Map 6 surrounding beacons for each room
3. **Find Matches** → Calculate KL divergence, select best matched rooms
4. **Relabel** → Sample from matched majority, relabel to minority, concatenate
5. **Feature Engineering** → Window data (1-2 seconds), extract 128 features
6. **Train & Evaluate** → Random Forest on augmented data, predict on test windows, propagate labels to frames, evaluate with macro F1-score

---

## Key Implementation Notes

1. **Random seeds:** Always set random seeds for reproducibility in:
   - Downsampling for KL calculation
   - Sampling records for relabeling
   - Random Forest training

2. **Full vs Partial matching:** Always prefer full matching (complete 6 beacons). Only use partial matching if necessary.

3. **KL divergence direction:** Calculate D_KL(P || Q) where P = minority, Q = majority

4. **Test data:** Never augment test data. Only augment training data.

5. **Relabeling scope:** Only relabel the location column. Keep all RSSI values unchanged.

6. **Window size:** 1-2 seconds non-overlapping (dataset specific - paper used 45s)

7. **Feature count:** 25 beacons × 5 statistics + 3 temporal = 128 features total

8. **Competition metric:** Macro F1-score (equal weight for all classes)

9. **Label propagation:** After prediction on test windows, assign predicted label to ALL frame-level records within each window
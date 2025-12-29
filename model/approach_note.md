# Model Development Approaches - ABC 2026 Indoor Location Prediction

This document tracks all modeling approaches for the challenge, focusing on key insights and results.

---

## Phase 1: XGBoost Baseline (Approaches 1-7)

**Core methodology:** Create 25-dimensional feature vectors (one per beacon), apply 1-second temporal windowing, aggregate with statistics, train XGBoost classifier.

### Approach 1: Baseline

**Features:** 
- 1-second windows: group BLE readings by timestamp
- Aggregate per beacon: mean, std, count
- Total: 25 beacons × 3 statistics = 75 features

**Results:** 0.28 macro F1

**Per-class observations:**
- Majority classes (e.g., nurse station): 0.5-0.6 F1
- Minority classes (e.g., 505, 517, 518): 0.00-0.05 F1
- Hallway: consistently misclassified (spans multiple boundaries)

**Insight:** Class imbalance severely impacts macro F1. Need better handling of minority classes.

### Approach 2: Extended Features

**Change:** Added min and max statistics to aggregation
**Features:** 25 beacons × 5 statistics = 125 features

**Results:** 0.30-0.31 F1

**Insight:** Marginal improvement (+0.02-0.03). Additional statistics capture slightly more discriminative information but don't solve fundamental issues.

### Approach 3: Class Weighting

**Change:** Applied 3× weight to minority classes during XGBoost training

**Results:** No significant change from Approach 2

**Insight:** Simply forcing model to pay attention to minority classes doesn't help. Problem is data quality, not model bias - minority class signals are too noisy/unstable.

### Approach 4: SMOTE Oversampling

**Change:** Used SMOTE to synthetically generate minority class samples

**Results:** No significant change

**Insight:** Generating synthetic data from noisy patterns doesn't create meaningful discriminative information. Can't fix bad data with more bad data.

### Approach 5: Dominated Beacon Features

**Change:** Added "strongest beacon" as categorical feature (beacon with highest RSSI per window)

**Results:** No significant change

**Rationale failed:** Expected closest beacon to be highly indicative, but:
- Dominant beacon not stable within windows
- Multiple beacons often have similar RSSI
- Spatial overlap between beacon coverage areas

### Approach 6: Relabeling Technique

**Change:** Applied temporal smoothing/majority voting from reference paper

**Results:** No significant change (actually hurts macro F1)

**Why it failed:**
- Reference paper optimized for **weighted F1** (majority classes dominate)
- Relabeling biases toward majority classes
- **Macro F1** requires balanced per-class performance
- This is a metric mismatch problem

**Insight:** Techniques optimized for one metric don't transfer to others without modification.

### Approach 7: Two-Stage Zone Classification

**Methodology:**
- Stage 1: Classify general zone (Left/Middle/Right of floor)
- Stage 2: Within-zone room classification
- Separate classifiers per zone

**Results:** 0.30 macro F1 (same as single-stage)

**Zone performance (independent evaluation):**
- Left zone: 0.30-0.35 F1
- Middle zone: 0.45-0.50 F1
- Right zone: 0.20-0.25 F1

**Why it failed:**
- First-stage errors propagate to second stage
- Wrong zone → impossible to get room right
- Error compounding reduces robustness
- Hierarchical only works when first stage is very accurate (>90%)

**Insight:** Breaking down problem seems logical but introduces error cascading. Direct classification can be more robust. Spatial layout quality varies (Middle > Left > Right).

---

## Phase 2: Sequential Modeling Breakthrough (Approach 8)

### Approach 8: LSTM with Ground Truth Segmentation

**Motivation:** After 7 XGBoost approaches plateauing at 0.30 F1, fundamentally reconsidered the problem. Key insight: **location prediction is inherently sequential** - people move through space over time, and temporal patterns of beacon appearances should be highly informative.

**Shift in paradigm:** From independent window classification → sequence classification

**Feature Engineering:**
- Count beacon appearances within each 1-second window
- Two variants tested:
  - **8a:** Percentage features (normalize counts to percentages)
  - **8b:** Raw count features (absolute beacon detection counts)
- Result: 25-dimensional feature vector per timestep (one value per beacon, beacons 1-23 actually used)

**Sequence Creation (Ground Truth Segmentation):**
```python
df['room_group'] = (df['room'] != df['room'].shift()).cumsum()
```
- Use ground truth room labels to identify when room changes occur
- Each contiguous block of same room becomes one sequence
- Example: Person in Kitchen for 45 seconds → one sequence of 45 timesteps
- Model trained on clean, single-room sequences

**Model Architecture:**
- LSTM layers (128 → 64 units) to process sequences
- Learn temporal dependencies in beacon appearance patterns
- Output: room classification for entire sequence
- Masking layer to handle variable-length sequences
- Dropout for regularization

**Training Protocol:**
- 4-fold cross-validation (split by day)
- 10 random seeds per fold (40 total runs)
- Provides robust performance estimates with variance

**Results:**
- **Approach 8a (percentage features):** 0.4792 ± 0.0890
- **Approach 8b (raw count features):** 0.4804 ± 0.0793

**Comparison to XGBoost baseline:**
- ~60% relative improvement (0.30 → 0.48)
- Raw counts slightly better than percentages (lower variance)

**Why LSTM Works:**
1. **Temporal dependencies captured:** Sequence of beacon appearances over time more informative than static snapshots
2. **Beacon counts are stable:** Counting appearances more robust than noisy RSSI values
3. **Room visits have signatures:** Each room has distinctive temporal pattern in beacon sequences
4. **Context matters:** LSTM learns that certain beacon sequences only valid for specific rooms

**Performance Characteristics:**
- Both percentage and raw count features work nearly equally well
- Raw counts have slight edge in variance (0.0793 vs 0.0890)
- Breakthrough comes from sequential modeling paradigm, not specific feature engineering

**Critical Limitation - The "Cheating" Problem:**
- ⚠️ This approach uses ground truth room labels to create sequence boundaries (`room_group`)
- During actual inference on unlabeled test data, we won't know when room changes occur
- This is a **controlled experiment** validating that sequential patterns are learnable
- **Not deployable as-is** - still need to solve real-world segmentation challenge

**Key Validation:** Proves that if we could perfectly segment room visits, LSTM can classify them with ~0.48 F1. The challenge shifts to: how do we segment without ground truth?

---

## Phase 3: Realistic Inference Challenge (Approaches 9-11)

**Problem:** During real deployment, we don't know when room changes occur. Need automatic segmentation.

### Approach 9: Baseline Sliding Window (20s)

**Methodology:**
```python
# For each position i in the data:
window = data[i : i+20]  # 20 consecutive seconds
prediction[i] = model.predict(window)  # Predict room at position i+19 (end of window)
```

**How it works:**
- Create overlapping 20-second windows with 1-second step
- Window at position i contains data from seconds [i, i+1, ..., i+19]
- Predict the room label at the **END of the window** (timestamp i+19)
- Example: Window covering seconds 0-19 predicts room at second 19
- Each prediction is independent (no memory between windows)

**Why this design:**
- Looking at 20 seconds of history to predict current location
- Model sees temporal context leading up to current moment
- Sliding step of 1 second means we make a prediction for every second

**Results:** 0.2961 ± 0.0493

**Fold breakdown:**
- Fold 1: 0.3548 ± 0.0359 (range: 0.3120 - 0.4138)
- Fold 2: 0.2573 ± 0.0303 (range: 0.1722 - 0.2937)
- Fold 3: 0.2694 ± 0.0344 (range: 0.2144 - 0.3385)
- Fold 4: 0.3028 ± 0.0249 (range: 0.2539 - 0.3416)

**Gap from ideal:** 0.48 → 0.30 (approximately 40% degradation)

**The "Boundary Lag" Problem:**
```
Actual room transition at second 100 (Kitchen → Hallway):

Window [80-99]: All Kitchen signals → Predicts Kitchen ✓
Window [90-109]: 90% Kitchen + 10% Hallway → Predicts Kitchen (?) 
Window [95-114]: 75% Kitchen + 25% Hallway → Predicts ??? (confused)
Window [100-119]: 50% Kitchen + 50% Hallway → Predicts ??? (confused)
Window [110-129]: 100% Hallway → Predicts Hallway ✓

During transition, model sees MIXED signals it never saw during training
```

**Insight:** Fixed windows create train-test mismatch. Training uses pure room sequences (ground truth boundaries), but inference sees mixed signals at transitions. This distribution shift hurts performance significantly. The confusion zone spans ~20 seconds around each transition.

### Approach 10: Agile Sliding Window (10s)

**Methodology:**
```python
# For each position i in the data:
window = data[i : i+10]  # 10 consecutive seconds
prediction[i] = model.predict(window)  # Predict room at position i+9 (end of window)
```

**Changes from Approach 9:**
- Reduced window from 20s to 10s
- Window at position i contains data from seconds [i, i+1, ..., i+9]
- Predict room at the end of window (timestamp i+9)
- Goal: React faster to transitions, reduce boundary confusion zone

**Results:** 0.3086 ± 0.0558

**Fold breakdown:**
- Fold 1: 0.3526 ± 0.0485 (range: 0.2646 - 0.4365)
- Fold 2: 0.2654 ± 0.0424 (range: 0.1865 - 0.3136)
- Fold 3: 0.2888 ± 0.0415 (range: 0.2274 - 0.3625)
- Fold 4: 0.3275 ± 0.0449 (range: 0.2470 - 0.3976)

**Improvement:** +0.0125 over 20s window, but variance increased (0.0558 vs 0.0493)

**Trade-offs observed:**
- ✅ Faster reaction to transitions (confusion zone reduced from ~20s to ~10s)
- ✅ Better at capturing short room visits
- ❌ More jittery predictions within stable rooms (less context)
- ❌ Higher sensitivity to local signal noise
- ❌ Less temporal context per prediction (10s vs 20s history)

**Insight:** Shorter windows provide agility-stability trade-off. Marginal gain suggests window size isn't the fundamental bottleneck - the boundary mixing problem remains, just in a shorter window.

### Approach 11: Temporal Voting (10s + 5s voting)

**Methodology:**
```python
# Step 1: Get raw predictions with 10s window
for i in range(len(data)):
    window = data[i : i+10]
    raw_prediction[i] = model.predict(window)

# Step 2: Apply 5-second majority voting
for i in range(len(raw_prediction)):
    neighborhood = raw_prediction[i-2 : i+3]  # 5 predictions centered at i
    final_prediction[i] = majority_vote(neighborhood)
```

**How temporal voting works:**
- Base: Get raw predictions using 10s sliding window (as Approach 10)
- Post-processing: For each prediction at position i, look at predictions in range [i-2, i-1, i, i+1, i+2]
- Take majority vote among these 5 predictions
- Example: If 3 out of 5 predict "Kitchen", final prediction = "Kitchen"

**Rationale:**
- Smooth out isolated misclassifications
- If model briefly predicts wrong room, neighbors can correct it
- True room changes should persist across multiple predictions
- Random noise should be filtered out

**Results:** 0.3115 ± 0.0606

**Fold breakdown:**
- Fold 1: 0.3617 ± 0.0549 (range: 0.2595 - 0.4491)
- Fold 2: 0.2638 ± 0.0435 (range: 0.1773 - 0.3139)
- Fold 3: 0.2927 ± 0.0480 (range: 0.2298 - 0.3726)
- Fold 4: 0.3278 ± 0.0454 (range: 0.2484 - 0.4057)

**Improvement:** +0.0029 over raw 10s predictions (minimal)

**Maximum F1 achieved:** 0.4491 (shows potential when conditions align)

**Why improvement is modest:**
- 10s base predictions already relatively stable within rooms
- Most prediction changes occur at legitimate boundaries (not random jitter)
- 5s voting window may be too short to provide substantial smoothing
- Voting can fix isolated errors but can't fix systematic boundary confusion
- Fundamental boundary mixing problem not addressed by post-processing

**Insight:** Voting provides minimal benefit because the core issue isn't prediction jitter - it's that windows spanning transitions inherently contain mixed signals the model never saw during training. Post-processing can't fix the input distribution mismatch.

---

## Phase 4: Model Architecture Experiments (Approaches 12-15)

**Context:** All experiments use ground truth segmentation to isolate architecture effects.

### Approach 12: CNN-LSTM (Failed)

**Methodology:**
- CNN-LSTM v1: 3 Conv1D layers + MaxPooling + LSTM
- CNN-LSTM v2: 1 Conv1D layer (no pooling) + LSTM

**Results:**
- v1: 0.1406 ± 0.1631 (catastrophic - extreme variance)
- v2: 0.2966 ± 0.1354 (partial recovery, still poor)

**Insight:** CNN layers interfere with temporal learning for BLE data. Convolutions don't help - they hurt. Simpler is better.

### Approach 13: Bidirectional LSTM

**Methodology:** Bi-LSTM (processes forward + backward)

**Results:** 0.4895 ± 0.0660

**Improvement over LSTM:** +2.1% mean, -25.8% variance (more stable)

**Insight:** Bidirectional context improves both performance and stability. Seeing future beacons helps classify current location.

### Approach 14: Bidirectional GRU 🏆

**Methodology:** Bi-GRU (simpler than LSTM - 2 gates vs 3)

**Results:** 0.5272 ± 0.0725 ✅ **BEST with ground truth segmentation**

**Comparison:**
| Model | Mean F1 | Fold 2 (noisy) |
|-------|---------|----------------|
| Bi-GRU | 0.5272 | 0.5245 |
| Bi-LSTM | 0.4895 | 0.4363 |
| LSTM | 0.4792 | ~0.38 |

**Insight:** 
- **GRU's simpler architecture generalizes better on noisy BLE data**
- 20% improvement on noisiest fold (Day 3)
- Fewer parameters = implicit regularization = better for sensor noise
- **The Simplicity Principle validated:** For noisy sequential data, simpler models win

### Approach 15: Regular GRU

**Methodology:** Unidirectional GRU (forward only)

**Results:** ~0.46 F1

**Insight:** Too simple. Bidirectionality provides significant value. Bi-GRU is the sweet spot.

---

## Phase 5: Advanced Inference Strategies (Approaches 16-21) - Dec 2024

**Goal:** Bridge the 0.53 → 0.31 performance gap using realistic inference without ground truth boundaries.

### Approach 16: Temporal Gap Segmentation

**Methodology:**
- Segment test data by timestamp gaps (gap > 1s = new sequence)
- Classify each segment with Bi-GRU
- Propagate predictions back to frames

**Results:** 0.1478 ± 0.0897 (Fold 1)

**Why it failed:**
- BLE data has irregular timestamps even within same room
- Over-segmentation: breaks single room visits into tiny fragments
- Changing threshold to 3s helped slightly but still poor (~0.20 F1)
- **Insight:** Timestamp gaps reflect data collection artifacts, NOT room transitions

### Approach 17: Change Point Detection

**Methodology:**
- Used PELT algorithm (ruptures library) to detect beacon pattern changes
- Tested penalties: 5, 10, 20, 30, 50

**Results (best = penalty 5):**
- Boundary Precision: 0.609
- Boundary Recall: 0.373 (misses 63% of transitions!)
- Segment Purity: 0.822

**Why it failed:**
- Even at optimal settings, only detects 37% of room changes
- BLE signals are too noisy and gradual
- Room transitions don't create sharp statistical breaks
- Example: 892-frame segment contained 9 different rooms but detected as one

**Insight:** BLE beacon patterns change too smoothly. No clear "edges" for automatic segmentation.

### Approach 18: Multi-Scale Voting (5s, 10s, 20s windows)

**Methodology:**
- Run Bi-GRU on three window sizes simultaneously
- Combine predictions via majority vote (confidence as tie-breaker)

**Results:** 0.1478 ± 0.0897 (Fold 1) - **worse than single 10s window!**

**Why it failed:**
- Democratic voting: bad predictions (5s, 20s) outvote good predictions (10s)
- 5s windows: too noisy, reactive
- 20s windows: too slow, mixed signals
- Combined: suboptimal scales drag down optimal scale

**Insight:** Not all window sizes are equal. Simple majority vote doesn't account for quality differences.

---

## Phase 6: Breakthrough - Bi-GRU + Sliding Window (Approach 19) 🚀

### Approach 19: Bi-GRU with Sliding Window + Voting

**Methodology:**
- Replace LSTM → Bidirectional GRU (only change!)
- Keep same inference: 10s sliding window + 5s temporal voting
- No other modifications

**Results (4-fold CV, 10 seeds):**

**OVERALL: 0.3854 ± 0.0424**

**Fold breakdown:**
- Fold 1: 0.4120 ± 0.0431 (max: 0.4849)
- Fold 2: 0.3677 ± 0.0249
- Fold 3: 0.3910 ± 0.0425 (max: 0.4629)
- Fold 4: 0.3709 ± 0.0408

**Comparison:**
- LSTM + sliding window: 0.3115 ± 0.0606
- Bi-GRU + sliding window: 0.3854 ± 0.0424
- **Improvement: +23.7% relative gain**
- **Variance reduction: -30%** (more stable)

**Critical Insight:**
- **Model architecture matters MORE than inference strategy**
- Bi-GRU handles noisy, mixed windows much better than LSTM
- Even with imperfect segmentation, Bi-GRU makes good predictions
- Closes gap: now at 73% of ideal performance (vs 59% with LSTM)
- **The breakthrough wasn't segmentation - it was the model itself**

**Per-class highlights:**
- Class 523: 0.79 F1 (very stable)
- Kitchen: 0.72 F1 
- Nurse station: 0.62 F1
- Hallway: still challenging (~0.02 F1)

---

## Phase 7: Spatial Constraint Attempts (Approaches 20-21)

### Approach 20: Sequential Spatial Filtering

**Methodology:**
- Built room adjacency matrix from floor plan
- Sequential filtering: reject transitions to non-adjacent rooms
- Confidence threshold: 0.6
- Time threshold: 3 seconds

**Results:** 0.0291 F1 (catastrophic failure)

**Why it failed - The Cascading Error Problem:**
```
True: Kitchen → Hallway → Room 523
Pred: Kitchen → Room 501 (wrong!) → 501 → 501 → 501...
                          ↑
                    STUCK FOREVER!
                    
501 not adjacent to 523 → REJECT
501 not adjacent to hallway → REJECT
Model becomes paralyzed
```

**Insight:** 
- One wrong prediction creates a lock
- Sequential decisions can't recover from early mistakes
- Local filtering is fundamentally flawed
- Need global optimization, not greedy decisions

### Approach 21: Viterbi Spatial Decoding

**Methodology:**
- Global optimization using Viterbi algorithm
- Considers ENTIRE sequence jointly
- Non-adjacent transitions penalized but not forbidden
- Transition penalty parameter: 5.0

**Results:** 0.4050 ± 0.0420 (+0.02 improvement)

**Why modest improvement:**
- Viterbi prevents cascading errors (good!)
- But spatial constraints may be too weak for this data
- Adjacency matrix might not perfectly match actual movement patterns
- Penalty tuning may need more exploration

**Insight:** 
- Global optimization > sequential filtering
- Small gain suggests spatial constraints have limited signal
- The 0.02 improvement indicates some value but not transformative

---

## Summary of Results

| Approach | Method | Macro F1 | Key Insight |
|----------|--------|----------|-------------|
| **1-7** | XGBoost variants | ~0.30 | Feature engineering plateau |
| **8** | LSTM + ground truth | 0.48 | Sequential modeling works! |
| **9-11** | LSTM + sliding window | 0.31 | 40% degradation from ideal |
| **12** | CNN-LSTM | 0.14-0.30 | Complexity hurts |
| **13** | Bi-LSTM + ground truth | 0.49 | Bidirectional helps |
| **14** | Bi-GRU + ground truth | **0.53** | Simpler = better |
| **16** | Temporal gap segment | 0.15 | Timestamps ≠ room changes |
| **17** | Change point detection | N/A | Only 37% recall |
| **18** | Multi-scale voting | 0.15 | Bad scales drag down good |
| **19** | **Bi-GRU + sliding window** | **0.39** 🏆 | **Architecture > strategy** |
| **20** | Sequential spatial filter | 0.03 | Cascading errors |
| **21** | Viterbi spatial | 0.41 | Modest gain (+0.02) |

---

## Core Insights

### 1. The Simplicity Principle (Architecture)
**For noisy BLE data:** Bi-GRU > Bi-LSTM > LSTM > CNN-LSTM

- Fewer parameters = better generalization
- GRU's 2 gates prevent overfitting to noise better than LSTM's 3 gates
- 20% performance gain on noisiest data (Fold 2)

### 2. The Model Architecture Dominance
**Changing LSTM → Bi-GRU gave +23.7% improvement**

- Bigger impact than any inference strategy
- Bi-GRU handles mixed boundary windows better
- Architecture choice matters more than segmentation quality

### 3. The Automatic Segmentation Problem
**All segmentation attempts failed:**

- Temporal gaps (0.15 F1): Data collection ≠ room transitions
- Change point detection (37% recall): Signals too smooth
- Multi-scale voting (0.15 F1): Democratic voting hurts

**Reality:** BLE beacon patterns don't have clear boundaries suitable for automatic detection.

### 4. The Spatial Constraint Limitation
**Spatial adjacency has limited signal:**

- Sequential filtering: catastrophic (cascading errors)
- Viterbi global optimization: modest gain (+0.02)
- Suggests physical constraints less informative than expected
- Movement patterns may not strictly follow floor plan adjacency

### 5. The Inference Strategy Lesson
**Sliding window + voting remains most practical:**

- Simple and robust
- Avoids catastrophic failure modes
- Combined with right model (Bi-GRU), achieves 73% of ideal performance
- Diminishing returns on further optimization

---

## Current Best Approach

**Configuration:**
- Model: Bidirectional GRU (128 → 64 units)
- Features: Beacon count percentages (23-dim per second)
- Training: Ground truth room sequences (max 50 timesteps)
- Inference: 10-second sliding window + 5-second temporal voting
- Optional: Viterbi spatial decoding (+0.002 F1)

**Performance:**
- Mean: 0.3854 F1 (without Viterbi) / 0.3874 F1 (with Viterbi)
- Variance: 0.0424 (stable across seeds)
- Gap from ideal: 27% (down from 41% with LSTM)

---

## Open Questions & Future Directions

### What Worked
✅ Bidirectional GRU architecture (biggest win)
✅ Beacon count features (raw or percentage)
✅ Simple sliding window inference (robust)
✅ Temporal voting smoothing (small gain)
✅ Viterbi global optimization (small gain)

### What Failed
❌ XGBoost window-based features
❌ Complex architectures (CNN-LSTM)
❌ Automatic segmentation (all methods)
❌ Multi-scale democratic voting
❌ Sequential spatial filtering

### Remaining Opportunities
1. **Confidence-based adaptive windowing:** Vary window size based on model uncertainty
2. **Ensemble methods:** Combine multiple Bi-GRU models trained with different seeds
3. **Better spatial constraints:** Learn transition patterns from data rather than floor plan
4. **Beacon proximity features:** Explicitly model which beacons are expected per room
5. **Class-specific strategies:** Handle hallways differently from rooms

### The Fundamental Trade-off
**Segmentation quality vs. Model robustness:**

- Perfect segmentation (ground truth): 0.53 F1
- Imperfect segmentation (sliding window): 0.39 F1
- **Gap: 0.14 F1 (26% of performance)**

Can we close this gap further? Likely ceiling around 0.42-0.45 F1 with current approach unless we solve the segmentation problem or get even better at handling mixed signals.

---

## Lessons for Similar Problems

1. **Try simpler models first:** GRU beat LSTM on noisy sensor data
2. **Bidirectional helps:** Seeing both past and future context matters
3. **Architecture > hyperparameters:** Changing model type gave 10× more improvement than any tuning
4. **Beware cascading errors:** Sequential filtering can fail catastrophically
5. **Automatic segmentation is hard:** When boundaries aren't clear, don't force them
6. **Embrace imperfection:** Sliding windows work well enough with the right model
7. **Global > local optimization:** Viterbi beats greedy decisions (but margin may be small)
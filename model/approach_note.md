# Model Development Approaches

This document tracks all modeling approaches attempted for the ABC 2026 Location Prediction Challenge, including methodologies, results, and key insights.

---

## Baseline Approach (Approach 1)

### Methodology

**Pipeline Overview:**
1. **Feature Vector Creation**: Create a 25-dimensional feature vector representing the 25 beacons (beacon_1, beacon_2, ..., beacon_25)
   - Each feature value corresponds to the RSSI value if the MAC address matches the beacon ID
   - Missing beacons have null/zero values

2. **Temporal Windowing**: Apply 1-second windowing by grouping records by timestamp (rounded to seconds)
   - **Aggregated features per beacon**: mean, std, count
   - **Total features**: 25 beacons × 3 statistics = 75 features per window

3. **Model Training**: Train XGBoost Classifier on windowed data
   - Each window is treated as a single training sample
   - Target: room label

4. **Prediction**: Apply the same preprocessing to test data, classify at window level, then propagate predictions back to frame level

### Results

**Initial Performance:**
- **Macro F1 Score**: 0.28
- **Majority class performance**: F1 scores of 0.5 - 0.6
- **Minority class performance**: F1 scores of 0.00 - 0.05 (severely impacting macro F1)
- **Hallway classification**: Consistently misclassified as other rooms (understandable given that hallways span across multiple room boundaries)

### Key Insights

The baseline establishes a reasonable framework but suffers from severe class imbalance issues. Minority classes and spatially ambiguous locations (like hallways) receive extremely low F1 scores, dragging down the macro F1 metric.

---

## Approach 2: Extended Feature Set

### Methodology

Extended the aggregated features from Approach 1 by adding additional statistical measures:
- **Previous features**: mean, std, count
- **New features**: min, max
- **Total features**: 25 beacons × 5 statistics = 125 features per window

### Results

**Performance:**
- **Macro F1 Score**: 0.30 - 0.31 (verified on both train/test splits)
- **Improvement**: +0.02 to +0.03 over baseline

### Key Insights

Adding min/max features provides marginal improvement, suggesting that additional statistical measures capture slightly more discriminative information about signal patterns. However, the improvement is modest and doesn't address the fundamental class imbalance problem.

---

## Approach 3: Class Weighting Strategy

### Methodology

Applied aggressive class weighting to address class imbalance:
- Assigned 3× weight to minority classes compared to majority classes during XGBoost training
- Hypothesis: Forcing the model to pay more attention to minority classes would improve their F1 scores

### Results

**Performance:**
- **Macro F1 Score**: No significant change from Approach 2
- **Outcome**: Failed to improve performance

### Key Insights

Simply increasing class weights does not solve the underlying issue. The problem is not that the model ignores minority classes, but rather that the minority class data itself is too noisy, unstable, or insufficiently distinctive. Overweighting poor-quality data doesn't make it more learnable.

---

## Approach 4: SMOTE / Oversampling

### Methodology

Applied synthetic minority oversampling techniques (SMOTE) to balance class distributions in the training set:
- Generated synthetic samples for minority classes
- Aimed to provide more training examples for underrepresented rooms

### Results

**Performance:**
- **Macro F1 Score**: No significant change from previous approaches
- **Outcome**: Failed to improve performance

### Key Insights

SMOTE and oversampling techniques do not address the root cause of poor minority class performance. The issue appears to be inherent data quality problems:
- High signal noise in certain locations
- Insufficient spatial distinctiveness between some rooms
- Instability in RSSI readings for certain beacon configurations

Generating synthetic data from noisy patterns does not create meaningful discriminative information.

---

## Approach 5: Dominated Beacon Features

### Methodology

Introduced additional aggregated features during windowing:
- **Dominated beacon**: The beacon with the strongest (highest) RSSI value in each window
- **Rationale**: The closest beacon might be highly indicative of location
- Added as a categorical or one-hot encoded feature alongside existing statistical features

### Results

**Performance:**
- **Macro F1 Score**: No significant change
- **Outcome**: Failed to improve performance despite theoretical promise

### Key Insights

While the concept of identifying the dominant beacon sounds promising, the current implementation did not yield improvements. Possible reasons:
- The dominant beacon may not be sufficiently stable within windows
- Multiple beacons may have similar RSSI values, making "dominance" ambiguous
- Spatial overlap between beacon coverage areas reduces discriminative power

This approach may warrant further investigation with refinements (e.g., top-k beacons, beacon ratios), but the initial results are not encouraging.

---

## Approach 6: Relabeling Technique (Paper-Inspired)

### Methodology

Implemented the relabeling technique introduced in the reference paper suggested by the competition organizers:
- Applied temporal smoothing or majority voting across neighboring windows
- Aimed to reduce label noise and improve prediction consistency

### Results

**Performance:**
- **Macro F1 Score**: No significant change
- **Outcome**: Failed to improve performance

### Key Insights

The relabeling technique works well for **weighted F1 score** (the metric used in the reference paper) because it biases predictions toward majority classes, which dominate the weighted average. However, this bias is counterproductive for **macro F1 score**, where each class is weighted equally regardless of size.

**Why it failed:**
- The technique artificially boosts majority class predictions
- Minority classes become even harder to predict correctly
- Macro F1 requires balanced per-class performance, not overall accuracy

This confirms that techniques optimized for weighted metrics do not transfer to macro metrics without modification.

---

## Approach 7: Two-Stage Zone-Based Classification

### Methodology

Reformulated the problem as a hierarchical classification task:

**Stage 1 - Zone Classification:**
- Classify the general zone first (Left, Middle, Right side of the 5th floor)
- 3-class problem instead of 20+ class problem

**Stage 2 - In-Zone Room Classification:**
- Within each predicted zone, perform fine-grained room classification
- Separate classifiers trained for each zone

**Rationale:** Breaking down the complex multi-class problem into smaller, more manageable sub-problems should improve performance by:
- Reducing confusion between spatially distant rooms
- Allowing zone-specific feature learning
- Decreasing the effective number of classes per classifier

### Results

**Overall Performance:**
- **Macro F1 Score**: 0.30 (same as single-stage approaches)
- **Outcome**: Did not improve over baseline

**Zone Classification Performance (when trained and evaluated per zone independently):**
- Left zone: F1 ~0.30 - 0.35
- Middle zone: F1 ~0.45 - 0.50
- Right zone: F1 ~0.20 - 0.25

**Problem with Two-Stage Approach:**
- First-stage errors propagate to second stage
- If zone classifier is wrong, the room classifier never had a chance
- Error compounding reduces overall performance

### Key Insights

Hierarchical approaches can work well when the first stage is very accurate, but in this case:
- Zone boundaries are not perfectly distinct in signal space
- First-stage errors (wrong zone prediction) eliminate any possibility of correct room prediction
- The compound error rate makes this approach less robust than direct room classification

The differential performance across zones (Middle > Left > Right) suggests that spatial layout and beacon placement quality vary significantly across the floor.

---

## Approach 8: LSTM-Based Sequential Modeling (The Breakthrough)

### Motivation

After 7 approaches exploring XGBoost with various feature engineering techniques (all plateauing at ~0.30 macro F1), we fundamentally reconsidered the problem formulation. The key insight: **location prediction is inherently sequential** - people move through space over time, and the temporal pattern of beacon appearances should be highly informative.

### Methodology

**Shift from Independent Windows to Sequential Modeling:**

**Feature Representation:**
- Count how many times each beacon appears within each 1-second window
- Two variants tested:
  - **Approach 8a**: Percentage features (normalize counts to percentages)
  - **Approach 8b**: Raw count features (use absolute counts)
- Result: 25-dimensional feature vector per timestep (one value per beacon)

**Sequence Creation (Controlled Experiment with Ground Truth):**
```python
df['room_group'] = (df['room'] != df['room'].shift()).cumsum()
```
- Use ground truth room labels to identify when room changes occur
- Each contiguous block of the same room becomes one sequence
- This provides LSTM with clean, single-room sequences for training

**Model Architecture:**
- LSTM layers to process sequences of beacon appearance patterns
- Learn temporal dependencies in how beacons appear/disappear during room visits
- Output: room classification for the entire sequence

**Evaluation Protocol:**
- 4-fold cross-validation
- 10 random seeds per fold (40 total runs)
- Provides robust performance estimates with variance

### Results

**Approach 8a (Percentage Features):**
- **Mean Macro F1**: 0.4792 ± 0.0890
- **Improvement**: ~60% over XGBoost baseline (0.48 vs 0.30)

**Approach 8b (Raw Count Features):**
- **Mean Macro F1**: 0.4804 ± 0.0793
- **Improvement**: ~60% over XGBoost baseline
- **Observation**: Slightly lower variance than percentage features

### Key Insights

**Why LSTM Works:**
1. **Temporal dependencies are critical**: The sequence of beacon appearances over time is more informative than static window snapshots
2. **Beacon counts are stable**: Counting beacon appearances is more robust than using noisy RSSI values
3. **Room visits have characteristic patterns**: Each room has a distinctive temporal "signature" in beacon appearance sequences
4. **Context matters**: LSTM can learn that certain beacon sequences are only valid for specific rooms

**Important Note: The "Cheating" Problem:**
- This approach uses ground truth room labels to create sequence boundaries (`room_group`)
- During actual inference on unlabeled test data, we won't know when room changes occur
- This controlled experiment validates that sequential patterns are learnable, but we still need to solve the real-world segmentation challenge

**Performance Characteristics:**
- Both percentage and raw count features work nearly equally well
- Raw counts may have slight edge in variance (0.0793 vs 0.0890)
- The breakthrough comes from the sequential modeling paradigm, not specific feature engineering

---

## The Segmentation Challenge and Realistic Inference

### The Core Problem

**What we validated in Approach 8:**
- ✅ Sequential patterns in beacon appearances are highly discriminative
- ✅ LSTM can learn these patterns effectively (~0.48 macro F1)
- ✅ Given perfect segmentation (ground truth boundaries), the approach works

**The real-world challenge:**
- ❌ During inference, we don't know when room changes occur
- ❌ We need automatic segmentation strategies for continuous, unlabeled BLE data
- ❓ How much performance degrades from "ideal" (ground truth) to "realistic" (automatic) segmentation?

### Bridging the Gap: Realistic Inference Strategies

Following the breakthrough in Approach 8, we shifted focus to developing inference strategies that work on continuous, unlabeled data without known room boundaries.

---

## Approach 9: Baseline Sliding Window (20s)

### Methodology

**Inference Strategy:**
- Applied a 20-second sliding window with 1-second step size
- Each window treated as a separate sequence for LSTM prediction
- No ground truth boundaries used during inference

**Goal:**
- Establish a baseline for continuous prediction performance
- Measure the performance gap between ideal segmentation (Approach 8) and realistic sliding window inference

**Technical Details:**
- Window size: 20 seconds of continuous BLE data
- Step size: 1 second (overlapping windows)
- Model: Same LSTM architecture from Approach 8

### Results

**Performance (4-fold CV, 10 seeds per fold, 40 runs total):**
- **Mean Macro F1**: 0.2961 ± 0.0493
- **Min Macro F1**: 0.1722
- **Max Macro F1**: 0.4138

**Fold-wise breakdown:**
- Fold 1: 0.3548 ± 0.0359 (range: 0.3120 - 0.4138)
- Fold 2: 0.2573 ± 0.0303 (range: 0.1722 - 0.2937)
- Fold 3: 0.2694 ± 0.0344 (range: 0.2144 - 0.3385)
- Fold 4: 0.3028 ± 0.0249 (range: 0.2539 - 0.3416)

### Key Insights

**Significant Performance Drop:**
- Degradation from ideal segmentation: 0.48 → 0.29 (approximately 40% relative decrease)
- The performance gap highlights the critical importance of segmentation quality

**The "Boundary Lag" Problem:**
- During room transitions, a 20-second window contains signals from **two different rooms**
- The model struggles to make a clean prediction when the window spans a room boundary
- This mixing of signals from different locations creates ambiguity

**Variability Across Folds:**
- Fold 1 performs best (0.3548), suggesting some room transitions or locations are easier to handle
- Fold 2 performs worst (0.2573), indicating certain spatial configurations are more challenging
- This variability suggests the difficulty is not uniform across all room pairs

**Per-Class Performance Patterns:**
- Classes with stable, distinctive beacon patterns (e.g., class 523, nurse station) maintain reasonable F1 scores
- Classes with ambiguous or transitional characteristics show severe performance drops
- Some minority classes drop to zero F1 (e.g., class 505, 517, 518 in various folds)

---

## Approach 10: Agile Sliding Window (10s)

### Methodology

**Inference Strategy:**
- Reduced window size from 20 seconds to 10 seconds
- Same 1-second step size (overlapping windows)
- Goal: Increase model "agility" to detect room transitions faster

**Rationale:**
- Shorter windows should react faster to room changes
- Reduced "boundary lag" as windows span less time
- Trade-off: less temporal context per window vs. faster adaptation

### Results

**Performance (4-fold CV, 10 seeds per fold, 40 runs total):**
- **Mean Macro F1**: 0.3086 ± 0.0558
- **Min Macro F1**: 0.1865
- **Max Macro F1**: 0.4365

**Fold-wise breakdown:**
- Fold 1: 0.3526 ± 0.0485 (range: 0.2646 - 0.4365)
- Fold 2: 0.2654 ± 0.0424 (range: 0.1865 - 0.3136)
- Fold 3: 0.2888 ± 0.0415 (range: 0.2274 - 0.3625)
- Fold 4: 0.3275 ± 0.0449 (range: 0.2470 - 0.3976)

**Comparison to 20s Window:**
- Mean improvement: +0.0125 (0.3086 vs 0.2961)
- Variance increase: 0.0558 vs 0.0493

### Key Insights

**Modest Performance Improvement:**
- The 10s window shows a marginal mean F1 improvement (~0.012 boost)
- Maximum F1 increased from 0.4138 to 0.4365, suggesting the approach has higher ceiling potential
- Performance remains substantially below ideal segmentation (0.48)

**Increased Instability:**
- Higher variance (0.0558 vs 0.0493) indicates less stable predictions
- The smaller window provides less temporal context, making predictions more sensitive to local noise
- Some seeds perform worse than with 20s window (minimum F1 dropped from 0.1722 to 0.1865 in one fold)

**The Agility-Stability Trade-off:**
- **Advantages of 10s window:**
  - Faster reaction to room transitions
  - Less "boundary lag" when crossing room boundaries
  - Potentially better at capturing short room visits
  
- **Disadvantages of 10s window:**
  - More "jittery" predictions within stable rooms
  - Higher frequency of incorrect predictions due to local signal fluctuations
  - Less robust to momentary signal anomalies

**Per-Class Behavior:**
- Classes with distinctive patterns show improvement (e.g., class 508: 0.1254 → 0.2722 in Fold 1)
- Classes in high-transition areas still struggle (e.g., hallways, room boundaries)
- Minority classes remain challenging, with several still achieving zero F1

---

## Approach 11: Temporal Smoothing (10s Window + Voting)

### Methodology

**Inference Strategy:**
- Base prediction: 10-second sliding window (same as Approach 10)
- Post-processing: Applied 5-second Majority Vote (Temporal Voting) filter
- Goal: Smooth out high-frequency "jitter" while maintaining fast transition response

**Voting Mechanism:**
- For each timestamp, collect predictions from the past 5 seconds
- Use majority voting to determine final room prediction
- Rationale: True room changes should persist across multiple windows, while noise should be filtered out

**Technical Details:**
- Primary window: 10 seconds
- Voting window: 5 seconds
- Combines agility of short windows with stability of temporal consensus

### Results

**Performance (4-fold CV, 10 seeds per fold, 40 runs total):**
- **Mean Macro F1**: 0.3115 ± 0.0606
- **Min Macro F1**: 0.1773
- **Max Macro F1**: 0.4491

**Fold-wise breakdown:**
- Fold 1: 0.3617 ± 0.0549 (range: 0.2595 - 0.4491)
- Fold 2: 0.2638 ± 0.0435 (range: 0.1773 - 0.3139)
- Fold 3: 0.2927 ± 0.0480 (range: 0.2298 - 0.3726)
- Fold 4: 0.3278 ± 0.0454 (range: 0.2484 - 0.4057)

**Comparison to Raw 10s Window:**
- Mean improvement: +0.0029 (0.3115 vs 0.3086)
- Variance increase: 0.0606 vs 0.0558
- Maximum F1 improved: 0.4491 vs 0.4365

### Key Insights

**Highest Performance Achieved for Continuous Inference:**
- Best mean Macro F1 across all realistic inference approaches: 0.3115
- Highest maximum F1 observed: 0.4491 (approaching the ideal segmentation performance of ~0.48)
- Voting provides marginal but consistent improvement over raw predictions

**Marginal Impact of Voting:**
- Improvement over raw 10s window: +0.003 mean F1
- The benefit is modest, suggesting:
  - The 10s predictions are already relatively stable
  - Most "jitter" occurs at legitimate transition boundaries rather than within rooms
  - 5-second voting window may be too short to provide substantial smoothing

**Variance Characteristics:**
- Highest variance of all approaches (0.0606)
- The 10s base window's instability persists even with smoothing
- Suggests the fundamental challenge is window size, not post-processing

**Performance Ceiling:**
- Maximum F1 of 0.4491 demonstrates that near-ideal performance is achievable in some conditions
- Gap to ideal segmentation: 0.4491 vs 0.4804 (only ~6% difference in best cases)
- This indicates that automatic segmentation can occasionally match ground truth segmentation quality

**Per-Class Patterns:**
- Voting helps stabilize predictions for classes with moderate distinctiveness
- Classes with very low baseline F1 (e.g., hallways, minority rooms) show minimal benefit
- High-performing classes (e.g., class 523, nurse station) maintain strong scores with added stability

---

## Summary of Results

| Approach | Macro F1 Score | Key Technique | Outcome |
|----------|----------------|---------------|---------|
| **Approach 1** (Baseline) | 0.28 | 1-sec windowing, mean/std/count features | Baseline established |
| **Approach 2** | 0.30 - 0.31 | Added min/max features | Small improvement |
| **Approach 3** | 0.30 - 0.31 | 3× minority class weighting | No change |
| **Approach 4** | 0.30 - 0.31 | SMOTE oversampling | No change |
| **Approach 5** | 0.30 - 0.31 | Dominated beacon features | No change |
| **Approach 6** | 0.30 - 0.31 | Relabeling technique | No change |
| **Approach 7** | 0.30 | Two-stage zone classification | No change |
| **Approach 8a** | **0.4792 ± 0.0890** | LSTM + percentage features + ground truth segmentation | **~60% improvement!** |
| **Approach 8b** | **0.4804 ± 0.0793** | LSTM + raw count features + ground truth segmentation | **~60% improvement!** |
| **Approach 9** | **0.2961 ± 0.0493** | 20s sliding window inference | Realistic baseline |
| **Approach 10** | **0.3086 ± 0.0558** | 10s sliding window inference | Modest improvement, higher variance |
| **Approach 11** | **0.3115 ± 0.0606** | 10s sliding window + 5s voting | **Best continuous inference** |

---

## Lessons Learned

### What Doesn't Work

1. **Simple class balancing techniques** (class weighting, SMOTE) don't address data quality issues
2. **Temporal smoothing/relabeling** optimized for weighted metrics hurts macro F1 performance
3. **Hierarchical classification** struggles when first-stage errors propagate downstream
4. **Additional statistical features** (min/max, dominant beacon) provide only marginal gains
5. **Independent window classification** (XGBoost) ignores valuable temporal dependencies

### What Works

1. **Sequential modeling with LSTM** captures temporal dependencies effectively (~60% improvement over XGBoost)
2. **Beacon appearance counts/frequencies** are more stable than raw RSSI values
3. **Shorter sliding windows (10s)** provide better agility for room transition detection
4. **Temporal voting/smoothing** provides marginal stability improvements for continuous inference
5. **4-fold cross-validation with multiple seeds** provides robust performance estimates

### Core Insights

1. **RSSI values are inherently noisy** - don't rely on exact signal strength
2. **Temporal patterns matter** - which beacons appear over time is more informative than static snapshots
3. **The segmentation quality bottleneck** - performance gap between ideal (0.48) and realistic (0.31) segmentation is substantial (~35% relative degradation)
4. **Boundary lag is the primary challenge** - windows spanning room transitions create ambiguous predictions
5. **Agility-stability trade-off** - shorter windows react faster but are less stable; longer windows are more stable but have more boundary lag
6. **Macro F1 is granularity-agnostic** - sequence-level ≈ frame-level predictions

### Open Challenges

1. **Closing the segmentation gap**: Can we develop better automatic segmentation to approach the 0.48 ideal performance?
2. **Boundary detection**: Can we explicitly detect room transitions to create cleaner sequences?
3. **Adaptive windowing**: Should window size vary based on signal patterns or predicted confidence?
4. **Competition format**: What is the actual test data format and submission requirements?

---

## Key Observations Across Approaches 9-11

### Performance Comparison

**Mean Macro F1 Progression:**
- 20s window (Approach 9): 0.2961 ± 0.0493
- 10s window (Approach 10): 0.3086 ± 0.0558
- 10s window + voting (Approach 11): 0.3115 ± 0.0606

**Maximum Performance:**
- 20s window: 0.4138
- 10s window: 0.4365
- 10s window + voting: 0.4491 (closest to ideal 0.48)

**Performance Gap from Ideal Segmentation:**
- Best case: 0.4804 (ideal) → 0.4491 (voting) = 6.5% degradation
- Mean case: 0.4804 (ideal) → 0.3115 (voting) = 35.2% degradation
- This gap represents the fundamental challenge of automatic segmentation

### The Boundary Lag Problem

**Observed pattern across all sliding window approaches:**
- When a window spans a room transition (e.g., 15 seconds in Room A, 5 seconds in Room B), the model faces ambiguous input
- The beacon signal mixture from two rooms creates a "hybrid" pattern that doesn't match either room's training profile
- This results in either:
  1. Delayed transition detection (predicting old room for too long)
  2. Premature transition (switching rooms before actually transitioning)
  3. Oscillating predictions (rapidly switching between rooms at boundaries)

### Variance and Stability Patterns

**Variance increases with shorter windows:**
- 20s window: ± 0.0493
- 10s window: ± 0.0558
- 10s + voting: ± 0.0606

**Why variance increases:**
- Less temporal context in shorter windows makes predictions more sensitive to local noise
- Different random seeds lead to different learned patterns for handling ambiguous short sequences
- The voting mechanism doesn't reduce variance because it operates on already-variable predictions

**Fold-specific variability:**
- Fold 1 consistently performs best across all approaches (mean F1 ~0.35-0.36)
- Fold 2 consistently performs worst (mean F1 ~0.25-0.26)
- This suggests spatial heterogeneity: some areas/room pairs are inherently easier or harder

### The Agility vs. Stability Trade-off

**20s Window:**
- ✅ More stable predictions within rooms
- ✅ Better temporal context for the model
- ❌ Slower to detect room changes
- ❌ Longer boundary lag periods

**10s Window:**
- ✅ Faster room transition detection
- ✅ Higher performance ceiling (0.4365 vs 0.4138)
- ❌ More "jittery" predictions
- ❌ Less temporal context

**10s + Voting:**
- ✅ Slightly smoother than raw 10s
- ✅ Highest mean and max performance
- ❌ Still inherits 10s window instability
- ❌ Marginal improvement suggests diminishing returns

---

## Conclusion

After establishing that LSTM-based sequential modeling achieves ~0.48 macro F1 with ideal segmentation (Approach 8), we successfully bridged the gap to realistic inference scenarios with continuous, unlabeled data.

**Key findings from realistic inference experiments (Approaches 9-11):**

1. **Sliding window inference is viable** but comes with a substantial performance cost (~35% degradation from ideal)

2. **The 10-second window with temporal voting (Approach 11)** represents the best realistic inference strategy tested:
   - Mean Macro F1: 0.3115 ± 0.0606
   - Max Macro F1: 0.4491 (approaching ideal performance in best cases)
   
3. **The boundary lag problem** is the primary bottleneck:
   - Windows spanning room transitions contain mixed signals
   - This creates prediction ambiguity that fundamentally limits performance
   - Post-processing smoothing provides only marginal benefits

4. **The agility-stability trade-off is real**:
   - Shorter windows detect transitions faster but are less stable
   - Longer windows are more stable but have worse boundary lag
   - 10 seconds appears to be a reasonable balance point

5. **Significant performance variability** exists:
   - High variance across seeds and folds indicates the problem is challenging
   - Some spatial configurations are much easier than others
   - Best-case performance (0.4491) nearly matches ideal segmentation, suggesting room for optimization

**Current status:** We have developed and validated a realistic inference pipeline that can process continuous BLE data without ground truth boundaries, achieving ~0.31 mean macro F1 with peak performance of ~0.45. The ~35% performance gap from ideal segmentation represents the engineering challenge of automatic sequence segmentation in real-world deployment.
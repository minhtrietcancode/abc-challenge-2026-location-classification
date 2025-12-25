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
- **Left Zone**: Macro F1 = 0.51
- **Middle Zone**: Macro F1 = 0.44  
- **Right Zone**: Macro F1 = 0.32
- **Zone Classification Accuracy**: 0.86

### Key Insights

**Interesting observation:** When we train and evaluate models on individual zones in isolation (knowing the true zone), performance is significantly better (0.32 - 0.51) than the full problem (0.30). This suggests that:
- Zone-specific patterns exist and are learnable
- The main bottleneck is distinguishing between zones accurately
- Within-zone classification is more tractable

**Why the approach failed overall:**
- Despite 86% zone classification accuracy, the 14% zone misclassification compounds errors
- Misclassifying the zone in Stage 1 guarantees room misclassification in Stage 2
- Error propagation from Stage 1 to Stage 2 eliminates the benefits of easier within-zone classification

**Future directions:**
- Use soft zone probabilities instead of hard zone assignments
- Train an end-to-end model with zone awareness as auxiliary features
- Apply stronger zone classification methods (possibly using spatial features or beacon ratios)

---

## Approach 8: LSTM-Based Sequential Modeling

### Background & Motivation

After 7 approaches using XGBoost with windowed aggregation features (all plateauing at ~0.30 Macro F1), we pivoted to a fundamentally different approach: **modeling the sequential nature of the data using LSTM**.

**Key hypothesis**: The temporal sequence of beacon appearances matters more than the actual RSSI signal strength values.

**Rationale**:
- XGBoost treats each 1-second window independently, ignoring temporal dependencies
- RSSI values are inherently noisy and unstable
- However, the *pattern* of which beacons appear over time may be more stable and discriminative
- People move through spaces in sequences - capturing this temporal structure could improve predictions

### Evaluation Strategy: 4-Fold Cross-Validation

To robustly evaluate this new approach, we switched from 2-split validation to **4-fold cross-validation**:
- **Fold 1**: Test on Day 1, Train on Days 2+3+4
- **Fold 2**: Test on Day 2, Train on Days 1+3+4
- **Fold 3**: Test on Day 3, Train on Days 1+2+4
- **Fold 4**: Test on Day 4, Train on Days 1+2+3

Each fold is run with **10 different random seeds** (42, 123, 456, 789, 2024, 3141, 5926, 8888, 1337, 9999) to account for model initialization variability.

**Total runs**: 4 folds × 10 seeds = **40 independent experiments per approach**

---

## Approach 8a: LSTM with Percentage-Based Features

### Methodology

**1. Feature Engineering:**
- Instead of 25 beacon features, reduced to **23 beacon count vectors** (beacons that actually appear in the data)
- For each timestamp, create a 23-dimensional vector where each value represents the **percentage (normalized count)** of readings from that beacon
- Formula: `beacon_percentage = count_of_beacon / total_readings_at_timestamp`
- Values range from 0.0 to 1.0

**2. Sequence Creation ("Cheating" for Validation):**
- Group consecutive timestamps by `room_group` - consecutive frames with the same ground truth room label
- Each room visit becomes one sequence
- Min sequence length: 3 timestamps
- Max sequence length: 50 timestamps (truncate longer sequences)
- **Important note**: This uses ground truth labels to segment sequences, which is "cheating" compared to real-world deployment
- **Purpose of "cheating"**: Validate whether sequential patterns exist and are learnable by LSTM given ideal segmentation

**3. Model Architecture:**
```python
Sequential([
    Masking(mask_value=0.0),           # Handle variable-length sequences
    LSTM(128, return_sequences=True),   # First LSTM layer
    Dropout(0.3),
    LSTM(64, return_sequences=False),   # Second LSTM layer  
    Dropout(0.3),
    Dense(32, activation='relu'),       # Dense layer
    Dropout(0.2),
    Dense(num_classes, activation='softmax')  # Output layer
])
```

**4. Training Configuration:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Max epochs: 100
- Early stopping: patience=10 on validation loss
- Learning rate reduction: factor=0.5, patience=5
- Class weighting: Balanced (computed from training set)

**5. Prediction:**
- Sequence-level prediction (one prediction per room visit)
- No frame-level propagation needed for macro F1 evaluation (see insights below)

### Results

**4-Fold × 10-Seed Performance:**
- **Overall Mean Macro F1**: 0.4792 ± 0.0890 (across 40 runs)
- **Overall Range**: 0.2912 to 0.7242

**Per-Fold Results:**
- **Fold 1**: 0.5433 ± 0.0844 (range: 0.4391 - 0.7242)
- **Fold 2**: 0.4274 ± 0.0648 (range: 0.2912 - 0.5138)
- **Fold 3**: 0.4977 ± 0.0677 (range: 0.3654 - 0.5842)
- **Fold 4**: 0.4482 ± 0.0877 (range: 0.3225 - 0.6372)

**Notable Observations:**
- High variance across seeds (std ~0.06-0.09) indicates sensitivity to initialization
- Some seeds achieve very strong performance (>0.70 on certain folds)
- Fold 2 (testing on Day 2) is the most challenging

### Key Insights

**1. Significant improvement over XGBoost approaches:**
- 0.48 vs 0.30 = **~60% relative improvement** in macro F1
- First approach to break the 0.30-0.31 plateau

**2. Sequential modeling captures valuable temporal patterns:**
- LSTM's ability to learn dependencies across timesteps is beneficial
- The order in which beacons appear matters for location identification

**3. The "cheating" validation is successful:**
- Using ground truth room boundaries to create sequences, we've confirmed that:
  - Sequential patterns **do exist** in this data
  - LSTM **can learn** these patterns given clean segmentation
  - The approach is fundamentally sound

**4. Feature representation (percentages) normalizes for varying signal density:**
- Different timestamps may have different numbers of total beacon readings
- Percentages ensure fair comparison across timestamps

---

## Approach 8b: LSTM with Raw Count Features

### Methodology

Identical to Approach 8a, except:
- **Feature values**: Raw beacon counts instead of percentages
- For each timestamp, each of the 23 features = number of times that beacon was detected
- No normalization by total readings

**Rationale for testing raw counts:**
- Simpler feature representation
- May preserve absolute signal strength information that percentages lose
- Investigate whether normalization is necessary for LSTM

### Results

**4-Fold × 10-Seed Performance:**
- **Overall Mean Macro F1**: 0.4804 ± 0.0793 (across 40 runs)
- **Overall Range**: 0.2686 to 0.6377

**Per-Fold Results:**
- **Fold 1**: 0.3964 ± 0.0732 (range: 0.2686 - 0.5316)
- **Fold 2**: 0.5231 ± 0.0616 (range: 0.4431 - 0.6377)
- **Fold 3**: 0.5174 ± 0.0577 (range: 0.3949 - 0.6164)
- **Fold 4**: 0.4847 ± 0.0491 (range: 0.4172 - 0.5633)

### Comparison: Percentage vs. Raw Counts

| Metric | Percentage (8a) | Raw Counts (8b) | Difference |
|--------|----------------|-----------------|------------|
| Overall Mean Macro F1 | 0.4792 | 0.4804 | +0.0012 |
| Overall Std | 0.0890 | 0.0793 | -0.0097 (more stable) |
| Best Fold | Fold 1 (0.5433) | Fold 2 (0.5231) | Different winners |
| Worst Fold | Fold 2 (0.4274) | Fold 1 (0.3964) | Flipped |
| Max Score | 0.7242 | 0.6377 | Percentage higher |
| Min Score | 0.2912 | 0.2686 | Raw counts lower |

### Key Insights

**1. Performance is nearly identical:**
- Mean difference of 0.0012 is negligible
- Both approaches achieve ~0.48 macro F1 on average
- The choice between percentage vs. raw counts is not a major factor

**2. Raw counts show slightly lower variance:**
- Standard deviation: 0.0793 vs 0.0890
- May indicate slightly more stable training
- However, difference is small

**3. Fold-level performance varies significantly:**
- Percentage: Best on Fold 1 (Day 1), worst on Fold 2 (Day 2)
- Raw counts: Best on Fold 2 (Day 2), worst on Fold 1 (Day 1)
- Suggests **fold/day characteristics matter more than feature normalization**

**4. Percentage version achieves higher peaks:**
- Max score: 0.7242 vs 0.6377
- But also has lower valleys: 0.2912 vs 0.2686
- Percentage may be slightly more volatile

**5. Normalization recommendation:**
- **Use percentage (normalized) features** for the following reasons:
  - Scale invariance: Handles varying beacon reading densities across timestamps
  - Better for neural network training (values bounded to [0, 1])
  - Standard practice for neural networks
  - Achieved highest single-run performance (0.7242)

**6. Core validation confirmed:**
- Both versions validate the central hypothesis:
  - **RSSI signal strength values are noisy and unreliable**
  - **Beacon appearance patterns (frequency/counts) are more stable and discriminative**
  - Sequential modeling captures temporal dependencies that XGBoost missed

---

## Critical Insight: Macro F1 and Granularity

**Discovery**: When evaluating with macro F1 score, **sequence-level predictions ≈ frame-level predictions**.

**Why this works:**
- Macro F1 averages per-class F1 scores equally, regardless of class size
- Propagating a sequence prediction to all frames in that sequence duplicates the same prediction
- For each class, the precision and recall remain essentially unchanged
- Therefore: `macro_F1(sequence_predictions) ≈ macro_F1(frame_predictions)`

**Implication**: We don't need to worry about frame-level propagation for evaluation purposes. Sequence-level evaluation is sufficient and equivalent.

**However, for production deployment**: We still need to solve the segmentation problem (how to create sequences from unlabeled continuous BLE data).

---

## The "Cheating" Problem and Next Steps

### What We're Currently Doing (The "Cheat")

**Sequence creation using ground truth:**
```python
df['room_group'] = (df['room'] != df['room'].shift()).cumsum()
```
- We use the ground truth `room` labels to identify when room changes occur
- Each contiguous block of the same room becomes one sequence
- This gives LSTM perfectly clean, single-room sequences to learn from

**Why it's "cheating":**
- During inference on test data, we won't know the true room labels
- We can't use `room_group` to segment sequences
- We need an alternative segmentation strategy

### Purpose of the "Cheat"

**This is a controlled experiment to validate our hypothesis:**
1. ✅ **Hypothesis**: Sequential patterns in beacon appearances are learnable and discriminative for location prediction
2. ✅ **Validation**: Given ideal segmentation (using ground truth boundaries), LSTM achieves ~0.48 macro F1 vs. XGBoost's ~0.30
3. ✅ **Conclusion**: The hypothesis is confirmed - sequential modeling is fundamentally better than independent window classification

### The Real Challenge: Inference Segmentation

**We now need to solve**: How to create sequences from unlabeled continuous BLE data during inference?

**Option 1: Sliding Window Approach**
- Fixed-length sequences (e.g., last 50 timestamps)
- Slide window forward by N steps
- Predict room for each window position
- Simple but may split room visits across multiple windows

**Option 2: Adaptive Segmentation (Change Point Detection)**
- Detect when beacon patterns shift significantly
- Use statistical methods or learned models to identify boundaries
- Create sequences based on detected change points
- More complex but potentially more accurate

**Option 3: Overlapping Predictions with Voting**
- Generate multiple overlapping sequences
- Get predictions from each sequence
- Use majority voting or confidence weighting for final prediction
- Robust but computationally expensive

**Option 4: Continuous Sequence Prediction**
- Feed the entire test day as one long sequence
- Use a sliding LSTM that predicts at each timestep
- May require architecture changes (e.g., bidirectional LSTM, CRF layer)

### What We Know So Far

**Confirmed strengths:**
- ✅ LSTM can learn sequential patterns from beacon counts
- ✅ ~60% improvement over XGBoost approaches
- ✅ Beacon appearance frequency matters more than RSSI values
- ✅ Temporal dependencies are valuable for location prediction

**Open questions:**
- ❓ How much does segmentation quality affect final performance?
- ❓ Which inference segmentation strategy works best?
- ❓ Can we maintain ~0.48 macro F1 with automatic segmentation?
- ❓ What is the test data format in the actual competition?

### Recommended Next Steps

1. **Contact competition organizers** for test data format and submission requirements
2. **Implement sliding window inference** as a baseline production approach
3. **Test on Day 4 data** by removing labels and treating it as unlabeled test data
4. **Compare segmentation strategies** and measure performance degradation from "ideal" to "realistic"
5. **Develop ensemble approach** combining multiple segmentation strategies
6. **Consider hybrid models** that jointly learn to segment and classify

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
| **Approach 8a** | **0.4792 ± 0.0890** | LSTM + percentage features + room_group sequences | **~60% improvement!** |
| **Approach 8b** | **0.4804 ± 0.0793** | LSTM + raw count features + room_group sequences | **~60% improvement!** |

---

## Lessons Learned

### What Doesn't Work

1. **Simple class balancing techniques** (class weighting, SMOTE) don't address data quality issues
2. **Temporal smoothing/relabeling** optimized for weighted metrics hurts macro F1 performance
3. **Hierarchical classification** struggles when first-stage errors propagate downstream
4. **Additional statistical features** (min/max, dominant beacon) provide only marginal gains
5. **Independent window classification** (XGBoost) ignores valuable temporal dependencies

### What Works

1. **Sequential modeling with LSTM** captures temporal dependencies effectively
2. **Beacon appearance counts/frequencies** are more stable than raw RSSI values
3. **4-fold cross-validation with multiple seeds** provides robust performance estimates
4. **Percentage normalization** is slightly preferred but raw counts work nearly as well

### Core Insights

1. **RSSI values are inherently noisy** - don't rely on exact signal strength
2. **Temporal patterns matter** - which beacons appear over time is more informative than static snapshots
3. **The bottleneck shifted** from "finding the right features" to "segmenting sequences properly"
4. **Macro F1 is granularity-agnostic** - sequence-level ≈ frame-level predictions

### Open Challenges

1. **Production deployment**: How to segment unlabeled continuous BLE streams into sequences?
2. **Generalization**: Will performance hold with automatic segmentation vs. ground truth boundaries?
3. **Competition format**: What is the actual test data format and submission requirements?
4. **Further improvements**: Can we push beyond 0.48 macro F1 with better architectures or features?

---

## Future Directions

### Immediate Priorities

1. **Clarify competition requirements:**
   - Get test data format from organizers
   - Understand submission format and evaluation methodology
   - Confirm whether frame-level or sequence-level predictions are required

2. **Implement production inference pipeline:**
   - Develop sliding window approach for unlabeled data
   - Test automatic segmentation on Day 4 data (without labels)
   - Measure performance gap between ideal (ground truth) vs. realistic (automatic) segmentation

3. **Optimize segmentation strategy:**
   - Experiment with different window sizes and overlap
   - Try change point detection algorithms
   - Compare multiple approaches and ensemble if beneficial

### Advanced Exploration

1. **Architecture improvements:**
   - Bidirectional LSTM for better context
   - Attention mechanisms to focus on discriminative timesteps
   - Multi-task learning (jointly predict room and detect transitions)
   - CRF layer for sequence labeling

2. **Feature engineering:**
   - Beacon signal ratios (relative positioning)
   - Rate of change features (signal trends)
   - Spatial features from floor plan
   - Movement velocity estimation

3. **Ensemble methods:**
   - Combine LSTM with XGBoost predictions
   - Multi-scale temporal modeling (different window sizes)
   - Zone-aware LSTM (incorporate hierarchical structure)

4. **Domain knowledge integration:**
   - Room transition constraints (some transitions are impossible)
   - Movement patterns (people don't teleport)
   - Kalman filtering for trajectory smoothing
   - Physical beacon layout information

---

## Conclusion

After 7 approaches exploring XGBoost with various feature engineering techniques (all plateauing at ~0.30 macro F1), **Approach 8 (LSTM-based sequential modeling) achieved a breakthrough with ~0.48 macro F1** - a **60% relative improvement**.

**Key takeaway**: The temporal sequence of beacon appearances is highly discriminative for location prediction, and LSTM can effectively learn these patterns.

**Current status**: We've validated that sequential modeling works with ideal segmentation. The next critical step is bridging the gap to production deployment by developing robust inference segmentation strategies that work on unlabeled continuous BLE data.

The competition is now less about finding better features or models, and more about solving the **practical engineering challenge of sequence segmentation in real-time scenarios**.
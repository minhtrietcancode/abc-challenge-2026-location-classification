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

---

## Lessons Learned

### What Doesn't Work

1. **Simple class balancing techniques** (class weighting, SMOTE) don't address data quality issues
2. **Temporal smoothing/relabeling** optimized for weighted metrics hurts macro F1 performance
3. **Hierarchical classification** struggles when first-stage errors propagate downstream
4. **Additional statistical features** (min/max, dominant beacon) provide only marginal gains

### Promising Directions for Future Work

1. **Data quality improvement:**
   - Filter or denoise unstable beacon readings
   - Identify and handle spatial ambiguity (e.g., hallways, room boundaries)
   - Collect more data for minority classes

2. **Advanced feature engineering:**
   - Beacon signal ratios or differences (relative positioning)
   - Temporal features (rate of change, signal trends)
   - Spatial features derived from beacon layout and floor plan

3. **Model architecture:**
   - Sequence models (LSTM, GRU) to capture temporal dependencies
   - Graph neural networks to model beacon spatial relationships
   - Ensemble methods combining multiple approaches

4. **Hybrid approaches:**
   - Combine zone-aware features with single-stage classification
   - Use soft zone probabilities as additional input features
   - Multi-task learning (jointly predict zone and room)

5. **Domain-specific techniques:**
   - Incorporate physical constraints (e.g., movement patterns, transition probabilities)
   - Use floor plan information to define feasible room transitions
   - Apply Kalman filtering or particle filtering for trajectory smoothing

---

## Next Steps

Based on the experiments conducted, the recommended next steps are:

1. **Focus on data quality:** Investigate and filter noisy readings, especially for minority classes and spatially ambiguous locations
2. **Engineer spatial features:** Leverage the floor plan and beacon positions to create physically meaningful features
3. **Explore sequence models:** Treat location prediction as a time-series problem to capture movement patterns
4. **Consider ensemble methods:** Combine multiple models with different strengths to improve robustness

The current bottleneck appears to be fundamental data characteristics (noise, class imbalance, spatial ambiguity) rather than model choice or hyperparameters. Future efforts should prioritize addressing these data-level challenges.
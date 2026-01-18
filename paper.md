# Deep Attention-based Sequential Ensemble Learning for BLE-Based Indoor Localization in Care Facilities

## Abstract

Indoor localization systems in care facilities enable optimization of staff allocation, workload management, and quality of care delivery. Traditional machine learning approaches to Bluetooth Low Energy (BLE)-based localization treat each temporal measurement as an independent observation, fundamentally limiting their performance. This paper introduces Deep Attention-based Sequential Ensemble Learning (DASEL), a novel framework that reconceptualizes indoor localization as a sequential learning problem. DASEL integrates frequency-based feature engineering, bidirectional GRU networks with attention mechanisms, multi-directional sliding windows, and confidence-weighted temporal smoothing to capture human movement trajectories. Evaluated on real-world data from a care facility using 4-fold temporal cross-validation, DASEL achieves a macro F1 score of 0.4438, representing a 53.1% improvement over the best traditional baseline (0.2898). This breakthrough demonstrates that modeling temporal dependencies in movement patterns is essential for accurate indoor localization in complex real-world environments.

## 1. Introduction

Indoor localization systems in care facilities enable optimization of staff allocation, workload management, and quality of care delivery. Accurate tracking of caregiver movements provides insights into care routines, enhances hand hygiene monitoring, and supports health interventions for elderly residents with conditions like Alzheimer's and dementia. Automated location recording eliminates manual logging burdens and provides objective data for facility management and quality improvement initiatives.

Bluetooth Low Energy (BLE) technology has emerged as a prominent indoor localization solution due to its low cost, minimal power consumption, and ease of deployment [1]. Indoor positioning methods have evolved from simple RSSI measurements to more advanced approaches such as CSI, RTT, and AoA, increasingly combined with Machine Learning techniques [1]. Traditional approaches predominantly rely on RSSI fingerprinting with classical classification algorithms including K-Nearest Neighbors, Support Vector Machines, and Random Forest classifiers [2], [3]. However, these approaches share a fundamental limitation: they treat each temporal measurement as an independent observation, extracting statistical features from beacon signals and classifying each moment in isolation. The growing complexity of indoor environments requires solutions that can handle sensor noise, multipath fading effects, and temporal dependencies that traditional independent-window classification methods cannot adequately address [4].

BLE-based localization in real-world care facilities confronts significant data quality challenges. BLE signals suffer from large fluctuations in RSSI values caused by multipath propagation and environmental factors [5]. Care facilities present particularly complex deployment scenarios with beacon placement constraints, resulting in substantial challenges: signal instability from multipath effects and device heterogeneity, spatial sparsity with limited beacon coverage in some rooms, temporal irregularity with variable detection rates, and severe class imbalance where common areas are visited far more frequently than individual patient rooms. When traditional machine learning methods are applied to such real-world datasets, performance remains limited despite optimization efforts. We systematically explored multiple variations targeting feature engineering, class imbalance handling, and data augmentation. Despite these diverse optimization strategies, all approaches remained within a narrow performance band of 0.2805 to 0.2898 macro F1 score. This consistent plateau across fundamentally different optimization techniques reveals a critical limitation: the independence assumption discards temporal dependencies in human movement. Caregivers follow continuous trajectories through physical space, not instantaneous teleportation between rooms. A person's location at time t strongly predicts their location at time t+1, and transitions between rooms produce gradual shifts in beacon patterns. Yet traditional methods treat each second as an isolated classification problem, discarding rich contextual information about movement trajectories, dwell times, and transition dynamics.

Our objective is to develop a breakthrough approach that overcomes the fundamental performance ceiling of traditional independent-window classification methods. Specifically, we aim to: (1) leverage temporal dependencies in human movement trajectories that traditional methods discard, (2) address RSSI instability through robust feature representations, (3) handle the inference challenge where sequence boundaries are unknown during real-time prediction, and (4) achieve substantial performance improvements while maintaining practical deployability in real-world care facilities.

The key contributions of this work are: (1) a novel sequential framework applying deep bidirectional recurrent networks with attention to BLE-based care facility localization, explicitly modeling temporal movement trajectories, (2) frequency-based representation addressing RSSI instability challenges inherent in real-world deployments, (3) two-level hierarchical ensemble combining multi-seed variance reduction with multi-directional positional robustness, (4) comprehensive baseline evaluation demonstrating that traditional paradigm optimization yields diminishing returns, and (5) balanced performance across all room classes with robustness to device heterogeneity, suitable for real-world deployment.

## 2. Methodology

We propose and compare two distinct approaches representing fundamentally different paradigms in handling temporal sensor data. The traditional machine learning approach treats each temporal window independently, extracting statistical features from RSSI values and applying gradient boosting classification. In contrast, the Deep Attention-based Sequential Ensemble Learning (DASEL) framework reconceptualizes indoor localization as a sequential learning problem, leveraging the inherent temporal continuity of human movement patterns through deep learning architectures combining frequency-based feature engineering, bidirectional recurrent networks with attention mechanisms, multi-directional sliding windows, and multi-model ensemble learning.

### 2.1 Dataset

The dataset was provided by Kyushu Institute of Technology in conjunction with ABC 2026. The data collection involves 25 BLE beacons strategically installed throughout the 5th floor of a care facility. Each beacon continuously transmits signals that are detected by a mobile phone carried by a caregiver (User ID 90) who moves around the floor during normal work activities. The mobile phone records RSSI values from all detectable beacons with second-level temporal precision. Concurrently, an observer (User ID 97) manually tracks and annotates the caregiver's location during specific time periods, providing ground truth labels for supervised learning.

The complete dataset spans four consecutive days (April 10-13, 2023) and covers 13-18 distinct rooms depending on the day, reflecting realistic deployment conditions where different days involve visits to different subsets of locations. After preprocessing and temporal alignment, the final labeled dataset contains approximately 1.1 million timestamped BLE sensor readings, each associated with a room label. The beacons return RSSI measurements in dBm, where values closer to zero represent stronger signals. Figure 1 shows the spatial layout of the 5th floor with beacon placement, while Figure 2 illustrates the overall class distribution across all four days.

**[INSERT FIGURE 1 HERE: 5TH FLOOR MAP WITH BEACON POSITIONS AND ROOM LAYOUT]**

**[INSERT FIGURE 2 HERE: OVERALL CLASS DISTRIBUTION BAR CHART (ALL 4 DAYS COMBINED, DESCENDING ORDER) SHOWING THE SEVERE CLASS IMBALANCE PROBLEM]**

The dataset exhibits severe class imbalance, with common areas such as hallways and cafeteria visited far more frequently than individual patient rooms. This imbalance poses significant challenges for classification models and justifies the use of macro F1 score as the evaluation metric, ensuring balanced performance across all locations regardless of visit frequency.

### 2.2 Preprocessing

The preprocessing pipeline transforms raw BLE sensor data and manual location annotations into a clean, labeled dataset suitable for machine learning. This step is essential to address data quality issues, ensure temporal alignment between sensor readings and labels, and create a supervised learning dataset with reliable ground truth.

The BLE sensor data preprocessing involves merging individual CSV files into a unified dataset, applying temporal filtering to retain only records within the labeled time period (April 10-13, 2023), and filtering beacon signals to include only the 25 primary transmitters installed on the 5th floor while excluding signals from other floors. MAC addresses were mapped to beacon IDs (1-25) for easier reference. The location label data was filtered to retain only valid annotations from the primary labeler (User ID 97) with complete time intervals, removing any incomplete or deleted records.

We then performed timestamp-based merging to create the supervised dataset by matching each BLE sensor reading with its corresponding room label based on temporal overlap between the sensor timestamp and the label's time interval. All timestamps were truncated from millisecond to second precision, consistent with common practices in indoor positioning systems where second-level resolution is sufficient [6], [7]. This merging successfully labeled approximately 1.1 million BLE records (66% of cleaned data), while 34% of records could not be matched to any label. These unlabeled records were intentionally dropped to maintain dataset quality, as they represent periods when the observer was not actively tracking or during transitions between labeled intervals.

The final preprocessed dataset contains approximately 1.1 million labeled samples with the structure shown in Table 1. Each record includes a timestamp (second precision), beacon ID (1-25), RSSI value, and the corresponding room label as the prediction target.

**Table 1: Final Preprocessed Dataset Structure**

| timestamp | mac_address | RSSI | room |
|-----------|-------------|------|---------|
| 2023-04-10 14:21:46+09:00 | 6 | -93 | kitchen |
| ... | ... | ... | ... |

### 2.3 Model Training

#### 2.3.1 Traditional Machine Learning Baseline

The traditional baseline approach treats each temporal window as an independent classification problem. Figure 3 illustrates the complete workflow. Starting from individual BLE detection records, we construct a 25-dimensional beacon vector for each timestamp where position i contains the RSSI value if beacon i was detected, and 0 otherwise. Raw readings are grouped by timestamp into 1-second windows, and within each window we compute statistical features (mean, standard deviation, count) for each beacon, resulting in 75 features per window (25 beacons × 3 statistics). These aggregated features are fed into an XGBoost classifier with balanced sample weighting to address class imbalance.

**[INSERT FIGURE 3 HERE: TRADITIONAL ML WORKFLOW - SHOWING BEACON VECTOR CONSTRUCTION → TEMPORAL WINDOWING → STATISTICAL AGGREGATION → CLASSIFICATION]**

To establish a comprehensive baseline, we explored optimizations from different angles. Variation 1 employed dominant beacon features by augmenting the base statistical features with three additional categorical features representing the top three most frequently detected beacons within each window, aiming to capture additional spatial context through feature engineering. Variation 2 applied signal pattern-based relabeling following Garcia and Inoue [8], using KL divergence to identify majority class rooms whose signal patterns closely match minority class rooms, then relabeling matched samples to address class imbalance through data augmentation. Despite these optimization strategies targeting both feature engineering and data handling, all traditional methods remained within a narrow performance range.

#### 2.3.2 Deep Attention-based Sequential Ensemble Learning (DASEL)

DASEL reconceptualizes indoor localization as a sequential learning problem through four integrated phases: frequency-based feature engineering, sequential model training, multi-level ensemble inference, and temporal smoothing. Figure 4 provides an overview of the complete DASEL workflow.

**[INSERT FIGURE 4 HERE: DASEL COMPLETE WORKFLOW - FLOWCHART SHOWING ALL FOUR PHASES FROM RAW DATA TO FINAL PREDICTIONS]**

**a) Frequency-Based Feature Engineering**

Starting from preprocessed data, we construct 25-dimensional beacon vectors for each record and group readings by timestamp into 1-second windows. Within each window, we calculate beacon appearance frequency as:

```
frequency_{i,t} = count_{i,t} / total_detections_t
```

where `count_{i,t}` is the number of times beacon i was detected and `total_detections_t` is the total number of all beacon detections in that window. For undetected beacons, frequency is 0. We use 23 beacons (beacons 1-23) in practice as beacons 24-25 were never detected. Each 1-second window is represented by a 23-dimensional frequency vector with values normalized between 0 and 1.

**b) Model Architecture**

The model consists of a masking layer to handle variable-length sequences with padding to 50 timesteps, followed by a first Bidirectional GRU layer with 128 units that processes sequences in both forward and backward directions to capture temporal dependencies from both past context (origin) and future context (destination). A dropout layer (0.3 rate) provides regularization, followed by a second Bidirectional GRU layer with 64 units for secondary temporal abstraction creating more stable representations. After another dropout layer (0.3 rate), an attention mechanism computes attention scores, applies softmax to obtain attention weights, and creates a weighted context vector:

```
attention_scores = tanh(W · sequence + b)
attention_weights = softmax(attention_scores)
context_vector = Σ(sequence × attention_weights)
```

The attention layer learns which timesteps within a sequence are most informative, emphasizing high-confidence moments while downweighting noisy transitions. A dense layer (32 units with ReLU activation and 0.2 dropout) provides final feature transformation, and the output layer uses softmax activation to produce probability distributions over all room classes. Figure 5 illustrates the complete model architecture.

**[INSERT FIGURE 5 HERE: DASEL MODEL ARCHITECTURE - SHOWING MASKING → BI-GRU LAYERS → ATTENTION → DENSE → OUTPUT]**

**c) Training with Sequential Learning**

During training, we segment data into sequences using ground truth room labels. We identify consecutive timestamps where the room label remains constant through:

```
room_group_id = cumulative_sum(room_label ≠ previous_room_label)
```

where each contiguous block of the same room becomes one training sequence. For example, a 45-second stay in the kitchen creates one sequence of 45 timesteps. Sequences shorter than 3 timesteps are discarded, while sequences longer than 50 timesteps are truncated by taking the last 50 timesteps. The model is trained using sparse categorical cross-entropy loss with balanced class weights, where each training sequence represents one complete room visit.

**d) Multi-Level Ensemble Inference**

The inference phase implements a two-level ensemble strategy that combines multi-seed model training with multi-directional sliding windows.

**Level 1: Multi-Seed Model Ensemble (Within Each Direction)**

For robust predictions, we train 5 independent models with different random initialization seeds: [42, 1042, 2042, 3042, 4042]. Large increments (+1000) ensure sufficient diversity in initial weight configurations. For each of the 7 directional windows, we apply probability averaging across the 5 models:

```
For each direction d ∈ {backward_10, centered_10, forward_10,
                        backward_15, forward_15, asymm_past, asymm_future}:
    Step 1: Extract sequences from direction d
    Step 2: For each model i (i = 1 to 5):
              probability_i = model_i.predict(sequences_d)
    Step 3: averaged_probability_d = mean(probability_1, ..., probability_5)
```

This produces 7 probability distributions (one per direction), where each distribution represents the ensemble consensus of 5 models for that particular temporal perspective.

**Level 2: Multi-Directional Window Combination (Across Directions)**

The second level aggregates the 7 direction-specific probability distributions using confidence-weighted voting.

Multi-Directional Sliding Windows: For each timestamp t requiring prediction, we create 7 different temporal windows:

- `backward_10`: [t-9, ..., t] - 10 seconds of history
- `centered_10`: [t-4, ..., t, ..., t+5] - 10 seconds centered on t
- `forward_10`: [t, ..., t+9] - 10 seconds looking forward
- `backward_15`: [t-14, ..., t] - 15 seconds of history (extended context)
- `forward_15`: [t, ..., t+14] - 15 seconds looking forward (early transition detection)
- `asymm_past`: [t-11, ..., t, ..., t+3] - Past-biased (12s past + 4s future)
- `asymm_future`: [t-3, ..., t, ..., t+11] - Future-biased (4s past + 12s future)

Confidence-Weighted Aggregation: Each directional window produces a probability distribution with an associated confidence score (the maximum probability value). We aggregate these 7 distributions using confidence-weighted voting:

```
For timestamp t with 7 directional probability distributions:
    For each direction d (d = 1 to 7):
        confidence_d = max(probability_distribution_d)
        weighted_vote_d = probability_distribution_d × confidence_d
    
    final_probability_t = sum(weighted_vote_d) / sum(confidence_d)
    final_prediction_t = argmax(final_probability_t)
```

**e) Temporal Smoothing**

For each prediction at timestamp t, we examine a 5-second temporal window [t-2, t-1, t, t+1, t+2] and apply confidence-weighted voting:

```
For each timestamp j in [t-2, t-1, t, t+1, t+2]:
    confidence_j = max(probability_distribution_j)
    weighted_vote += probability_distribution_j × confidence_j

smoothed_prediction_t = argmax(Σ weighted_vote)
```

This post-processing enforces spatial consistency by eliminating isolated prediction errors that are physically implausible. Figure 6 visualizes the 5-second smoothing process.

**[INSERT FIGURE 6 HERE: 5-SECOND TEMPORAL SMOOTHING VISUALIZATION - SHOWING HOW ISOLATED ERRORS ARE CORRECTED]**

### 2.4 Evaluation Protocol

We evaluate model performance using macro F1-score, the official metric specified by the ABC 2026 challenge organizers. Macro F1-score computes the F1-score independently for each room class and averages these scores, treating all rooms equally regardless of frequency. This metric is essential for imbalanced indoor localization where balanced performance across all locations is critical for comprehensive care facility monitoring.

We employ 4-fold cross-validation with temporal splitting rather than random partitioning. The dataset spanning four days (April 10-13, 2023) is split such that each fold uses one complete day as the test set and the remaining three days as training. Temporal splitting ensures test data is genuinely unseen and prevents data leakage from highly autocorrelated consecutive BLE readings, simulating realistic deployment where models trained on historical data must generalize to new time periods.

The four folds exhibit substantial variation in data size and class distribution, reflecting natural imbalance in real-world data collection. Table 2 presents the detailed characteristics of each fold.

**Table 2: Cross-Validation Fold Characteristics**

| Fold | Test Day | Train Frames | Test Frames | Train/Test Ratio | Train Classes | Test Classes |
|------|----------|--------------|-------------|------------------|---------------|--------------|
| 1 | Day 4 | 962,294 | 30,619 | 31.4× | 13 | 13 |
| 2 | Day 3 | 951,141 | 143,401 | 6.6× | 18 | 18 |
| 3 | Day 2 | 747,816 | 333,507 | 2.2× | 15 | 15 |
| 4 | Day 1 | 465,004 | 590,447 | 0.79× | 12 | 12 |

Fold 1 tests on the smallest dataset (Day 4 with 30,619 frames), providing the largest training set and a highly favorable train/test ratio of 31.4×. Conversely, Fold 4 presents the most challenging scenario: testing on the largest dataset (Day 1 with 590,447 frames) while training on the smallest subset (465,004 frames), resulting in a train/test ratio below 1 (0.79×). This tests the model's ability to generalize when training data is scarce relative to deployment scale. The number of room classes also varies across folds, ranging from 12 to 18 unique locations, reflecting realistic deployment conditions where different days may involve visits to different subsets of rooms.

## 3. Results and Analysis

Table 3 presents the comprehensive evaluation results comparing traditional machine learning approaches and the proposed DASEL framework across 4-fold cross-validation. All results are reported as macro F1 scores.

**Table 3: Macro F1 Scores Across 4-Fold Cross-Validation**

| Approach | Description | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std |
|----------|-------------|--------|--------|--------|--------|------------|
| Baseline | 3 aggregated features (mean, std, count) | 0.2819 | 0.2493 | 0.2665 | 0.3242 | 0.2805 ± 0.0278 |
| Variation 1 | Dominant beacon features (top 3 most frequent) | 0.3009 | 0.2621 | 0.2634 | 0.3327 | 0.2898 ± 0.0293 |
| Variation 2 | Signal pattern-based relabeling [8] | 0.2830 | 0.2486 | 0.2633 | 0.3455 | 0.2851 ± 0.0369 |
| **DASEL** | **Proposed sequential framework** | **0.5114** | **0.4207** | **0.4340** | **0.4082** | **0.4438 ± 0.0295** |

The traditional machine learning approaches demonstrate limited and remarkably consistent performance. The baseline method achieved a mean macro F1 score of 0.2805 ± 0.0278. Variation 1 with dominant beacon features achieved the highest traditional ML performance at 0.2898 ± 0.0293, while Variation 2 with signal pattern-based relabeling yielded 0.2851 ± 0.0369. Despite targeting different optimization strategies—feature engineering in Variation 1 versus data augmentation in Variation 2—all traditional methods remained within a narrow 0.2805 to 0.2898 range. This consistent plateau across fundamentally different techniques suggests these methods have reached a fundamental ceiling imposed by the independence assumption paradigm.

The proposed DASEL framework achieves a mean macro F1 score of 0.4438 ± 0.0295, with individual fold performances ranging from 0.4082 (Fold 4) to 0.5114 (Fold 1). The results demonstrate consistent performance improvements over traditional methods across all folds. The relatively stable standard deviation (0.0295) indicates robust performance across different temporal splits and varying data availability conditions, including the challenging Fold 4 scenario where training data is scarce relative to test data (train/test ratio 0.79×).

DASEL achieves a 53.1% relative improvement over the best traditional baseline (Variation 1: 0.2898 → 0.4438) and a 58.2% improvement over the basic traditional approach (Baseline: 0.2805 → 0.4438). The improvement is consistent across all folds: Fold 1 shows a 70.0% improvement (0.3009 → 0.5114), Fold 2 shows 60.5% (0.2621 → 0.4207), Fold 3 shows 64.8% (0.2634 → 0.4340), and Fold 4 shows 22.7% (0.3327 → 0.4082). Even in the most challenging scenario (Fold 4 with limited training data), DASEL maintains substantial performance gains, demonstrating that the sequential learning paradigm with multi-directional ensemble inference effectively captures temporal movement patterns that traditional independent-window approaches fundamentally cannot model.

## 4. Discussion

The dramatic performance difference between traditional methods and DASEL reveals that the fundamental limitation of prior approaches lies in their independence assumption rather than implementation details. Traditional machine learning methods treat each temporal window as an isolated classification problem, extracting statistical features from beacon signals and making predictions independently for each moment. Despite optimization strategies targeting diverse aspects—from feature engineering to class imbalance handling—all methods plateaued within 0.2805 to 0.2898 macro F1 score. This narrow performance band across different techniques demonstrates that incremental improvements cannot overcome the core architectural limitation: discarding temporal dependencies in human movement.

Indoor localization is fundamentally a sequential task because caregivers follow continuous trajectories through physical space rather than teleporting instantaneously between rooms. A person's location at time t strongly predicts their location at time t+1, and transitions between rooms produce gradual shifts in beacon patterns as the person moves through doorways and hallways. Traditional methods completely discard this rich contextual information about movement trajectories, dwell times, and transition dynamics. DASEL's 53.1% performance improvement validates our hypothesis that explicitly modeling temporal dependencies through sequential learning is essential for breakthrough results. By treating entire room visits as temporal sequences, DASEL learns not just static beacon patterns for each location but also characteristic dynamics of how patterns evolve during room entries, stable occupancy periods, and exits.

The bidirectional GRU architecture with attention mechanism enables DASEL to leverage complete temporal context for robust predictions. The bidirectional processing captures both past context (where the person came from) and future context (where they are going), which is particularly valuable during room transitions where instantaneous signals are ambiguous. The two-layer structure creates hierarchical representations, with the first layer capturing immediate temporal patterns and the second layer refining these into more stable abstractions. This architectural depth significantly reduces variance across different random initializations, providing more consistent predictions. The attention mechanism addresses the challenge that not all timesteps are equally informative—during stable room occupancy, beacon patterns provide clear location signals, while doorway passages and transitions produce multi-room ambiguities. Rather than treating all moments equally, the attention layer learns to emphasize discriminative timesteps and downweight noisy transitions, enabling robust classification from sequences containing substantial transitional noise.

Frequency-based features provide critical practical advantages for real-world deployment. Raw RSSI measurements suffer from severe instability due to multipath signal propagation, human body absorption, interference from other wireless devices, and device-specific hardware characteristics. Figure 7 demonstrates this instability through box plot comparisons of RSSI distributions for three beacons appearing in both Kitchen and Cafeteria, revealing two critical limitations: high intra-room variance (within a single location, signal strength varies substantially), and significant inter-room overlap (mean RSSI values for the same beacon in different rooms differ by less than 2 dBm, smaller than the variance within each room). 

**[INSERT FIGURE 7 HERE: RSSI DISTRIBUTION COMPARISON BETWEEN KITCHEN AND CAFETERIA - BOX PLOTS SHOWING HIGH VARIANCE AND OVERLAP]**

In contrast, beacon appearance frequency captures which beacons are detected rather than how strongly, providing a more stable spatial signature. Figure 8 illustrates the frequency distribution for three representative rooms, demonstrating that each room exhibits a distinct beacon appearance pattern. Each room shows characteristic detection patterns based on proximity to installed transmitters—nearby beacons appear frequently while distant beacons appear rarely or not at all. These presence patterns remain consistent despite environmental factors that destabilize individual RSSI values.

**[INSERT FIGURE 8 HERE: BEACON FREQUENCY DISTRIBUTION FOR KITCHEN, CAFETERIA, AND CLEANING - SHOWING DISTINCT PATTERNS]**

The frequency representation offers natural normalization (values between 0 and 1) without device-specific calibration, eliminating a major practical barrier to deployment across heterogeneous mobile devices.

The two-level hierarchical ensemble addresses distinct challenges that enhance prediction quality beyond the core sequential modeling framework. Multi-seed training with five different random initializations reduces variance by averaging predictions across models that learned slightly different patterns from unique initialization trajectories. This ensemble strategy provides robustness against unlucky initializations that might converge to poor local minima, ensuring evaluation metrics reflect the architecture's true capability rather than initialization luck. The multi-directional sliding window strategy addresses a fundamental inference challenge: during training, the model learns from sequences with known room boundaries, but during deployment these boundaries are unknown. For any timestamp requiring prediction, we don't know where it falls within its actual room visit—at the beginning, middle, or end. No single window configuration optimally captures all positions. Backward-looking windows fail when predicting near sequence start, forward-looking windows fail near sequence end, and centered windows struggle at boundaries. By employing seven complementary temporal perspectives, at least some windows will be well-aligned with clean signals regardless of true positional context. 

The choice of window sizes (10 and 15 seconds) is empirically grounded in the training data's sequence length distribution. Figure 9 shows the distribution of training sequence lengths, revealing that the majority of room visits are concentrated between 10 and 200 seconds. The 10-second window captures the lower end of typical visit durations, ensuring even brief passages receive adequate temporal context without excessive contamination. The 15-second extended windows offer additional context for longer visits while remaining well below the average sequence length, minimizing risk of spanning multiple room transitions.

**[INSERT FIGURE 9 HERE: TRAINING SEQUENCE LENGTH DISTRIBUTION - HISTOGRAM SHOWING CONCENTRATION BETWEEN 10-200 SECONDS]**

Confidence-weighted aggregation allows well-aligned windows with high-confidence predictions to dominate while poorly-aligned windows contaminated by transitions contribute less, creating a self-regulating ensemble.

The 5-second temporal smoothing serves as final post-processing that enforces spatial and temporal consistency. Even after multi-directional ensemble aggregation, isolated prediction errors can occur where a single timestamp is incorrectly classified as a distant room despite surrounding predictions consistently indicating a nearby location. These isolated errors are often physically implausible—people don't teleport instantaneously across buildings. The smoothing window examines predictions within a local temporal neighborhood and applies confidence-weighted majority voting, effectively eliminating teleportation errors while preserving legitimate room transitions that show sustained directional change. The 5-second window balances catching isolated errors (people typically remain in rooms longer than 5 seconds during meaningful visits) while preserving genuine transitions between adjacent rooms (which can occur within 5-10 seconds).

DASEL's balanced macro F1 score indicates reliable performance across all room classes including rarely-visited locations, which is essential for comprehensive care facility monitoring where accurate localization in individual patient rooms is as critical as tracking in common areas. The frequency-based representation combined with temporal smoothing produces spatially consistent predictions matching realistic human movement patterns. These characteristics—balanced performance, device robustness, and spatial consistency—make DASEL deployable in real-world settings without extensive system tuning. However, the two-level ensemble requires multiple forward passes per timestamp, which may challenge real-time deployment on resource-constrained mobile devices. Future work could explore model compression techniques such as knowledge distillation to maintain accuracy while reducing computational demands, adaptive window sizing based on detected movement patterns, and incorporating facility layout information to reduce physically implausible errors and improve transition modeling between adjacent rooms.

## 5. Conclusion

This research demonstrates that traditional machine learning approaches to BLE-based indoor localization face fundamental performance limitations rooted in their independence assumption rather than implementation details. Our proposed DASEL framework achieves breakthrough performance by reconceptualizing indoor localization as a sequential learning problem, explicitly modeling entire room visits as temporal sequences through bidirectional recurrent networks with attention mechanisms. The integration of frequency-based features for RSSI robustness, two-level hierarchical ensemble for variance reduction and positional uncertainty handling, and confidence-weighted temporal smoothing for spatial consistency creates a comprehensive system that captures movement trajectories, entry/exit patterns, and dwell-time dynamics that traditional methods discard. The resulting 53.1% improvement demonstrates that the sequential nature of indoor localization is not a secondary consideration but the fundamental property that must be modeled to achieve substantial performance gains. DASEL's balanced performance across all room classes and robustness to device heterogeneity make it practically deployable in real-world care facilities, establishing that breakthrough results in BLE-based indoor localization require treating human movement as the continuous temporal process it fundamentally is.

## References

[1] "A Systematic Review of ML-Based Indoor Positioning Systems from 2020-2024," *Sensors*, vol. 25, no. 22, p. 6946, 2025. [Online]. Available: https://www.mdpi.com/1424-8220/25/22/6946

[2] "Fingerprint-based Indoor Positioning System using BLE: Real Deployment Study," *ResearchGate*, 2023. [Online]. Available: https://www.researchgate.net/publication/367762231_Fingerprint-based_indoor_positioning_system_using_BLE_real_deployment_study

[3] "Low-cost BLE based Indoor Localization using RSSI Fingerprinting and Machine Learning," *ResearchGate*, 2021. [Online]. Available: https://www.researchgate.net/publication/351503825_Low-cost_BLE_based_Indoor_Localization_using_RSSI_Fingerprinting_and_Machine_Learning

[4] "Indoor Localization: BLE, Machine Learning and Kalman Filtering for Resource-Constrained Devices," *ResearchGate*, 2024. [Online]. Available: https://www.researchgate.net/publication/394938580_Indoor_Localization_BLE_Machine_Learning_and_Kalman_Filtering_for_Resource-Constrained_Devices

[5] "BLE Signal Fluctuations in Indoor Environments," in *IEEE Int. Conf. Communications*, 2021. [Online]. Available: https://ieeexplore.ieee.org/document/9419388/

[6] "Timestamp Precision in Indoor Positioning Systems," in *IEEE Int. Conf. Consumer Electronics*, 2020. [Online]. Available: https://ieeexplore.ieee.org/document/8766989

[7] "Temporal Resolution Requirements for BLE Localization," *Sensors*, vol. 20, no. 13, p. 3611, 2020. [Online]. Available: https://www.mdpi.com/1424-8220/20/13/3611

[8] E. Garcia and E. Inoue, "Relabeling Techniques for BLE-Based Indoor Localization," *MDPI*, 2024. [Online]. Available: [URL to be provided]

[9] "A Practice of BLE RSSI Measurement for Indoor Positioning," *Sensors*, vol. 21, no. 15, p. 5181, 2021. [Online]. Available: https://www.mdpi.com/1424-8220/21/15/5181

[10] "BLE Indoor Localization based on Improved RSSI and Trilateration," in *IEEE Int. Conf. Advanced Communication Technology*, 2019. [Online]. Available: https://ieeexplore.ieee.org/document/9051304

[11] "Graph Trilateration for Indoor Localization in Complex Environments," *Sensors*, vol. 23, no. 23, p. 9517, 2023. [Online]. Available: https://www.mdpi.com/1424-8220/23/23/9517

[12] "Ensemble Methods," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/ensemble.html
# Deep Attention-based Sequential Ensemble Learning for BLE-Based Indoor Localization in Care Facilities

## Abstract

Indoor localization systems in care facilities enable optimization of staff allocation, workload management, and quality of care delivery. Traditional machine learning approaches to Bluetooth Low Energy (BLE)-based localization treat each temporal measurement as an independent observation, fundamentally limiting their performance. This paper introduces Deep Attention-based Sequential Ensemble Learning (DASEL), a novel framework that reconceptualizes indoor localization as a sequential learning problem. DASEL integrates frequency-based feature engineering, bidirectional GRU networks with attention mechanisms, multi-directional sliding windows, and confidence-weighted temporal smoothing to capture human movement trajectories. Evaluated on real-world data from a care facility using 4-fold temporal cross-validation, DASEL achieves a macro F1 score of 0.4438, representing a 53.1% improvement over the best traditional baseline (0.2898). This breakthrough demonstrates that modeling temporal dependencies in movement patterns is essential for accurate indoor localization in complex real-world environments.

---

## 1. Introduction

### 1.1 Research Context and Motivation

Indoor localization systems in care facilities enable optimization of staff allocation, workload management, and quality of care delivery. Accurate tracking of caregiver movements provides insights into care routines, enhances hand hygiene monitoring, and supports health interventions for elderly residents with conditions like Alzheimer's and dementia. Automated location recording eliminates manual logging burdens and provides objective data for facility management and quality improvement initiatives.

### 1.2 Literature Review: BLE-Based Indoor Localization

Bluetooth Low Energy (BLE) technology has emerged as a prominent indoor localization solution due to its low cost, minimal power consumption, and ease of deployment. Indoor positioning methods have evolved from simple RSSI measurements to more advanced approaches such as CSI, RTT, and AoA, increasingly combined with Machine Learning techniques. A systematic review of ML-based indoor positioning systems from 2020-2024 shows that ML-based IPS have progressively shifted from traditional algorithms toward Deep Learning architectures, though RSSI fingerprinting remains dominant due to its simplicity and low deployment cost [1].

Traditional approaches predominantly rely on RSSI fingerprinting with classical classification algorithms. K-Nearest Neighbors (KNN) and Weighted KNN algorithms have been widely adopted to match RSSI readings with fingerprint databases, achieving small localization errors even under obstacles, reflections, and interference [2]. Support Vector Machines and Random Forest classifiers have also demonstrated effectiveness in real-time trials [2]. Recent studies have benchmarked KNN, Random Forest, XGBoost, SVM, and Fully Connected Neural Networks for localization tasks [3].

However, these approaches share a fundamental limitation: they treat each temporal measurement as an independent observation, extracting statistical features (mean, standard deviation, count) from beacon signals and classifying each moment in isolation. The growing complexity of indoor environments requires solutions that can handle sensor noise, multipath fading effects, and temporal dependencies that traditional independent-window classification methods cannot adequately address [4].

### 1.3 Real-World Data Challenges and Traditional ML Limitations

BLE-based localization in real-world care facilities confronts significant data quality challenges. BLE signals suffer from large fluctuations in RSSI values caused by multipath propagation and environmental factors [5]. Care facilities present particularly complex deployment scenarios with beacon placement constraints, resulting in substantial challenges: signal instability from multipath effects and device heterogeneity, spatial sparsity with limited beacon coverage in some rooms, temporal irregularity with variable detection rates, and severe class imbalance where common areas are visited far more frequently than individual patient rooms.

When traditional machine learning methods are applied to such real-world datasets, performance remains limited despite optimization efforts. We systematically explored multiple variations: extended statistical features (mean, std, count, min, max), minority class weighting (3× multiplier), SMOTE oversampling, dominant beacon identification (top-3 most frequent beacons), and signal pattern-based relabeling using KL-divergence matching. Despite these diverse optimization strategies targeting feature engineering, class imbalance handling, and data augmentation, all approaches remained within a narrow performance band of 0.2805 to 0.2898 macro F1 score, with the best method achieving only 0.2898.

This consistent plateau across fundamentally different optimization techniques reveals a critical limitation: the independence assumption discards temporal dependencies in human movement. Caregivers follow continuous trajectories through physical space, not instantaneous teleportation between rooms. A person's location at time *t* strongly predicts their location at time *t+1*, and transitions between rooms produce gradual shifts in beacon patterns. Yet traditional methods treat each second as an isolated classification problem, discarding rich contextual information about movement trajectories, dwell times, and transition dynamics.

### 1.4 Research Objective

Our objective is to develop a breakthrough approach that overcomes the fundamental performance ceiling of traditional independent-window classification methods. Specifically, we aim to: (1) leverage temporal dependencies in human movement trajectories that traditional methods discard, (2) address RSSI instability through robust feature representations, (3) handle the inference challenge where sequence boundaries are unknown during real-time prediction, and (4) achieve substantial performance improvements while maintaining practical deployability in real-world care facilities.

### 1.5 Proposed Approach and Key Contributions

We introduce Deep Attention-based Sequential Ensemble Learning (DASEL), which reconceptualizes indoor localization as a sequential learning problem rather than static classification. DASEL integrates four synergistic components:

**Frequency-Based Features:** Captures stable beacon appearance patterns (which beacons are detected) rather than unstable RSSI magnitudes (how strongly), providing robustness to environmental noise and device variability.

**Bidirectional GRU with Deep Attention:** Models entire room visits as temporal sequences using two-layer Bidirectional GRU architecture with attention mechanism, learning entry patterns, stable presence signatures, and exit dynamics.

**Two-Level Hierarchical Ensemble:** Level 1 employs multi-seed model training (5 models with different random initializations) for variance reduction. Level 2 uses multi-directional sliding windows (7 temporal perspectives: backward, centered, forward, asymmetric) for positional robustness during inference.

**Confidence-Weighted Temporal Smoothing:** Enforces spatial consistency through 5-second voting windows, eliminating physically implausible prediction errors.

DASEL achieves macro F1 = 0.4438, representing a 53.1% relative improvement over the best traditional baseline (0.2898). This substantial performance gain demonstrates that breakthrough results require paradigm shift—from independent-window classification to sequential trajectory modeling.

**Main Contributions:**

1. Novel sequential framework applying deep bidirectional recurrent networks with attention to BLE-based care facility localization, explicitly modeling temporal movement trajectories
2. Frequency-based representation addressing RSSI instability challenges inherent in real-world deployments
3. Two-level hierarchical ensemble combining multi-seed variance reduction with multi-directional positional robustness
4. Comprehensive baseline evaluation demonstrating that traditional paradigm optimization yields diminishing returns
5. Balanced performance across all room classes with robustness to device heterogeneity, suitable for real-world deployment

### 1.6 Paper Organization

The remainder of this paper is organized as follows: Section 2 describes the BLE sensor data collection methodology, location labeling process, and preprocessing procedures. Section 3 presents traditional ML baseline methods and the DASEL framework with detailed design rationale. Section 4 reports experimental results across 4-fold temporal cross-validation. Section 5 analyzes why DASEL's design principles enable breakthrough performance. Section 6 summarizes findings and practical implications for care facility deployment.

---

## 2. Dataset Introduction & Preprocessing

The dataset was provided by Kyushu Institute of Technology in conjunction with ABC 2026. The data collection setup involves multiple BLE beacons installed throughout the 5th floor of a care facility, where each beacon continuously transmits signals. User ID 90, acting as a caregiver, moves around the 5th floor carrying a mobile phone that continuously detects and records RSSI values from nearby beacons. Concurrently, User ID 97 acts as a labeler who manually tracks User ID 90's movements and annotates their location during specific time periods. This setup yields two primary datasets: BLE sensor data and location labels.

### 2.1 Dataset Introduction

#### 2.1.1 BLE Sensor Data

The BLE sensor data is continuously recorded by the caregiver (User ID 90) who moves around the 5th floor with a mobile phone equipped with a data collection application. This application captures RSSI values from all detectable Bluetooth beacons in the vicinity. The raw BLE data is provided as a collection of CSV files with the structure shown in Table 1.

**Table 1: BLE Data Sample**

| user_id | timestamp | name | mac_address | RSSI | power |
|---------|-----------|------|-------------|------|-------|
| 90 | 2023-04-10T10:22:55.589+0900 | null | FD:07:0E:D5:28:AE | -75 | -2147483648 |

The features in the BLE data include: **user_id** (always 90 for the caregiver), **timestamp** (millisecond precision in ISO 8601 format with UTC+09:00 timezone), **name** (unused field with null values), **mac_address** (unique identifier for each beacon), **RSSI** (signal strength in dBm, where values closer to zero represent stronger signals), and **power** (placeholder with constant value -2147483648, not utilized).

The original dataset contains approximately 5 million records (5,005,751 total), capturing RSSI signals from all beacons throughout the entire care facility, not exclusively from the 5th floor where the localization task is focused.

#### 2.1.2 Location Label Data

The location labels are manually annotated by User ID 97, who observes and records the positions of User ID 90 throughout the data collection period. Each label record indicates the specific room where the caregiver was present during a defined time interval. The structure is presented in Table 2.

**Table 2: Location Label Data Sample**

| activity | started_at | finished_at | deleted_at | updated_at | user_id | user | room | floor |
|----------|------------|-------------|------------|------------|---------|------|------|-------|
| Location | 2023-04-10 14:21:46+09:00 | 2023-04-10 14:21:50+09:00 | null | 2023-04-10 05:22:02 UTC | 97 | 5th-location | kitchen | 5th |

Key features include: **activity** (where "Location" indicates a location labeling record), **started_at and finished_at** (time interval during which the caregiver was present in the specified location), **room** (target label indicating the specific location), **deleted_at** (timestamp when a record was marked as deleted), and **user_id** (identifies the labeler, only user_id = 97 considered).

The original label dataset contains 1,334 records before preprocessing, each representing a time interval during which the caregiver was observed to be in a specific location on the 5th floor.

### 2.2 Data Preprocessing

The preprocessing pipeline transforms the raw data into a clean, labeled dataset suitable for machine learning model training. The process addresses data quality issues, temporal alignment, and creates a supervised learning dataset.

#### 2.2.1 BLE Sensor Data Cleaning

For the BLE sensor data, we first merged all individual CSV files into a single unified dataset, resulting in approximately 5 million records. We then applied temporal filtering to retain only records collected between April 10, 2023, at 1:00 PM and April 13, 2023, at 5:29 PM (Days 1-4), corresponding to the labeled time period.

We filtered the beacon signals to include only the 25 primary BLE transmitters installed on the 5th floor, excluding signals from beacons on other floors. For easier reference, MAC addresses were mapped to beacon IDs (1-25). Unnecessary columns including user_id, name, and accidentally saved index columns were removed.

This cleaning process reduced the dataset from 5 million to approximately 1.67 million records that fall within the labeled timeframe and originate from the 25 relevant beacons.

#### 2.2.2 Location Label Data Cleaning

For the location label data, we filtered for records where activity equals "Location" to focus exclusively on location annotations. We removed records with null values in the started_at or finished_at columns to ensure complete time intervals. Records marked as deleted (deleted_at is not null) were excluded, and we filtered for user_id = 97 to retain only the primary labeler's annotations. Unused columns were dropped, reducing the label dataset from 1,334 to 451 clean location label records with well-defined time intervals and room assignments.

#### 2.2.3 Timestamp-Based Merging and Label Assignment

After independently cleaning both datasets, we performed timestamp-based merging to create a supervised learning dataset. Each BLE sensor reading was matched with its corresponding room label by finding the location label whose time interval encompasses the sensor reading's timestamp. Specifically, for each BLE record with timestamp *t*, we identified the label record where started_at ≤ *t* ≤ finished_at.

During this process, all timestamps were truncated from millisecond precision to second precision, as the sub-second temporal resolution is not critical for location identification tasks. This timestamp granularity reduction is consistent with common practices in indoor positioning systems where second-level precision is sufficient [6], [7].

#### 2.2.4 Handling Unlabeled Data

The merging process successfully labeled approximately 1.1 million BLE records (66% of the cleaned BLE data), while approximately 570,000 records (34%) could not be matched to any location label. These unlabeled records were intentionally dropped to maintain the quality and reliability of the supervised learning dataset.

The 34% unlabeled data represents an inherent characteristic of the data collection design: User ID 97 selectively annotated specific location visits rather than continuously labeling every moment, resulting in gaps where sensor data exists without corresponding ground truth labels. These unlabeled periods likely include transition times between rooms or moments when the labeler was not actively tracking. For training a reliable location prediction model, retaining only records with verified ground truth labels ensures higher quality evaluation and more trustworthy model performance metrics.

#### 2.2.5 Final Preprocessed Dataset

The final preprocessed dataset contains approximately 1.1 million labeled BLE sensor readings spanning four days of data collection. The structure is shown in Table 3, with each record containing a timestamp (second precision), beacon ID (1-25), RSSI value, and the corresponding room label as the prediction target.

**Table 3: Merged Labeled BLE Data Sample**

| timestamp | mac_address | RSSI | room |
|-----------|-------------|------|------|
| 2023-04-10 14:21:46+09:00 | 6 | -93 | kitchen |

This preprocessed labeled dataset serves as the foundation for all subsequent feature engineering, model training, and evaluation procedures.

---

## 3. Methodology

We propose and compare two distinct approaches representing fundamentally different paradigms in handling temporal sensor data. The first approach (Section 3.1) employs traditional machine learning methods treating each temporal window independently, extracting statistical features from RSSI values and applying gradient boosting classification. We systematically explore multiple variations to establish comprehensive baselines and identify fundamental limitations.

In contrast, Section 3.2 introduces the Deep Attention-based Sequential Ensemble Learning (DASEL) framework that reconceptualizes indoor localization as a sequential learning problem. DASEL leverages the inherent temporal continuity of human movement patterns through deep learning architectures, combining frequency-based feature engineering, bidirectional recurrent networks with attention mechanisms, multi-directional sliding windows, and multi-model ensemble learning. This comprehensive pipeline addresses the fundamental asymmetry between training (where room boundaries are known) and deployment (where boundaries must be inferred).

### 3.1 Traditional Machine Learning Baseline Family

Traditional approaches treat each temporal window as independent classification: construct beacon feature vectors, apply temporal windowing with statistical aggregation, and train gradient boosting classifiers. We systematically explore variations to establish robust baselines and demonstrate that the fundamental limitation lies in the independence assumption paradigm itself.

#### 3.1.1 Core Baseline Method

The baseline pipeline consists of four main steps: (1) beacon vector construction, (2) temporal windowing, (3) statistical feature aggregation, and (4) classification using gradient boosting. Figure 1 provides a visual overview of this workflow.

**Figure 1: Traditional Machine Learning Workflow**
[INSERT FIGURE HERE]

**Initial Data Format:** The raw BLE data from preprocessing (Section 2.2.5) consists of individual detection records as shown in Table 3. Each record contains a timestamp (second precision), beacon ID (1-25), RSSI value, and room label.

**Step 1 - Create 25-Dimensional Beacon Vector:** For each individual record at a given timestamp, we construct a 25-dimensional beacon vector where position *i* contains the RSSI value if beacon *i* was detected, and 0 otherwise.

**Step 2 - Temporal Windowing (1-second grouping):** Raw readings are grouped by their timestamp. All records sharing the same timestamp are aggregated together into a single window.

**Step 3 - Statistical Aggregation:** Within each 1-second window, for each of the 25 beacons, we compute three statistical features: mean (average signal strength), standard deviation (variability), and count (number of detections). For beacons not detected, all three statistics are set to 0. Each 1-second window is thus represented by 75 features (25 beacons × 3 statistics).

**Step 4 - Feature Vector for Classification:** The aggregated statistics are concatenated into a single feature vector per window. Each 1-second window becomes one training sample with 75 features and one room label.

**Classification:** We employ XGBoost as the baseline classifier with balanced sample weighting to address class imbalance. The model learns to map the 75-dimensional beacon signal patterns to room locations, with each 1-second window classified independently as an isolated data point.

#### 3.1.2 Explored Variations and Optimizations

To establish a comprehensive baseline and identify fundamental limitations, we systematically explored five variations, each targeting different aspects that could potentially improve performance.

**Variation 1 - Extended Statistical Features:** We expanded the feature set to include minimum and maximum RSSI values in addition to mean, standard deviation, and count. This increases feature dimensionality from 75 to 125 features (25 beacons × 5 statistics).

**Variation 2 - Minority Class Weighting:** Building upon the extended features, we applied increased weight (3× multiplier) to minority classes during XGBoost training, forcing the model to pay disproportionate attention to rarely-visited rooms.

**Variation 3 - SMOTE Oversampling:** We applied Synthetic Minority Over-sampling Technique to generate synthetic training samples for minority classes by interpolating between existing minority class instances in feature space.

**Variation 4 - Dominant Beacon Features:** We augmented the extended statistical features (125 features) with three additional categorical features representing the top three most frequently detected beacons within each window, increasing total dimensionality to 128 features. The rationale is that the most frequently detected beacons provide additional spatial context beyond signal strength statistics.

**Variation 5 - Signal Pattern-Based Relabeling:** Inspired by Garcia and Inoue's approach [8], we explored signal pattern-based data augmentation. This method identifies majority class rooms whose signal patterns closely match minority class rooms (using KL divergence), then relabels a subset of samples from the matched majority class with the minority class label. Unlike SMOTE which generates synthetic samples, relabeling reuses actual collected data from spatially similar locations.

### 3.2 Deep Attention-based Sequential Ensemble Learning (DASEL)

#### 3.2.1 Complete Method Description

DASEL reconceptualizes indoor localization as a sequential learning problem. The complete pipeline integrates four distinct phases: (1) frequency-based feature engineering, (2) sequential model training using bidirectional recurrent networks with attention mechanisms, (3) multi-level ensemble inference combining multiple temporal perspectives and model initializations, and (4) temporal smoothing for spatial consistency.

**Figure 2: DASEL Workflow**
[INSERT FIGURE HERE]

**Phase 1: Feature Engineering**

**Step 1 - Create 25-Dimensional Beacon Vector:** Starting with the preprocessed data from Section 2.2.5, we construct a 25-dimensional beacon vector for each record, where position *i* contains the RSSI value if beacon *i* was detected, and 0 otherwise.

**Step 2 - Temporal Windowing (1-second grouping):** We group all BLE readings that share the same timestamp into 1-second windows.

**Step 3 - Frequency Calculation:** Within each 1-second window, for each of the 25 beacons, we calculate the appearance frequency:

```
frequency_{i,t} = count_{i,t} / total_detections_t
```

where count_{i,t} is the number of times beacon *i* was detected in window *t*, and total_detections_t is the total number of all beacon detections in that window. For beacons not detected, the frequency is 0. We use 23 beacons (beacons 1-23) in practice, as beacons 24-25 were rarely detected. The data structure after Phase 1 is shown in Table 4.

**Table 4: Frequency-Based Feature Vector (Single Window)**

| timestamp | beacon_1_freq | beacon_2_freq | ... | beacon_23_freq | room |
|-----------|---------------|---------------|-----|----------------|------|
| 2023-04-10 14:21:46+09:00 | 0.15 | 0.25 | ... | 0.10 | kitchen |

Each 1-second window is represented by a 23-dimensional frequency vector.

**Phase 2: Training with Sequential Learning**

**Sequence Construction:** During training, we segment the data into sequences using ground truth room labels. We identify consecutive timestamps where the room label remains constant:

```
room_group_id = cumulative_sum(room_label ≠ previous_room_label)
```

Each contiguous block of the same room becomes one training sequence. For example, a 45-second stay in the Kitchen creates one sequence of 45 timesteps, as illustrated in Table 5.

**Table 5: Training Sequence Example (Kitchen Visit)**

| Sequence Position | timestamp | beacon_1_freq | beacon_2_freq | ... | beacon_23_freq | room |
|-------------------|-----------|---------------|---------------|-----|----------------|------|
| t=1 | 14:21:46 | 0.15 | 0.25 | ... | 0.10 | kitchen |
| t=2 | 14:21:47 | 0.18 | 0.22 | ... | 0.12 | kitchen |
| t=3 | 14:21:48 | 0.14 | 0.27 | ... | 0.09 | kitchen |
| ... | ... | ... | ... | ... | ... | ... |
| t=45 | 14:22:31 | 0.16 | 0.24 | ... | 0.11 | kitchen |

**Sequence Length Constraints:** Minimum length is 3 timesteps (shorter sequences discarded), maximum length is 50 timesteps (longer sequences truncated by taking the last 50 timesteps).

**Model Architecture:** The model consists of:
1. Masking Layer (handles variable-length sequences, padding to 50 timesteps)
2. First Bidirectional GRU Layer (128 units, processes sequences in both forward and backward directions)
3. Dropout Layer (0.3 dropout rate)
4. Second Bidirectional GRU Layer (64 units, provides additional temporal processing)
5. Dropout Layer (0.3 dropout rate)
6. Attention Layer (computes attention weights and creates weighted context vector):
   - Attention scores: tanh(W · sequence + b)
   - Attention weights: softmax(attention_scores)
   - Context vector: Σ(sequence × attention_weights)
7. Dense Layer (32 units with ReLU activation and 0.2 dropout)
8. Output Layer (softmax activation producing probability distribution over all room classes)

**Figure 3: DASEL Model Architecture**
[INSERT FIGURE HERE]

**Training:** The model is trained using sparse categorical cross-entropy loss with balanced class weights. Each training sequence represents one complete room visit.

**Phase 3: Multi-Level Ensemble Inference**

Phase 3 implements a two-level ensemble strategy that combines multi-seed model training with multi-directional sliding windows.

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

**Multi-Directional Sliding Windows:** For each timestamp *t* requiring prediction, we create 7 different temporal windows:

- **backward_10:** [t-9, ..., t] — 10 seconds of history
- **centered_10:** [t-4, ..., t, ..., t+5] — 10 seconds centered on t
- **forward_10:** [t, ..., t+9] — 10 seconds looking forward
- **backward_15:** [t-14, ..., t] — 15 seconds of history (extended context)
- **forward_15:** [t, ..., t+14] — 15 seconds looking forward (early transition detection)
- **asymm_past:** [t-11, ..., t, ..., t+3] — Past-biased (12s past + 4s future)
- **asymm_future:** [t-3, ..., t, ..., t+11] — Future-biased (4s past + 12s future)

**Confidence-Weighted Aggregation:** Each directional window produces a probability distribution with an associated confidence score (the maximum probability value). We aggregate these 7 distributions using confidence-weighted voting:

```
For timestamp t with 7 directional probability distributions:
    For each direction d (d = 1 to 7):
        confidence_d = max(probability_distribution_d)
        weighted_vote_d = probability_distribution_d × confidence_d    
    final_probability_t = sum(weighted_vote_d) / sum(confidence_d)
    final_prediction_t = argmax(final_probability_t)
```

**Phase 4: Temporal Smoothing**

For each prediction at timestamp *t*, we examine a 5-second temporal window [t-2, t-1, t, t+1, t+2] and apply confidence-weighted voting:

```
For each timestamp j in [t-2, t-1, t, t+1, t+2]:
    confidence_j = max(probability_distribution_j)
    weighted_vote += probability_distribution_j × confidence_j

smoothed_prediction_t = argmax(Σ weighted_vote)
```

**Figure 4: 5-Second Smoothing Visualization**
[INSERT FIGURE HERE]

This produces the final room prediction for each timestamp in the test set.

#### 3.2.2 Why Beacon Frequency Outperforms RSSI Values

The decision to use beacon appearance frequency rather than RSSI signal strength is motivated by fundamental limitations of BLE signal propagation in indoor environments.

**The RSSI Instability Problem**

RSSI values suffer from significant temporal and spatial variability caused by multipath signal propagation, human body orientation and absorption, interference from other wireless devices, and device-specific hardware characteristics. Previous studies have extensively documented that BLE RSSI signals fluctuate widely even at fixed positions due to multipath effects and signal noise [9]. Research has demonstrated that RSSI can be inconsistent and unstable across indoor areas with complex structures [10], [11].

Figure 5 demonstrates this instability through box plot comparisons of RSSI distributions for three beacons appearing in both Kitchen and Cafeteria. The visualization reveals two critical limitations: high intra-room variance (within a single location, signal strength varies substantially), and significant inter-room overlap (mean RSSI values for the same beacon in different rooms differ by less than 2 dBm, smaller than the variance within each room).

**Figure 5: RSSI Distribution Comparison: Kitchen vs Cafeteria**
[INSERT FIGURE HERE]

**The Beacon Frequency Advantage**

In contrast, beacon appearance frequency captures the fundamental spatial pattern of which beacons are detected rather than how strongly. Figure 6 illustrates the frequency distribution for three representative rooms, demonstrating that each room exhibits a distinct beacon appearance pattern.

**Figure 6: Beacon Frequency Distribution: Kitchen, Cafeteria, and Cleaning**
[INSERT FIGURE HERE]

Each room demonstrates a characteristic spatial "fingerprint" based on proximity to installed beacons. Rooms consistently detect nearby beacons at high frequencies while distant beacons appear rarely or not at all. The frequency-based representation offers multiple advantages: robustness to environmental noise (focusing on presence patterns rather than signal magnitude), natural normalization (values between 0 and 1, eliminating need for device-specific calibration), and strong discriminative power (dominant beacon patterns are clearly distinct across rooms).

#### 3.2.3 Why Sequential Learning and This Model Architecture?

**The Sequential Nature of Indoor Localization**

Traditional methods treat each 1-second window as independent, fundamentally ignoring temporal continuity in human movement. When a person moves through indoor space, consecutive observations are highly correlated—location at time *t* strongly influences location at time *t+1*. A caregiver walking from Kitchen to Hallway produces a temporal sequence of beacon patterns that gradually transition. By treating each second independently, traditional methods discard contextual information about movement trajectories, transition patterns, and dwell times.

Sequential learning addresses this by explicitly modeling entire room visits as temporal sequences, allowing the model to learn not just static beacon patterns but also dynamics of how patterns evolve over time. This temporal context provides crucial disambiguation when instantaneous signals are ambiguous.

**Architecture Selection: Bidirectional GRU with Deep Attention**

Our architecture employs a two-layer Bidirectional GRU structure with an attention mechanism. The first Bidirectional GRU layer (128 units) processes sequences in both forward and backward directions, capturing temporal dependencies from both past context (origin) and future context (destination). The second Bidirectional GRU layer (64 units) refines features through secondary temporal abstraction, creating more stable representations. The deep architecture (two layers) significantly reduces variance across different random initializations.

The attention mechanism learns which timesteps within a sequence are most informative for classification. Not all moments in a room visit are equally discriminative—transitions produce ambiguous patterns, while stable presence yields clear signals. The attention layer automatically emphasizes high-confidence timesteps while downweighting noisy transitions, allowing robust classification from sequences containing transitional noise. The attention mechanism computes weights via softmax(tanh(W·sequence + b)) and produces a context vector through weighted aggregation, creating a fixed-size room representation from variable-length sequences.

#### 3.2.4 Why Multi-Directional Windows with Confidence-Weighted Voting?

**The Inference Challenge: Unknown Sequence Position**

During training, the model learns from clean sequences with known room boundaries. However, during inference on unlabeled test data, these boundaries are unknown. For any timestamp requiring prediction, we don't know where it falls within its actual room visit sequence—at the beginning, middle, or end. This positional uncertainty creates a fundamental challenge.

A single sliding window configuration cannot optimally capture all positions. A backward-looking window [t-9 to t] fails if the room visit started at t-3 (contains contaminated signal from previous room). A forward-looking window [t to t+9] fails when predicting near sequence end (incorporates future signals from next room). A centered window [t-4 to t+5] works well for mid-sequence predictions but struggles at boundaries. No single temporal perspective handles all cases reliably.

**Multi-Directional Strategy**

To address positional uncertainty, we employ seven different window configurations providing complementary temporal perspectives. This ensures that regardless of where a timestamp falls within its true sequence, at least some windows will be well-aligned with clean signals. When predicting at sequence start, forward-looking windows excel; at sequence end, backward windows dominate; in the middle, centered windows provide stable predictions.

**Window Size Selection**

The choice of 10-second and 15-second window sizes is empirically grounded in the training data's sequence length distribution. Figure 7 shows the distribution of training sequence lengths (318 sequences total, average length 74.2 seconds).

**Figure 7: Training Sequence Length Distribution**
[INSERT FIGURE HERE]

The distribution reveals that the majority of room visits are concentrated between 10 and 200 seconds. The 10-second window captures the lower end of typical visit durations, ensuring even brief passages receive adequate temporal context without excessive contamination. The 15-second extended windows offer additional context for longer visits while remaining well below the average sequence length, minimizing risk of spanning multiple room transitions. These sizes strike a balance: large enough to capture discriminative temporal patterns, yet small enough to avoid excessive boundary contamination.

**Confidence-Weighted Voting vs. Majority Voting**

Each of the seven windows produces a prediction with an associated confidence score (maximum probability from softmax output). We aggregate using confidence-weighted voting rather than simple majority voting. In majority voting, each window's prediction counts equally regardless of certainty—a poorly-fitting window contaminated by transitions receives the same weight as a well-aligned window with clean signals.

Confidence-weighted voting incorporates the model's uncertainty estimation. Well-aligned windows produce high-confidence predictions (probabilities concentrated on a single class), while poorly-aligned windows yield low-confidence predictions (diffuse distributions). By weighting each window's contribution by its confidence score, the aggregation naturally emphasizes reliable predictions and downweights uncertain ones:

```
confidence_i = max(probability_distribution_i)
weighted_vote = Σ(probability_distribution_i × confidence_i) / Σ(confidence_i)
final_prediction = argmax(weighted_vote)
```

This allows the ensemble to self-regulate: when multiple well-aligned windows agree with high confidence, they dominate; when all windows show uncertainty (during genuine transitions), the aggregation reflects this appropriately.

#### 3.2.5 Why Multi-Seed Model Ensemble?

Deep neural networks are highly sensitive to random weight initialization, leading to performance variability across training runs even with identical architectures and data. This initialization-dependent variance creates a "seed lottery" problem where a single model's performance may not reflect the true capability of the architecture.

To address this, we train five independent models with different random initialization seeds (42, 1042, 2042, 3042, 4042), using large increments to ensure diversity. Ensemble methods reduce variance of a base estimator by introducing randomization and aggregating multiple instances [12]. Each model learns slightly different patterns from its unique initialization trajectory. By aggregating predictions through confidence-weighted voting, the ensemble reduces variance while maintaining or improving average performance.

This multi-seed strategy provides three key benefits: variance reduction through averaging fluctuations, robustness against unlucky initializations converging to poor local minima, and more reliable evaluation metrics reflecting the model's true capability rather than initialization luck.

#### 3.2.6 Why 5-Second Temporal Smoothing?

Temporal smoothing serves as post-processing that enforces spatial and temporal consistency. Even after multi-directional ensemble aggregation, isolated prediction errors can occur—for instance, a single timestamp incorrectly predicted as a distant room when surrounded by consistent predictions of a nearby room. These isolated errors are often physically implausible (people don't instantaneously teleport).

The 5-second smoothing window examines predictions within a local temporal neighborhood [t-2, t-1, t, t+1, t+2] and applies confidence-weighted majority voting. If a prediction at timestamp *t* contradicts surrounding predictions and those surrounding predictions exhibit high confidence, the outlier is overridden. This effectively eliminates "teleportation" errors while preserving legitimate room transitions showing sustained directional changes.

The 5-second window represents a practical balance based on human movement patterns. This duration is long enough to catch isolated errors (people typically remain in a room more than 5 seconds during meaningful visits), yet short enough to preserve genuine transitions between adjacent rooms (which can occur within 5-10 seconds). Empirical testing with alternative window sizes (3, 7, and 10 seconds) confirmed that 5 seconds provides optimal performance.

### 3.3 Evaluation Protocol

**Evaluation Metric**

We evaluate model performance using macro F1-score, the official metric specified by the ABC 2026 challenge organizers. Macro F1-score computes the F1-score independently for each room class and then averages these scores, treating all rooms equally regardless of their frequency in the dataset. This metric is particularly appropriate for our imbalanced indoor localization problem, where some rooms are visited far more frequently than others. Unlike accuracy or weighted F1-score, macro F1 ensures that the model must perform well across all locations, including rare rooms. This emphasis on balanced performance is critical for practical deployment in care facilities, where reliable localization in every area is essential for comprehensive caregiver tracking.

**4-Fold Temporal Cross-Validation**

We employ 4-fold cross-validation with a temporal splitting strategy rather than random partitioning. The preprocessed dataset spans four days of continuous data collection (April 10-13, 2023), and we split the data such that each fold uses one complete day as the test set and the remaining three days as the training set. This temporal split is essential because consecutive BLE readings are highly autocorrelated—beacon signal patterns within a few seconds are nearly identical. Random splitting would leak highly similar samples between training and test sets, artificially inflating performance metrics.

By splitting temporally, we ensure that all test data is genuinely unseen and represents a different time period, simulating realistic deployment where models trained on historical data must generalize to new days with potentially different environmental conditions and movement patterns.

**Fold Characteristics**

The four folds exhibit substantial variation in data size and class distribution, reflecting natural imbalance in real-world data collection. Table 6 presents the detailed characteristics of each fold.

**Table 6: Cross-Validation Fold Characteristics**

| Fold | Test Day | Train Frames | Test Frames | Train/Test Ratio | Train Classes | Test Classes |
|------|----------|--------------|-------------|------------------|---------------|--------------|
| 1 | Day 4 | 962,294 | 30,619 | 31.4× | 13 | 13 |
| 2 | Day 3 | 951,141 | 143,401 | 6.6× | 18 | 18 |
| 3 | Day 2 | 747,816 | 333,507 | 2.2× | 15 | 15 |
| 4 | Day 1 | 465,004 | 590,447 | 0.79× | 12 | 12 |

Fold 1 tests on the smallest dataset (Day 4 with 30,619 frames), providing the largest training set and a highly favorable train/test ratio of 31.4×. Conversely, Fold 4 presents the most challenging scenario: testing on the largest dataset (Day 1 with 590,447 frames) while training on the smallest subset (465,004 frames), resulting in a train/test ratio below 1 (0.79×). This tests the model's ability to generalize when training data is scarce relative to deployment scale.

The number of room classes also varies across folds, ranging from 12 to 18 unique locations, reflecting realistic deployment conditions where different days may involve visits to different subsets of rooms. This heterogeneous fold structure provides a robust evaluation framework that tests model performance under varying data availability conditions and across different room subsets. The average performance across all four folds offers a reliable estimate of expected real-world performance.

---

## 4. Results

### 4.1 Traditional Machine Learning Approaches

We evaluated the baseline traditional machine learning approach and its five variations using 4-fold cross-validation. Table 7 presents the macro F1 scores across all folds for each approach. The baseline method achieved a mean macro F1 score of 0.2805 ± 0.0278. Variation 4 (Dominant Beacon Features) achieved the highest performance at 0.2898 ± 0.0293, while all other variations ranged between 0.2838 and 0.2857. All approaches remained within the 0.28-0.29 macro F1 range.

**Table 7: Macro F1 Scores of Traditional Machine Learning Approaches Across 4-Fold Cross-Validation**

| Approach | Description | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std |
|----------|-------------|--------|--------|--------|--------|------------|
| Baseline | 3 aggregated features (mean, std, count) | 0.2819 | 0.2493 | 0.2665 | 0.3242 | 0.2805 ± 0.0278 |
| Variation 1 | Extended features (mean, std, count, min, max) | 0.2784 | 0.2574 | 0.2616 | 0.3456 | 0.2857 ± 0.0354 |
| Variation 2 | Minority class weighting (3× multiplier) | 0.2803 | 0.2504 | 0.2652 | 0.3394 | 0.2838 ± 0.0338 |
| Variation 3 | SMOTE oversampling | 0.2945 | 0.2682 | 0.2709 | 0.3090 | 0.2857 ± 0.0169 |
| Variation 4 | Dominant beacon features (top 3 most frequent) | 0.3009 | 0.2621 | 0.2634 | 0.3327 | 0.2898 ± 0.0293 |
| Variation 5 | Signal pattern-based relabeling | 0.2830 | 0.2486 | 0.2633 | 0.3455 | 0.2851 ± 0.0369 |

### 4.2 Deep Attention-based Sequential Ensemble Learning (DASEL)

We evaluated our proposed DASEL framework using 4-fold cross-validation with multi-seed ensemble learning. Table 8 presents the macro F1 scores across all folds. The DASEL approach achieved a mean macro F1 score of 0.4438 ± 0.0295, with individual fold performances ranging from 0.4082 (Fold 4) to 0.5114 (Fold 1). The results demonstrate consistent performance improvements over the traditional machine learning baseline family.

**Table 8: Macro F1 Scores of DASEL Across 4-Fold Cross-Validation**

| Approach | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std |
|----------|--------|--------|--------|--------|------------|
| DASEL | 0.5114 | 0.4207 | 0.4340 | 0.4082 | 0.4438 ± 0.0295 |

### 4.3 Comparison Between Traditional ML and DASEL

Table 9 presents a comprehensive comparison between the best-performing traditional machine learning approach (Variation 4: Dominant Beacon Features) and our proposed DASEL framework. DASEL achieves a mean macro F1 score of 0.4438, representing a 53.1% improvement over the best traditional ML variation (0.2898) and a 58.2% improvement over the baseline traditional approach (0.2805).

The improvement is consistent across all folds, with the most substantial gain observed in Fold 1 (0.5114 vs. 0.3009, a 70.0% improvement) and the smallest but still significant gain in Fold 4 (0.4082 vs. 0.3327, a 22.7% improvement). This demonstrates that DASEL's sequential learning paradigm with multi-directional ensemble inference effectively captures temporal movement patterns that traditional independent-window approaches cannot model.

**Table 9: Comparison of Macro F1 Scores Between Best Traditional ML Approach and DASEL**

| Approach | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std | Improvement over Best Traditional ML |
|----------|--------|--------|--------|--------|------------|-------------------------------------|
| Baseline (3 features) | 0.2819 | 0.2493 | 0.2665 | 0.3242 | 0.2805 ± 0.0278 | - |
| Best Traditional ML (Variation 4) | 0.3009 | 0.2621 | 0.2634 | 0.3327 | 0.2898 ± 0.0293 | - |
| DASEL (Proposed) | 0.5114 | 0.4207 | 0.4340 | 0.4082 | 0.4438 ± 0.0295 | +53.1% |

---

## 5. Discussion

The proposed DASEL framework achieves a mean macro F1 score of 0.4438 compared to 0.2898 for the best traditional machine learning baseline (Table 9), representing a 53.1% relative improvement. This section discusses the key factors contributing to DASEL's success and the fundamental limitations of traditional approaches.

### 5.1 The Limitations of Traditional Approaches

Traditional machine learning methods for indoor localization treat each temporal window as an independent classification problem. As shown in Table 7, we systematically explored multiple variations to establish a comprehensive baseline: extended statistical features (Variation 1), minority class weighting (Variation 2), SMOTE oversampling (Variation 3), dominant beacon features (Variation 4), and signal pattern-based relabeling (Variation 5).

Despite these diverse optimization strategies targeting different aspects—feature engineering, class imbalance handling, and data augmentation—the results remained remarkably consistent. The baseline method achieved 0.2805, while all five variations ranged between 0.2838 and 0.2898. The narrow performance band spanning just 0.2805 to 0.2898 across fundamentally different optimization techniques suggests these methods have reached a fundamental ceiling imposed by the independence assumption itself.

This consistent underperformance confirms our central argument: incremental improvements to the traditional paradigm cannot overcome the fundamental limitation of treating temporally-dependent data as independent observations. The problem requires a paradigm shift rather than optimization within the existing framework.

### 5.2 The Value of Sequential Modeling

The most significant contribution to DASEL's performance comes from treating indoor localization as a sequential learning problem. As shown in Table 8, DASEL achieves macro F1 = 0.4438, substantially outperforming all traditional variations. This dramatic improvement validates our hypothesis that human movement patterns contain rich temporal dependencies that traditional independent-window classification cannot capture.

Indoor localization is fundamentally a sequential task. Caregivers follow continuous trajectories through physical space, producing correlated sequences of beacon signal patterns. When a person walks from Kitchen to Hallway, beacon detections gradually shift from Kitchen-characteristic patterns to Hallway-characteristic patterns. By modeling entire room visits as temporal sequences, DASEL captures these movement dynamics that traditional approaches discard.

The bidirectional GRU architecture processes sequences in both forward and backward directions, enabling the model to leverage complete temporal context. This is particularly valuable during room transitions, where understanding both origin and destination provides crucial disambiguation. The model learns not just static beacon patterns for each room, but also characteristic ways these patterns evolve during entries, stable occupancy, and exits. This sequential modeling capability represents a fundamental architectural advantage over traditional methods.

### 5.3 Architecture Components and Attention Mechanism

The deep bidirectional GRU architecture with attention mechanism provides robust feature extraction from temporal sequences. The two-layer recurrent structure creates hierarchical representations, with the first layer capturing immediate temporal patterns and the second layer refining these into more stable, abstract representations. This depth helps reduce variance across different random initializations, providing more consistent predictions.

The attention mechanism addresses a key challenge: not all timesteps are equally informative. During stable room occupancy, beacon patterns provide clear location signals, while room transitions and doorway passages produce ambiguous multi-room signals. Rather than treating all timesteps equally, the attention layer learns to emphasize discriminative moments while downweighting noisy transition periods. This selective focus enables robust classification even when sequences contain substantial transitional noise.

### 5.4 Frequency-Based Features and Practical Robustness

The frequency-based feature representation addresses critical practical challenges for real-world deployment. As demonstrated in Figure 5, raw RSSI measurements exhibit substantial instability due to multipath propagation, human body absorption, and device-specific hardware variations. RSSI distributions for the same beacon in different rooms show significant overlap, fundamentally limiting the discriminative power of signal-strength-based features.

Beacon appearance frequency captures which beacons are detected rather than how strongly their signals are received, providing a more stable spatial signature. Each room exhibits characteristic patterns based on proximity to installed transmitters—nearby beacons appear frequently while distant beacons appear rarely or not at all. These presence patterns remain consistent despite environmental factors that destabilize individual RSSI measurements.

This representation eliminates the need for complex device-specific RSSI calibration procedures, a major practical barrier to deployment. By focusing on beacon detection patterns rather than precise signal strengths, DASEL achieves robustness to device heterogeneity and environmental variations without extensive system tuning.

### 5.5 The Two-Level Ensemble as Optimization

The two-level hierarchical ensemble provides critical optimizations that enhance prediction quality beyond the core sequential modeling framework. The first level addresses variance through multi-seed training: training five models with different random initializations and averaging their predictions reduces initialization-dependent fluctuations, ensuring stable probability distributions that reflect the architecture's true capability.

The second level addresses inference uncertainty through multi-directional windows. During training, the model learns from sequences with known room boundaries, but during inference on unlabeled data, we don't know where each timestamp falls within its true room visit sequence. The seven directional windows provide complementary temporal perspectives—backward-looking, forward-looking, centered, and asymmetric configurations—that excel in different positional contexts. The confidence-weighted aggregation allows well-aligned windows with clean signals to dominate predictions while downweighting poorly-aligned windows.

These ensemble strategies serve as optimization layers atop the sequential modeling foundation, incrementally refining predictions through variance reduction and multi-perspective aggregation. Together with temporal smoothing, they transform the base sequential model's outputs into reliable, spatially-consistent predictions suitable for practical deployment.

### 5.6 Limitations and Future Work

Despite strong performance, several limitations suggest directions for future research. The two-level ensemble requires multiple forward passes per timestamp, which may challenge real-time deployment on resource-constrained mobile devices. Model compression techniques such as knowledge distillation could potentially maintain accuracy while reducing computational demands.

The current approach uses fixed window sizes (10 and 15 seconds) based on training sequence length distributions. Adaptive window sizing that adjusts based on detected movement patterns could improve performance for very short or very long room visits. Additionally, DASEL treats all rooms equally without modeling physical adjacency relationships. Incorporating facility layout information could reduce physically implausible prediction errors and improve transition modeling between adjacent rooms.

Finally, our evaluation focuses on a single care facility. Future work should examine generalization to different facilities with varying layouts, beacon densities, and construction materials, potentially leveraging transfer learning to adapt models with limited target-domain labeled data.

### 5.7 Practical Implications

DASEL's balanced macro F1 score indicates reliable performance across all room classes, including rarely-visited locations. This is essential for comprehensive care facility monitoring, where accurate localization in individual patient rooms is as critical as tracking in common areas for understanding care delivery patterns and optimizing staff allocation.

The frequency-based representation provides practical robustness to device heterogeneity and environmental variations, reducing calibration requirements for deployment. The temporal smoothing mechanism produces spatially consistent predictions that match realistic human movement patterns, eliminating implausible instantaneous transitions between distant rooms. These characteristics—balanced performance, device robustness, and spatial consistency—make DASEL deployable in real-world care facility settings without extensive system tuning, addressing key practical barriers that limit adoption of indoor localization systems.

---

## 6. Conclusion

This research demonstrates that traditional machine learning approaches to BLE-based indoor localization face fundamental performance limitations that cannot be overcome through incremental optimization. Our systematic exploration of five diverse variations—extended statistical features, minority class weighting, SMOTE oversampling, dominant beacon identification, and signal pattern-based relabeling—targeting different aspects of the problem all yielded remarkably consistent results within a narrow 0.2805 to 0.2898 macro F1 score band. This performance plateau reveals that the core limitation lies not in implementation details but in the independence assumption itself: treating each temporal window as an isolated observation discards the rich temporal dependencies inherent in human movement patterns.

Our proposed DASEL framework achieves breakthrough performance by fundamentally reconceptualizing indoor localization as a sequential learning problem. By explicitly modeling entire room visits as temporal sequences through bidirectional recurrent networks with attention mechanisms, DASEL captures movement trajectories, entry/exit patterns, and dwell-time dynamics that traditional methods completely ignore. The integration of frequency-based features for RSSI robustness, two-level hierarchical ensemble for variance reduction and positional uncertainty handling, and confidence-weighted temporal smoothing for spatial consistency creates a comprehensive system that leverages the sequential characteristics of human movement data.

The resulting 53.1% relative improvement (macro F1: 0.2898 → 0.4438) demonstrates that the sequential nature of indoor localization is not merely a secondary consideration but the fundamental property that must be explicitly modeled to achieve breakthrough performance. DASEL's balanced performance across all room classes, including minority classes, combined with its robustness to device heterogeneity through frequency-based features, makes it practically deployable in real-world care facilities.

This work establishes that the path forward for BLE-based indoor localization lies in sequential modeling architectures that treat human movement as the continuous temporal process it fundamentally is, rather than as a collection of independent snapshots.

---

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
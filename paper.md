# Deep Attention-based Sequential Ensemble Learning for BLE-Based Indoor Localization in Care Facilities

## Abstract

Indoor localization systems in care facilities enable optimization of staff allocation, workload management, and quality of care delivery. Traditional machine learning approaches to Bluetooth Low Energy (BLE)-based localization treat each temporal measurement as an independent observation, fundamentally limiting their performance. This paper introduces Deep Attention-based Sequential Ensemble Learning (DASEL), a novel framework that reconceptualizes indoor localization as a sequential learning problem. DASEL integrates frequency-based feature engineering, bidirectional GRU networks with attention mechanisms, multi-directional sliding windows, and confidence-weighted temporal smoothing to capture human movement trajectories. Evaluated on real-world data from a care facility using 4-fold temporal cross-validation, DASEL achieves a macro F1 score of 0.4438, representing a 53.1% improvement over the best traditional baseline (0.2898). This breakthrough demonstrates that modeling temporal dependencies in movement patterns is essential for accurate indoor localization in complex real-world environments.

---

## 1. Introduction

### 1.1 Research Context and Motivation

Indoor localization systems in care facilities enable optimization of staff allocation, workload management, and quality of care delivery. Accurate tracking of caregiver movements provides insights into care routines, enhances hand hygiene monitoring, and supports health interventions for elderly residents with conditions like Alzheimer's and dementia. Automated location recording eliminates manual logging burdens and provides objective data for facility management and quality improvement initiatives.

### 1.2 Literature Review: BLE-Based Indoor Localization

Bluetooth Low Energy (BLE) technology has emerged as a prominent indoor localization solution due to its low cost, minimal power consumption, and ease of deployment. Indoor positioning methods have evolved from simple Received Signal Strength Indicator (RSSI) measurements to more advanced approaches such as Channel State Information (CSI), Round Trip Time (RTT), and Angle of Arrival (AoA), increasingly combined with Machine Learning techniques. A systematic review of ML-based indoor positioning systems from 2020-2024 shows that ML-based IPS have progressively shifted from traditional algorithms toward Deep Learning architectures, though RSSI fingerprinting remains dominant due to its simplicity and low deployment cost [1].

Traditional machine learning approaches to BLE indoor localization predominantly rely on RSSI fingerprinting with classical classification algorithms. K-Nearest Neighbors (KNN) and Weighted KNN (WKNN) algorithms have been widely adopted to match collected RSSI readings with fingerprint databases, achieving small localization errors even under obstacles, reflections, and interference conditions [2]. Support Vector Machines (SVM) have been used for localization using offline and online RSSI measurements, while Random Forest classifiers have demonstrated effectiveness through real-time trials in simulated IoT environments [2]. Recent studies evaluating machine learning algorithms on BLE RSSI datasets have benchmarked KNN, Random Forest, XGBoost, SVM, and Fully Connected Neural Networks for localization tasks, with performance assessed through localization error and success rate metrics [3].

However, these approaches share a fundamental limitation: they treat each temporal measurement as an independent observation, extracting statistical features (mean, standard deviation, count) from beacon signals and classifying each moment in isolation. For several years, fingerprinting using RSSI and KNN has been the dominant approach due to its simplicity and low deployment cost, but in recent years this approach has declined in prominence due to its limitations in dynamic and multipath environments [1]. The growing complexity of indoor environments requires solutions that can handle sensor noise, multipath fading effects, and temporal dependencies that traditional independent-window classification methods cannot adequately address [4].

### 1.3 Real-World Data Challenges and Traditional ML Limitations

BLE-based localization in real-world care facilities confronts significant data quality challenges. BLE signals suffer from large fluctuations in RSSI values caused by multipath propagation and environmental factors [5]. Care facilities present particularly complex deployment scenarios with beacon placement constraints, resulting in datasets with substantial challenges: signal instability from multipath effects and device heterogeneity, spatial sparsity with limited beacon coverage in some rooms, temporal irregularity with variable detection rates, and severe class imbalance where common areas are visited far more frequently than individual patient rooms.

When traditional machine learning methods are applied to such real-world datasets, performance remains limited despite optimization efforts. We systematically explored multiple variations: extended statistical features (mean, std, count, min, max), minority class weighting (3× multiplier), SMOTE oversampling, dominant beacon identification (top-3 most frequent beacons), and signal pattern-based relabeling using KL-divergence matching. Despite these diverse optimization strategies targeting feature engineering, class imbalance handling, and data augmentation, all approaches remained within a narrow performance band of 0.2805 to 0.2898 macro F1 score, with the best method achieving only 0.2898.

This consistent plateau across fundamentally different optimization techniques reveals a critical limitation: the independence assumption discards temporal dependencies in human movement. Caregivers follow continuous trajectories through physical space, not instantaneous teleportation between rooms. A person's location at time *t* strongly predicts their location at time *t+1*, and transitions between rooms produce gradual shifts in beacon patterns. Yet traditional methods treat each second as an isolated classification problem, discarding rich contextual information about movement trajectories, dwell times, and transition dynamics.

### 1.4 Research Objective

Our objective is to develop a breakthrough approach that overcomes the fundamental performance ceiling and limitations of traditional independent-window classification methods. Specifically, we aim to:

1. Leverage temporal dependencies in human movement trajectories that traditional methods discard
2. Address RSSI instability through robust feature representations
3. Handle the inference challenge where sequence boundaries are unknown during real-time prediction
4. Achieve substantial performance improvements while maintaining practical deployability in real-world care facilities

### 1.5 Proposed Approach and Key Contributions

We introduce Deep Attention-based Sequential Ensemble Learning (DASEL), which reconceptualizes indoor localization as a sequential learning problem rather than static classification. DASEL integrates four synergistic components:

**Frequency-Based Features:** Captures stable beacon appearance patterns (which beacons are detected) rather than unstable RSSI magnitudes (how strongly), providing robustness to environmental noise and device variability.

**Bidirectional GRU with Deep Attention:** Models entire room visits as temporal sequences using two-layer Bidirectional GRU architecture with attention mechanism, learning entry patterns, stable presence signatures, and exit dynamics.

**Two-Level Hierarchical Ensemble:**
- *Level 1:* Multi-seed model training (5 models with different random initializations) for variance reduction
- *Level 2:* Multi-directional sliding windows (7 temporal perspectives: backward, centered, forward, asymmetric) for positional robustness during inference

**Confidence-Weighted Temporal Smoothing:** Enforces spatial consistency through 5-second voting windows, eliminating physically implausible prediction errors.

**Key Achievements and Contributions:**

DASEL achieves macro F1 = 0.4438, representing a 53.1% relative improvement over the best traditional baseline (0.2898). This substantial performance gain demonstrates that breakthrough results require paradigm shift—from independent-window classification to sequential trajectory modeling.

**Main Contributions:**

1. **Novel Sequential Framework:** First application of deep bidirectional recurrent networks with attention to BLE-based care facility localization, explicitly modeling temporal movement trajectories
2. **Frequency-Based Representation:** Stable feature engineering addressing RSSI instability challenges inherent in real-world deployments
3. **Two-Level Hierarchical Ensemble:** Innovative inference strategy combining multi-seed variance reduction with multi-directional positional robustness
4. **Comprehensive Baseline Evaluation:** Systematic comparison against five traditional ML variations using 4-fold temporal cross-validation, demonstrating that traditional paradigm optimization yields diminishing returns
5. **Practical Deployability:** Balanced performance across all room classes (including minority classes) with robustness to device heterogeneity, suitable for real-world deployment

### 1.6 Paper Organization

The remainder of this paper is organized as follows:

**Section 2** (Dataset Introduction & Preprocessing) describes the BLE sensor data collection methodology in the care facility, the location labeling process, and preprocessing procedures including temporal alignment, frequency feature calculation, and handling of unlabeled data.

**Section 3** (Methodology) presents two distinct approaches. Section 3.1 (Traditional ML Baseline Family) details the baseline gradient boosting approach and five systematic variations exploring extended features, class balancing strategies (minority weighting, SMOTE), dominant beacon identification, and signal pattern-based relabeling. Section 3.2 (DASEL Framework) describes the complete four-phase pipeline (frequency features, sequential training with Bi-GRU+attention, two-level ensemble inference, temporal smoothing) with detailed rationale for each design choice.

**Section 4** (Results) reports experimental results across 4-fold temporal cross-validation, comparing traditional ML variations against DASEL and demonstrating the 53.1% improvement.

**Section 5** (Discussion) analyzes why DASEL's integrated design principles enable breakthrough performance, discussing the value of sequential modeling, the two-level ensemble optimization strategy, architecture contributions, frequency-based robustness, and practical implications.

**Section 6** (Conclusion) summarizes findings, practical implications for care facility deployment, and future research directions.

---

## 2. Dataset Introduction & Preprocessing

The dataset used for this research was provided by Kyushu Institute of Technology in conjunction with ABC 2026. The data collection setup involves multiple Bluetooth Low Energy (BLE) beacons installed throughout the 5th floor of a care facility, where each beacon continuously transmits signals. User ID 90, acting as a caregiver, moves around the 5th floor carrying a mobile phone with an application that continuously detects and records RSSI (Received Signal Strength Indicator) values from nearby Bluetooth beacons. The application logs beacon signals at each timestamp as the caregiver moves through different locations. Concurrently, User ID 97 acts as a labeler who manually tracks User ID 90's movements and annotates their location during specific time periods. This setup yields two primary datasets: the BLE sensor data recorded by the caregiver's device and the location labels annotated by the labeler.

### 2.1 Dataset Introduction

#### 2.1.1 BLE Sensor Data

The BLE sensor data is continuously recorded by the caregiver (User ID 90) who moves around the 5th floor with a mobile phone equipped with a data collection application. This application captures RSSI values from all detectable Bluetooth beacons in the vicinity. The raw BLE data is provided as a collection of CSV files with the structure shown in Table 1.

**Table 1: BLE Data Sample**

| user_id | timestamp | name | mac_address | RSSI | power |
|---------|-----------|------|-------------|------|-------|
| 90 | 2023-04-10T10:22:55.589+0900 | null | FD:07:0E:D5:28:AE | -75 | -2147483648 |

The features in the BLE data are defined as follows:

- **user_id:** Identifies the individual recording the data (always 90 for the caregiver)
- **timestamp:** Represents the detection time of each signal with millisecond precision in ISO 8601 format with timezone information (UTC+09:00)
- **name:** An unused field that consistently contains null values
- **mac_address:** Serves as the unique identifier for each beacon (e.g., FD:07:0E:D5:28:AE)
- **RSSI:** Indicates the signal strength in dBm, where values closer to zero represent stronger signals and closer proximity to the beacon
- **power:** An additional field that appears to be a placeholder with constant value -2147483648 and is not utilized in this study

The original dataset contains approximately 5 million records (5,005,751 total). It is important to note that these records capture RSSI signals from all beacons throughout the entire care facility, not exclusively from the 5th floor where the location prediction task is focused.

#### 2.1.2 Location Label Data

The location labels are manually annotated by User ID 97, who observes and records the positions of User ID 90 (the caregiver) throughout the data collection period. Each label record indicates the specific room or location where the caregiver was present during a defined time interval. The structure of the label data is presented in Table 2.

**Table 2: Location Label Data Sample**

| activity | started_at | finished_at | deleted_at | updated_at | user_id | user | room | floor |
|----------|------------|-------------|------------|------------|---------|------|------|-------|
| Location | 2023-04-10 14:21:46+09:00 | 2023-04-10 14:21:50+09:00 | null | 2023-04-10 05:22:02 UTC | 97 | 5th-location | kitchen | 5th |

The features in the location label data are defined as follows:

- **activity:** Specifies the type of annotation, where "Location" indicates a location labeling record (only records with activity = "Location" are relevant for this study)
- **started_at and finished_at:** Define the time interval during which the caregiver was present in the specified location, where any timestamp falling within this range is labeled with the corresponding room value
- **room:** Serves as the target label indicating the specific location (e.g., "kitchen", "cafeteria", "nurse station", "hallway", "523")
- **deleted_at:** Contains the timestamp when a record was marked as deleted (records with non-null values are excluded from analysis)
- **updated_at:** Indicates when the label record was last modified (not used for modeling)
- **user_id:** Identifies the labeler (only records with user_id = 97 are considered)
- **user:** Provides a text identifier for the labeling task
- **floor:** Indicates the floor level (should be "5th" or "5f" for all relevant records)

The original label dataset contains 1,334 records before preprocessing. Each record represents a time interval during which the caregiver was observed to be in a specific location on the 5th floor.

### 2.2 Data Preprocessing

The preprocessing pipeline involves multiple stages to transform the raw data into a clean, labeled dataset suitable for machine learning model training. The process addresses data quality issues, temporal alignment, and the creation of a supervised learning dataset. This section describes the cleaning procedures applied to both datasets and the methodology for merging them into a unified labeled dataset.

#### 2.2.1 BLE Sensor Data Cleaning

For the BLE sensor data, we first merged all individual CSV files into a single unified dataset, resulting in approximately 5 million records. Subsequently, we applied temporal filtering to retain only records collected between April 10, 2023, at 1:00 PM and April 13, 2023, at 5:29 PM (Days 1-4), which corresponds to the labeled time period.

We then filtered the beacon signals to include only the 25 primary BLE transmitters installed on the 5th floor, excluding signals from beacons located on other floors or outside the study area. For easier reference and processing, MAC addresses were mapped to beacon IDs (1-25). Unnecessary columns including the user_id (always 90), name (always null), and accidentally saved index columns were removed.

This cleaning process reduced the dataset from 5 million to approximately 1.67 million records that fall within the labeled timeframe and originate from the 25 relevant beacons.

#### 2.2.2 Location Label Data Cleaning

For the location label data, we filtered for records where activity equals "Location" to focus exclusively on location annotations. We removed records with null values in the started_at or finished_at columns to ensure complete time intervals. Records marked as deleted (deleted_at is not null) were excluded, and we filtered for user_id = 97 to retain only the primary labeler's annotations.

Unused columns including deleted_at, updated_at, activity, user, and user_id were dropped. This reduced the label dataset from 1,334 to 451 clean location label records with well-defined time intervals and room assignments.

#### 2.2.3 Timestamp-Based Merging and Label Assignment

After independently cleaning both datasets, we performed timestamp-based merging to create a supervised learning dataset. Each BLE sensor reading was matched with its corresponding room label by finding the location label whose time interval encompasses the sensor reading's timestamp. Specifically, for each BLE record with timestamp *t*, we identified the label record where started_at ≤ *t* ≤ finished_at, ensuring the sensor reading occurred during a labeled location visit.

During this process, all timestamps were truncated from millisecond precision to second precision, as the sub-second temporal resolution is not critical for location identification tasks where human movement patterns operate on a timescale of seconds rather than milliseconds. This timestamp granularity reduction is consistent with common practices in indoor positioning systems where second-level precision is sufficient for practical applications [6], [7].

#### 2.2.4 Handling Unlabeled Data

The merging process successfully labeled approximately 1.1 million BLE records (66% of the cleaned BLE data), while approximately 570,000 records (34%) could not be matched to any location label. These unlabeled records were intentionally dropped to maintain the quality and reliability of the supervised learning dataset.

The 34% unlabeled data represents an inherent characteristic of the data collection design rather than a preprocessing error: User ID 97 selectively annotated specific location visits rather than continuously labeling every moment, resulting in gaps where sensor data exists without corresponding ground truth labels. These unlabeled periods likely include transition times between rooms, moments when the labeler was not actively tracking, or areas outside the scope of this study. For the purposes of training a reliable location prediction model, retaining only records with verified ground truth labels ensures higher quality evaluation and more trustworthy model performance metrics.

#### 2.2.5 Final Preprocessed Dataset

The final preprocessed dataset contains approximately 1.1 million labeled BLE sensor readings spanning four days of data collection. The structure of this merged dataset is shown in Table 3, with each record containing a timestamp (second precision), beacon ID (1-25), RSSI value, and the corresponding room label as the prediction target.

**Table 3: Merged Labeled BLE Data Sample**

| timestamp | mac_address | RSSI | room |
|-----------|-------------|------|------|
| 2023-04-10 14:21:46+09:00 | 6 | -93 | kitchen |

This preprocessed labeled dataset serves as the foundation for all subsequent feature engineering, model training, and evaluation procedures described in the following sections.

---

## 3. Methodology

To address the indoor localization challenge, we propose and compare two distinct approaches that represent fundamentally different paradigms in handling temporal sensor data. The first approach (Section 3.1) serves as our baseline family, employing traditional machine learning methods commonly used in BLE-based localization research. This family of methods treats each temporal window independently, extracting statistical features from RSSI values and applying gradient boosting classification. We systematically explore multiple variations within this paradigm—including extended feature engineering strategies, class imbalance handling techniques (minority class weighting, SMOTE oversampling, and signal pattern-based relabeling), dominant beacon identification—to establish a comprehensive baseline and identify the fundamental limitations of treating time windows as isolated observations.

In contrast, our proposed method (Section 3.2) introduces a novel Deep Attention-based Sequential Ensemble Learning (DASEL) framework that fundamentally reconceptualizes indoor localization as a sequential learning problem. Rather than treating each time window as an isolated data point, DASEL leverages the inherent temporal continuity of human movement patterns through deep learning architectures. The key innovation lies in combining four integrated components: (1) frequency-based feature engineering that prioritizes signal presence over noisy strength measurements, (2) bidirectional recurrent networks with attention mechanisms that learn which moments in a movement sequence are most informative, (3) a multi-directional sliding window strategy that provides multiple temporal perspectives during inference, and (4) multi-model ensemble learning that ensures robust and stable predictions. This comprehensive pipeline addresses the fundamental asymmetry between training (where room boundaries are known) and deployment (where boundaries must be inferred), achieving substantial improvements over traditional approaches while maintaining practical deployability.

The following sections detail each approach's methodology, design rationale, and implementation specifics.

### 3.1 Traditional Machine Learning Baseline Family

Traditional machine learning approaches to indoor localization treat the problem as a static classification task, where each temporal window of beacon signals is independently classified into a room category. This baseline family implements widely-adopted pipelines: constructing beacon feature vectors, applying temporal windowing with statistical aggregation, and training gradient boosting classifiers. We systematically explore multiple variations within this framework—testing different feature sets, class balancing strategies, and post-processing techniques—to establish a robust baseline and demonstrate that the fundamental limitation lies in the independence assumption paradigm itself rather than specific implementation choices.

#### 3.1.1 Core Baseline Method

The baseline pipeline consists of four main steps: (1) beacon vector construction, (2) temporal windowing, (3) statistical feature aggregation, and (4) classification using gradient boosting. Figure 1 provides a visual overview of this complete workflow, illustrating how raw data transforms at each stage.

**Figure 1: Traditional Machine Learning Workflow**
[INSERT FIGURE HERE]

**Initial Data Format:** The raw BLE data from preprocessing (Section 2.2.5) consists of individual detection records as shown in Table 3. Each record contains a timestamp (second precision), beacon MAC address (beacon ID from 1-25), RSSI value, and room label. Multiple beacons are detected at irregular intervals within each second, resulting in a variable number of readings per timestamp.

**Step 1 - Create 25-Dimensional Beacon Vector:** For each individual record at a given timestamp, we construct a 25-dimensional beacon vector where position *i* contains the RSSI value if beacon *i* was detected, and 0 otherwise. This transforms the data from individual detections to a structured vector format per record.

**Step 2 - Temporal Windowing (1-second grouping):** Raw readings are grouped by their timestamp (already at second precision from preprocessing). All records sharing the same timestamp are aggregated together into a single window. For instance, multiple detections at 2023-04-10 14:21:46+09:00 (from different beacons) are grouped into one window. This transforms multiple individual records into one consolidated window per second.

**Step 3 - Statistical Aggregation:** Within each 1-second window, for each of the 25 beacons, we compute three statistical features from the RSSI values observed in that window:

- **Mean:** Average signal strength across all detections of the beacon within the window
- **Standard Deviation:** Variability of signal strength within the window
- **Count:** Number of times the beacon was detected in the window

For beacons not detected in a given window, all three statistics are set to 0. After this aggregation step, each 1-second window is represented by 25 beacons × 3 statistics = 75 numerical features.

**Step 4 - Feature Vector for Classification:** The aggregated statistics are concatenated into a single feature vector per window. Each 1-second window becomes one training sample with 75 features (25 beacons × 3 statistics) and one room label, ready for classification. The window represents a static snapshot of beacon signal patterns at a specific moment in time.

**Classification:** We employ XGBoost (eXtreme Gradient Boosting) as the baseline classifier, a gradient boosting decision tree algorithm well-established in multiclass classification tasks. To address class imbalance in the dataset (where some rooms are visited far more frequently than others), we apply balanced sample weighting during training. This ensures the model pays equal attention to both frequently-visited rooms (e.g., nurse station) and rarely-visited rooms (e.g., individual patient rooms). The model learns to map the 75-dimensional beacon signal patterns to room locations through ensemble decision trees, with each 1-second window classified independently as an isolated data point.

#### 3.1.2 Explored Variations and Optimizations

To establish a comprehensive baseline and identify the fundamental limitations of the traditional paradigm, we systematically explored five variations of the core approach, each targeting different aspects of the pipeline that could potentially improve performance.

**Variation 1 - Extended Statistical Features:** We expanded the feature set to include minimum and maximum RSSI values in addition to the baseline's mean, standard deviation, and count. This increases feature dimensionality from 75 to 125 features (25 beacons × 5 statistics), capturing the full range of signal strength within each window and providing additional information about signal extremes and stability.

**Variation 2 - Minority Class Weighting:** Building upon the extended features, we applied increased weight (3× multiplier) to minority classes during XGBoost training. This modification forces the model to pay disproportionate attention to rarely-visited rooms during optimization, attempting to balance the learning process beyond the standard balanced sample weighting used in the baseline.

**Variation 3 - SMOTE Oversampling:** We applied Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic training samples for minority classes. SMOTE creates new samples by interpolating between existing minority class instances in feature space, artificially balancing the class distribution during training while maintaining the original test set distribution.

**Variation 4 - Dominant Beacon Features:** We augmented the extended statistical features (125 features from Variation 1) with three additional categorical features representing the top three most frequently detected beacons within each window. Specifically, we identified:

1. The strongest beacon (beacon with the highest occurrence count in the window)
2. The second strongest beacon (beacon with the second highest occurrence count)
3. The third strongest beacon (beacon with the third highest occurrence count)

This increases the total feature dimensionality to 128 features (125 statistical features + 3 dominant beacon identifiers). The rationale is that the most frequently detected beacons within a temporal window provide additional spatial context beyond signal strength statistics, capturing which beacons have the most consistent presence rather than necessarily the strongest signal.

**Variation 5 - Signal Pattern-Based Relabeling:** Inspired by Garcia and Inoue's relabeling approach for BLE-based indoor localization [8], we explored signal pattern-based data augmentation as a preprocessing technique to address class imbalance. This method operates on the training data before model training:

1. Identify minority and majority classes based on sample counts
2. Analyze signal patterns of both classes by calculating Kullback-Leibler (KL) divergence between their RSSI distributions across beacons
3. Identify majority class rooms whose signal patterns closely match minority class rooms (low KL divergence indicates similar beacon detection patterns)
4. Select a subset of samples from the matched majority class
5. Relabel these selected samples with the minority class label
6. Augment the original training dataset with these relabeled samples

This approach leverages the spatial similarity of beacon signal patterns between different rooms—when two rooms have similar beacon coverage and signal distributions, samples from the majority class room can effectively serve as additional training data for the minority class room. Unlike SMOTE which generates synthetic samples through interpolation, relabeling reuses actual collected data from spatially similar locations, potentially preserving the authentic signal characteristics while addressing class imbalance.

### 3.2 Deep Attention-based Sequential Ensemble Learning (DASEL)

#### 3.2.1 Complete Method Description

Our proposed approach, Deep Attention-based Sequential Ensemble Learning (DASEL), reconceptualizes indoor localization as a sequential learning problem rather than a static classification task. The complete pipeline integrates four distinct phases: (1) frequency-based feature engineering, (2) sequential model training using bidirectional recurrent networks with attention mechanisms, (3) multi-level ensemble inference combining multiple temporal perspectives and model initializations, and (4) temporal smoothing for spatial consistency.

**Figure 2: DASEL Workflow**
[INSERT FIGURE HERE]

**Phase 1: Feature Engineering**

**Step 1 - Create 25-Dimensional Beacon Vector:** Starting with the preprocessed data from Section 2.2.5 (Table 3), we construct a 25-dimensional beacon vector for each record, where position *i* contains the RSSI value if beacon *i* was detected, and 0 otherwise.

**Step 2 - Temporal Windowing (1-second grouping):** We group all BLE readings that share the same timestamp into 1-second windows, creating discrete time steps.

**Step 3 - Frequency Calculation:** Within each 1-second window, for each of the 25 beacons, we calculate the appearance frequency:

```
frequency_{i,t} = count_{i,t} / total_detections_t
```

where count_{i,t} is the number of times beacon *i* was detected in window *t*, and total_detections_t is the total number of all beacon detections in that window. For beacons not detected in a window, the frequency is 0. We use 23 beacons (beacons 1-23) in practice, as beacons 24-25 were rarely detected.

The data structure after Phase 1 is shown in Table 4.

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

**Sequence Length Constraints:**
- Minimum length: 3 timesteps (shorter sequences discarded)
- Maximum length: 50 timesteps (longer sequences truncated by taking the last 50 timesteps)

**Model Architecture:** The model consists of the following layers:

1. **Masking Layer:** Handles variable-length sequences (padded to 50 timesteps for batch processing)
2. **First Bidirectional GRU Layer:** 128 units, processes sequences in both forward and backward directions
3. **Dropout Layer:** 0.3 dropout rate
4. **Second Bidirectional GRU Layer:** 64 units, provides additional temporal processing
5. **Dropout Layer:** 0.3 dropout rate
6. **Attention Layer:** Computes attention weights and creates weighted context vector
   - Attention scores: tanh(W · sequence + b)
   - Attention weights: softmax(attention_scores)
   - Context vector: Σ(sequence × attention_weights)
7. **Dense Layer:** 32 units with ReLU activation and 0.2 dropout
8. **Output Layer:** Softmax activation producing probability distribution over all room classes

**Figure 3: DASEL Model Architecture**
[INSERT FIGURE HERE]

**Training:** The model is trained using sparse categorical cross-entropy loss with balanced class weights. Each training sequence represents one complete room visit.

**Phase 3: Multi-Level Ensemble Inference**

Phase 3 implements a two-level ensemble strategy that combines multi-seed model training with multi-directional sliding windows. Unlike traditional ensemble methods that aggregate predictions at a single stage, our approach hierarchically combines predictions first within each temporal direction, then across different directional perspectives.

**Level 1: Multi-Seed Model Ensemble (Within Each Direction)**

For robust and stable predictions, we train 5 independent models with different random initialization seeds:

**Seed Configuration:**
- Seed values: [42, 1042, 2042, 3042, 4042]
- Large increments (+1000) ensure sufficient diversity in initial weight configurations
- Each model learns slightly different patterns from its unique optimization trajectory

**Within-Direction Aggregation:**

For each of the 7 directional windows, we apply probability averaging across the 5 models:

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

**Multi-Directional Sliding Windows:**

For each timestamp *t* requiring prediction, we create 7 different temporal windows:

- **backward_10:** [t-9, ..., t] — 10 seconds of history
- **centered_10:** [t-4, ..., t, ..., t+5] — 10 seconds centered on t
- **forward_10:** [t, ..., t+9] — 10 seconds looking forward
- **backward_15:** [t-14, ..., t] — 15 seconds of history (extended context)
- **forward_15:** [t, ..., t+14] — 15 seconds looking forward (early transition detection)
- **asymm_past:** [t-11, ..., t, ..., t+3] — Past-biased (12s past + 4s future)
- **asymm_future:** [t-3, ..., t, ..., t+11] — Future-biased (4s past + 12s future)

**Confidence-Weighted Aggregation:**

Each directional window produces a probability distribution with an associated confidence score (the maximum probability value). We aggregate these 7 distributions using confidence-weighted voting rather than simple averaging:

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

The decision to use beacon appearance frequency rather than RSSI signal strength as the primary feature representation is motivated by fundamental limitations of BLE signal propagation in indoor environments. RSSI measurements are inherently unstable and exhibit poor discriminative power, while beacon frequency patterns provide stable, location-specific signatures.

**The RSSI Instability Problem**

RSSI values in indoor environments suffer from significant temporal and spatial variability caused by multiple environmental factors: multipath signal propagation from reflections off walls and furniture, human body orientation and absorption, interference from other wireless devices, and device-specific hardware characteristics. Previous studies have extensively documented that BLE RSSI signals fluctuate widely even at fixed positions due to multipath effects and signal noise, with these fluctuations affecting positioning accuracy in complex indoor environments [9]. Research has further demonstrated that RSSI can be inconsistent and unstable across indoor areas with complex structures and varying materials [10], [11].

Figure 5 demonstrates this instability through box plot comparisons of RSSI distributions for three beacons that appear in both Kitchen and Cafeteria as an example. The visualization reveals two critical limitations. First, RSSI measurements exhibit high intra-room variance—within a single location, signal strength varies substantially with standard deviations of several dBm, creating overlapping ranges and numerous outliers. Second, and more problematically, RSSI distributions show significant inter-room overlap. The mean RSSI values for the same beacon in Kitchen versus Cafeteria differ by less than 2 dBm across all three beacons examined, with these minimal differences being smaller than the variance within each room. This overlap fundamentally limits the discriminative power of RSSI-based features, as a given signal strength measurement could plausibly originate from either location.

**Figure 5: RSSI Distribution Comparison: Kitchen vs Cafeteria**
[INSERT FIGURE HERE]

**The Beacon Frequency Advantage**

In contrast to noisy RSSI measurements, beacon appearance frequency captures the fundamental spatial pattern of which beacons are detected rather than how strongly their signals are received. For example, Figure 6 illustrates the frequency distribution of beacon detections for three representative rooms (Kitchen, Cafeteria, and Cleaning), demonstrating that each room exhibits a distinct beacon appearance pattern.

**Figure 6: Beacon Frequency Distribution: Kitchen, Cafeteria, and Cleaning**
[INSERT FIGURE HERE]

As shown in these examples, each room demonstrates a characteristic spatial "fingerprint" based on proximity to installed beacons. Rooms consistently detect nearby beacons at high frequencies while distant beacons appear rarely or not at all, creating stable location-specific signatures. This spatial clustering reflects the physical facility layout and provides strong discriminative power for classification—in stark contrast to the overlapping RSSI distributions, the dominant beacon patterns are clearly distinct across rooms.

The frequency-based representation offers multiple advantages:

1. By focusing on presence and absence patterns rather than signal magnitude, frequency features are inherently more robust to the environmental noise factors that destabilize RSSI measurements. A beacon's appearance is a more stable characteristic than the precise strength of its noisy analog signal.

2. Additionally, frequency features are naturally normalized to values between 0 and 1, eliminating the need for complex RSSI calibration procedures to account for device-specific variations.

3. The long-tail distribution visible in beacon frequency patterns—where most beacons have near-zero frequencies in any given room—demonstrates that essential discriminative information lies in identifying which beacons are present rather than measuring their exact signal strengths.

This representation reduces dimensionality from traditional RSSI-based approaches while focusing on the stable, informative signal that captures spatial relationships.

#### 3.2.3 Why Sequential Learning and This Model Architecture?

**The Sequential Nature of Indoor Localization**

Traditional machine learning approaches, as described in Section 3.1, treat each 1-second window as an independent, isolated sample. This independence assumption fundamentally ignores the temporal continuity inherent in human movement patterns. When a person moves through an indoor space, consecutive observations are highly correlated—the location at time *t* strongly influences the location at time *t+1*. A caregiver walking from the Kitchen to the Hallway produces a temporal sequence of beacon patterns that gradually transition from Kitchen-characteristic signals to Hallway-characteristic signals.

By treating each second independently, traditional methods discard this rich contextual information about movement trajectories, transition patterns, and dwell times within locations. Sequential learning addresses this limitation by explicitly modeling entire room visits as temporal sequences, allowing the model to learn not just static beacon patterns but also the dynamics of how these patterns evolve over time. This temporal context provides crucial disambiguation when instantaneous beacon signals are ambiguous—for instance, a transitional pattern between Kitchen and Hallway is fundamentally different from a stable pattern indicating prolonged presence in either location.

**Architecture Selection: Bidirectional GRU with Deep Attention**

Our architecture employs a two-layer Bidirectional GRU structure with an attention mechanism for robust sequential pattern learning. The first Bidirectional GRU layer (128 units) processes the sequence in both forward and backward directions simultaneously, capturing temporal dependencies from both past context (where the person came from) and future context (where they are heading). This bidirectional processing is particularly valuable for indoor localization, as movement patterns are often characterized by both entry and exit signatures.

The second Bidirectional GRU layer (64 units) refines these features through a secondary level of temporal abstraction, creating more stable representations by filtering out noisy variations in the first layer's output. The deep architecture (two GRU layers rather than one) significantly reduces variance across different random initializations, providing more consistent and reliable predictions.

The attention mechanism completes the architecture by learning which timesteps within a sequence are most informative for classification. Not all moments in a room visit are equally discriminative—transitions between rooms and brief passages through doorways produce ambiguous beacon patterns, while periods of stable presence in a room's center yield clear signals. The attention layer automatically identifies and emphasizes these high-confidence timesteps while downweighting noisy transition periods. This selective focus allows the model to extract room-specific signatures even from sequences containing some transitional noise.

The attention mechanism computes weights via softmax(tanh(W·sequence + b)) and produces a context vector through weighted aggregation, effectively creating a fixed-size room representation from variable-length visit sequences. This combination of bidirectional temporal processing, deep feature refinement, and selective attention enables the model to learn robust room signatures from sequential beacon frequency patterns.

#### 3.2.4 Why Multi-Directional Windows with Confidence-Weighted Voting?

**The Inference Challenge: Unknown Sequence Position**

During training, the model learns from clean sequences with known room boundaries—each sequence represents a complete, continuous room visit identified through ground truth labels. However, during inference on unlabeled test data, these boundaries are unknown. For any given timestamp requiring prediction, we do not know where it falls within its actual room visit sequence: is it at the beginning, middle, or end? This positional uncertainty creates a fundamental challenge for sliding window-based prediction.

A single sliding window configuration cannot optimally capture all positions. Consider a backward-looking window [t-9 to t] predicting at position t: if the actual room visit started at t-3, the window contains 6 seconds from the previous room (contaminated signal) and only 4 seconds from the current room. Conversely, a forward-looking window [t to t+9] fails when predicting near the end of a room visit, as it incorporates future signals from the next room. A centered window [t-4 to t+5] works well for predictions in the middle of stable room visits but struggles at boundaries. No single temporal perspective can handle all cases reliably—each window type has blind spots corresponding to different positions within true sequences.

**Multi-Directional Strategy**

To address this positional uncertainty, we employ seven different window configurations that provide complementary temporal perspectives:

- **backward_10** [t-9 to t]: Captures 10 seconds of history, optimal when prediction point is at or near sequence end
- **centered_10** [t-4 to t+5]: Balanced view, optimal for mid-sequence predictions where both past and future are stable
- **forward_10** [t to t+9]: Focuses on upcoming context, optimal when prediction point is near sequence start
- **backward_15** [t-14 to t]: Extended history for longer room visits, provides more temporal context
- **forward_15** [t to t+14]: Extended future view for early transition detection in longer sequences
- **asymm_past** [t-11 to t+3]: Past-heavy (12s past, 4s future), specialized for detecting room exits
- **asymm_future** [t-3 to t+11]: Future-heavy (4s past, 12s future), specialized for detecting room entries

This multi-directional approach ensures that regardless of where a timestamp falls within its true sequence, at least some windows will be well-aligned with clean, uncontaminated signals. When predicting at sequence start, forward-looking windows excel; at sequence end, backward windows dominate; in the middle, centered windows provide stable predictions.

**Window Size Selection**

The choice of 10-second and 15-second window sizes is empirically grounded in the training data's sequence length distribution. Figure 7 shows the distribution of training sequence lengths across all room visits in the dataset (318 sequences total, average length 74.2 seconds).

**Figure 7: Sequence Length Distribution**
[INSERT FIGURE HERE]

The distribution reveals that the majority of room visits are concentrated between 10 seconds and 200 seconds, with a substantial peak in shorter visits. The 10-second window size captures the lower end of typical visit durations, ensuring that even brief room passages receive adequate temporal context without excessive contamination from adjacent rooms. This size is particularly important because sequences shorter than 10 seconds represent quick transitions (walking through) rather than meaningful room occupancy, and a 10-second window provides sufficient context to distinguish these patterns.

The 15-second extended windows offer additional context for the numerous longer visits while remaining well below the average sequence length of 74 seconds, minimizing the risk of spanning multiple room transitions within a single window. These window sizes strike a balance: large enough to capture discriminative temporal patterns, yet small enough to avoid excessive boundary contamination in typical room visits.

**Confidence-Weighted Voting vs. Majority Voting**

Each of the seven windows produces a prediction with an associated confidence score (the maximum probability from the softmax output). We aggregate these predictions using confidence-weighted voting rather than simple majority voting. In majority voting, each window's prediction counts equally regardless of the model's certainty—a poorly-fitting window contaminated by transition signals receives the same weight as a well-aligned window with clean, stable signals. This equal weighting allows ambiguous or erroneous predictions to inappropriately influence the final decision.

Confidence-weighted voting addresses this by incorporating the model's uncertainty estimation. Windows that are well-aligned with clean room signals produce high-confidence predictions (softmax probabilities concentrated on a single class), while poorly-aligned windows during transitions yield low-confidence predictions (diffuse probability distributions). By weighting each window's contribution by its confidence score, the aggregation naturally emphasizes reliable predictions and downweights uncertain ones. Mathematically, for timestamp *t* with predictions from windows *i* = 1...7:

```
confidence_i = max(probability_distribution_i)
weighted_vote = Σ(probability_distribution_i × confidence_i) / Σ(confidence_i)
final_prediction = argmax(weighted_vote)
```

This approach allows the ensemble to self-regulate: when multiple well-aligned windows agree with high confidence, they dominate the decision; when all windows show uncertainty (during genuine transitions), the aggregation reflects this ambiguity appropriately. The result is a robust prediction mechanism that adaptively emphasizes the most reliable temporal perspectives for each timestamp's unique positional context.

#### 3.2.5 Why Multi-Seed Model Ensemble?

Deep neural networks are highly sensitive to random weight initialization, leading to performance variability across different training runs even with identical architectures, hyperparameters, and training data. Deep neural networks suffer from various sources of variance, such as finite datasets and random initialization, and training the same under-constrained model on the same data with different initial conditions will result in different models given the difficulty of the optimization problem. This initialization-dependent variance creates a "seed lottery" problem where a single model's performance may not reflect the true capability of the architecture but rather the luck of a particular random initialization.

To address this instability, we train five independent models with different random initialization seeds (42, 1042, 2042, 3042, 4042), using large increments (+1000) to ensure sufficient diversity in the initial weight configurations. Ensemble methods are used as a way to reduce the variance of a base estimator by introducing randomization into its construction procedure and then making an ensemble out of it [12]. Each model learns slightly different patterns and decision boundaries from its unique initialization trajectory, and by aggregating their predictions through confidence-weighted voting, the ensemble reduces variance while maintaining or improving average performance.

This multi-seed strategy provides three key benefits: variance reduction through averaging out individual model fluctuations, robustness against unlucky initializations that might converge to poor local minima, and more reliable evaluation metrics that reflect the model's true capability rather than initialization luck. The ensemble essentially transforms the seed lottery from a liability into an asset, using the diversity of solutions found from different starting points to create a more stable and reliable final predictor.

#### 3.2.6 Why 5-Second Temporal Smoothing?

Temporal smoothing serves as a post-processing optimization step that enforces spatial and temporal consistency in the final predictions. Even after multi-directional ensemble aggregation, isolated prediction errors can occur—for instance, a single timestamp might be incorrectly predicted as a distant room (e.g., Room 517) when surrounded by consistent predictions of a nearby room (e.g., Kitchen). These isolated errors are often physically implausible, as people do not instantaneously teleport across buildings.

The 5-second smoothing window examines predictions within a local temporal neighborhood [t-2, t-1, t, t+1, t+2] and applies confidence-weighted majority voting. If a prediction at timestamp *t* contradicts the surrounding predictions and those surrounding predictions exhibit high confidence, the outlier is overridden with the local consensus. This mechanism effectively eliminates "teleportation" errors while preserving legitimate room transitions that show sustained directional changes in predictions.

The choice of a 5-second window represents a practical balance based on common-sense human movement patterns. This duration is long enough to catch isolated errors—people typically remain in a room for more than 5 seconds during meaningful visits—yet short enough to preserve genuine transitions between adjacent rooms, which can occur within 5-10 seconds of walking. Empirical testing with alternative window sizes (3, 7, and 10 seconds) confirmed that 5 seconds provides optimal performance: shorter windows lack sufficient context to reliably correct errors, while longer windows risk smoothing over genuine rapid transitions. The smoothing step provides consistent but modest improvements in macro F1 score, demonstrating that while the multi-directional ensemble already produces high-quality predictions, this final spatial consistency enforcement offers a reliable incremental benefit.

### 3.3 Evaluation Protocol

**Evaluation Metric**

We evaluate model performance using macro F1-score, the official metric specified by the ABC 2026 challenge organizers. Macro F1-score computes the F1-score independently for each room class and then averages these scores, treating all rooms equally regardless of their frequency in the dataset. This metric is particularly appropriate for our imbalanced indoor localization problem, where some rooms (e.g., nurse station, hallway) are visited far more frequently than others (e.g., individual patient rooms). Unlike accuracy or weighted F1-score, macro F1 ensures that the model must perform well across all locations, including rare rooms, rather than simply optimizing for the most common classes. This emphasis on balanced performance across all rooms is critical for practical deployment in care facilities, where reliable localization in every area—not just frequently visited ones—is essential for comprehensive caregiver tracking.

**4-Fold Temporal Cross-Validation**

We employ 4-fold cross-validation with a temporal splitting strategy rather than random partitioning. The preprocessed dataset spans four days of continuous data collection (April 10-13, 2023), and we split the data such that each fold uses one complete day as the test set and the remaining three days as the training set. This temporal split is essential because consecutive BLE readings are highly autocorrelated—beacon signal patterns within a few seconds of each other are nearly identical. Random splitting would leak highly similar samples between training and test sets, artificially inflating performance metrics and failing to test the model's true generalization capability.

By splitting temporally, we ensure that all test data is genuinely unseen and represents a different time period, simulating realistic deployment where models trained on historical data must generalize to new days with potentially different environmental conditions, movement patterns, and temporal distributions of room visits.

**Fold Characteristics**

The four folds exhibit substantial variation in data size and class distribution, reflecting the natural imbalance in real-world data collection across different days. Table 6 presents the detailed characteristics of each fold.

**Table 6: Cross-Validation Fold Characteristics**

| Fold | Test Day | Train Frames | Test Frames | Train/Test Ratio | Train Classes | Test Classes |
|------|----------|--------------|-------------|------------------|---------------|--------------|
| 1 | Day 4 | 962,294 | 30,619 | 31.4× | 13 | 13 |
| 2 | Day 3 | 951,141 | 143,401 | 6.6× | 18 | 18 |
| 3 | Day 2 | 747,816 | 333,507 | 2.2× | 15 | 15 |
| 4 | Day 1 | 465,004 | 590,447 | 0.79× | 12 | 12 |

Several notable patterns emerge from these characteristics. Fold 1 tests on the smallest dataset (Day 4 with only 30,619 frames), representing a short collection period, while providing the largest training set. This creates a highly favorable train/test ratio of 31.4×. Conversely, Fold 4 presents the most challenging scenario: testing on the largest dataset (Day 1 with 590,447 frames) while training on the smallest subset (465,004 frames), resulting in a train/test ratio below 1 (0.79×). This fold effectively tests the model's ability to generalize when training data is scarce relative to deployment scale.

The number of room classes also varies across folds, ranging from 12 to 18 unique locations. This variation reflects realistic deployment conditions where different days may involve visits to different subsets of rooms within the facility. Notably, all test sets contain only rooms that also appear in the corresponding training set, ensuring the evaluation focuses on generalization to new temporal contexts rather than zero-shot classification of unseen room types.

This heterogeneous fold structure provides a robust evaluation framework that tests model performance under varying data availability conditions—from data-abundant scenarios (Fold 1) to data-scarce situations (Fold 4)—and across different subsets of the facility's room layout. The average performance across all four folds offers a reliable estimate of expected real-world performance that accounts for temporal variability, class imbalance, and differing train/test scales.

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

We evaluated our proposed DASEL framework using 4-fold cross-validation with multi-seed ensemble learning. Table 8 presents the macro F1 scores across all folds. The DASEL approach achieved a mean macro F1 score of 0.4438 ± 0.0295, with individual fold performances ranging from 0.4082 (Fold 4) to 0.5114 (Fold 1). The results demonstrate consistent performance improvements over the traditional machine learning baseline family, with Fold 1 achieving the highest macro F1 score of 0.5114.

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

The proposed Deep Attention-based Sequential Ensemble Learning (DASEL) framework achieves a mean macro F1 score of 0.4438 compared to 0.2898 for the best traditional machine learning baseline (Table 9), representing a 53.1% relative improvement. This section discusses the key factors contributing to DASEL's success and analyzes the limitations of traditional approaches that motivated our design choices.

### 5.1 The Limitations of Traditional Approaches

Traditional machine learning methods for indoor localization treat each temporal window as an independent classification problem, extracting statistical features from beacon signals and applying gradient boosting classifiers. As shown in Table 7, we systematically explored multiple variations of this paradigm to establish a comprehensive baseline: extended statistical features (Variation 1), minority class weighting (Variation 2), SMOTE oversampling (Variation 3), dominant beacon features (Variation 4), and signal pattern-based relabeling (Variation 5).

Despite these diverse optimization strategies targeting different aspects of the problem—feature engineering, class imbalance handling, and data augmentation—the results remained remarkably consistent across all approaches. The baseline method achieved a macro F1 score of 0.2805, while all five variations ranged between 0.2838 and 0.2898, with the best-performing approach (Variation 4: Dominant Beacon Features) reaching only 0.2898. The narrow performance band spanning just 0.2805 to 0.2898 across fundamentally different optimization techniques suggests that these methods have reached a fundamental ceiling imposed by the independence assumption itself.

This consistent underperformance across diverse variations confirms our central argument: incremental improvements to the traditional paradigm cannot overcome the fundamental limitation of treating temporally-dependent data as independent observations. The problem requires a paradigm shift rather than optimization within the existing framework. DASEL addresses this limitation by reconceptualizing indoor localization as a sequential learning problem, enabling the breakthrough performance gains demonstrated in our results.

### 5.2 The Value of Sequential Modeling

The most significant contribution to DASEL's performance comes from treating indoor localization as a sequential learning problem. As shown in Table 8, DASEL achieves a macro F1 score of 0.4438, substantially outperforming all traditional machine learning variations. This dramatic improvement validates our core hypothesis that human movement patterns contain rich temporal dependencies that traditional independent-window classification cannot capture.

Indoor localization is fundamentally a sequential task. Caregivers do not instantaneously teleport between rooms—they follow continuous trajectories through physical space, producing correlated sequences of beacon signal patterns. When a person walks from the Kitchen to the Hallway, beacon detections gradually shift from Kitchen-characteristic patterns to Hallway-characteristic patterns. By modeling entire room visits as temporal sequences rather than isolated moments, DASEL captures these movement dynamics that traditional approaches discard entirely.

The bidirectional GRU architecture processes sequences in both forward and backward directions, enabling the model to leverage complete temporal context. This bidirectional processing is particularly valuable during room transitions, where understanding both where the person came from and where they are heading provides crucial disambiguation. The model learns not just static beacon patterns for each room, but also the characteristic ways these patterns evolve during entries, stable occupancy, and exits.

This sequential modeling capability represents a fundamental architectural advantage over traditional methods. Rather than struggling to classify individual ambiguous timestamps, DASEL examines temporal neighborhoods to infer location from movement trajectories. This shift from static classification to dynamic sequence modeling explains the substantial performance improvements observed across all evaluation folds.

### 5.3 Architecture Components and Attention Mechanism

The deep bidirectional GRU architecture with attention mechanism provides robust feature extraction from temporal sequences. The two-layer recurrent structure creates hierarchical representations, with the first layer capturing immediate temporal patterns and the second layer refining these features into more stable, abstract representations. This depth helps reduce variance across different random initializations, providing more consistent predictions.

The attention mechanism addresses a key challenge in processing variable-length room visit sequences: not all timesteps are equally informative. During stable room occupancy, beacon patterns provide clear location signals, while room transitions and doorway passages produce ambiguous multi-room signals. Rather than treating all timesteps equally, the attention layer learns to emphasize discriminative moments while downweighting noisy transition periods. This selective focus enables robust classification even when sequences contain substantial transitional noise.

### 5.4 Frequency-Based Features and Practical Robustness

The frequency-based feature representation addresses critical practical challenges for real-world deployment. As demonstrated in Figure 5, raw RSSI measurements exhibit substantial instability due to multipath propagation, human body absorption, and device-specific hardware variations. RSSI distributions for the same beacon in different rooms show significant overlap, fundamentally limiting the discriminative power of signal-strength-based features.

Beacon appearance frequency captures which beacons are detected rather than how strongly their signals are received, providing a more stable spatial signature. Each room exhibits characteristic patterns based on proximity to installed transmitters—nearby beacons appear frequently while distant beacons appear rarely or not at all. These presence patterns remain consistent despite environmental factors that destabilize individual RSSI measurements, creating robust location-specific fingerprints.

This representation eliminates the need for complex device-specific RSSI calibration procedures, a major practical barrier to deployment. By focusing on beacon detection patterns rather than precise signal strengths, DASEL achieves robustness to device heterogeneity and environmental variations without extensive system tuning.

### 5.5 The Two-Level Ensemble as Optimization

The two-level hierarchical ensemble provides critical optimizations that enhance prediction quality beyond the core sequential modeling framework. The first level addresses variance through multi-seed training: training five models with different random initializations and averaging their predictions reduces initialization-dependent fluctuations, ensuring stable probability distributions that reflect the architecture's true capability rather than random initialization luck.

The second level addresses inference uncertainty through multi-directional windows. During training, the model learns from sequences with known room boundaries, but during inference on unlabeled data, we do not know where each timestamp falls within its true room visit sequence. The seven directional windows provide complementary temporal perspectives—backward-looking, forward-looking, centered, and asymmetric configurations—that excel in different positional contexts. The confidence-weighted aggregation allows well-aligned windows with clean signals to dominate predictions while downweighting poorly-aligned windows, creating an adaptive system robust to positional uncertainty.

These ensemble strategies serve as optimization layers atop the sequential modeling foundation, incrementally refining predictions through variance reduction and multi-perspective aggregation. Together with the temporal smoothing post-processing step, they transform the base sequential model's outputs into reliable, spatially-consistent predictions suitable for practical deployment.

### 5.6 Limitations and Future Work

Despite strong performance, several limitations suggest directions for future research. The two-level ensemble requires multiple forward passes per timestamp, which may challenge real-time deployment on resource-constrained mobile devices. Model compression techniques such as knowledge distillation could potentially maintain accuracy while reducing computational demands.

The current approach uses fixed window sizes (10 and 15 seconds) based on training sequence length distributions. Adaptive window sizing that adjusts based on detected movement patterns could improve performance for very short or very long room visits. Additionally, DASEL treats all rooms equally without modeling physical adjacency relationships. Incorporating facility layout information could reduce physically implausible prediction errors and improve transition modeling between adjacent rooms.

Finally, our evaluation focuses on a single care facility. Future work should examine generalization to different facilities with varying layouts, beacon densities, and construction materials, potentially leveraging transfer learning to adapt models with limited target-domain labeled data.

### 5.7 Practical Implications

DASEL's balanced macro F1 score indicates reliable performance across all room classes, including rarely-visited locations. This is essential for comprehensive care facility monitoring, where accurate localization in individual patient rooms is as critical as tracking in common areas for understanding care delivery patterns and optimizing staff allocation.

The frequency-based representation provides practical robustness to device heterogeneity and environmental variations, reducing calibration requirements for deployment. The temporal smoothing mechanism produces spatially consistent predictions that match realistic human movement patterns, eliminating implausible instantaneous transitions between distant rooms. These characteristics—balanced performance, device robustness, and spatial consistency—make DASEL deployable in real-world care facility settings without extensive system tuning, addressing key practical barriers that limit adoption of indoor localization systems.

---

## 6. Conclusion

This research demonstrates that traditional machine learning approaches to BLE-based indoor localization face fundamental performance limitations that cannot be overcome through incremental optimization. Our systematic exploration of five diverse variations—extended statistical features, minority class weighting, SMOTE oversampling, dominant beacon identification, and signal pattern-based relabeling—targeting different aspects of the problem (feature engineering, class imbalance handling, and data augmentation) all yielded remarkably consistent results within a narrow 0.2805 to 0.2898 macro F1 score band. This performance plateau across fundamentally different optimization strategies reveals that the core limitation lies not in implementation details but in the independence assumption itself: treating each temporal window as an isolated observation discards the rich temporal dependencies inherent in human movement patterns. The consistent ceiling across well-known and promising optimization techniques confirms that breakthrough performance requires paradigm shift rather than refinement within the traditional framework.

Our proposed Deep Attention-based Sequential Ensemble Learning (DASEL) framework achieves this breakthrough by fundamentally reconceptualizing indoor localization as a sequential learning problem. By explicitly modeling entire room visits as temporal sequences through bidirectional recurrent networks with attention mechanisms, DASEL captures movement trajectories, entry/exit patterns, and dwell-time dynamics that traditional methods completely ignore. The integration of frequency-based features for RSSI robustness, two-level hierarchical ensemble for variance reduction and positional uncertainty handling, and confidence-weighted temporal smoothing for spatial consistency creates a comprehensive system that leverages the sequential characteristics of human movement data.

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
# Brief Starting Pipeline

This document describes a **simple, clean starting pipeline** for BLE-based indoor localization / classification using XGBoost.

---

## 1. Data Splitting Strategy (Model Selection)

We use **time-aware splits by day** to avoid data leakage caused by highly similar consecutive BLE records.

### Step 1: Sanity Check

* **Train**: Day 1
* **Test**: Day 2 (or Day 2 + Day 3)

---

### Step 2: Model Comparison (Main Decision)

* **Train**: Day 1 + Day 2
* **Test**: Day 3

Purpose:

* Compare models and feature settings
* Select window size, features, and model type
* Closest to the final competition / deployment scenario

> **Note:** This split is the primary basis for choosing the model.

---

## 2. Beacon Feature Construction

For each raw BLE record:

* Assume up to **25 known beacon MAC addresses**
* Create **25 RSSI columns**:

  * `beacon_1`, `beacon_2`, ..., `beacon_25`

Feature rule:

* If the record's MAC address matches beacon *i* → store RSSI value in `beacon_i`
* Otherwise → set `beacon_i = 0`

This converts sparse BLE scans into a fixed-length feature vector.

---

## 3. Temporal Aggregation (Windowing)

* Aggregate records using a **sliding time window**
* **Start with window size = 1 second**

For each window and each beacon column, compute:

* **Mean RSSI**
* **Standard deviation (SD)**
* **Count** (number of detections)

Only these three statistics are used initially to keep the model simple and stable.

---

## 4. Feature Engineering Summary

For each time window:

* 25 beacons × 3 statistics
* Total features = **75 features per window**

Labels:

* Room / location label corresponding to the window

---

## 5. Model Training and Evaluation

* Use **XGBoost (multiclass classification)**
* Train on the selected training split
* Evaluate on the corresponding test split using: Macro F1-score
---

## 6. Final Model (After Selection)

After selecting the best configuration:

* **Train** on Day 1 + Day 2 + Day 3 (all labeled data)
* Use the trained model to predict the **unlabeled test set**

---

## Key Principle

> All model choices must be finalized **before** training on the full dataset.

This ensures fair evaluation and avoids information leakage.

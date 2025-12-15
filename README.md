# ABC 2026 Challenge - Activity & Location Recognition

This year's challenge contains 2 problems:
- **Location Prediction**: Predict location based on labeled RSSI (Received Signal Strength Indicator) values from Bluetooth beacon signals
- **Activity Prediction**: Predict activity based on recorded sensor features (data has not been released yet)

---

## Repository Structure
```
ABC2026/
├── Dataset/                          # Original raw data
│   ├── BLE Data/                     # Raw BLE sensor CSV files
│   ├── 5f_label_loc_train.csv        # Location labels
│   └── 5th floor map.png             # Facility floor plan
├── data_prep/                        # Data cleaning notebooks
│   ├── label_data_cleaning.ipynb     # Label dataset preprocessing
│   ├── ble_data_merging.ipynb        # Merge all BLE CSV files
│   ├── ble_data_cleaning.ipynb       # BLE data preprocessing
│   └── add_label_ble_data.ipynb      # Timestamp-based labeling of BLE data
├── cleaned_dataset/                  # Processed data outputs
│   ├── cleaned_label_loc.csv         # Cleaned location labels
│   ├── merged_ble_data.csv           # All BLE data merged into one file
│   ├── cleaned_ble_data.csv          # Final cleaned BLE dataset
│   └── labelled_ble_data.csv         # BLE data with room labels
├── analysis/                         # Exploratory data analysis
│   ├── basic_analysis.ipynb          # Analysis of individual label and BLE files
│   └── labelled_ble_data_analysis.ipynb  # Analysis of labeled BLE dataset
├── .gitignore                        # Git ignore rules for large files
└── README.md                         # This file
```

---

## Location Prediction Problem

### Context

The data collection setup involves two people performing complementary tasks:

1. **User ID 90 (Data Collector)**: A caregiver who moves around the 5th floor of a care facility
   - Carries a mobile phone with an app that continuously records RSSI values from nearby Bluetooth beacons
   - The app logs beacon signals at each timestamp as they move through different locations
   
2. **User ID 97 (Labeler)**: A person who tracks User 90's movements and annotates their location
   - Records which room/location User 90 was in during specific time ranges
   - Creates ground truth labels by noting the start and end timestamps for each location visit

3. **Infrastructure**: Multiple Bluetooth Low Energy (BLE) beacons are installed throughout the 5th floor
   - Each beacon continuously transmits signals
   - The caregiver's phone detects these signals and records their strength (RSSI)
   - Different beacon signal patterns correspond to different locations

### Official Description

From the challenge organizers:

> Data gathering in care facility (5th floor)
> - Beacons installed in facility as transmitter
> - Mobile phone with app carried by caregiver acts as receiver
> - Every location visited, RSSI is detected by mobile app
> - Accelerometer data is collected from the application on the mobile phone of the caregiver
> 
> User IDs used in Mobile App, each user carries a different phone:
> - User ID 90: for 5th floor caregiver (collects sensor data - location and accelerometer)
> - User ID 97: for 5th location labeling

---

## Provided Materials

### 1. BLE Data (Dataset/BLE Data/)

A collection of CSV files containing raw RSSI sensor readings. These files may have originally been a single file split into smaller chunks for easier handling.

**File naming pattern**: userbleid_90_[timestamp]_[id].csv

**Format** (no header):
```
user_id, timestamp, name, mac_address, rssi, power
```

**Column descriptions**:
- user_id: Always 90 (the caregiver collecting data)
- timestamp: When the signal was detected (format: YYYY-MM-DDTHH:MM:SS.mmm+TZTZ)
- name: Appears to be null/empty in the data
- mac_address: Unique identifier of the beacon (e.g., FD:07:0E:D5:28:AE)
- rssi: Signal strength in dBm (typically negative values; closer to 0 = stronger signal = closer proximity)
- power: Additional signal information (often appears as -2147483648, likely a placeholder/null value)

**Example record**:
```
90,2023-04-10T10:22:55.589+0900,null,FD:07:0E:D5:28:AE,-75,-2147483648
```

### 2. Location Labels (Dataset/5f_label_loc_train.csv)

Ground truth labels indicating which room User 90 was in during specific time ranges.

**Format**:
```
,Unnamed: 0,activity,started_at,finished_at,deleted_at,updated_at,user_id,user,room,floor
```

**Column descriptions**:
- (first column): Row index (can be ignored)
- Unnamed: 0: Another index column (can be ignored)
- activity: Label type (appears to be "Location" for location labeling tasks)
- started_at: Timestamp when the person entered this location
- finished_at: Timestamp when the person left this location
- deleted_at: Database field for soft deletes (typically empty, can be ignored)
- updated_at: When this label record was last updated (can be ignored for modeling)
- user_id: ID of the labeller, either 97 or 91 here, but mainly 97 here
- user: Username or identifier string (e.g., "5th-location")
- room: **[TARGET LABEL]** The room/location name (e.g., "kitchen", "cafeteria", "nurse station", "hallway", "523")
- floor: Floor identifier (should be "5th" or "5f" for all records in this dataset)

**Example record**:
```
170,170,Location,2023-04-10 14:21:46+09:00,2023-04-10 14:21:50+09:00,,2023-04-10 05:22:02 UTC,97,5th-location,kitchen,5th
```

This indicates: "Between 14:21:46 and 14:21:50 on April 10, 2023, the caregiver was in the kitchen on the 5th floor."

### 3. Facility Map (Dataset/5th floor map.png)

A floor plan showing:
- Layout of rooms on the 5th floor
- Locations where beacons are installed
- Spatial relationships between different locations

### 4. Tutorial Notebooks

Two Jupyter notebooks provided by the challenge organizers:
1. **Processing the Location Label**: Guide on how to parse and use the label CSV file
2. **Checking the Location Data**: Guide on preprocessing BLE data

---

## Data Cleaning (data_prep/)

The data preparation process consists of four main steps:

### 1. Label Data Cleaning (`label_data_cleaning.ipynb`)

Starting with the raw label file `5f_label_loc_train.csv` (1,334 records), we performed the following cleaning operations:

**Filtering steps**:
- Removed unnecessary index columns (`Unnamed: 0.1`, `Unnamed: 0`)
- Filtered for records where `activity == "Location"` (location labeling tasks only)
- Removed records with null values in `started_at` or `finished_at` columns
- Excluded records marked as deleted (`deleted_at` is not null)
- Filtered for `user_id == 97` (5th floor location labeling only)
- Dropped unused columns: `deleted_at`, `updated_at`, `activity`, `user`, `user_id`

**Result**: 451 clean location label records saved to `cleaned_dataset/cleaned_label_loc.csv` with columns:
- `started_at`: Entry timestamp for the location
- `finished_at`: Exit timestamp for the location  
- `room`: Target label (location name)
- `floor`: Floor identifier

### 2. BLE Data Merging (`ble_data_merging.ipynb`)

The raw BLE data is split across multiple CSV files in `Dataset/BLE Data/`. These were merged into a single file:

**Process**:
- Read all `userbleid_90_*.csv` files from the BLE Data directory
- Concatenated all records maintaining the original schema
- Saved as `cleaned_dataset/merged_ble_data.csv`

**Result**: ~5 million BLE signal records in a single file

### 3. BLE Data Cleaning (`ble_data_cleaning.ipynb`)

Starting with the merged BLE data (~5 million records), we performed extensive cleaning:

**Temporal filtering**:
- Converted `timestamp` column to proper datetime format (with timezone UTC+09:00)
- Filtered for records between 2023-04-10 13:00:00 and 2023-04-13 17:29:59
  - We did this step as instructed in the provided tutorial notebook on how to process ble data as no documentations are provided
  - Day 1 (1 PM onwards) through Day 4 (until 5:30 PM)

**Beacon filtering**:
- From hundreds of unique MAC addresses detected, selected only 25 key beacons
- These 25 beacons correspond to the main BLE transmitters installed on the 5th floor
- Mapped MAC addresses to beacon IDs (1-25) for easier reference

**Column cleanup**:
- Removed `Unnamed: 0` (accidentally saved index column)
- Removed `user_id` column (all records are from user 90)
- Removed `name` column (always null)

**Result**: 1,673,395 cleaned BLE records saved to `cleaned_dataset/cleaned_ble_data.csv` with columns:
- `timestamp`: Detection time (timezone-aware datetime)
- `mac address`: Beacon ID (1-25)
- `RSSI`: Signal strength in dBm
- `power`: Additional signal information (typically -2147483648)

**Beacon distribution** (number of readings per beacon):
- Beacon 4: 380,092 readings
- Beacon 9: 330,508 readings  
- Beacon 14: 186,595 readings
- Beacon 19: 133,965 readings
- (... and 21 other beacons with varying frequencies)

### 4. BLE Data Labeling (`add_label_ble_data.ipynb`)

After cleaning both the location labels and BLE sensor data, we merged them to create a labeled dataset for supervised learning.

#### Objective
Match each BLE sensor reading with its corresponding room label based on timestamp alignment.

#### Approach
We perform a **timestamp-based join** between the BLE sensor data and location labels:

1. **Merge Strategy**: Use `pd.merge_asof()` with `direction='backward'` to find the most recent label where `started_at <= BLE_timestamp`
2. **Validation**: Ensure `BLE_timestamp <= finished_at` to guarantee the timestamp falls within the labeled time range
3. **Filtering**: Keep only records with valid room labels (drop unlabeled data)

#### Why Drop Unlabeled Records?

**Decision: Drop 34% of BLE data without matching labels**

**Rationale**:
- **Supervised Learning Requirement**: We need labeled data to train a room classification model
- **Data Collection Design**: 
  - User 90 (caregiver) continuously collected BLE sensor data
  - User 97 (labeler) selectively annotated specific location visits
  - This creates gaps where sensor data exists but no ground truth label is available
- **Unlabeled records represent**:
  - Transition periods between rooms
  - Times when the labeler wasn't actively tracking
  - Areas outside the scope of this challenge
- **Not a preprocessing error**: The 34% gap is inherent to the dataset design, not a mistake in our code

#### Data Cleaning Steps
1. ✅ Convert timestamps to datetime format
2. ✅ Match BLE readings with room labels using time ranges
3. ✅ Validate timestamp falls within [started_at, finished_at]
4. ✅ Drop records without valid labels
5. ✅ Remove unnecessary columns (power, started_at, finished_at)

#### Result
- **Original BLE records**: ~1.67M
- **Labeled records retained**: ~1.10M (66%)
- **Records dropped**: ~0.57M (34%)

**Output file**: `cleaned_dataset/labelled_ble_data.csv` with columns:
- `timestamp`: Detection time
- `mac address`: Beacon ID (1-25)
- `RSSI`: Signal strength in dBm
- `room`: **[TARGET LABEL]** Room/location name

**File format**:
```csv
timestamp,mac address,RSSI,room
2023-04-10 14:21:46+09:00,6,-93,kitchen
```

---

## Exploratory Data Analysis (analysis/)

After preparing the cleaned and labeled datasets, we conducted exploratory analysis to understand the data characteristics.

### 1. Basic Analysis (`basic_analysis.ipynb`)

Initial exploration of individual data files:
- Examined structure and format of raw label data
- Analyzed distribution of BLE sensor readings
- Investigated temporal patterns in data collection
- Explored beacon signal characteristics

### 2. Labeled BLE Data Analysis (`labelled_ble_data_analysis.ipynb`)

Comprehensive analysis of the merged labeled dataset:
- Room distribution and class balance
- Beacon signal patterns across different locations
- RSSI value distributions per room
- Temporal patterns and visit durations
- Feature engineering insights for model development

---

## Task Overview

**Objective**: Build a machine learning model that predicts which room a person is in based on RSSI signal patterns from multiple beacons.

**Training approach**:
1. Use the labeled BLE dataset (`labelled_ble_data.csv`) which already has room labels matched to each sensor reading
2. Aggregate and structure RSSI values from different beacons as features
3. Engineer time-based and signal-strength features
4. Train a classification model to predict room labels from RSSI patterns

**Key challenge**: Different beacons are detected at different locations with varying signal strengths. The model must learn which combination of beacon signals corresponds to each room.

---

## Data Issues & Notes

**Important**: Not all BLE data files have corresponding labels. For example:
- BLE data may be collected starting at 10:22 AM
- Labels may only begin at 2:21 PM (14:21)
- Only use BLE data files whose timestamps overlap with the labeled time ranges

This has been addressed in the data cleaning process by:
1. Filtering the BLE data to match the labeled time range (2023-04-10 13:00:00 to 2023-04-13 17:29:59)
2. Performing timestamp-based matching to align BLE readings with room labels
3. Dropping 34% of BLE readings that fall outside labeled time ranges (expected behavior given the data collection design)
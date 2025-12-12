# ABC 2026 Challenge - Activity & Location Recognition

This year's challenge contains 2 problems:
- **Location Prediction**: Predict location based on labeled RSSI (Received Signal Strength Indicator) values from Bluetooth beacon signals
- **Activity Prediction**: Predict activity based on recorded sensor features (data has not been released yet)

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

## Task Overview

**Objective**: Build a machine learning model that predicts which room a person is in based on RSSI signal patterns from multiple beacons.

**Training approach**:
1. Merge BLE data with location labels based on timestamps
2. For each BLE reading, find the corresponding room label where started_at <= BLE_timestamp <= finished_at
3. Aggregate and structure RSSI values from different beacons as features
4. Train a classification model to predict room labels from RSSI patterns

**Key challenge**: Different beacons are detected at different locations with varying signal strengths. The model must learn which combination of beacon signals corresponds to each room.

---

## Data Issues & Notes

**Important**: Not all BLE data files have corresponding labels. For example:
- BLE data may be collected starting at 10:22 AM
- Labels may only begin at 2:21 PM (14:21)
- Only use BLE data files whose timestamps overlap with the labeled time ranges
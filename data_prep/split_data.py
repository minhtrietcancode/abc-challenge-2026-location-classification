"""
4-Fold Temporal Data Split Generator

This script:
1. Loads the full 4-day dataset
2. Creates 4 temporal folds (leave-one-day-out)
3. Saves each fold as train.csv and test.csv

Folds:
- Fold 1: Train [Days 1,2,3] → Test [Day 4]
- Fold 2: Train [Days 1,2,4] → Test [Day 3]
- Fold 3: Train [Days 1,3,4] → Test [Day 2]
- Fold 4: Train [Days 2,3,4] → Test [Day 1]
"""

import pandas as pd
import os
from pathlib import Path

print("="*70)
print("4-FOLD TEMPORAL DATA SPLIT GENERATOR")
print("="*70)

# ==================== CONFIGURATION ====================
INPUT_FILE = "cleaned_dataset/labelled_ble_data.csv"
OUTPUT_DIR = "cleaned_dataset/split_data"

# ==================== LOAD DATA ====================
print("\n[1/4] Loading full dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df):,} records")

# Parse timestamp and extract date
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# Assign day numbers (1-4)
unique_dates = sorted(df['date'].unique())
date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
df['day_number'] = df['date'].map(date_to_day)

print(f"\n📅 Date to Day mapping:")
for date, day_num in date_to_day.items():
    day_records = len(df[df['day_number'] == day_num])
    print(f"   Day {day_num}: {date} ({day_records:,} records)")

# Drop the temporary columns (keep only original format)
df_clean = df.drop(['date', 'day_number'], axis=1)

# ==================== CREATE FOLDS ====================
print("\n[2/4] Creating 4 temporal folds...")

# Define fold configurations
fold_configs = [
    {"name": "fold1", "train_days": [1, 2, 3], "test_day": 4},
    {"name": "fold2", "train_days": [1, 2, 4], "test_day": 3},
    {"name": "fold3", "train_days": [1, 3, 4], "test_day": 2},
    {"name": "fold4", "train_days": [2, 3, 4], "test_day": 1},
]

for config in fold_configs:
    fold_name = config["name"]
    train_days = config["train_days"]
    test_day = config["test_day"]
    
    print(f"\n{fold_name.upper()}: Train days {train_days} → Test day {test_day}")
    
    # Split data
    train_df = df[df['day_number'].isin(train_days)].copy()
    test_df = df[df['day_number'] == test_day].copy()
    
    # Remove day_number column (keep original format)
    train_df = train_df.drop(['date', 'day_number'], axis=1)
    test_df = test_df.drop(['date', 'day_number'], axis=1)
    
    # Get unique rooms in train and test
    train_rooms = set(train_df['room'].unique())
    test_rooms = set(test_df['room'].unique())
    common_rooms = train_rooms & test_rooms
    
    # Filter to common rooms only
    train_df_filtered = train_df[train_df['room'].isin(common_rooms)].reset_index(drop=True)
    test_df_filtered = test_df[test_df['room'].isin(common_rooms)].reset_index(drop=True)
    
    print(f"   Train: {len(train_df):,} records → {len(train_df_filtered):,} after filtering")
    print(f"   Test:  {len(test_df):,} records → {len(test_df_filtered):,} after filtering")
    print(f"   Common rooms: {len(common_rooms)}")
    print(f"   Train-only rooms: {train_rooms - test_rooms if train_rooms - test_rooms else 'None'}")
    print(f"   Test-only rooms: {test_rooms - train_rooms if test_rooms - train_rooms else 'None'}")
    
    # Create output directory
    fold_dir = Path(OUTPUT_DIR) / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train and test files
    train_path = fold_dir / "train.csv"
    test_path = fold_dir / "test.csv"
    
    train_df_filtered.to_csv(train_path, index=False)
    test_df_filtered.to_csv(test_path, index=False)
    
    print(f"   ✓ Saved: {train_path}")
    print(f"   ✓ Saved: {test_path}")

# ==================== SUMMARY ====================
print("\n" + "="*70)
print("[3/4] SUMMARY - 4 FOLDS CREATED")
print("="*70)

summary_data = []
for config in fold_configs:
    fold_name = config["name"]
    train_days = config["train_days"]
    test_day = config["test_day"]
    
    # Load saved files to verify
    fold_dir = Path(OUTPUT_DIR) / fold_name
    train_df = pd.read_csv(fold_dir / "train.csv")
    test_df = pd.read_csv(fold_dir / "test.csv")
    
    summary_data.append({
        'Fold': fold_name,
        'Train Days': str(train_days),
        'Test Day': test_day,
        'Train Records': len(train_df),
        'Test Records': len(test_df),
        'Train Timestamps': train_df['timestamp'].nunique(),
        'Test Timestamps': test_df['timestamp'].nunique(),
        'Common Rooms': train_df['room'].nunique()
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# ==================== DIRECTORY STRUCTURE ====================
print("\n" + "="*70)
print("[4/4] DIRECTORY STRUCTURE")
print("="*70)

print(f"\n{OUTPUT_DIR}/")
for config in fold_configs:
    fold_name = config["name"]
    print(f"├── {fold_name}/")
    print(f"│   ├── train.csv")
    print(f"│   └── test.csv")

# ==================== USAGE INSTRUCTIONS ====================
print("\n" + "="*70)
print("USAGE IN YOUR PIPELINE")
print("="*70)

print("""
To use in your pipeline, simply change the data loading paths:

# For Fold 1 (recommended - Day 4 test)
train_df = pd.read_csv("cleaned_dataset/split_data/fold1/train.csv")
test_df = pd.read_csv("cleaned_dataset/split_data/fold1/test.csv")

# For Fold 2 (Day 3 test)
train_df = pd.read_csv("cleaned_dataset/split_data/fold2/train.csv")
test_df = pd.read_csv("cleaned_dataset/split_data/fold2/test.csv")

# For Fold 3 (Day 2 test)
train_df = pd.read_csv("cleaned_dataset/split_data/fold3/train.csv")
test_df = pd.read_csv("cleaned_dataset/split_data/fold3/test.csv")

# For Fold 4 (Day 1 test)
train_df = pd.read_csv("cleaned_dataset/split_data/fold4/train.csv")
test_df = pd.read_csv("cleaned_dataset/split_data/fold4/test.csv")

RECOMMENDATION:
- Use fold1 as your PRIMARY result (Days 1+2+3 → Day 4)
- Run all 4 folds for cross-validation if needed
""")

print("\n" + "="*70)
print("✅ ALL FOLDS CREATED SUCCESSFULLY!")
print("="*70)
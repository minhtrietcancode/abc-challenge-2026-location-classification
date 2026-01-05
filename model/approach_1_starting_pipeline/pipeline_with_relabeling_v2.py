"""
Indoor Localization Pipeline with KL Divergence Relabeling
Based on the paper: "Relabeling for Indoor Localization Using Stationary Beacons in Nursing Care Facilities"
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BEACON LAYOUT MAPPING - Based on 5th Floor Map
# ============================================================================
# Define the 6 surrounding beacons for each room
# Format: [front_left, side_left, source, front, side_right, front_right]

ROOM_BEACON_MAPPING = {
    # Top row rooms - Using string room numbers matching your dataset
    '501': {
        'source': 1,
        'six_beacons': [None, None, 1, 13, 2, 15]  # fl, sl, s, f, sr, fr
    },
    '502': {
        'source': 2,
        'six_beacons': [1, 13, 2, 15, 3, 16]
    },
    '503': {
        'source': 3,
        'six_beacons': [2, 15, 3, 16, 5, 17]
    },
    '505': {
        'source': 5,
        'six_beacons': [3, 16, 5, 17, 6, None]
    },
    '506': {
        'source': 6,
        'six_beacons': [5, 17, 6, None, None, None]
    },
    '507': {
        'source': 7,
        'six_beacons': [None, None, 7, 9, 8, 18]
    },
    '508': {
        'source': 8,
        'six_beacons': [7, 9, 8, 18, 10, 20]
    },
    '510': {
        'source': 10,
        'six_beacons': [8, 18, 10, 20, 11, 21]
    },
    '511': {
        'source': 11,
        'six_beacons': [10, 20, 11, 21, 12, 22]
    },
    '512': {
        'source': 12,
        'six_beacons': [11, 21, 12, 22, None, 23]
    },
    
    # Bottom row rooms (513-523)
    '513': {
        'source': 13,
        'six_beacons': [None, None, 13, 1, 15, 2]
    },
    '515': {
        'source': 15,
        'six_beacons': [13, 1, 15, 2, 16, 3]
    },
    '516': {
        'source': 16,
        'six_beacons': [15, 2, 16, 3, 17, 5]
    },
    '517': {
        'source': 17,
        'six_beacons': [16, 3, 17, 5, None, 6]
    },
    '518': {
        'source': 18,
        'six_beacons': [None, 9, 18, 7, 20, 8]
    },
    '520': {
        'source': 20,
        'six_beacons': [9, 18, 20, 8, 21, 10]
    },
    '521': {
        'source': 21,
        'six_beacons': [18, 20, 21, 10, 22, 11]
    },
    '522': {
        'source': 22,
        'six_beacons': [20, 21, 22, 11, 23, 12]
    },
    '523': {
        'source': 23,
        'six_beacons': [21, 22, 23, 12, None, None]
    },
    
    # Common areas - matching your dataset naming
    'kitchen': {
        'source': 14,
        'six_beacons': [None, None, 14, None, None, None]
    },
    'nurse station': {
        'source': 9,
        'six_beacons': [None, None, 9, None, None, None]
    },
    'cafeteria': {
        'source': 25,
        'six_beacons': [None, None, 25, 4, None, None]
    },
    'hallway': {
        'source': 19,  # Beacon 19 is in the hallway area
        'six_beacons': [None, None, 19, None, None, None]
    },
    'cleaning': {
        'source': 19,  # Near cleaning/elevator area
        'six_beacons': [None, None, 19, None, None, None]
    }
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_and_filter_fold(i):
    """Load and filter fold data to keep only common labels"""
    train_dir = f'../../cleaned_dataset/split_data/fold{i}/train.csv'  
    test_dir = f'../../cleaned_dataset/split_data/fold{i}/test.csv'   
    
    # Load the data 
    train_df = pd.read_csv(train_dir)
    test_df = pd.read_csv(test_dir)

    # Get all of the unique labels for train / test sets 
    train_labels = list(train_df['room'].unique())
    test_labels = list(test_df['room'].unique())

    # Labels that appear in BOTH train and test
    common_labels = list(set(train_labels) & set(test_labels))

    # Filter to just keep the records with labels in common labels list
    train_df = train_df[train_df['room'].isin(common_labels)].reset_index(drop=True)
    test_df  = test_df[test_df['room'].isin(common_labels)].reset_index(drop=True)

    return train_df, test_df


def add_beacon_features(df, num_beacons=25):
    """Add 25 beacon features (beacon_1, beacon_2, ..., beacon_25)"""
    df = df.copy()

    for i in range(1, num_beacons + 1):
        df[f'beacon_{i}'] = df['RSSI'].where(df['mac address'] == i, 0)

    return df


def aggregate_by_timestamp(df):
    """
    Aggregate beacon data by timestamp (1-second windows)
    
    For each beacon column:
    - If ALL values are 0: set mean, std, min, max, count = 0
    - If ANY non-zero values exist: calculate statistics ONLY on non-zero values
    """    
    df = df.copy()
    
    # Group by timestamp
    grouped = df.groupby('timestamp')
    
    result_rows = []
    
    for timestamp, group in grouped:
        # Initialize row with timestamp and room
        row_data = {
            'timestamp': timestamp,
            'room': group['room'].iloc[0]
        }
        
        # Process each beacon column (beacon_1 to beacon_25)
        for beacon_id in range(1, 26):
            beacon_col = f'beacon_{beacon_id}'
            
            # Get all values for this beacon in this window
            beacon_values = group[beacon_col].values
            
            # Filter to get only non-zero values
            non_zero_values = beacon_values[beacon_values != 0]
            
            # Check if we have any non-zero values
            if len(non_zero_values) > 0:
                # Calculate statistics on non-zero values ONLY
                row_data[f'{beacon_col}_mean'] = non_zero_values.mean()
                row_data[f'{beacon_col}_std'] = non_zero_values.std() if len(non_zero_values) > 1 else 0.0
                row_data[f'{beacon_col}_min'] = non_zero_values.min()
                row_data[f'{beacon_col}_max'] = non_zero_values.max()
                row_data[f'{beacon_col}_count'] = len(non_zero_values)
            else:
                # All values are zero - set everything to 0
                row_data[f'{beacon_col}_mean'] = 0.0
                row_data[f'{beacon_col}_std'] = 0.0
                row_data[f'{beacon_col}_min'] = 0.0
                row_data[f'{beacon_col}_max'] = 0.0
                row_data[f'{beacon_col}_count'] = 0
        
        result_rows.append(row_data)
    
    # Create DataFrame from result rows
    windowed_df = pd.DataFrame(result_rows)
    
    # Filter out completely empty windows (all beacons are 0)
    beacon_mean_cols = [f'beacon_{i}_mean' for i in range(1, 26)]
    valid_windows = windowed_df[beacon_mean_cols].sum(axis=1) != 0
    
    removed_count = (~valid_windows).sum()
    windowed_df = windowed_df[valid_windows].reset_index(drop=True)
    
    print(f"  Total windows after aggregation: {len(windowed_df)}")
    print(f"  Removed {removed_count} empty windows (all beacons = 0)")
    
    return windowed_df


# ============================================================================
# RELABELING FUNCTIONS
# ============================================================================

def identify_minority_classes(train_df, n_minority=2):
    """Identify the top N minority classes (rooms with fewest samples)
    Only considers rooms that are in the ROOM_BEACON_MAPPING (patient rooms)
    """
    # Filter to only rooms in our beacon mapping
    valid_rooms = [room for room in train_df['room'].unique() if room in ROOM_BEACON_MAPPING]
    filtered_df = train_df[train_df['room'].isin(valid_rooms)]
    
    room_counts = filtered_df['room'].value_counts()
    minority_rooms = room_counts.nsmallest(n_minority).index.tolist()
    
    print(f"\n  📊 Class Distribution (Patient Rooms Only):")
    for room in room_counts.index:
        count = room_counts[room]
        status = "🔴 MINORITY" if room in minority_rooms else ""
        print(f"     {room}: {count} samples {status}")
    
    # Also show rooms not in mapping (for info)
    unmapped_rooms = [room for room in train_df['room'].unique() if room not in ROOM_BEACON_MAPPING]
    if unmapped_rooms:
        print(f"\n  ℹ️  Rooms not in mapping (excluded from relabeling): {unmapped_rooms}")
    
    return minority_rooms


def extract_six_beacon_pattern(df, room_name, statistic='mean'):
    """
    Extract the 6 surrounding beacon pattern for a specific room
    
    Args:
        df: Windowed dataframe with beacon statistics
        room_name: Name of the room (e.g., '501', '502')
        statistic: Which statistic to use ('mean', 'std', 'min', 'max', 'count')
    
    Returns:
        DataFrame with 6 beacon columns (replacing None with 0)
        None if room not in mapping or no data
    """
    if room_name not in ROOM_BEACON_MAPPING:
        print(f"  ⚠️  Room {room_name} not in beacon mapping - skipping")
        return None
    
    room_data = df[df['room'] == room_name].copy()
    
    if len(room_data) == 0:
        return None
    
    six_beacons = ROOM_BEACON_MAPPING[room_name]['six_beacons']
    
    # Create pattern dataframe with 6 columns
    pattern_cols = []
    for i, beacon_id in enumerate(six_beacons):
        col_name = f'beacon_pos_{i}'  # beacon_pos_0 to beacon_pos_5
        
        if beacon_id is None:
            # No beacon at this position - fill with 0
            room_data[col_name] = 0.0
        else:
            # Extract the statistic for this beacon
            beacon_col = f'beacon_{beacon_id}_{statistic}'
            room_data[col_name] = room_data[beacon_col]
        
        pattern_cols.append(col_name)
    
    return room_data[pattern_cols + ['room']]


def calculate_kl_divergence(pattern_minority, pattern_majority):
    """
    Calculate KL divergence between minority and majority class patterns
    
    D_KL(P || Q) = Σ P(i) log(P(i)/Q(i))
    
    Args:
        pattern_minority: DataFrame with 6 beacon columns for minority class
        pattern_majority: DataFrame with 6 beacon columns for majority class
    
    Returns:
        Total KL divergence (sum across all 6 beacons)
    """
    # Downsample majority to match minority sample size
    n_minority = len(pattern_minority)
    n_majority = len(pattern_majority)
    
    if n_majority > n_minority:
        pattern_majority = pattern_majority.sample(n=n_minority, random_state=42)
    elif n_minority > n_majority:
        pattern_minority = pattern_minority.sample(n=n_majority, random_state=42)
    
    total_kl = 0.0
    beacon_cols = [f'beacon_pos_{i}' for i in range(6)]
    
    for col in beacon_cols:
        # Get values for this beacon position
        p_values = pattern_minority[col].values
        q_values = pattern_majority[col].values
        
        # Convert to absolute values (RSSI are negative)
        p_values = np.abs(p_values)
        q_values = np.abs(q_values)
        
        # Add small epsilon to avoid log(0) and division by zero
        epsilon = 1e-10
        p_values = p_values + epsilon
        q_values = q_values + epsilon
        
        # Create histograms with the same bins for both distributions
        # This ensures they have the same length
        bins = 50  # Number of bins
        
        # Get the range for bins (use combined range)
        all_values = np.concatenate([p_values, q_values])
        value_min = all_values.min()
        value_max = all_values.max()
        
        # Create histogram bins
        bin_edges = np.linspace(value_min, value_max, bins + 1)
        
        # Calculate histograms
        p_hist, _ = np.histogram(p_values, bins=bin_edges)
        q_hist, _ = np.histogram(q_values, bins=bin_edges)
        
        # Add epsilon to avoid zero counts
        p_hist = p_hist.astype(float) + epsilon
        q_hist = q_hist.astype(float) + epsilon
        
        # Normalize to create probability distributions
        p_dist = p_hist / p_hist.sum()
        q_dist = q_hist / q_hist.sum()
        
        # Calculate KL divergence for this beacon
        # Both arrays now have the same length (bins)
        kl = entropy(p_dist, q_dist)
        
        # Handle potential inf/nan values
        if np.isnan(kl) or np.isinf(kl):
            kl = 0.0
        
        total_kl += kl
    
    return total_kl


def has_complete_six_beacons(room_name):
    """Check if a room has all 6 surrounding beacons (Full Matching)"""
    if room_name not in ROOM_BEACON_MAPPING:
        return False
    
    six_beacons = ROOM_BEACON_MAPPING[room_name]['six_beacons']
    return all(beacon is not None for beacon in six_beacons)


def find_best_match_kl(minority_room, train_df, matching_type='full'):
    """
    Find the best matching room for relabeling using KL divergence
    
    Args:
        minority_room: Name of minority room
        train_df: Training dataframe
        matching_type: 'full' (only complete 6 beacons) or 'partial' (all rooms)
    
    Returns:
        best_match_room: Name of the best matching room
        kl_divergences: Dict of all KL divergences calculated
    """
    print(f"\n  🔍 Finding match for {minority_room} using {matching_type.upper()} matching...")
    
    # Extract minority pattern
    minority_pattern = extract_six_beacon_pattern(train_df, minority_room, statistic='mean')
    
    if minority_pattern is None or len(minority_pattern) == 0:
        print(f"  ⚠️  No data for {minority_room}")
        return None, {}
    
    # Get candidate rooms
    all_rooms = train_df['room'].unique()
    candidate_rooms = []
    
    for room in all_rooms:
        if room == minority_room:
            continue
        
        if matching_type == 'full':
            # Only consider rooms with complete 6 beacons
            if has_complete_six_beacons(room):
                candidate_rooms.append(room)
        else:  # partial
            # Consider all rooms
            if room in ROOM_BEACON_MAPPING:
                candidate_rooms.append(room)
    
    print(f"  📋 Candidate rooms: {candidate_rooms}")
    
    # Calculate KL divergence for each candidate
    kl_divergences = {}
    
    for candidate_room in candidate_rooms:
        candidate_pattern = extract_six_beacon_pattern(train_df, candidate_room, statistic='mean')
        
        if candidate_pattern is None or len(candidate_pattern) == 0:
            continue
        
        kl = calculate_kl_divergence(minority_pattern, candidate_pattern)
        kl_divergences[candidate_room] = kl
        print(f"     {candidate_room}: KL = {kl:.4f}")
    
    if len(kl_divergences) == 0:
        print(f"  ⚠️  No valid candidates found")
        return None, {}
    
    # Find room with minimum KL divergence
    best_match_room = min(kl_divergences, key=kl_divergences.get)
    print(f"  ✅ Best match: {best_match_room} (KL = {kl_divergences[best_match_room]:.4f})")
    
    return best_match_room, kl_divergences


def relabel_data(train_df, minority_room, match_room, n_samples_to_add):
    """
    Relabel samples from match_room to minority_room
    
    Args:
        train_df: Original training dataframe
        minority_room: Target minority room
        match_room: Source room to relabel from
        n_samples_to_add: Number of samples to relabel
    
    Returns:
        relabeled_df: New samples with relabeled room
    """
    print(f"\n  🔄 Relabeling {n_samples_to_add} samples from {match_room} to {minority_room}...")
    
    # Get samples from match room
    match_samples = train_df[train_df['room'] == match_room].copy()
    
    # Sample n_samples_to_add
    if len(match_samples) < n_samples_to_add:
        print(f"  ⚠️  Only {len(match_samples)} samples available, using all")
        relabeled_samples = match_samples
    else:
        relabeled_samples = match_samples.sample(n=n_samples_to_add, random_state=42)
    
    # Update the room label
    relabeled_samples['room'] = minority_room
    
    print(f"  ✅ Created {len(relabeled_samples)} relabeled samples")
    
    return relabeled_samples


def apply_relabeling(train_df, n_minority=2, matching_type='full', augmentation_factor=1.0):
    """
    Apply full relabeling pipeline to training data
    
    Args:
        train_df: Training dataframe (after windowing)
        n_minority: Number of minority classes to augment
        matching_type: 'full' or 'partial'
        augmentation_factor: How much to increase minority class (1.0 = double the size)
    
    Returns:
        augmented_df: Training data with relabeled samples added
        relabeling_info: Dictionary with relabeling details
    """
    print(f"\n{'='*80}")
    print(f"APPLYING RELABELING (Matching: {matching_type.upper()}, Factor: {augmentation_factor})")
    print(f"{'='*80}")
    
    # Step 1: Identify minority classes
    minority_rooms = identify_minority_classes(train_df, n_minority=n_minority)
    
    # Step 2: Find best matches and relabel
    relabeling_info = {}
    all_relabeled_samples = []
    
    for minority_room in minority_rooms:
        # Find best match
        best_match, kl_divs = find_best_match_kl(minority_room, train_df, matching_type=matching_type)
        
        if best_match is None:
            print(f"  ⚠️  Could not find match for {minority_room}, skipping...")
            continue
        
        # Calculate how many samples to add
        minority_count = len(train_df[train_df['room'] == minority_room])
        n_to_add = int(minority_count * augmentation_factor)
        
        # Relabel samples
        relabeled = relabel_data(train_df, minority_room, best_match, n_to_add)
        all_relabeled_samples.append(relabeled)
        
        # Store info
        relabeling_info[minority_room] = {
            'best_match': best_match,
            'kl_divergence': kl_divs.get(best_match, None),
            'original_count': minority_count,
            'added_count': len(relabeled),
            'new_count': minority_count + len(relabeled)
        }
    
    # Step 3: Combine original and relabeled data
    if len(all_relabeled_samples) > 0:
        relabeled_df = pd.concat(all_relabeled_samples, ignore_index=True)
        augmented_df = pd.concat([train_df, relabeled_df], ignore_index=True)
        
        print(f"\n  📊 AUGMENTATION SUMMARY:")
        print(f"     Original samples: {len(train_df)}")
        print(f"     Relabeled samples: {len(relabeled_df)}")
        print(f"     Total samples: {len(augmented_df)}")
    else:
        print(f"\n  ⚠️  No relabeling performed")
        augmented_df = train_df
    
    return augmented_df, relabeling_info


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_evaluate_fold(fold_num, train_df, test_df, use_relabeling=True, 
                        matching_type='full', augmentation_factor=1.0):
    """
    Train and evaluate model on a fold with optional relabeling
    
    Args:
        fold_num: Fold number
        train_df: Training dataframe (raw)
        test_df: Test dataframe (raw)
        use_relabeling: Whether to apply relabeling
        matching_type: 'full' or 'partial'
        augmentation_factor: How much to increase minority class
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING FOLD {fold_num}")
    print(f"{'='*80}\n")
    
    # Step 1: Add beacon features
    print("Adding 25 beacon vector features for both sets...")
    train_df = add_beacon_features(train_df)
    test_df = add_beacon_features(test_df)

    # Step 2: Apply windowing
    print("Applying windowing for both sets...")
    windowed_train_df = aggregate_by_timestamp(train_df)
    windowed_test_df = aggregate_by_timestamp(test_df)

    # Step 3: Apply relabeling (if enabled)
    relabeling_info = None
    if use_relabeling:
        windowed_train_df, relabeling_info = apply_relabeling(
            windowed_train_df, 
            n_minority=2,
            matching_type=matching_type,
            augmentation_factor=augmentation_factor
        )
    
    # Step 4: Prepare features
    feature_cols = [col for col in windowed_train_df.columns 
                    if col not in ['room', 'timestamp']]

    X_train = windowed_train_df[feature_cols]
    y_train = windowed_train_df['room']

    # Step 5: Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Step 6: Calculate sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)

    # Step 7: Train XGBoost
    print(f"\n{'='*80}")
    print("TRAINING XGBOOST MODEL")
    print(f"{'='*80}")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        objective='multi:softmax',
        num_class=len(y_train.unique()),
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )

    xgb_model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    print("✅ Training completed!")
    
    # Step 8: Evaluate on test set
    print(f"\n{'='*80}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*80}")
    
    X_test = windowed_test_df[feature_cols]
    y_test = windowed_test_df['room']
    y_test_encoded = label_encoder.transform(y_test)
    
    y_pred_encoded = xgb_model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Calculate metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=label_encoder.classes_)
    per_class_f1_dict = {label: f1 for label, f1 in zip(label_encoder.classes_, per_class_f1)}
    
    print(f"\n📊 RESULTS:")
    print(f"   Macro F1 Score: {macro_f1:.4f}")
    print(f"   Weighted F1 Score: {weighted_f1:.4f}")
    
    return {
        'fold': fold_num,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1_dict,
        'classes': label_encoder.classes_,
        'relabeling_info': relabeling_info
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("INDOOR LOCALIZATION WITH KL DIVERGENCE RELABELING")
    print("="*80)
    
    # Load all 4 folds
    print("\n📁 Loading data for all 4 folds...")
    train_df_1, test_df_1 = load_and_filter_fold(1)
    train_df_2, test_df_2 = load_and_filter_fold(2)
    train_df_3, test_df_3 = load_and_filter_fold(3)
    train_df_4, test_df_4 = load_and_filter_fold(4)
    print("✅ All folds loaded!")
    
    folds = {
        1: (train_df_1, test_df_1),
        2: (train_df_2, test_df_2),
        3: (train_df_3, test_df_3),
        4: (train_df_4, test_df_4)
    }
    
    # Train and evaluate with relabeling
    results_with_relabeling = {}
    
    for fold_num, (train_df, test_df) in folds.items():
        result = train_evaluate_fold(
            fold_num, 
            train_df, 
            test_df,
            use_relabeling=True,
            matching_type='full',  # Use 'full' matching (complete 6 beacons)
            augmentation_factor=1.0  # Double the minority class size
        )
        results_with_relabeling[fold_num] = result
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL FOLDS")
    print("="*80)
    
    macro_f1_scores = []
    weighted_f1_scores = []
    
    for fold_num in [1, 2, 3, 4]:
        result = results_with_relabeling[fold_num]
        macro_f1_scores.append(result['macro_f1'])
        weighted_f1_scores.append(result['weighted_f1'])
        
        print(f"\nFold {fold_num}:")
        print(f"  Macro F1: {result['macro_f1']:.4f}")
        print(f"  Weighted F1: {result['weighted_f1']:.4f}")
        
        if result['relabeling_info']:
            print(f"  Relabeling Applied:")
            for minority_room, info in result['relabeling_info'].items():
                print(f"    {minority_room}: {info['original_count']} → {info['new_count']} samples")
                print(f"      (matched with {info['best_match']}, KL={info['kl_divergence']:.4f})")
    
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS:")
    print(f"  Mean Macro F1: {np.mean(macro_f1_scores):.4f} ± {np.std(macro_f1_scores):.4f}")
    print(f"  Mean Weighted F1: {np.mean(weighted_f1_scores):.4f} ± {np.std(weighted_f1_scores):.4f}")
    print(f"{'='*80}")
    
    # Save detailed results
    with open('relabeling_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("INDOOR LOCALIZATION RESULTS WITH KL DIVERGENCE RELABELING\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        f.write("Matching Type: FULL (complete 6 beacons only)\n")
        f.write("Augmentation Factor: 1.0 (double minority class)\n")
        f.write("Number of Minority Classes: 2\n\n")
        
        f.write("RESULTS PER FOLD:\n")
        f.write("-"*80 + "\n")
        
        for fold_num in [1, 2, 3, 4]:
            result = results_with_relabeling[fold_num]
            f.write(f"\nFold {fold_num}:\n")
            f.write(f"  Macro F1: {result['macro_f1']:.4f}\n")
            f.write(f"  Weighted F1: {result['weighted_f1']:.4f}\n")
            
            f.write(f"\n  Per-class F1 scores:\n")
            for class_name in sorted(result['per_class_f1'].keys()):
                f.write(f"    {class_name:20s}: {result['per_class_f1'][class_name]:.4f}\n")
            
            if result['relabeling_info']:
                f.write(f"\n  Relabeling Details:\n")
                for minority_room, info in result['relabeling_info'].items():
                    f.write(f"    {minority_room}:\n")
                    f.write(f"      Best Match: {info['best_match']}\n")
                    f.write(f"      KL Divergence: {info['kl_divergence']:.4f}\n")
                    f.write(f"      Original Samples: {info['original_count']}\n")
                    f.write(f"      Added Samples: {info['added_count']}\n")
                    f.write(f"      New Total: {info['new_count']}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"SUMMARY STATISTICS:\n")
        f.write(f"{'='*80}\n")
        f.write(f"Mean Macro F1: {np.mean(macro_f1_scores):.4f} ± {np.std(macro_f1_scores):.4f}\n")
        f.write(f"Mean Weighted F1: {np.mean(weighted_f1_scores):.4f} ± {np.std(weighted_f1_scores):.4f}\n")
        f.write(f"Min Macro F1: {np.min(macro_f1_scores):.4f}\n")
        f.write(f"Max Macro F1: {np.max(macro_f1_scores):.4f}\n")
    
    print("\n✅ Detailed results saved to 'relabeling_results.txt'")
    
    return results_with_relabeling


if __name__ == "__main__":
    results = main()

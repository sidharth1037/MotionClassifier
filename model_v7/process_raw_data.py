import pandas as pd
import numpy as np
import os
import re

# ========== CONFIG ==========
DATA_DIR = "dataset"
OUT_FILE = "motion_windows_v7.npz"
WINDOW_SIZE = 96               # samples (approx 1.92 seconds @ 50Hz)
STEP_SIZE = 48                 # 50% overlap

# Activity classes
CLASS_MAP = {
    'walking': 0,
    'upstairs': 1,
    'downstairs': 2,
    'idle': 3
}

# ==============================================================================
# NEW IN V7: FEATURE ENGINEERING
# ==============================================================================
# V6 used 4 features: AccX, AccY, AccZ, Magnitude
# V7 adds 2 new features that specifically help distinguish walking from upstairs:
#
# Feature 5: Jerk_Z (vertical jerk = derivative of Z-axis acceleration)
#   - WHY: Upstairs produces sharp, periodic vertical jolts when you push off
#     each step. Walking has smoother Z-axis transitions. The jerk (rate of
#     change) makes these sudden vertical changes much more prominent.
#   - HOW: jerk_z[t] = acc_z[t] - acc_z[t-1], with jerk_z[0] = 0
#
# Feature 6: AccZ_detrended (gravity-removed vertical acceleration)
#   - WHY: The raw Z-axis is dominated by gravity (~9.8 m/s²). This masks the
#     dynamic vertical motion that differs between walking and upstairs.
#     Upstairs has larger dynamic vertical oscillations (you're actually
#     lifting your body upward each step). Walking has smaller ones.
#   - HOW: Subtract a rolling mean (window=10 samples = 0.2s) from Z-axis.
#     This removes gravity + slow drift, leaving only the fast dynamic motion.
#
# Total features: 6 = [AccX, AccY, AccZ, Magnitude, JerkZ, AccZ_detrended]
# ==============================================================================

DETREND_WINDOW = 10  # Rolling mean window for gravity removal (10 samples = 0.2s at 50Hz)


def compute_engineered_features(data_3axis):
    """
    Given raw 3-axis accelerometer data (N, 3), compute all 6 features.
    
    Input:  (N, 3) array of [AccX, AccY, AccZ]
    Output: (N, 6) array of [AccX, AccY, AccZ, Magnitude, JerkZ, AccZ_detrended]
    """
    acc_x = data_3axis[:, 0]
    acc_y = data_3axis[:, 1]
    acc_z = data_3axis[:, 2]
    
    # Feature 4: Magnitude (same as v6)
    mag = np.linalg.norm(data_3axis, axis=1)
    
    # Feature 5: Jerk_Z — first-order difference of Z-axis
    # jerk_z[t] = acc_z[t] - acc_z[t-1]
    # First sample has no previous, so we set it to 0
    jerk_z = np.zeros_like(acc_z)
    jerk_z[1:] = np.diff(acc_z)
    
    # Feature 6: AccZ_detrended — Z-axis with running mean subtracted
    # This removes the gravity component (~9.8 m/s²) and slow drift,
    # leaving only the dynamic vertical acceleration from body movement.
    # We use a simple moving average with a small window (0.2s) to preserve
    # step-level dynamics while removing gravity.
    #
    # Using np.convolve for the rolling mean, with 'same' padding to keep
    # the array length identical. Edge effects are minimal since windows
    # are extracted from the middle of longer recordings.
    kernel = np.ones(DETREND_WINDOW) / DETREND_WINDOW
    acc_z_rolling_mean = np.convolve(acc_z, kernel, mode='same')
    acc_z_detrended = acc_z - acc_z_rolling_mean
    
    # Stack all 6 features: (N, 6)
    return np.column_stack((acc_x, acc_y, acc_z, mag, jerk_z, acc_z_detrended))


def load_data(data_dir):
    segments = []
    labels = []
    users = [] 
    
    for label_name, label_idx in CLASS_MAP.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"⚠ Warning: Folder '{folder_path}' not found. Skipping class {label_name}.")
            continue
            
        print(f"Processing {label_name}...")
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not files:
            print(f"⚠ Warning: No CSV files found in '{folder_path}'")
            continue
        
        for f in files:
            # Extract User ID
            user_match = re.search(r'(\d+)', f)
            user_id = int(user_match.group(1)) if user_match else 0
            
            file_path = os.path.join(folder_path, f)
            try:
                df = pd.read_csv(file_path)
                
                # Auto-detect columns (X, Y, Z accelerometer)
                cols = [c for c in df.columns if 'x' in c.lower() or 'y' in c.lower() or 'z' in c.lower()][:3]
                if len(cols) < 3:
                    print(f"  ⚠ Skipping {f}: Found only {len(cols)} acceleration columns (need 3)")
                    continue

                data = df[cols].values
                
                # Validate data quality
                if len(data) < WINDOW_SIZE:
                    print(f"  ⚠ Skipping {f}: Too short ({len(data)} samples < {WINDOW_SIZE})")
                    continue
                
                # Check for NaN values
                if np.any(np.isnan(data)):
                    print(f"  ⚠ Warning: {f} contains NaN values. Filling with 0.")
                    data = np.nan_to_num(data, nan=0.0)
                
                # ==========================================================
                # V7 CHANGE: Compute all 6 features instead of just 4
                # Old (v6): AccX, AccY, AccZ, Magnitude
                # New (v7): AccX, AccY, AccZ, Magnitude, JerkZ, AccZ_detrended
                # ==========================================================
                data = compute_engineered_features(data)

                # Sliding Window with overlap
                window_count = 0
                for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
                    window = data[i : i + WINDOW_SIZE]
                    segments.append(window)
                    labels.append(label_idx)
                    users.append(user_id)
                    window_count += 1
                
                if window_count > 0:
                    print(f"  ✓ {f}: {window_count} windows extracted")
                else:
                    print(f"  ⚠ {f}: No windows generated (file too short)")
                    
            except Exception as e:
                print(f"  ✗ Error reading {f}: {e}")

    return np.array(segments), np.array(labels), np.array(users)

# Run
print("="*70)
print("DATA PROCESSING V7 - Window Extraction + Feature Engineering")
print("="*70)
print(f"Config: WINDOW_SIZE={WINDOW_SIZE}, STEP_SIZE={STEP_SIZE}")
print(f"Classes: {list(CLASS_MAP.keys())}")
print(f"Features: AccX, AccY, AccZ, Magnitude, JerkZ, AccZ_detrended (6 total)")
print()

X, y, users = load_data(DATA_DIR)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Total windows extracted: {len(X)}")
print(f"Shape: X={X.shape}, y={y.shape}, users={users.shape}")
print(f"Features: AccX, AccY, AccZ, Magnitude, JerkZ, AccZ_detrended")
print(f"User IDs found: {sorted(np.unique(users).astype(int))}")

# Class distribution
print("\nClass Distribution:")
for class_name, class_idx in CLASS_MAP.items():
    count = np.sum(y == class_idx)
    pct = (count / len(y) * 100) if len(y) > 0 else 0
    print(f"  {class_name:12} (label {class_idx}): {count:6} windows ({pct:5.1f}%)")

# Validate output
if len(X) == 0:
    print("\n✗ ERROR: No data extracted! Check your dataset folder.")
    exit(1)

print(f"\n{'='*70}")
np.savez_compressed(OUT_FILE, X=X, y=y, users=users)
print(f"✓ Saved to {OUT_FILE}")
print(f"Next: Run 'split_normalize.py' to create LOSO dataset with augmentation")
print(f"{'='*70}")

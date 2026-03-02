import pandas as pd
import numpy as np
import os
import re

# ========== CONFIG ==========
DATA_DIR = "dataset"
OUT_FILE = "motion_windows_v6.npz"
WINDOW_SIZE = 96               # samples (approx 1.92 seconds @ 50Hz) — faster predictions
STEP_SIZE = 48                 # 50% overlap

# Activity classes
CLASS_MAP = {
    'walking': 0,
    'upstairs': 1,
    'downstairs': 2,
    'idle': 3
}

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
                
                # Add Magnitude as 4th feature
                mag = np.linalg.norm(data, axis=1, keepdims=True)
                data = np.hstack((data, mag)) 

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
print("DATA PROCESSING - Window Extraction")
print("="*70)
print(f"Config: WINDOW_SIZE={WINDOW_SIZE}, STEP_SIZE={STEP_SIZE}")
print(f"Classes: {list(CLASS_MAP.keys())}")
print()

X, y, users = load_data(DATA_DIR)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Total windows extracted: {len(X)}")
print(f"Shape: X={X.shape}, y={y.shape}, users={users.shape}")
print(f"Features: AccX, AccY, AccZ, Magnitude")
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
print(f"Next: Run 'split_normalize.py' to create LOSO dataset")
print(f"{'='*70}")
import pandas as pd
import numpy as np
import os
from scipy import stats

# ========== CONFIG ==========
# Put your 1100s of data in folders named by class:
# dataset/
#   ├── walking/
#   │     ├── file1.csv ...
#   ├── upstairs/
#   │     ├── file1.csv ...
#   └── downstairs/
#         ├── file1.csv ...
DATA_DIR = "dataset" 
OUT_FILE = "motion_windows.npz"
WINDOW_SIZE = 100      # 100 samples (approx 2 seconds @ 50Hz)
STEP_SIZE = 50         # 50% overlap (new window every 50 samples)
# ============================

def load_data(data_dir):
    segments = []
    labels = []
    
    # Map folder names to labels
    class_map = {'walking': 0, 'upstairs': 1, 'downstairs': 2}
    
    for label_name, label_idx in class_map.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found.")
            continue
            
        print(f"Processing {label_name}...")
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        for f in files:
            file_path = os.path.join(folder_path, f)
            try:
                # Read CSV (Assuming columns: "Time (s)", "Acceleration x", "Acceleration y", "Acceleration z")
                # Adjust column names below if yours are different!
                df = pd.read_csv(file_path)
                
                # Select only X, Y, Z columns
                # NOTE: Update these strings to match your exact CSV headers
                if 'Acceleration x (m/s^2)' in df.columns:
                     # common syntax for some apps
                    cols = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
                else:
                    # fallback or edit manually
                    cols = [c for c in df.columns if 'x' in c.lower() or 'y' in c.lower() or 'z' in c.lower()][:3]

                data = df[cols].values
                
                # FEATURE ENGINEERING: Add Magnitude column
                # Mag = sqrt(x^2 + y^2 + z^2)
                # This helps the model identify intensity regardless of phone orientation
                mag = np.linalg.norm(data, axis=1, keepdims=True)
                data = np.hstack((data, mag)) # Now shape is (N, 4)

                # Create Sliding Windows
                for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
                    window = data[i : i + WINDOW_SIZE]
                    segments.append(window)
                    labels.append(label_idx)
                    
            except Exception as e:
                print(f"Error reading {f}: {e}")

    X = np.array(segments)
    y = np.array(labels)
    
    return X, y

# Run
X, y = load_data(DATA_DIR)
print(f"\nFinal Shape: X={X.shape}, y={y.shape}")
print(f"Features per step: {X.shape[2]} (AccX, AccY, AccZ, Magnitude)")

np.savez_compressed(OUT_FILE, X=X, y=y)
print(f"Saved to {OUT_FILE}. Now run your 'split_normalize.py' on this file.")
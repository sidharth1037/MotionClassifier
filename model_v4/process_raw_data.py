import pandas as pd
import numpy as np
import os
import re

# ========== CONFIG ==========
DATA_DIR = "dataset" 
OUT_FILE = "motion_windows_v4.npz"
WINDOW_SIZE = 128      # INCREASED: 128 samples (approx 2.56 seconds @ 50Hz)
STEP_SIZE = 64         # 50% overlap 
# ============================

def load_data(data_dir):
    segments = []
    labels = []
    users = [] 
    
    class_map = {
        'walking': 0, 
        'upstairs': 1, 
        'downstairs': 2,
        'idle': 3
    }
    
    for label_name, label_idx in class_map.items():
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' not found. Skipping class {label_name}.")
            continue
            
        print(f"Processing {label_name}...")
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        for f in files:
            # Extract User ID
            user_match = re.search(r'(\d+)', f)
            user_id = int(user_match.group(1)) if user_match else 0
            
            file_path = os.path.join(folder_path, f)
            try:
                df = pd.read_csv(file_path)
                
                # Auto-detect columns
                cols = [c for c in df.columns if 'x' in c.lower() or 'y' in c.lower() or 'z' in c.lower()][:3]
                if len(cols) < 3: continue

                data = df[cols].values
                
                # Add Magnitude
                mag = np.linalg.norm(data, axis=1, keepdims=True)
                data = np.hstack((data, mag)) 

                # Sliding Window
                for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
                    window = data[i : i + WINDOW_SIZE]
                    segments.append(window)
                    labels.append(label_idx)
                    users.append(user_id) 
                    
            except Exception as e:
                print(f"Error reading {f}: {e}")

    return np.array(segments), np.array(labels), np.array(users)

# Run
X, y, users = load_data(DATA_DIR)
print(f"\nFinal Shape: X={X.shape}, y={y.shape}")
print(f"User IDs found: {np.unique(users)}") 
print(f"Features: AccX, AccY, AccZ, Magnitude")

np.savez_compressed(OUT_FILE, X=X, y=y, users=users)
print(f"Saved to {OUT_FILE}. Now run 'split_normalize'.")
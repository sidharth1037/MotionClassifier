import os
import pandas as pd
import numpy as np

# === CONFIG ===
DATA_FOLDER = "C:\\Users\\sidha\\Desktop\\Main Project\\raw_data"   # root folder
WINDOW_SIZE = 100          # number of samples per window (2s at 50Hz)
STEP_SIZE = 50             # overlap
OUTPUT_FILE = "C:\\Users\\sidha\\Desktop\\Main Project\\raw_data\\motion_windows.npz"

# Label map based on folder names
LABEL_MAP = {
    "walking": 0,
    "stairs_up": 1,
    "stairs_down": 2
}

def load_csv(file_path):
    """Load CSV into pandas dataframe."""
    df = pd.read_csv(file_path, header=0)
    return df

def create_windows(df, label):
    """Create sliding windows from dataframe."""
    data = df.values  # convert to numpy
    windows = []
    labels = []

    for start in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = data[start:end]
        windows.append(window)
        labels.append(label)

    return np.array(windows), np.array(labels)

all_windows = []
all_labels = []

# === Traverse subfolders ===
for folder, label in LABEL_MAP.items():
    folder_path = os.path.join(DATA_FOLDER, folder)
    if not os.path.exists(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            filepath = os.path.join(folder_path, file)

            # load csv
            df = load_csv(filepath)
            print(filepath, "NaN count:", df.isna().sum().sum())


            # remove "Time" column if exists
            if "Time (s)" in df.columns:
                df = df.drop(columns=["Time (s)"])

            # create windows
            X, y = create_windows(df, label)
            all_windows.append(X)
            all_labels.append(y)

# === Stack everything ===
X = np.vstack(all_windows)
y = np.hstack(all_labels)

print("Final dataset shape:", X.shape, y.shape)
# X.shape → (num_windows, WINDOW_SIZE, num_features)

# Save for training
np.savez(OUTPUT_FILE, X=X, y=y)
print(f"Saved dataset to {OUTPUT_FILE}")

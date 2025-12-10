import numpy as np
import pandas as pd
from keras.models import load_model
import os
import collections

# ================= CONFIG =================
MODEL_FILE = "motion_model_1.0.h5"
WINDOW_SIZE = 100
# Mean and std from your screenshot
mean = np.array([0.2693, 3.3516, 9.3094, -0.0018, 0.0002, 0.0031])
std  = np.array([0.9552, 1.1246, 2.2726, 0.3554, 0.2906, 0.2520])

# Paths to last CSV of each class
csv_files = {
    "walking": "stairs_up/up6.csv",
    "stairs_up": "walking/walking11.csv",
    "stairs_down": "stairs_down/down5.csv"
}
# ==========================================

# Load model
model = load_model(MODEL_FILE)
print("Loaded model:", MODEL_FILE)

for label, file_path in csv_files.items():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Load CSV
    df = pd.read_csv(file_path)
    if "Time (s)" in df.columns:
        df = df.drop(columns=["Time (s)"])
    data = df.values  # shape (num_samples, 6)

    # Create windows
    windows = []
    for i in range(len(data) - WINDOW_SIZE + 1):
        windows.append(data[i:i+WINDOW_SIZE])
    X_new = np.array(windows)

    # Normalize
    X_new = (X_new - mean) / std

    # Predict
    preds = model.predict(X_new)
    pred_classes = np.argmax(preds, axis=1)

    # Majority vote
    counter = collections.Counter(pred_classes)
    final_class = counter.most_common(1)[0][0]

    print(f"\nFile: {file_path}")
    print("Window prediction counts:", dict(counter))
    print(f"Final predicted class: {final_class} ({label})")

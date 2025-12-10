import numpy as np
import pandas as pd
from keras.models import load_model
import collections
import os

# ================= CONFIG =================
MODEL_FILE = "motion_model_1.0.h5"
WINDOW_SIZE = 100
# Mean and std from your training stats
mean = np.array([0.2693, 3.3516, 9.3094, -0.0018, 0.0002, 0.0031])
std  = np.array([0.9552, 1.1246, 2.2726, 0.3554, 0.2906, 0.2520])

CLASS_NAMES = {0: "walking", 1: "stairs_up", 2: "stairs_down"}

# Path to the CSV file to test
csv_file = "walking/walking12.csv"  # <-- set your file path here
# ==========================================

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file not found: {csv_file}")

# Load model
model = load_model(MODEL_FILE)
print("Loaded model:", MODEL_FILE)

# Load CSV
df = pd.read_csv(csv_file)
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
final_class_idx = counter.most_common(1)[0][0]
final_class_name = CLASS_NAMES[final_class_idx]

print(f"\nFile: {csv_file}")
print("Window prediction counts:", dict(counter))
print(f"Final predicted class: {final_class_name}")

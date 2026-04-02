"""
Run this AFTER training to generate the final model_meta_v7.json
with the correct mean/std values from the training dataset.

Usage: python generate_meta_json.py
"""
import numpy as np
import json
import os

DATASET_FILE = "motion_dataset_loso_v7.npz"
OUTPUT_FILE = "model_meta_v7.json"

assert os.path.exists(DATASET_FILE), f"ERROR: {DATASET_FILE} not found. Run split_normalize.py first."

data = np.load(DATASET_FILE)
mean = data["mean"].tolist()
std = data["std"].tolist()

# ==============================================================================
# V7 CHANGES:
# - feature_order now has 6 features (was 4)
# - Added units for new features
# - Updated input_shape to [1, 96, 6]
# - Updated model_version to 7.0
#
# IMPORTANT FOR ANDROID APP:
# Your Android code needs to compute JerkZ and AccZ_detrended in real-time:
#   JerkZ[t] = AccZ[t] - AccZ[t-1]  (first-order difference)
#   AccZ_detrended[t] = AccZ[t] - rolling_mean(AccZ, window=10)
# These must be computed BEFORE normalization (subtract mean, divide by std).
# ==============================================================================

meta = {
    "model_version": "7.0",
    "mean": [round(m, 4) for m in mean],
    "std": [round(s, 4) for s in std],
    "feature_order": ["acc_x", "acc_y", "acc_z", "acc_mag", "jerk_z", "acc_z_detrended"],
    "units": {
        "acc_x": "m/s^2",
        "acc_y": "m/s^2",
        "acc_z": "m/s^2",
        "acc_mag": "m/s^2",
        "jerk_z": "m/s^2/sample",
        "acc_z_detrended": "m/s^2"
    },
    "classes": ["walking", "upstairs", "downstairs", "idle"],
    "class_map": {
        "0": "walking",
        "1": "upstairs",
        "2": "downstairs",
        "3": "idle"
    },
    "window_size": 96,
    "step_size": 48,
    "sample_rate": 50,
    "input_shape": [1, 96, 6],
    "engineered_features": {
        "jerk_z": {
            "description": "First-order difference of Z-axis acceleration",
            "formula": "jerk_z[t] = acc_z[t] - acc_z[t-1], jerk_z[0] = 0"
        },
        "acc_z_detrended": {
            "description": "Z-axis with rolling mean subtracted (gravity removal)",
            "formula": "acc_z_detrended[t] = acc_z[t] - rolling_mean(acc_z, window=10)",
            "rolling_window": 10
        }
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"Generated {OUTPUT_FILE}")
print(f"  mean: {meta['mean']}")
print(f"  std:  {meta['std']}")
print(f"  features: {meta['feature_order']}")
print(f"  window_size: {meta['window_size']}")
print(f"  input_shape: {meta['input_shape']}")
print(f"\nCopy this file to your Android app's assets/model/ folder.")
print(f"\nIMPORTANT: Your Android app must compute jerk_z and acc_z_detrended")
print(f"in real-time before feeding data to the model. See 'engineered_features'")
print(f"in the JSON for formulas.")

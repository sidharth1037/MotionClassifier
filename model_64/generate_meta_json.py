"""
Run this AFTER training to generate the final model_meta_v6_64.json
with the correct mean/std values from the training dataset.

Usage: python generate_meta_json.py
"""
import numpy as np
import json
import os

DATASET_FILE = "motion_dataset_loso_v6_64.npz"
OUTPUT_FILE = "model_meta_v6_64.json"

assert os.path.exists(DATASET_FILE), f"ERROR: {DATASET_FILE} not found. Run split_normalize.py first."

data = np.load(DATASET_FILE)
mean = data["mean"].tolist()
std = data["std"].tolist()

meta = {
    "model_version": "6.0_64",
    "mean": [round(m, 4) for m in mean],
    "std": [round(s, 4) for s in std],
    "feature_order": ["acc_x", "acc_y", "acc_z", "acc_mag"],
    "units": {
        "acc_x": "m/s^2",
        "acc_y": "m/s^2",
        "acc_z": "m/s^2",
        "acc_mag": "m/s^2"
    },
    "classes": ["walking", "upstairs", "downstairs", "idle"],
    "class_map": {
        "0": "walking",
        "1": "upstairs",
        "2": "downstairs",
        "3": "idle"
    },
    "window_size": 64,
    "step_size": 32,
    "sample_rate": 50,
    "input_shape": [1, 64, 4]
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"Generated {OUTPUT_FILE}")
print(f"  mean: {meta['mean']}")
print(f"  std:  {meta['std']}")
print(f"  window_size: {meta['window_size']}")
print(f"  input_shape: {meta['input_shape']}")
print(f"\nCopy this file to your Android app's assets/model/ folder.")

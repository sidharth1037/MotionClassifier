# process_raw_data.py - Documentation

## Overview
This script processes raw accelerometer CSV files from the `dataset/` folder and converts them into windowed sensor data samples suitable for deep learning. It extracts overlapping windows of fixed size from each CSV file, adds a magnitude feature, and saves everything as a compressed NumPy archive.

## Why This Approach?

### 1. **Sliding Window Technique**
- **Why**: Raw sensor data is continuous, but neural networks need fixed-size inputs
- **Benefit**: A sliding window with overlap (50%) maximizes data utilization and creates more training samples from limited data
- Example: A 5-second recording (250 samples @ 50Hz) with 50% overlap generates ~4 windows instead of just 1

### 2. **Feature Engineering (Magnitude)**
- **Why**: The magnitude (norm) of acceleration captures total motion intensity independent of direction
- **Formula**: `magnitude = √(accX² + accY² + accZ²)`
- **Benefit**: Provides the model with a 4th feature that captures acceleration magnitude, improving pattern recognition

### 3. **Separate File Organization**
- **Why**: Organizing by class (walking/, upstairs/, downstairs/, idle/) makes data loading intuitive and allows easy validation
- **Benefit**: Easy to spot missing or imbalanced classes during preprocessing

## Key Parameters

### Dataset Configuration
```python
DATA_DIR = "dataset"              # Location of organized CSV files
OUT_FILE = "motion_windows_v5.npz"  # Output compressed archive
WINDOW_SIZE = 128                 # Samples per window (approx 2.56s @ 50Hz)
STEP_SIZE = 64                    # Stride between windows (50% overlap)
```

**What You Can Change:**

| Parameter | Impact | Recommendation |
|-----------|--------|-----------------|
| `WINDOW_SIZE` | Larger = more context, but fewer samples | 64-256 samples typical for activity recognition |
| `STEP_SIZE` | Smaller = more overlap = more training data | Usually set to WINDOW_SIZE / 2 for 50% overlap |
| `WINDOW_SIZE / STEP_SIZE` | Controls overlap percentage | Typical: 2-4x step size (50-75% overlap) |
| `DATA_DIR` | Path to dataset folder | Must contain subdirectories: walking/, upstairs/, downstairs/, idle/ |

### Class Mapping
```python
CLASS_MAP = {
    'walking': 0,
    'upstairs': 1,
    'downstairs': 2,
    'idle': 3
}
```
These indices must match throughout the pipeline. Change only if you add/remove activity classes.

## General Structure

### Input Format
```
dataset/
├── walking/
│   ├── walking1.csv
│   ├── walking2.csv
│   └── ...
├── upstairs/
│   ├── upstairs1.csv
│   └── ...
├── downstairs/
├── idle/
```

**CSV Format**: Must contain acceleration columns (X, Y, Z)
```
x,y,z              # or AccX, AccY, AccZ - column names are auto-detected
1.2, 0.5, 9.8
1.3, 0.4, 9.7
...
```

### Processing Pipeline
```
1. Iterate each activity class folder
2. For each CSV file:
   a. Load data and extract X, Y, Z columns
   b. Calculate magnitude feature
   c. Create sliding windows
   d. Store with label and user ID
3. Save all windows to compressed archive
```

### Output Format
**File**: `motion_windows_v5.npz`
```python
X = shape (n_samples, 128, 4)      # 128 timesteps, 4 features (AccX, Y, Z, Magnitude)
y = shape (n_samples,)              # Class labels (0-3)
users = shape (n_samples,)          # User ID for each sample (for LOSO splitting)
```

## Important Considerations

### Data Quality Checks
1. **Minimum Length**: Files shorter than `WINDOW_SIZE` are skipped
   - This prevents window extraction errors
   - Consider preprocessing to concat very short files if data is scarce

2. **NaN Handling**: NaN values are replaced with 0.0
   - May mask data quality issues
   - Check raw data for corruption if many NaNs appear

3. **User ID Extraction**: Automatically extracts first digit(s) from filename
   - Example: `walking1.csv` → user_id = 1
   - Example: `user_05_walking.csv` → user_id = 5
   - **Critical for LOSO validation** (next step in pipeline)

### Optimization Tips

**To increase dataset size:**
- Decrease `STEP_SIZE` (more overlap = more windows)
- Ensure datasets folder has balanced classes
- Collect more recordings

**To reduce memory:**
- Increase `STEP_SIZE` (less overlap)
- Remove `users` array if you don't need LOSO validation
- Use downsampling before windowing

## Debugging

### Common Issues

**"No CSV files found in 'path'"**
- Check folder names match CLASS_MAP keys exactly
- Verify CSV files exist in subfolders

**"Too short (X samples < 128)"**
- Recording is too short for 128-sample window
- Either use shorter WINDOW_SIZE or collect longer recordings

**"Found only 1 acceleration columns (need 3)"**
- CSV doesn't have proper X, Y, Z columns
- Check column names in CSV header

**Few windows per user**
- Increase STEP_SIZE (less overlap)
- Collect longer duration recordings

## Next Steps
After running this script:
1. Run `split_normalize.py` to perform LOSO splitting and normalization
2. Then run `model_v5.py` to train the model

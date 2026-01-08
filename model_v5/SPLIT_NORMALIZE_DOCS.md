# split_normalize.py - Documentation

## Overview
This script splits the windowed sensor data into training and test sets using **Leave-One-Subject-Out (LOSO)** validation, then normalizes the features to zero-mean and unit-variance. It also converts labels to one-hot encoding. The result is a properly preprocessed dataset ready for neural network training.

## Why This Approach?

### 1. **Leave-One-Subject-Out (LOSO) Validation**
- **Why**: LOSO ensures the model evaluates on a completely unseen person's data
- **Problem It Solves**: Traditional random train/test splits leak user-specific patterns
  - If User 1 is in both train and test, the model just memorizes User 1's style
  - LOSO guarantees zero user overlap
- **How It Works**: 
  - One user's entire dataset → Test set
  - All other users' data → Training set
  - Model trained on 90% of users, tested on 1 held-out user
- **Benefit**: Simulates real-world deployment where the model encounters new people

### 2. **Feature Normalization (Z-score)**
- **Why**: Neural networks train faster and more stably with normalized inputs
- **Formula**: `X_normalized = (X - mean) / std`
- **Key Point**: Mean and std are computed from **training data only**
  - This prevents data leakage (test set statistics don't inform training)
- **Benefit**: 
  - Reduces internal covariate shift
  - Allows higher learning rates
  - Prevents features with large ranges from dominating

### 3. **One-Hot Encoding**
- **Why**: Neural network classification requires categorical output format
- **Example**: Class 1 (upstairs) becomes [0, 1, 0, 0]
- **Benefit**: Enables softmax cross-entropy loss calculation

## Key Parameters

### Configuration
```python
DATA_FILE = "motion_windows_v5.npz"        # Input from process_raw_data.py
OUT_FILE = "motion_dataset_loso_v5.npz"    # Output for model_v5.py
NUM_CLASSES = 4                             # Activities: Walking, Upstairs, Downstairs, Idle
MIN_TEST_SAMPLES = 50                       # Warning if test set smaller than this
MIN_CLASSES_IN_TEST = 3                     # Warning if test set missing this many classes
```

**What You Can Change:**

| Parameter | Impact | Notes |
|-----------|--------|-------|
| `NUM_CLASSES` | Changes one-hot encoding shape | Must match your class count (currently 4) |
| `MIN_TEST_SAMPLES` | Warning threshold only | Doesn't affect splitting, just alerts you |
| `MIN_CLASSES_IN_TEST` | Warning threshold only | Catches incomplete test sets |

⚠️ **Important**: `NUM_CLASSES` and `MIN_*` are warnings, not hard limits. The script runs even if triggered.

## General Structure

### Input Format
**File**: `motion_windows_v5.npz` (output from process_raw_data.py)
```python
X.shape = (n_windows, 128, 4)      # n_windows samples, 128 timesteps, 4 features
y.shape = (n_windows,)              # Class labels (0, 1, 2, 3)
users.shape = (n_windows,)          # User IDs for each window
```

### Processing Pipeline

```
1. Load windowed data (X, y, users)
   
2. SPLIT - Leave-One-Subject-Out Strategy:
   a. Identify all unique users from users array
   b. Select best test user:
      - Prefer user with all 4 activities
      - Fallback: user with most diverse activities
   c. Create masks:
      - train_mask = (users != selected_user)
      - test_mask = (users == selected_user)
   d. Split X and y using masks
   
3. NORMALIZE - Z-score normalization:
   a. Compute mean/std from training set only
   b. Apply to both training and test: (X - mean) / std
   c. Save mean/std for later inference
   
4. ENCODE - One-hot label encoding:
   a. Convert y (0,1,2,3) to one-hot (4-dimensional)
   b. Example: 1 → [0,1,0,0]
   
5. SAVE - Compressed archive with all components
```

### Output Format
**File**: `motion_dataset_loso_v5.npz`
```python
X_train = shape (n_train, 128, 4)           # Normalized training features
X_test = shape (n_test, 128, 4)             # Normalized test features
y_train = shape (n_train, 4)                # One-hot encoded training labels
y_test = shape (n_test, 4)                  # One-hot encoded test labels
mean = shape (4,)                           # Feature means for normalization
std = shape (4,)                            # Feature standard deviations
classes = shape (4,)                        # Unique class labels [0,1,2,3]
```

## Important Considerations

### User Selection Logic
The script intelligently selects which user becomes the test set:

```python
# Best case: User has all 4 activities
if activity_count == 4:
    use_this_user_as_test = True
    break  # Stop here, this is ideal

# Fallback: Use user with most diverse activities
if activity_count > current_best:
    current_best = activity_count
```

**Why This Matters**: 
- If test user missing an activity (e.g., no "Idle" data), the model can't be evaluated on it
- Script warns but continues (you may need to collect more data)

### Normalization Details

**Critical Point**: Mean/std are computed from training set ONLY
```python
train_flat = X_train.reshape(-1, 4)  # Flatten to 2D
mean = train_flat.mean(axis=0)       # Stats from train only!
std = train_flat.std(axis=0)         # This prevents data leakage

# Apply same normalization to test
X_test_normalized = (X_test - mean) / std  # Use training statistics
```

**Why**: Ensures test set normalization isn't influenced by test data statistics.

### Data Quality Warnings

The script issues warnings for:
```
⚠ WARNING: Test set very small (N < 50)
```
- Indicates limited test data for reliable evaluation
- Solution: Use smaller STEP_SIZE in process_raw_data.py to generate more windows

```
⚠ WARNING: Test set missing X classes
```
- Test user doesn't have all activity types
- Solution: Ensure all users in raw data collected all activities

### One-Hot Encoding Mapping

```
Class Index → One-Hot Vector
0 (Walking) → [1, 0, 0, 0]
1 (Upstairs) → [0, 1, 0, 0]
2 (Downstairs) → [0, 0, 1, 0]
3 (Idle) → [0, 0, 0, 1]
```

## Debugging

### Common Issues

**"ERROR: motion_windows_v5.npz not found"**
- Run process_raw_data.py first
- Verify output filename matches input filename here

**Test distribution shows missing classes (✗)**
- Test user doesn't have recordings for all activities
- Impact: Model can't be evaluated on that activity
- Solution: Collect data from that user or adjust STEP_SIZE to get more windows

**"Test set very small (N < 50)"**
- Limited windows for test set evaluation
- Less reliable performance estimate
- Solution: Increase data in process_raw_data.py or decrease STEP_SIZE

**Normalization values look extreme** (e.g., std = 1000)
- Check raw data for outliers or measurement errors
- Consider clipping extreme values before windowing

### Verification Checklist

After running, verify:
1. ✓ Training samples >> Test samples (typically 80/20 split or better)
2. ✓ All 4 classes present in both train and test
3. ✓ Feature means near 0, stds near 1 (after normalization)
4. ✓ No NaN values in output arrays

```python
# Quick verification
data = np.load("motion_dataset_loso_v5.npz")
print(data["X_train"].mean())  # Should be close to 0
print(data["X_train"].std())   # Should be close to 1
```

## Advanced Customization

### Alternative Split Strategies

**If you want random train/test instead of LOSO:**
Modify the split section:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```
⚠️ Not recommended for human activity recognition (user-specific patterns leak!)

### Alternative Normalization

**StandardScaler-compatible format:**
```python
# Current: Z-score normalization
X_normalized = (X - mean) / std

# Alternative: Min-Max (0-1 range)
X_minmax = (X - min) / (max - min)

# Alternative: Robust (resistant to outliers)
X_robust = (X - median) / IQR
```

## Next Steps
After running this script:
1. Verify output file `motion_dataset_loso_v5.npz` exists
2. Run `model_v5.py` to train the neural network
3. The model will automatically load and use this preprocessed data
